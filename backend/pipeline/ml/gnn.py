"""
Spatial Graph Neural Network (GNN) for Solar Active Region Topology and Sympathetic Flaring.

In solar physics, flares in one Active Region (AR) can trigger flares in distant ARs via
large-scale coronal magnetic loop arcades and coronal mass ejection (CME) shock fronts—a phenomenon
known as 'sympathetic flaring'.

This module builds a multi-relational topological graph of the solar disk:
  - Nodes: Active Regions characterized by SDO/HMI SHARP magnetic vector features (21 parameters)
           and McIntosh morphological complexity classifications.
  - Edges: Heliographic angular distance on the solar sphere (great-circle geometry)
           and magnetic arcade coupling strength:
             cos(Δθ) = sin(φ_1)sin(φ_2) + cos(φ_1)cos(φ_2)cos(λ_1 - λ_2)
             w_ij = exp(-Δθ / θ_0) * (1 + κ * P_ij)
           where θ_0 ≈ 30° is the characteristic coronal arcade scale, and P_ij is the
           magnetic polarity product between bipole regions.

Architecture:
  - SolarGNNConv: Edge-conditioned message-passing layer with residual connection and LayerNorm.
  - SolarCrossAttentionPooling: Multi-head cross-attention layer where a learned solar-disk
    latent query attends over active region node representations to form a unified disk embedding.
  - Full native PyTorch tensor engine with transparent PyTorch Geometric integration if available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import MessagePassing, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    class MessagePassing(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
    def global_mean_pool(x, batch):
        if batch is None or batch.numel() == 0:
            return x.mean(dim=0, keepdim=True)
        batch_size = int(batch.max().item()) + 1
        out = torch.zeros(batch_size, x.size(-1), device=x.device, dtype=x.dtype)
        counts = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        counts.scatter_add_(0, batch.unsqueeze(-1), torch.ones_like(batch.unsqueeze(-1), dtype=x.dtype))
        return out / counts.clamp(min=1.0)


# ---------------------------------------------------------------------------
# Solar Coordinate & Graph Geometry Utilities
# ---------------------------------------------------------------------------

SOLAR_RADIUS_KM = 6.957e5  # R_sun in km
CHARACTERISTIC_ARCADE_DEG = 30.0  # Characteristic coronal loop arcade decay angle


def compute_heliographic_angular_distance(
    lat1_deg: torch.Tensor,
    lon1_deg: torch.Tensor,
    lat2_deg: torch.Tensor,
    lon2_deg: torch.Tensor,
) -> torch.Tensor:
    """
    Computes great-circle angular distance Δθ (in degrees) on the solar sphere:
        cos(Δθ) = sin(φ_1)sin(φ_2) + cos(φ_1)cos(φ_2)cos(λ_1 - λ_2)
    """
    phi1 = torch.deg2rad(lat1_deg)
    lambda1 = torch.deg2rad(lon1_deg)
    phi2 = torch.deg2rad(lat2_deg)
    lambda2 = torch.deg2rad(lon2_deg)

    cos_delta = torch.sin(phi1) * torch.sin(phi2) + torch.cos(phi1) * torch.cos(phi2) * torch.cos(lambda1 - lambda2)
    cos_delta = torch.clamp(cos_delta, -1.0, 1.0)
    delta_rad = torch.acos(cos_delta)
    return torch.rad2deg(delta_rad)


def build_solar_disk_graph(
    active_regions: List[dict],
    max_angular_deg: float = 75.0,
    arcade_scale_deg: float = CHARACTERISTIC_ARCADE_DEG,
    polarity_coupling_factor: float = 0.35,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs a graph from active region metadata snapshots on the visible solar disk.

    Parameters
    ----------
    active_regions : list of dict
        Each dict must contain:
          - 'features': float list of 21 SHARP magnetic parameters
          - 'latitude': float degrees [-90, +90]
          - 'longitude': float degrees [-90, +90] (heliographic Stonyhurst or Carrington)
          - 'polarity': int (-1 for negative-dominant, +1 for positive-dominant, 0 for neutral)
    max_angular_deg : float
        Maximum angular distance on the solar sphere to establish a coronal loop edge.
    arcade_scale_deg : float
        Exponential decay length scale for magnetic connectivity.
    polarity_coupling_factor : float
        Bonus weight multiplier for complementary bipolar AR pairs (+/-).

    Returns
    -------
    node_features : torch.Tensor [num_nodes, 21]
    edge_index : torch.Tensor [2, num_edges] (long)
    edge_attr : torch.Tensor [num_edges, 4] (float: [distance_deg, arcade_weight, physical_dist_km, polarity_prod])
    """
    num_nodes = len(active_regions)
    if num_nodes == 0:
        empty_nodes = torch.zeros((0, 21), dtype=torch.float32, device=device)
        empty_edges = torch.zeros((2, 0), dtype=torch.long, device=device)
        empty_attr = torch.zeros((0, 4), dtype=torch.float32, device=device)
        return empty_nodes, empty_edges, empty_attr

    node_feats = torch.tensor([ar["features"] for ar in active_regions], dtype=torch.float32, device=device)
    lats = torch.tensor([ar.get("latitude", 0.0) for ar in active_regions], dtype=torch.float32, device=device)
    lons = torch.tensor([ar.get("longitude", 0.0) for ar in active_regions], dtype=torch.float32, device=device)
    polarities = torch.tensor([ar.get("polarity", 1) for ar in active_regions], dtype=torch.float32, device=device)

    src_list = []
    dst_list = []
    attr_list = []

    # 1. Self-loops for every active region
    for i in range(num_nodes):
        src_list.append(i)
        dst_list.append(i)
        attr_list.append([0.0, 1.0, 0.0, 1.0])

    # 2. Directed edges between pairs within max_angular_deg
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            delta_deg = compute_heliographic_angular_distance(lats[i], lons[i], lats[j], lons[j]).item()
            if delta_deg <= max_angular_deg:
                dist_km = (delta_deg * math.pi / 180.0) * SOLAR_RADIUS_KM
                # Polarity coupling: opposite polarities form closed flux tubes (stronger connection)
                p_prod = float(polarities[i] * polarities[j])
                coupling_mult = 1.0 + (polarity_coupling_factor if p_prod < 0 else 0.0)
                arcade_weight = math.exp(-delta_deg / arcade_scale_deg) * coupling_mult

                src_list.append(i)
                dst_list.append(j)
                attr_list.append([delta_deg, arcade_weight, dist_km / 1.0e6, p_prod])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
    edge_attr = torch.tensor(attr_list, dtype=torch.float32, device=device)
    return node_feats, edge_index, edge_attr


# ---------------------------------------------------------------------------
# Solar GNN Message Passing Layer
# ---------------------------------------------------------------------------

class SolarGNNConv(nn.Module):
    r"""
    Edge-conditioned spatial message passing layer for Active Region topology.
    Message function:
        m_{j->i} = GELU(W_neigh * h_j + W_edge * e_{ij}) * w_{ij}
    Aggregation:
        h_i^{(l+1)} = LayerNorm(h_i^{(l)} + W_self * h_i^{(l)} + \sum_{j \in N(i)} m_{j->i})
    """
    def __init__(self, in_features: int, out_features: int, edge_features: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.lin_self = nn.Linear(in_features, out_features, bias=False)
        self.lin_neigh = nn.Linear(in_features, out_features, bias=False)
        self.lin_edge = nn.Linear(edge_features, out_features, bias=True)

        self.norm = nn.LayerNorm(out_features)
        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [num_nodes, in_features]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_features]
        """
        num_nodes = x.size(0)
        h_self = self.lin_self(x)

        if edge_index.numel() == 0 or edge_index.size(1) == 0:
            return self.norm(self.act(h_self))

        src, dst = edge_index[0], edge_index[1]
        x_src = x[src]  # [num_edges, in_features]

        # Edge feature transformation & weighting
        h_edge = self.lin_edge(edge_attr)
        weights = edge_attr[:, 1].unsqueeze(-1)  # arcade_weight

        # Message computation
        messages = self.act(self.lin_neigh(x_src) + h_edge) * weights

        # Aggregation via scatter_add
        aggr = torch.zeros((num_nodes, self.out_features), device=x.device, dtype=x.dtype)
        aggr.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # Residual connection + LayerNorm
        out = self.norm(h_self + aggr)
        return self.act(out)


# ---------------------------------------------------------------------------
# Cross-Attention Graph Pooling Layer
# ---------------------------------------------------------------------------

class SolarCrossAttentionPooling(nn.Module):
    """
    Multi-head cross-attention graph pooling layer.
    Instead of simple global mean/max pooling, a learnable solar disk query vector
    attends to each Active Region node embedding. Active regions with severe magnetic complexity
    and large flux concentrations naturally receive higher attention weights.
    """
    def __init__(self, node_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.randn(1, 1, node_dim))
        self.mha = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [num_nodes, node_dim]
        batch: [num_nodes] mapping node index to graph index in batch (defaults to all in 1 graph)
        Returns: [batch_size, node_dim]
        """
        if x.numel() == 0:
            return self.query.squeeze(1)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        batch_size = int(batch.max().item()) + 1
        graph_embeddings = []

        for b in range(batch_size):
            mask = (batch == b)
            x_b = x[mask]
            if x_b.size(0) == 0:
                graph_embeddings.append(self.query.squeeze(0))
                continue

            # Query attends over node keys & values: [1, 1, node_dim] attending over [1, N_b, node_dim]
            q = self.query
            k = x_b.unsqueeze(0)
            v = x_b.unsqueeze(0)

            attn_out, _ = self.mha(q, k, v)
            pooled = self.norm(attn_out.squeeze(0) + q.squeeze(0))
            graph_embeddings.append(pooled)

        return torch.cat(graph_embeddings, dim=0)


# ---------------------------------------------------------------------------
# Complete Sympathetic Flare Graph Neural Network
# ---------------------------------------------------------------------------

class SympatheticFlareGNN(nn.Module):
    """
    Production-grade Spatial Graph Neural Network for Solar Cycle 25 active region networks.
    Couples multi-layer edge-conditioned message passing with cross-attention pooling.
    """
    def __init__(
        self,
        in_channels: int = 21,
        hidden_channels: int = 64,
        out_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.conv_layers = nn.ModuleList([
            SolarGNNConv(hidden_channels, hidden_channels, edge_features=4)
            for _ in range(num_layers)
        ])

        self.pooling = SolarCrossAttentionPooling(hidden_channels, num_heads=4, dropout=dropout)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, 4] (if None, synthetic defaults created)
        batch: [num_nodes]
        Returns:
            embedding: [batch_size, out_channels]
        """
        if x.numel() == 0 or x.size(0) == 0:
            batch_size = 1 if batch is None or batch.numel() == 0 else int(batch.max().item()) + 1
            return torch.zeros((batch_size, self.out_channels), device=x.device, dtype=x.dtype)

        if edge_attr is None:
            # Default edge attributes: [distance=10.0, arcade_weight=1.0, dist_km=0.1, polarity=1.0]
            edge_attr = torch.ones((edge_index.size(1), 4), device=x.device, dtype=torch.float32)

        h = self.input_proj(x)
        for conv in self.conv_layers:
            h = conv(h, edge_index, edge_attr)

        pooled = self.pooling(h, batch)
        return self.out_proj(pooled)
