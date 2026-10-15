"""
Spatial Graph Neural Network (GNN) for Solar Active Region Topology.
Nodes: Active Regions characterized by SDO/HMI SHARP magnetic vector features.
Edges: Heliographic angular distance and magnetic arcade connectivity.
"""
import math
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

SOLAR_RADIUS_KM = 6.957e5
CHARACTERISTIC_ARCADE_DEG = 30.0

def compute_heliographic_angular_distance(
    lat1_deg: torch.Tensor,
    lon1_deg: torch.Tensor,
    lat2_deg: torch.Tensor,
    lon2_deg: torch.Tensor,
) -> torch.Tensor:
    """Computes spherical great-circle separation angle in degrees."""
    phi1, lambda1 = torch.deg2rad(lat1_deg), torch.deg2rad(lon1_deg)
    phi2, lambda2 = torch.deg2rad(lat2_deg), torch.deg2rad(lon2_deg)
    cos_delta = torch.clamp(
        torch.sin(phi1) * torch.sin(phi2) + torch.cos(phi1) * torch.cos(phi2) * torch.cos(lambda1 - lambda2),
        -1.0, 1.0
    )
    return torch.rad2deg(torch.acos(cos_delta))

def build_solar_disk_graph(
    active_regions: List[dict],
    max_angular_deg: float = 75.0,
    arcade_scale_deg: float = CHARACTERISTIC_ARCADE_DEG,
    polarity_coupling_factor: float = 0.35,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constructs topological graph with magnetic arcade connectivity decay."""
    num_nodes = len(active_regions)
    if num_nodes == 0:
        return torch.zeros((0, 21), device=device), torch.zeros((2, 0), dtype=torch.long, device=device), torch.zeros((0, 4), device=device)
    node_feats = torch.tensor([ar["features"] for ar in active_regions], dtype=torch.float32, device=device)
    lats = torch.tensor([ar.get("latitude", 0.0) for ar in active_regions], dtype=torch.float32, device=device)
    lons = torch.tensor([ar.get("longitude", 0.0) for ar in active_regions], dtype=torch.float32, device=device)
    polarities = torch.tensor([ar.get("polarity", 1) for ar in active_regions], dtype=torch.float32, device=device)
    src_list, dst_list, attr_list = [], [], []
    for i in range(num_nodes):
        src_list.append(i); dst_list.append(i); attr_list.append([0.0, 1.0, 0.0, 1.0])
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j: continue
            delta_deg = compute_heliographic_angular_distance(lats[i], lons[i], lats[j], lons[j]).item()
            if delta_deg <= max_angular_deg:
                dist_km = (delta_deg * math.pi / 180.0) * SOLAR_RADIUS_KM
                p_prod = float(polarities[i] * polarities[j])
                coupling = 1.0 + (polarity_coupling_factor if p_prod < 0 else 0.0)
                arcade_weight = math.exp(-delta_deg / arcade_scale_deg) * coupling
                src_list.append(i); dst_list.append(j); attr_list.append([delta_deg, arcade_weight, dist_km / 1e6, p_prod])
    return node_feats, torch.tensor([src_list, dst_list], dtype=torch.long, device=device), torch.tensor(attr_list, dtype=torch.float32, device=device)

class SolarGNNConv(nn.Module):
    r"""Edge-conditioned spatial message passing layer for Active Region topology."""
    def __init__(self, in_features: int, out_features: int, edge_features: int = 4):
        super().__init__()
        self.lin_self = nn.Linear(in_features, out_features, bias=False)
        self.lin_neigh = nn.Linear(in_features, out_features, bias=False)
        self.lin_edge = nn.Linear(edge_features, out_features, bias=True)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        h_self = self.lin_self(x)
        if edge_index.numel() == 0 or edge_index.size(1) == 0:
            return self.norm(self.act(h_self))
        src, dst = edge_index[0], edge_index[1]
        x_src = x[src]
        h_edge = self.lin_edge(edge_attr)
        weights = edge_attr[:, 1].unsqueeze(-1)
        messages = self.act(self.lin_neigh(x_src) + h_edge) * weights
        aggr = torch.zeros((num_nodes, h_self.size(1)), device=x.device, dtype=x.dtype)
        aggr.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        return self.act(self.norm(h_self + aggr))
