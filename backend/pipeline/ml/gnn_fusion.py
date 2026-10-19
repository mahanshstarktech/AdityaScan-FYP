"""
Spatial-Temporal Multi-Modal Fusion Engine.
Couples Active Region Graph Neural Network (GNN) embeddings with TCN-BiLSTM temporal representations.
"""
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import SympatheticFlareGNN

class CrossModalAttentionGating(nn.Module):
    """Bidirectional cross-attention gating between temporal lightcurve features and spatial GNN topology."""
    def __init__(self, temporal_dim: int = 128, spatial_dim: int = 128, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.attn_temp_to_spat = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.proj_temp = nn.Linear(temporal_dim, hidden_dim)
        self.proj_spat = nn.Linear(spatial_dim, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, temporal_emb: torch.Tensor, spatial_emb: torch.Tensor) -> torch.Tensor:
        if temporal_emb.dim() == 2: temporal_emb = temporal_emb.unsqueeze(1)
        if spatial_emb.dim() == 2: spatial_emb = spatial_emb.unsqueeze(1)
        t_proj = self.proj_temp(temporal_emb)
        s_proj = self.proj_spat(spatial_emb)
        attn_out, _ = self.attn_temp_to_spat(query=t_proj, key=s_proj, value=s_proj)
        g = self.gate(torch.cat([t_proj, attn_out], dim=-1))
        fused = g * t_proj + (1.0 - g) * attn_out
        return self.norm(fused.squeeze(1))

class SolarFlareSpatialTemporalFusion(nn.Module):
    """End-to-end multi-modal architecture with 5 forecasting heads."""
    HORIZONS = ["30m", "1h", "3h", "6h", "24h"]

    def __init__(self, node_in_dim: int = 21, edge_dim: int = 4, temporal_in_dim: int = 21, hidden_dim: int = 128):
        super().__init__()
        self.gnn = SympatheticFlareGNN(node_in_dim=node_in_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, num_layers=3)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.cross_modal_gate = CrossModalAttentionGating(temporal_dim=hidden_dim, spatial_dim=hidden_dim, hidden_dim=hidden_dim)
        self.forecast_heads = nn.ModuleDict({
            h: nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 3),  # [No-Flare, M-Class, X-Class]
            ) for h in self.HORIZONS
        })

    def forward(self, node_feats, edge_index, edge_attr, temporal_series, batch=None) -> Dict[str, torch.Tensor]:
        spatial_emb = self.gnn(node_feats, edge_index, edge_attr, batch=batch)
        if temporal_series.dim() == 3:
            temporal_repr = self.temporal_encoder(temporal_series[:, -1, :])
        else:
            temporal_repr = self.temporal_encoder(temporal_series)
        fused_emb = self.cross_modal_gate(temporal_repr, spatial_emb)
        return {h: self.forecast_heads[h](fused_emb) for h in self.HORIZONS}
