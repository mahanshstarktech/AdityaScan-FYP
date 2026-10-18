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
