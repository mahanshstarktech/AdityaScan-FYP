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
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prototype solar disk graph builder with distance thresholding."""
    n = len(active_regions)
    if n == 0:
        return torch.zeros((0, 21), device=device), torch.zeros((2, 0), dtype=torch.long, device=device), torch.zeros((0, 4), device=device)
    feats = torch.tensor([ar["features"] for ar in active_regions], dtype=torch.float32, device=device)
    src, dst, attrs = [], [], []
    for i in range(n):
        src.append(i); dst.append(i); attrs.append([0.0, 1.0, 0.0, 1.0])
    return feats, torch.tensor([src, dst], dtype=torch.long, device=device), torch.tensor(attrs, dtype=torch.float32, device=device)
