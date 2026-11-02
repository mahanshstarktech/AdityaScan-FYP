"""
Dynamic Active Region Coronal Arcade Graph Builder for AdityaScan.

Physics:
  Coronal loops interconnect bipolar active regions across the solar photosphere.
  Magnetic connectivity decays exponentially with great-circle angular distance:
    w_ij = exp(-Δθ / θ_0) * (1 + κ * P_ij)
  where θ_0 ≈ 30° is the characteristic coronal arcade scale, and P_ij is the
  polarity coupling (+1 for complementary bipolar pairs, -1 for matching).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

SOLAR_RADIUS_KM = 6.957e5  # R_sun
CHARACTERISTIC_ARCADE_DEG = 30.0


@dataclass
class ActiveRegionNode:
    ar_id: int
    latitude_deg: float
    longitude_deg: float
    area_msh: float
    flux_density_gauss: float
    polarity: int  # -1 (neg), +1 (pos), 0 (neutral)
    features: List[float]


@dataclass
class CoronalArcadeEdge:
    source_ar: int
    target_ar: int
    angular_distance_deg: float
    physical_distance_km: float
    arcade_coupling_weight: float
    polarity_product: float


class ActiveRegionGraphBuilder:
    """
    Constructs multi-relational graphs on the visible solar disk.
    """

    def __init__(
        self,
        max_angular_deg: float = 75.0,
        arcade_scale_deg: float = CHARACTERISTIC_ARCADE_DEG,
        polarity_coupling_factor: float = 0.35,
    ) -> None:
        self.max_angular_deg = max_angular_deg
        self.arcade_scale_deg = arcade_scale_deg
        self.polarity_coupling = polarity_coupling_factor

    def compute_angular_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Great-circle angular separation on solar sphere (degrees)."""
        p1, l1 = math.radians(lat1), math.radians(lon1)
        p2, l2 = math.radians(lat2), math.radians(lon2)
        cos_d = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(l1 - l2)
        cos_d = max(-1.0, min(1.0, cos_d))
        return math.degrees(math.acos(cos_d))

    def build_graph(
        self, active_regions: List[ActiveRegionNode]
    ) -> Tuple[np.ndarray, np.ndarray, List[CoronalArcadeEdge]]:
        """
        Returns:
          node_feature_matrix: [N, D]
          adjacency_matrix: [N, N]
          edges: list of CoronalArcadeEdge
        """
        n = len(active_regions)
        if n == 0:
            return np.zeros((0, 21), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), []

        feature_dim = len(active_regions[0].features)
        node_feats = np.zeros((n, feature_dim), dtype=np.float32)
        for i, node in enumerate(active_regions):
            node_feats[i] = np.array(node.features, dtype=np.float32)

        adj = np.eye(n, dtype=np.float32)
        edges: List[CoronalArcadeEdge] = []

        for i in range(n):
            for j in range(i + 1, n):
                ar_i = active_regions[i]
                ar_j = active_regions[j]
                dist_deg = self.compute_angular_distance(
                    ar_i.latitude_deg, ar_i.longitude_deg, ar_j.latitude_deg, ar_j.longitude_deg
                )

                if dist_deg <= self.max_angular_deg:
                    dist_km = math.radians(dist_deg) * SOLAR_RADIUS_KM
                    pol_prod = float(ar_i.polarity * ar_j.polarity)
                    decay = math.exp(-dist_deg / self.arcade_scale_deg)
                    # Opposing polarities establish stronger reconnection arcades
                    weight = decay * (1.0 + self.polarity_coupling * (-pol_prod))

                    adj[i, j] = weight
                    adj[j, i] = weight

                    edges.append(
                        CoronalArcadeEdge(
                            source_ar=ar_i.ar_id,
                            target_ar=ar_j.ar_id,
                            angular_distance_deg=round(dist_deg, 2),
                            physical_distance_km=round(dist_km, 1),
                            arcade_coupling_weight=round(weight, 4),
                            polarity_product=pol_prod,
                        )
                    )

        return node_feats, adj, edges
