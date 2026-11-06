import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from pipeline.ml.active_region_graph_builder import (
    ActiveRegionGraphBuilder,
    ActiveRegionNode,
)
from pipeline.ml.loss import FocalBrierLossCalculator


def test_angular_distance():
    builder = ActiveRegionGraphBuilder()
    # Identical coordinates -> 0 distance
    d0 = builder.compute_angular_distance(0.0, 0.0, 0.0, 0.0)
    assert pytest.approx(d0, abs=1e-4) == 0.0

    # 90 degrees separation
    d90 = builder.compute_angular_distance(0.0, 0.0, 90.0, 0.0)
    assert pytest.approx(d90, abs=1e-4) == 90.0


def test_active_region_graph_construction():
    builder = ActiveRegionGraphBuilder(max_angular_deg=60.0)

    # Empty test
    nodes, adj, edges = builder.build_graph([])
    assert len(nodes) == 0
    assert len(edges) == 0

    # 2 Active regions: AR13664 (S19W42) and AR13668 (S15W10)
    ar1 = ActiveRegionNode(
        ar_id=13664, latitude_deg=-19.0, longitude_deg=-42.0, area_msh=1200.0,
        flux_density_gauss=850.0, polarity=1, features=[1.0] * 21
    )
    ar2 = ActiveRegionNode(
        ar_id=13668, latitude_deg=-15.0, longitude_deg=-10.0, area_msh=650.0,
        flux_density_gauss=450.0, polarity=-1, features=[2.0] * 21
    )

    nodes, adj, edges = builder.build_graph([ar1, ar2])
    assert nodes.shape == (2, 21)
    assert adj.shape == (2, 2)
    assert adj[0, 1] > 0.0
    assert adj[0, 1] == adj[1, 0]
    assert len(edges) == 1
    edge = edges[0]
    assert edge.polarity_product == -1.0
    assert edge.angular_distance_deg < 60.0


def test_focal_brier_loss():
    loss_calc = FocalBrierLossCalculator(gamma=2.0, brier_weight=0.20)
    # Perfect predictions -> loss should be very small
    preds_perfect = np.array([[0.99, 0.01], [0.01, 0.99]])
    labels = np.array([0, 1])
    loss_perf, metrics_perf = loss_calc.compute_loss(preds_perfect, labels)
    assert loss_perf < 0.05

    # Incorrect confident predictions -> high loss
    preds_wrong = np.array([[0.01, 0.99], [0.99, 0.01]])
    loss_wrong, metrics_wrong = loss_calc.compute_loss(preds_wrong, labels)
    assert loss_wrong > loss_perf
