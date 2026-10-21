import math
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

torch = pytest.importorskip("torch")

from pipeline.ml.gnn import (
    compute_heliographic_angular_distance,
    build_solar_disk_graph,
    SolarGNNConv,
    SolarCrossAttentionPooling,
    SympatheticFlareGNN,
)
from pipeline.ml.gnn_fusion import (
    CrossModalAttentionGating,
    SolarFlareSpatialTemporalFusion,
    MultiHorizonSolarLoss,
)


def test_heliographic_angular_distance():
    # Coincident points -> 0 deg
    lat1 = torch.tensor([0.0])
    lon1 = torch.tensor([0.0])
    d0 = compute_heliographic_angular_distance(lat1, lon1, lat1, lon1)
    assert pytest.approx(d0.item(), abs=1e-4) == 0.0

    # 90 degrees latitude separation along prime meridian
    lat2 = torch.tensor([90.0])
    lon2 = torch.tensor([0.0])
    d90 = compute_heliographic_angular_distance(lat1, lon1, lat2, lon2)
    assert pytest.approx(d90.item(), abs=1e-3) == 90.0

    # Opposite sides along equator (e.g. 45E and 45W) -> 90 deg
    lat_eq = torch.tensor([0.0])
    lon_w = torch.tensor([-45.0])
    lon_e = torch.tensor([45.0])
    d_eq = compute_heliographic_angular_distance(lat_eq, lon_w, lat_eq, lon_e)
    assert pytest.approx(d_eq.item(), abs=1e-3) == 90.0


def test_build_solar_disk_graph():
    # Empty input
    nodes, edges, attr = build_solar_disk_graph([])
    assert nodes.size(0) == 0
    assert edges.size(1) == 0

    # Two active regions: AR13664 (S19W42) and AR13668 (S15W10)
    # Distance is approx 32 degrees on the disk -> within max_angular_deg=75.0
    ar_list = [
        {"features": [1.0] * 21, "latitude": -19.0, "longitude": -42.0, "polarity": 1},
        {"features": [2.0] * 21, "latitude": -15.0, "longitude": -10.0, "polarity": -1},
    ]
    nodes, edges, attr = build_solar_disk_graph(ar_list, max_angular_deg=75.0)

    assert nodes.size(0) == 2
    assert nodes.size(1) == 21
    # 2 self-loops + 2 directed edges (0->1 and 1->0) = 4 edges total
    assert edges.size(1) == 4
    assert attr.size(0) == 4
    assert attr.size(1) == 4

    # Verify opposite polarities produce negative polarity product
    assert attr[2, 3].item() == -1.0


def test_solar_gnn_conv():
    conv = SolarGNNConv(in_features=16, out_features=32, edge_features=4)
    x = torch.randn(3, 16)
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 0]], dtype=torch.long)
    edge_attr = torch.randn(4, 4)

    out = conv(x, edge_index, edge_attr)
    assert out.size(0) == 3
    assert out.size(1) == 32
    assert not torch.isnan(out).any()


def test_solar_cross_attention_pooling():
    pool = SolarCrossAttentionPooling(node_dim=32, num_heads=4)
    x = torch.randn(5, 32)
    # 2 graphs in batch: graph 0 has 3 nodes, graph 1 has 2 nodes
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    pooled = pool(x, batch)
    assert pooled.size(0) == 2
    assert pooled.size(1) == 32
    assert not torch.isnan(pooled).any()


def test_sympathetic_flare_gnn():
    gnn = SympatheticFlareGNN(in_channels=21, hidden_channels=32, out_channels=64, num_layers=2)
    x = torch.randn(4, 21)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.randn(4, 4)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    out = gnn(x, edge_index, edge_attr, batch)
    assert out.size(0) == 2
    assert out.size(1) == 64
    assert not torch.isnan(out).any()


def test_spatial_temporal_fusion_and_loss():
    fusion_model = SolarFlareSpatialTemporalFusion(
        gnn_in_channels=21,
        gnn_hidden_channels=32,
        gnn_out_channels=64,
        temp_dim=64,
        wave_dim=32,
        fused_dim=64,
    )

    B = 2
    z_temp = torch.randn(B, 64)
    z_wave = torch.randn(B, 32)
    node_feats = torch.randn(6, 21)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    preds = fusion_model(z_temp, z_wave, node_feats, edge_index, batch=batch)
    assert len(preds) == 5
    for h in ["30m", "1h", "3h", "6h", "24h"]:
        assert h in preds
        logits, bounds = preds[h]
        assert logits.shape == (B, 3)
        assert bounds.shape == (B, 2)
        # Verify p_low <= p_high
        assert (bounds[:, 0] <= bounds[:, 1] + 1e-6).all()

    # Test multi-horizon loss computation
    loss_fn = MultiHorizonSolarLoss()
    targets = {h: torch.randint(0, 3, (B,)) for h in ["30m", "1h", "3h", "6h", "24h"]}
    total_loss, metrics = loss_fn(preds, targets)

    assert total_loss > 0.0
    assert not torch.isnan(total_loss)
    assert "loss_1h" in metrics
    assert "brier_1h" in metrics
