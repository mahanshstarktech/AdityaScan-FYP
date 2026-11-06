"""
Active Region Coronal Arcade Graph Message-Passing Throughput Benchmark.

Measures graph adjacency construction and message passing latency across
multi-region solar active topologies (simulating May 2024 severe storm conditions).
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from pipeline.ml.active_region_graph_builder import (
    ActiveRegionGraphBuilder,
    ActiveRegionNode,
)


def run_gnn_benchmark(num_iterations: int = 500) -> None:
    print(f"[*] Running Active Region Graph Construction Benchmark ({num_iterations} iterations)...")
    builder = ActiveRegionGraphBuilder(max_angular_deg=75.0, arcade_scale_deg=30.0)

    # Simulate 8 concurrent active regions across solar disk with 21 McIntosh/Hale features
    dummy_feats = [0.1 * i for i in range(21)]
    sample_nodes = [
        ActiveRegionNode(13664, -19.0, -42.0, 2150.0, 4.82, -1, dummy_feats),
        ActiveRegionNode(13668, -15.0, -28.0, 1400.0, 3.10, 1, dummy_feats),
        ActiveRegionNode(13663, 12.0, -35.0, 980.0, 2.50, 1, dummy_feats),
        ActiveRegionNode(13670, 22.0, 15.0, 750.0, 1.80, -1, dummy_feats),
        ActiveRegionNode(13671, -8.0, 45.0, 520.0, 1.20, 1, dummy_feats),
        ActiveRegionNode(13672, 5.0, -60.0, 410.0, 0.90, -1, dummy_feats),
        ActiveRegionNode(13673, -28.0, 10.0, 610.0, 1.40, 1, dummy_feats),
        ActiveRegionNode(13674, 18.0, 55.0, 480.0, 1.10, -1, dummy_feats),
    ]

    latencies_ms = []
    total_edges_sampled = []

    # Warmup
    for _ in range(10):
        builder.build_graph(sample_nodes)

    for _ in range(num_iterations):
        t0 = time.perf_counter()
        node_feats, adj, edges = builder.build_graph(sample_nodes)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        total_edges_sampled.append(len(edges))

    mean_ms = float(np.mean(latencies_ms))
    p50_ms = float(np.percentile(latencies_ms, 50))
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))
    mean_edges = float(np.mean(total_edges_sampled))

    print("\n" + "=" * 60)
    print("   ACTIVE REGION GRAPH ADJACENCY THROUGHPUT REPORT   ")
    print("=" * 60)
    print(f"Iterations:             {num_iterations}")
    print(f"Active Nodes per Graph: {len(sample_nodes)}")
    print(f"Mean Extracted Edges:   {mean_edges:.1f}")
    print(f"Mean Construction Time: {mean_ms:.3f} ms")
    print(f"P50 Latency:            {p50_ms:.3f} ms")
    print(f"P95 Latency:            {p95_ms:.3f} ms")
    print(f"P99 Latency:            {p99_ms:.3f} ms")
    print(f"Throughput:             {1000.0 / mean_ms:.1f} graphs/sec")
    print("=" * 60)


if __name__ == "__main__":
    run_gnn_benchmark()
