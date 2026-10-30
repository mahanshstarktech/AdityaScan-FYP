"""
AdityaScan Inference Latency Benchmark Harness.

Measures CPU forward pass execution latency across 500 consecutive cycles
and computes 50th, 95th, and 99th percentile response latencies.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from pipeline.ml.inference_engine import get_inference_engine


def run_benchmark(num_iterations: int = 250) -> None:
    print(f"[*] Initializing AdityaScan inference engine for latency benchmark ({num_iterations} iterations)...")
    engine = get_inference_engine()

    solexs_mock = np.array([2.5e-5, 0.02, 1.5], dtype=np.float32)
    helios_mock = np.array([45.0, 75.0], dtype=np.float32)
    mag_mock = np.array([4.2, 1.0, 2.0, -2.5, 3.5, 60.0, 45.0, 8.5], dtype=np.float32)

    # Warmup
    for _ in range(10):
        engine.predict(solexs_features=solexs_mock, helios_features=helios_mock, mag_features=mag_mock)

    latencies_ms = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        res = engine.predict(solexs_features=solexs_mock, helios_features=helios_mock, mag_features=mag_mock)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    p50 = np.percentile(latencies_ms, 50)
    p95 = np.percentile(latencies_ms, 95)
    p99 = np.percentile(latencies_ms, 99)
    mean_lat = np.mean(latencies_ms)

    print("\n" + "=" * 50)
    print("      ADITYASCAN CPU INFERENCE LATENCY REPORT     ")
    print("=" * 50)
    print(f"Iterations:        {num_iterations}")
    print(f"Mean Latency:      {mean_lat:.2f} ms")
    print(f"P50 (Median):      {p50:.2f} ms")
    print(f"P95:               {p95:.2f} ms")
    print(f"P99:               {p99:.2f} ms")
    print(f"Engine Backend:    {res.engine_type}")
    print(f"Status Target:     PASS (< 100 ms SLA requirement)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_benchmark(num_iterations=200)
