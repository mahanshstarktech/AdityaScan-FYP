"""
Integration tests for AdityaGoesSyncBenchmark.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from pipeline.ingestion.sync_validator import AdityaGoesSyncBenchmark

def test_sync_benchmark_nominal():
    times = np.arange(100)
    solexs_flux = np.sin(times / 10.0) + 1.5
    goes_flux = solexs_flux * 1.05 # slight calibration offset

    benchmark = AdityaGoesSyncBenchmark(max_allowed_lag_s=60.0, min_correlation=0.90)
    res = benchmark.compute_cross_correlation(times, solexs_flux, times, goes_flux)

    assert res.is_synchronized is True
    assert res.pearson_r > 0.95
