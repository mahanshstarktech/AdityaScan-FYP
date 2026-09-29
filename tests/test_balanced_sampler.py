"""
Unit tests for StratifiedSolarSampler class-imbalance rebalancing.
"""
import pytest
from pipeline.ml.balanced_sampler import compute_effective_num_weights, StratifiedSolarSampler

def test_effective_num_weights():
    counts = {"quiet": 12000, "C": 500, "M": 80, "X": 10}
    weights = compute_effective_num_weights(counts)
    assert weights["X"] > weights["M"] > weights["C"] > weights["quiet"]
    assert weights["X"] > 1.0

def test_stratified_sampler_distribution():
    labels = ["quiet"] * 1000 + ["C"] * 50 + ["M"] * 10 + ["X"] * 2
    sampler = StratifiedSolarSampler(labels, oversample_flare_ratio=5.0, num_samples=500)
    samples = list(iter(sampler))
    assert len(samples) == 500
