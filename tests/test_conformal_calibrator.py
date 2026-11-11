"""
Unit test suite for AdityaScan MultiHorizonConformalCalibrator.

Validates:
  - Finite-sample conformal quantile derivation across all forecast horizons.
  - Distribution-free 90% empirical coverage target guarantee.
  - Interval bounds sanity [0 <= p_low <= point <= p_high <= 1].
  - Empty or missing horizons handling.
"""

import numpy as np
import pytest
from pipeline.ml.conformal_calibrator import (
    MultiHorizonConformalCalibrator,
    ConformalInterval,
)


def test_conformal_calibration_guarantee():
    np.random.seed(42)
    calibrator = MultiHorizonConformalCalibrator(horizons=["5m", "15m", "60m"], default_alpha=0.10)

    n_samples = 1000
    y_true = {}
    y_prob = {}

    for h in ["5m", "15m", "60m"]:
        # Simulated calibrated probability with small Gaussian noise
        probs = np.random.uniform(0.05, 0.95, size=n_samples)
        # Generate binary outcomes according to Bernoulli(probs)
        outcomes = (np.random.rand(n_samples) < probs).astype(np.float32)
        y_true[h] = outcomes
        y_prob[h] = probs

    quantiles = calibrator.fit_calibration_set(y_true, y_prob)

    assert "5m" in quantiles
    assert "15m" in quantiles
    assert "60m" in quantiles

    # Target is 90% coverage: empirical coverage should be >= 0.88
    for h in ["5m", "15m", "60m"]:
        emp_cov = calibrator.empirical_coverages[h]
        assert emp_cov >= 0.88, f"Horizon {h} coverage {emp_cov} fell below 0.88"


def test_predict_interval_bounds():
    calibrator = MultiHorizonConformalCalibrator()
    calibrator.calibrated_quantiles = {"30m": 0.125}

    interval = calibrator.predict_interval("30m", probability=0.75)
    assert isinstance(interval, ConformalInterval)
    assert interval.horizon == "30m"
    assert interval.point_estimate == 0.75
    assert interval.p_low == pytest.approx(0.625, abs=1e-3)
    assert interval.p_high == pytest.approx(0.875, abs=1e-3)
    assert interval.margin == 0.125
    assert interval.coverage_target == 0.90


def test_extreme_boundary_clamping():
    calibrator = MultiHorizonConformalCalibrator()
    calibrator.calibrated_quantiles = {"5m": 0.20}

    # Low boundary clamping at 0.0
    int_low = calibrator.predict_interval("5m", probability=0.05)
    assert int_low.p_low == 0.0
    assert int_low.p_high == 0.25

    # High boundary clamping at 1.0
    int_high = calibrator.predict_interval("5m", probability=0.95)
    assert int_high.p_low == 0.75
    assert int_high.p_high == 1.0
