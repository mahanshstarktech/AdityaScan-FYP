"""
Unit tests for Continuous Morlet Wavelet Transform and QPP Precursor Detection.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from pipeline.physics.wavelet_qpp import (
    compute_morlet_cwt,
    extract_qpp_features,
)


def test_morlet_cwt_synthetic_monochromatic_sine():
    # Synthetic 60-second periodic oscillation in solar flux
    target_period = 60.0  # seconds
    dt = 2.0  # 2s cadence
    t = np.arange(0, 1200, dt)  # 20 minutes

    # Signal with 60s oscillation + small noise
    np.random.seed(42)
    signal = 1.0 + 0.2 * np.sin(2.0 * np.pi * t / target_period) + 0.02 * np.random.randn(len(t))

    scalogram, periods, coi = compute_morlet_cwt(
        t, signal, min_period_sec=15.0, max_period_sec=180.0, num_scales=48
    )

    assert scalogram.shape[0] == 48
    assert scalogram.shape[1] == len(t)
    assert len(periods) == 48
    assert len(coi) == len(t)

    # Global power peak should match 60s within 15% tolerance
    global_power = np.mean(scalogram, axis=1)
    peak_idx = int(np.argmax(global_power))
    detected_period = periods[peak_idx]

    assert pytest.approx(detected_period, rel=0.15) == target_period


def test_extract_qpp_features_significance():
    # Strong 120-second MHD oscillation burst
    dt = 2.0
    t = np.arange(0, 1800, dt)  # 30 mins
    oscillation = 0.5 * np.sin(2.0 * np.pi * t / 120.0)
    signal = 2.0 + oscillation

    res = extract_qpp_features(t, signal, min_period_sec=20.0, max_period_sec=250.0)

    assert pytest.approx(res.dominant_period_sec, rel=0.15) == 120.0
    assert res.is_statistically_significant is True
    assert res.quality_factor > 1.0
    assert res.spectral_energy_bandpass > 0.0
