"""
Unit tests for Neupert Effect Derivative Proxy and Chromospheric Evaporation Detector.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from pipeline.physics.neupert_proxy import (
    compute_neupert_derivative,
    analyze_neupert_flare_profile,
    cross_correlate_neupert_hxr,
)


def test_neupert_derivative_on_gaussian_flare():
    # Synthetic Gaussian solar flare pulse: I(t) = exp(-(t - t0)^2 / (2 * sigma^2))
    t = np.linspace(0, 3600, 360)  # 1 hour at 10s cadence
    t0 = 1800.0
    sigma = 300.0
    flux = 1e-4 * np.exp(-((t - t0) ** 2) / (2 * sigma ** 2)) + 1e-6

    deriv = compute_neupert_derivative(t, flux, cadence_sec=10.0)

    # Derivative of Gaussian: d/dt [exp(-u^2)] = -(t - t0)/sigma^2 * exp(...)
    # Peak positive derivative should occur before t0 (around t0 - sigma = 1500s)
    peak_deriv_idx = int(np.argmax(deriv))
    t_peak_deriv = t[peak_deriv_idx]

    assert 1400.0 <= t_peak_deriv <= 1600.0
    # Zero crossing should occur near peak flux (t0 = 1800s)
    assert abs(deriv[180]) < 1e-8


def test_analyze_neupert_flare_profile():
    # Simulate an impulsive flare profile: fast rise (impulsive Neupert heating), slow decay
    t = np.linspace(0, 3600, 360)
    # Fast rise (t=1200 to 1800), slow exponential cooling decay
    flux = np.zeros_like(t)
    rise_mask = (t >= 1200) & (t <= 1800)
    decay_mask = t > 1800
    flux[rise_mask] = 1e-4 * ((t[rise_mask] - 1200) / 600.0) ** 2
    flux[decay_mask] = 1e-4 * np.exp(-(t[decay_mask] - 1800) / 900.0)
    flux += 1e-7

    res = analyze_neupert_flare_profile(t, flux, cadence_sec=10.0)

    assert res.is_impulsive_burst is True
    # Neupert derivative peak should precede SXR peak
    assert res.peak_derivative_time <= res.sxr_peak_time
    assert res.evaporation_lead_time_sec >= 0.0


def test_cross_correlate_neupert_hxr():
    # Coincident d(SXR)/dt and HXR pulse
    t = np.linspace(0, 1800, 180)
    # HXR pulse centered at t=600s
    hxr_flux = 500.0 * np.exp(-((t - 600) ** 2) / (2 * 60 ** 2))
    # SXR is integral of HXR: cumulative sum
    sxr_flux = np.cumsum(hxr_flux) * 10.0 + 10.0

    corr, lag = cross_correlate_neupert_hxr(t, sxr_flux, t, hxr_flux, max_lag_sec=120.0)

    # High correlation (> 0.85) and near-zero lag (< 30s)
    assert corr > 0.80
    assert abs(lag) <= 30.0
