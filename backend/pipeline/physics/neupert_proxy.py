"""
Neupert Effect Derivative Proxy and Chromospheric Evaporation Detector.

The Neupert Effect (Neupert 1968, Veronig et al. 2002):
  In solar flares, non-thermal electrons accelerated during magnetic reconnection propagate
  down along magnetic field lines and deposit kinetic energy into the dense chromosphere via
  thick-target bremsstrahlung collisions, generating Hard X-ray (HXR) emission.
  This deposition causes explosive heating and overpressure, driving hot plasma upward into coronal
  loops (chromospheric evaporation) which emits thermal Soft X-rays (SXR).

  Under the thick-target model:
      d(I_SXR(t)) / dt ≈ k * F_HXR(t)

  Therefore, the time derivative of soft X-ray irradiance (from Aditya-L1 SoLEXS or NOAA GOES)
  provides an instantaneous physical proxy for non-thermal particle acceleration even before peak
  thermal soft X-ray emission is observed.

Key Features:
  - Savitzky-Golay numerical smoothing filter to suppress high-cadence telemetry noise
    while preserving impulsive sub-minute reconnection bursts.
  - Pearson cross-correlation and time lag determination between d(SXR)/dt and HEL1OS HXR.
  - Explosive chromospheric evaporation precursor trigger for the early-warning triage system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.signal import savgol_filter


@dataclass
class NeupertAnalysisResult:
    """Stores quantitative metrics from Neupert effect derivative analysis."""
    times_unix: np.ndarray
    sxr_flux: np.ndarray
    neupert_derivative: np.ndarray
    peak_derivative_time: float
    peak_derivative_val: float
    sxr_peak_time: float
    evaporation_lead_time_sec: float  # Time difference between dI/dt peak and SXR peak
    is_impulsive_burst: bool
    hxr_cross_correlation: Optional[float] = None
    hxr_lag_sec: Optional[float] = None


def compute_neupert_derivative(
    times_unix: Union[List[float], np.ndarray],
    flux_sxr: Union[List[float], np.ndarray],
    window_length: int = 15,
    polyorder: int = 2,
    cadence_sec: float = 60.0,
) -> np.ndarray:
    """
    Computes smoothed numerical derivative d(SXR)/dt using a Savitzky-Golay filter.

    Parameters
    ----------
    times_unix : array-like
        Observation timestamps in seconds.
    flux_sxr : array-like
        Soft X-ray flux (e.g. W/m^2 from GOES or counts/s from SoLEXS).
    window_length : int
        Filter window size (must be odd).
    polyorder : int
        Polynomial order for fitting.
    cadence_sec : float
        Nominal sampling interval in seconds if times are non-uniform.

    Returns
    -------
    derivative : np.ndarray
        First derivative dI/dt with noise suppression.
    """
    flux = np.asarray(flux_sxr, dtype=np.float64)
    n = len(flux)
    if n < 4:
        return np.zeros_like(flux)

    # Ensure window_length is odd and does not exceed array length
    wl = min(window_length, n if n % 2 != 0 else n - 1)
    if wl <= polyorder:
        wl = polyorder + 1 if (polyorder + 1) % 2 != 0 else polyorder + 2

    if wl > n:
        wl = max(3, n if n % 2 != 0 else n - 1)
        poly = min(1, wl - 1)
    else:
        poly = polyorder

    # Savitzky-Golay derivative (deriv=1, delta=cadence_sec)
    try:
        deriv = savgol_filter(flux, window_length=wl, polyorder=poly, deriv=1, delta=cadence_sec)
    except Exception:
        # Fallback to standard central difference
        deriv = np.gradient(flux, cadence_sec)

    return deriv


def analyze_neupert_flare_profile(
    times_unix: Union[List[float], np.ndarray],
    flux_sxr: Union[List[float], np.ndarray],
    background_threshold_sigma: float = 3.0,
    cadence_sec: float = 60.0,
) -> NeupertAnalysisResult:
    """
    Analyzes a flare time series to extract Neupert precursor timing and evaporation lead time.
    """
    times = np.asarray(times_unix, dtype=np.float64)
    flux = np.asarray(flux_sxr, dtype=np.float64)

    if len(times) == 0:
        return NeupertAnalysisResult(
            times_unix=np.array([]),
            sxr_flux=np.array([]),
            neupert_derivative=np.array([]),
            peak_derivative_time=0.0,
            peak_derivative_val=0.0,
            sxr_peak_time=0.0,
            evaporation_lead_time_sec=0.0,
            is_impulsive_burst=False,
        )

    deriv = compute_neupert_derivative(times, flux, cadence_sec=cadence_sec)

    # Find peak SXR flux
    idx_sxr_peak = int(np.argmax(flux))
    t_sxr_peak = float(times[idx_sxr_peak])

    # Find peak derivative up to or near SXR peak (reconnection heating phase)
    search_cutoff = min(len(times), idx_sxr_peak + 5)
    idx_deriv_peak = int(np.argmax(deriv[:search_cutoff]))
    t_deriv_peak = float(times[idx_deriv_peak])
    val_deriv_peak = float(deriv[idx_deriv_peak])

    # Lead time in seconds (Neupert peak precedes SXR peak in impulsive flares)
    lead_time = max(0.0, t_sxr_peak - t_deriv_peak)

    # Background noise level in first 20% of window
    baseline_n = max(3, int(len(flux) * 0.2))
    baseline_std = float(np.std(deriv[:baseline_n]))
    baseline_mean = float(np.mean(deriv[:baseline_n]))

    is_impulsive = val_deriv_peak > (baseline_mean + background_threshold_sigma * max(1e-12, baseline_std))

    return NeupertAnalysisResult(
        times_unix=times,
        sxr_flux=flux,
        neupert_derivative=deriv,
        peak_derivative_time=t_deriv_peak,
        peak_derivative_val=val_deriv_peak,
        sxr_peak_time=t_sxr_peak,
        evaporation_lead_time_sec=lead_time,
        is_impulsive_burst=is_impulsive,
    )


def cross_correlate_neupert_hxr(
    times_sxr: np.ndarray,
    flux_sxr: np.ndarray,
    times_hxr: np.ndarray,
    flux_hxr: np.ndarray,
    max_lag_sec: float = 300.0,
) -> Tuple[float, float]:
    """
    Computes Pearson cross-correlation between d(SXR)/dt and actual HEL1OS HXR counts.
    Returns (max_correlation, optimal_lag_sec).
    """
    if len(times_sxr) == 0 or len(times_hxr) == 0:
        return 0.0, 0.0

    # Interpolate HXR to SXR timestamps
    common_times = times_sxr
    hxr_interp = np.interp(common_times, times_hxr, flux_hxr, left=0.0, right=0.0)
    sxr_deriv = compute_neupert_derivative(common_times, flux_sxr)

    # Normalize zero-mean unit-variance
    d_norm = (sxr_deriv - np.mean(sxr_deriv)) / (np.std(sxr_deriv) + 1e-10)
    h_norm = (hxr_interp - np.mean(hxr_interp)) / (np.std(hxr_interp) + 1e-10)

    # Full cross-correlation
    corr = np.correlate(d_norm, h_norm, mode="full") / len(d_norm)
    lags = np.arange(-len(d_norm) + 1, len(d_norm))

    cadence = float(common_times[1] - common_times[0]) if len(common_times) > 1 else 60.0
    lags_sec = lags * cadence

    valid_mask = np.abs(lags_sec) <= max_lag_sec
    if not np.any(valid_mask):
        return float(np.max(corr)), 0.0

    best_idx = np.argmax(corr[valid_mask])
    best_corr = float(corr[valid_mask][best_idx])
    best_lag = float(lags_sec[valid_mask][best_idx])

    return max(-1.0, min(1.0, best_corr)), best_lag
