"""
Continuous Morlet Wavelet Transform for Quasi-Periodic Pulsation (QPP) Detection.

In solar flares and pre-flare active region dynamics, magnetohydrodynamic (MHD) oscillations
(e.g., fast magnetoacoustic sausage and kink modes, slow standing waves) in coronal magnetic loops
modulate reconnection rates and produce Quasi-Periodic Pulsations (QPP).

Observable signature:
  Periodic intensity fluctuations in soft X-ray, hard X-ray, and microwave emissions
  with periods ranging between 10 seconds and 300 seconds during flare precursor phases
  (Nakariakov & Melnikov 2009, Inglis et al. 2016).

This module implements:
  - Continuous Wavelet Transform (CWT) using the standard complex Morlet wavelet (omega_0 = 6.0).
  - Cone of Influence (COI) calculation to mask edge artifacts.
  - Global wavelet power spectrum and red-noise 95% confidence level assessment (Torrence & Compo 1998).
  - Automated QPP feature extraction (dominant period, spectral power, Q-factor) for downstream ML models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class QPPFeatureResult:
    """Stores quantitative features extracted from QPP wavelet transform."""
    dominant_period_sec: float
    max_power: float
    is_statistically_significant: bool  # Exceeds 95% red-noise confidence level
    quality_factor: float              # Q = f_0 / delta_f
    spectral_energy_bandpass: float    # Integrated power in 10-300s band
    scalogram: np.ndarray              # 2D power array [num_scales, num_times]
    periods: np.ndarray                # 1D array of Fourier periods in seconds
    coi_mask: np.ndarray               # 2D boolean mask where True = valid (inside COI)


def morlet_wavelet(t: np.ndarray, s: float, omega0: float = 6.0) -> np.ndarray:
    """
    Complex Morlet mother wavelet at scale s:
        psi(eta) = pi^(-1/4) * exp(i * omega_0 * eta) * exp(-eta^2 / 2)
    where eta = t / s.
    """
    eta = t / s
    norm = np.pi ** (-0.25) / np.sqrt(s)
    return norm * np.exp(1j * omega0 * eta) * np.exp(-0.5 * eta ** 2)


def compute_morlet_cwt(
    times_sec: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    min_period_sec: float = 10.0,
    max_period_sec: float = 300.0,
    num_scales: int = 48,
    omega0: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Continuous Morlet Wavelet Transform over the target period band.

    Parameters
    ----------
    times_sec : array-like
        Sampling times in seconds (assumes approximately regular cadence dt).
    signal : array-like
        Solar flux time-series values.
    min_period_sec : float
        Shortest physical oscillation period (default: 10s).
    max_period_sec : float
        Longest physical oscillation period (default: 300s).
    num_scales : int
        Number of logarithmically spaced scales.
    omega0 : float
        Non-dimensional frequency parameter (standard: 6.0 for Fourier period alignment).

    Returns
    -------
    scalogram : np.ndarray [num_scales, num_times]
        Normalized wavelet power |W_n(s)|^2.
    periods : np.ndarray [num_scales]
        Fourier periods corresponding to each wavelet scale in seconds.
    coi : np.ndarray [num_times]
        Cone of Influence e-folding period at each time index.
    """
    x = np.asarray(signal, dtype=np.float64)
    t = np.asarray(times_sec, dtype=np.float64)
    n = len(x)

    if n < 4:
        return np.zeros((num_scales, n)), np.zeros(num_scales), np.zeros(n)

    dt = float(np.median(np.diff(t))) if n > 1 else 1.0
    dt = max(1e-3, dt)

    # Detrend and normalize signal
    x_detrend = x - np.mean(x)
    std_val = np.std(x_detrend)
    if std_val > 1e-12:
        x_norm = x_detrend / std_val
    else:
        x_norm = x_detrend

    # Scale conversion factor: for Morlet with omega0=6.0, Fourier period T = 1.033 * scale
    fourier_factor = (4.0 * np.pi) / (omega0 + np.sqrt(2.0 + omega0 ** 2))

    s_min = min_period_sec / fourier_factor
    s_max = max_period_sec / fourier_factor
    scales = np.geomspace(s_min, s_max, num=num_scales)
    periods = scales * fourier_factor

    # FFT-based convolution for high numerical efficiency
    # Next power of 2 for zero-padding to reduce circular convolution
    n_padded = 2 ** int(np.ceil(np.log2(n * 2)))
    omega = 2.0 * np.pi * np.fft.fftfreq(n_padded, d=dt)

    x_pad = np.pad(x_norm, (0, n_padded - n), mode="constant")
    x_fft = np.fft.fft(x_pad)

    scalogram = np.zeros((num_scales, n), dtype=np.float64)

    # Daughter wavelets in Fourier domain:
    #   psi_hat(omega) = pi^(-1/4) * (2*s)^(1/2) * H(omega) * exp(- (s*omega - omega0)^2 / 2)
    const = (np.pi ** -0.25) * np.sqrt(2.0 * np.pi * dt)
    for i, s in enumerate(scales):
        # Heaviside step function: only positive frequencies
        mask = omega > 0
        daughter_fft = np.zeros(n_padded, dtype=np.float64)
        daughter_fft[mask] = const * np.sqrt(s) * np.exp(-0.5 * (s * omega[mask] - omega0) ** 2)

        # Inverse FFT of product
        w_conv = np.fft.ifft(x_fft * daughter_fft)[:n]
        scalogram[i, :] = np.abs(w_conv) ** 2

    # Cone of influence: distance from edge where power drops by e^-2
    coi = fourier_factor / np.sqrt(2.0) * np.minimum(t - t[0], t[-1] - t)

    return scalogram, periods, coi


def extract_qpp_features(
    times_sec: Union[List[float], np.ndarray],
    signal: Union[List[float], np.ndarray],
    min_period_sec: float = 10.0,
    max_period_sec: float = 300.0,
    confidence_level: float = 0.95,
) -> QPPFeatureResult:
    """
    Extracts high-level solar flare QPP features from telemetry lightcurves.
    Assesses statistical significance against an autoregressive red-noise background.
    """
    scalogram, periods, coi = compute_morlet_cwt(
        times_sec, signal, min_period_sec=min_period_sec, max_period_sec=max_period_sec
    )

    n_scales, n_times = scalogram.shape
    if n_times == 0 or n_scales == 0:
        return QPPFeatureResult(
            dominant_period_sec=0.0,
            max_power=0.0,
            is_statistically_significant=False,
            quality_factor=0.0,
            spectral_energy_bandpass=0.0,
            scalogram=scalogram,
            periods=periods,
            coi_mask=np.zeros_like(scalogram, dtype=bool),
        )

    # 2D COI mask: True where period <= COI at time t
    coi_mask = periods[:, None] <= coi[None, :]

    # Global Wavelet Power Spectrum (time-average inside COI)
    valid_counts = np.sum(coi_mask, axis=1)
    masked_scalogram = np.where(coi_mask, scalogram, 0.0)
    global_ws = np.divide(
        np.sum(masked_scalogram, axis=1),
        np.maximum(1, valid_counts),
        where=valid_counts > 0,
    )

    # Theoretical red-noise lag-1 autocorrelation background
    x = np.asarray(signal, dtype=np.float64)
    x_c = x - np.mean(x)
    r1 = np.corrcoef(x_c[:-1], x_c[1:])[0, 1] if len(x_c) > 2 else 0.5
    r1 = max(0.0, min(0.92, float(r1))) if not np.isnan(r1) else 0.5

    # Red noise spectrum: P_k = (1 - r1^2) / (1 + r1^2 - 2*r1*cos(2*pi*f))
    fourier_freqs = 1.0 / periods
    dt = float(np.median(np.diff(times_sec))) if len(times_sec) > 1 else 1.0
    red_noise = (1.0 - r1 ** 2) / (1.0 + r1 ** 2 - 2.0 * r1 * np.cos(2.0 * np.pi * fourier_freqs * dt))

    # Chi-square 95% threshold (dof ≈ 2 for complex Morlet)
    # chi2.ppf(0.95, df=2) / 2 ≈ 5.991 / 2 ≈ 2.996
    significance_thresh = red_noise * 2.996

    # Dominant peak period
    peak_idx = int(np.argmax(global_ws))
    dom_period = float(periods[peak_idx])
    max_pwr = float(global_ws[peak_idx])
    is_sig = bool(max_pwr > significance_thresh[peak_idx])

    # Quality factor Q = f_0 / delta_f
    # Half-power width around dominant peak
    half_power = max_pwr / 2.0
    above_half = global_ws >= half_power
    f_band = fourier_freqs[above_half]
    delta_f = float(np.max(f_band) - np.min(f_band)) if len(f_band) > 1 else (1.0 / dom_period * 0.2)
    q_factor = float((1.0 / dom_period) / max(1e-6, delta_f))

    # Integrated spectral energy across target 10-300s band
    total_energy = float(np.sum(masked_scalogram))

    return QPPFeatureResult(
        dominant_period_sec=dom_period,
        max_power=max_pwr,
        is_statistically_significant=is_sig,
        quality_factor=q_factor,
        spectral_energy_bandpass=total_energy,
        scalogram=scalogram,
        periods=periods,
        coi_mask=coi_mask,
    )
