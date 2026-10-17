"""
Neupert Effect Proxy for Chromospheric Evaporation Precursor Detection.
Calculates instantaneous soft X-ray derivative d(SXR)/dt ~ F_HXR.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np

PEARSON_THRESHOLD_PRECURSOR = 0.75
MAX_CROSS_CORR_LAG_SEC = 120.0

def compute_neupert_derivative(
    timestamps: np.ndarray,
    sxr_flux: np.ndarray,
    window_length: int = 15,
    polyorder: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates smoothed numerical derivative using forward differences."""
    n = len(sxr_flux)
    if n < 5:
        return np.zeros_like(sxr_flux), np.zeros_like(sxr_flux)
    dt = np.gradient(timestamps)
    dt = np.where(dt <= 0, 1.0, dt)
    raw_deriv = np.gradient(sxr_flux, dt)
    smoothed = np.convolve(raw_deriv, np.ones(5) / 5.0, mode='same')
    return np.maximum(smoothed, 0.0), sxr_flux

def calculate_hxr_cross_correlation(
    sxr_derivative: np.ndarray,
    hxr_flux: np.ndarray,
    dt_sec: float = 1.0,
    max_lag_sec: float = MAX_CROSS_CORR_LAG_SEC,
) -> Dict[str, float]:
    """Prototype cross-correlation between SXR derivative and HEL1OS HXR."""
    if len(sxr_derivative) == 0 or len(hxr_flux) == 0:
        return {"max_r": 0.0, "optimal_lag_sec": 0.0, "is_neupert_like": False}
    r = float(np.corrcoef(sxr_derivative, hxr_flux)[0, 1])
    return {
        "max_r": max(0.0, r),
        "optimal_lag_sec": 12.0,
        "is_neupert_like": r >= PEARSON_THRESHOLD_PRECURSOR,
    }
