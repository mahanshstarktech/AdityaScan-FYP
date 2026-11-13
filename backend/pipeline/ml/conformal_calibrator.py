"""
Multi-Horizon Split Conformal Prediction Calibrator for AdityaScan.

Provides finite-sample, distribution-free statistical coverage guarantees:
  P(Y_{n+1} in C(X_{n+1})) >= 1 - alpha

Grounded in:
  - Vovk et al. (2005), Algorithmic Learning in a Random World.
  - Angelopoulos & Bates (2023), Conformal Prediction: A Gentle Introduction.
  - Applied to multi-horizon solar flare forecasting (5m, 10m, 15m, 30m, 60m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ConformalInterval:
    horizon: str
    point_estimate: float
    p_low: float
    p_high: float
    margin: float
    coverage_target: float


class MultiHorizonConformalCalibrator:
    """
    Calibrates multi-horizon solar flare forecast probabilities using split conformal prediction.
    """

    def __init__(
        self,
        horizons: Optional[List[str]] = None,
        default_alpha: float = 0.10,  # 90% target coverage
    ) -> None:
        self.horizons = horizons or ["5m", "10m", "15m", "30m", "60m"]
        self.default_alpha = default_alpha
        self.calibrated_quantiles: Dict[str, float] = {}
        self.empirical_coverages: Dict[str, float] = {}

    def fit_calibration_set(
        self,
        y_true: Dict[str, np.ndarray],
        y_prob: Dict[str, np.ndarray],
        alpha: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Computes finite-sample conformal quantile margins q_{1-alpha} for each horizon.

        Non-conformity score:
          s_i = |y_i - p_i|
        Conformal quantile level:
          level = ceil((n + 1) * (1 - alpha)) / n
        """
        cov_alpha = alpha if alpha is not None else self.default_alpha

        for h in self.horizons:
            if h not in y_true or h not in y_prob:
                continue

            y_t = np.asarray(y_true[h], dtype=np.float32)
            y_p = np.asarray(y_prob[h], dtype=np.float32)

            n = len(y_t)
            if n == 0:
                continue

            # Non-conformity scores: absolute residual
            scores = np.abs(y_t - y_p)

            # Conformal quantile level adjusted for finite-sample size
            quantile_idx = math.ceil((n + 1) * (1.0 - cov_alpha)) / n
            quantile_idx = min(1.0, max(0.0, quantile_idx))

            q_val = float(np.quantile(scores, quantile_idx, method="higher"))
            self.calibrated_quantiles[h] = round(q_val, 4)

            # Check empirical coverage on calibration set
            in_bounds = (y_t >= (y_p - q_val)) & (y_t <= (y_p + q_val))
            self.empirical_coverages[h] = round(float(np.mean(in_bounds)), 4)

        return self.calibrated_quantiles

    def predict_interval(
        self,
        horizon: str,
        probability: float,
        alpha: Optional[float] = None,
    ) -> ConformalInterval:
        """
        Applies calibrated conformal margin to derive interval [p_low, p_high].
        """
        target_cov = 1.0 - (alpha if alpha is not None else self.default_alpha)
        margin = self.calibrated_quantiles.get(horizon, 0.15)

        p_low = max(0.0, probability - margin)
        p_high = min(1.0, probability + margin)

        return ConformalInterval(
            horizon=horizon,
            point_estimate=round(probability, 4),
            p_low=round(p_low, 4),
            p_high=round(p_high, 4),
            margin=round(margin, 4),
            coverage_target=round(target_cov, 2),
        )

    def evaluate_test_coverage(
        self,
        y_test_true: Dict[str, np.ndarray],
        y_test_prob: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Evaluates empirical coverage on holdout test partition.
        """
        test_coverages = {}
        for h in self.horizons:
            if h not in y_test_true or h not in y_test_prob:
                continue
            margin = self.calibrated_quantiles.get(h, 0.15)
            y_t = np.asarray(y_test_true[h])
            y_p = np.asarray(y_test_prob[h])
            in_bounds = (y_t >= (y_p - margin)) & (y_t <= (y_p + margin))
            test_coverages[h] = round(float(np.mean(in_bounds)), 4)
        return test_coverages
