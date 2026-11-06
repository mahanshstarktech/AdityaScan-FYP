"""
Composite Class-Balanced Focal Loss with Brier Calibration Penalty for Solar Flare Forecasting.

Implements:
  1. Multi-class Focal Loss (Lin et al. 2017) with focusing parameter gamma = 2.0
     and inverse class frequency weighting to penalize dominant quiet-Sun classes.
  2. Brier Score Calibration penalty to minimize Expected Calibration Error (ECE).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


class FocalBrierLossCalculator:
    """
    Calculates composite focal and Brier loss metrics across multi-horizon forecasts.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha_weights: Optional[List[float]] = None,
        brier_weight: float = 0.25,
    ) -> None:
        self.gamma = gamma
        # Default alpha weights for [B/Quiet, C, M, X, X+]
        self.alpha_weights = alpha_weights if alpha_weights else [0.05, 0.15, 0.35, 0.75, 1.25]
        self.brier_weight = brier_weight

    def compute_loss(
        self,
        pred_probs: np.ndarray,  # [B, C]
        target_labels: np.ndarray,  # [B]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computes focal loss and Brier score.
        """
        B, C = pred_probs.shape
        eps = 1e-7
        probs = np.clip(pred_probs, eps, 1.0 - eps)

        focal_loss_sum = 0.0
        brier_score_sum = 0.0

        for b in range(B):
            target = int(target_labels[b])
            p_t = probs[b, target]
            alpha_t = self.alpha_weights[target] if target < len(self.alpha_weights) else 1.0

            # Focal loss: -alpha * (1 - p_t)^gamma * log(p_t)
            fl = -alpha_t * ((1.0 - p_t) ** self.gamma) * math.log(p_t)
            focal_loss_sum += fl

            # Brier score: sum_{c=1}^C (p_c - y_c)^2
            one_hot = np.zeros(C)
            one_hot[target] = 1.0
            bs = np.sum((probs[b] - one_hot) ** 2)
            brier_score_sum += bs

        mean_focal = focal_loss_sum / max(1, B)
        mean_brier = brier_score_sum / max(1, B)
        total_loss = mean_focal + self.brier_weight * mean_brier

        return float(total_loss), {
            "total_loss": round(total_loss, 4),
            "focal_loss": round(mean_focal, 4),
            "brier_score": round(mean_brier, 4),
        }
