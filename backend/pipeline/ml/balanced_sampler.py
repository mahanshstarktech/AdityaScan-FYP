"""
Solar Cycle 25 Stratified Dataset Sampler.
Combats 1:1200 extreme flare-to-quiet class imbalance.
"""
import numpy as np
import torch
from typing import Dict, List, Tuple

def compute_effective_num_weights(class_counts: Dict[str, int], beta: float = 0.9999) -> Dict[str, float]:
    """Compute Class-Balanced Loss weights based on effective number of samples."""
    weights = {}
    for c, n in class_counts.items():
        if n > 0:
            eff_num = (1.0 - (beta ** n)) / (1.0 - beta)
            weights[c] = float(1.0 / eff_num)
        else:
            weights[c] = 0.0
    total = sum(weights.values())
    return {c: float(w / total * len(weights)) for c, w in weights.items()}
