"""
Solar Cycle 25 Stratified Dataset Sampler.
Combats 1:1200 extreme flare-to-quiet class imbalance.
AdityaScan US-ML-02 core deliverable.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Tuple, Iterator, Optional
import numpy as np
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)

def compute_effective_num_weights(class_counts: Dict[str, int], beta: float = 0.9999) -> Dict[str, float]:
    """Compute Class-Balanced Loss weights based on effective number of samples (Cui et al.)."""
    weights = {}
    for c, n in class_counts.items():
        if n > 0:
            eff_num = (1.0 - (beta ** n)) / (1.0 - beta)
            weights[c] = float(1.0 / eff_num)
        else:
            weights[c] = 0.0
    total = sum(weights.values()) or 1.0
    return {c: float(w / total * len(weights)) for c, w in weights.items()}

class StratifiedSolarSampler(Sampler[int]):
    """
    Weighted random sampler for high-cadence multi-modal solar sequences.
    Overcomes the severe 1:1200 ratio of flare peak windows (X/M/C)
    against quiet-Sun baseline irradiance.
    """
    def __init__(
        self,
        labels: List[str],
        oversample_flare_ratio: float = 4.0,
        beta: float = 0.9999,
        num_samples: Optional[int] = None
    ) -> None:
        self.labels = labels
        self.oversample_flare_ratio = oversample_flare_ratio
        
        # Count classes
        counts: Dict[str, int] = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        
        self.class_weights = compute_effective_num_weights(counts, beta=beta)
        
        # Flare classes receive additional boost
        sample_weights = []
        for lab in labels:
            w = self.class_weights.get(lab, 1.0)
            if lab in ("X", "M", "C"):
                w *= self.oversample_flare_ratio
            sample_weights.append(w)
            
        self.weights = torch.as_tensor(sample_weights, dtype=torch.double)
        self.num_samples = num_samples or len(labels)

    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples
