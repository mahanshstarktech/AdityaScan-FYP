"""
Solar Cycle 25 Chronological Dataset Split Generator.
Guarantees forward-in-time train/val/test partitioning with zero data leakage.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

@dataclass
class TemporalSplitResult:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    quarantine_hours: float
    train_date_range: Tuple[str, str]
    val_date_range: Tuple[str, str]
    test_date_range: Tuple[str, str]
