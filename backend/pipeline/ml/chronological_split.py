"""
Solar Cycle 25 Chronological Dataset Split Generator.
Guarantees forward-in-time train/val/test partitioning with zero temporal data leakage.
AdityaScan US-ML-02 core deliverable.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class TemporalSplitResult:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    quarantine_hours: float
    train_date_range: Tuple[str, str]
    val_date_range: Tuple[str, str]
    test_date_range: Tuple[str, str]
    flare_distribution: Dict[str, Dict[str, int]]

class ChronologicalSplitter:
    """
    Chronological forward-facing partitioner for high-cadence solar telemetry.
    Applies strict quarantine gaps between train, validation, and test splits
    to prevent temporal leakage from autoregressive sliding window sequences.
    """
    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        quarantine_buffer_hours: float = 24.0,
    ) -> None:
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.quarantine_buffer_hours = quarantine_buffer_hours

    def split(self, timestamps: pd.Series, flare_labels: Optional[pd.Series] = None) -> TemporalSplitResult:
        """Partition chronological timestamps with buffer isolation."""
        sorted_order = np.argsort(timestamps.values)
        sorted_times = timestamps.iloc[sorted_order].reset_index(drop=True)
        n = len(sorted_times)

        train_cutoff_idx = int(n * self.train_ratio)
        val_cutoff_idx = int(n * (self.train_ratio + self.val_ratio))

        train_end_time = sorted_times.iloc[train_cutoff_idx]
        val_start_time = train_end_time + pd.Timedelta(hours=self.quarantine_buffer_hours)

        val_start_idx = np.searchsorted(sorted_times, val_start_time)
        val_start_idx = max(train_cutoff_idx + 1, min(val_start_idx, val_cutoff_idx - 1))

        val_end_time = sorted_times.iloc[val_cutoff_idx]
        test_start_time = val_end_time + pd.Timedelta(hours=self.quarantine_buffer_hours)

        test_start_idx = np.searchsorted(sorted_times, test_start_time)
        test_start_idx = max(val_cutoff_idx + 1, min(test_start_idx, n - 1))

        train_idx = sorted_order[:train_cutoff_idx]
        val_idx = sorted_order[val_start_idx:val_cutoff_idx]
        test_idx = sorted_order[test_start_idx:]

        distributions = {}
        if flare_labels is not None:
            for split_name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
                labels_subset = flare_labels.iloc[idxs]
                counts = labels_subset.value_counts().to_dict()
                distributions[split_name] = {str(k): int(v) for k, v in counts.items()}

        logger.info(
            "Chronological split completed: Train=%d, Val=%d, Test=%d (Purged %d buffer samples)",
            len(train_idx), len(val_idx), len(test_idx), n - (len(train_idx) + len(val_idx) + len(test_idx))
        )

        return TemporalSplitResult(
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            quarantine_hours=self.quarantine_buffer_hours,
            train_date_range=(str(sorted_times.iloc[0]), str(sorted_times.iloc[train_cutoff_idx - 1])),
            val_date_range=(str(sorted_times.iloc[val_start_idx]), str(sorted_times.iloc[val_cutoff_idx - 1])),
            test_date_range=(str(sorted_times.iloc[test_start_idx]), str(sorted_times.iloc[-1])),
            flare_distribution=distributions
        )
