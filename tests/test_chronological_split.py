"""
Unit tests for ChronologicalSplitter temporal leak isolation.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from pipeline.ml.chronological_split import ChronologicalSplitter

def test_chronological_splitter_no_leakage():
    base_time = pd.Timestamp("2024-01-01 00:00:00")
    times = pd.Series([base_time + pd.Timedelta(hours=i) for i in range(1000)])
    labels = pd.Series(np.random.choice(["quiet", "C", "M", "X"], size=1000, p=[0.90, 0.07, 0.02, 0.01]))

    splitter = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, quarantine_buffer_hours=24.0)
    result = splitter.split(times, labels)

    train_max_time = times.iloc[result.train_indices].max()
    val_min_time = times.iloc[result.val_indices].min()
    val_max_time = times.iloc[result.val_indices].max()
    test_min_time = times.iloc[result.test_indices].min()

    assert (val_min_time - train_max_time).total_seconds() >= 24 * 3600
    assert (test_min_time - val_max_time).total_seconds() >= 24 * 3600
    assert len(set(result.train_indices).intersection(set(result.val_indices))) == 0
    assert len(set(result.val_indices).intersection(set(result.test_indices))) == 0
