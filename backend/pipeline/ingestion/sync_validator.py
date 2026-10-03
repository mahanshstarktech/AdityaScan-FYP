"""
Aditya-L1 SoLEXS vs NOAA GOES-18 Telemetry Cross-Correlation Benchmark.
Quantifies sensor alignment, timing lag, and instrumental scale factors.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

@dataclass
class SyncBenchmarkResult:
    pearson_r: float
    time_lag_seconds: float
    flux_ratio: float
    is_synchronized: bool
    num_samples: int
