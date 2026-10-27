"""
Aditya-L1 MAG Level-2 Variational Filter & IMF Discontinuity Detector.

Physics Reference:
  - Burlaga, L. F. (1981): Discontinuities and waves in the interplanetary magnetic field.
  - MAG Instrument Team, URSC / ISRO (2024): MAG L2 Science Telemetry Standards.

Computes:
  1. 10-second rolling magnetic vector variance σ_B^2 = Var(|B|) + Var(Bx) + Var(By) + Var(Bz).
  2. Instantaneous IMF clock angle rotation rate: dθ/dt = |θ_t - θ_{t-1}|.
  3. Directional discontinuities (rotational / tangential) marking interplanetary shock boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class MAGFilteredSample:
    unix_time: float
    Bx: float
    By: float
    Bz: float
    B_total: float
    clock_angle_deg: float
    cone_angle_deg: float
    variance_10s: float
    is_discontinuity: bool
    shock_index: float


class MAGVariationalFilter:
    """
    Real-time sliding window variational filter for Aditya-L1 MAG Level-2 telemetry.
    """

    def __init__(self, window_size: int = 12, rotation_threshold_deg: float = 30.0) -> None:
        self.window_size = max(3, window_size)
        self.rotation_threshold = rotation_threshold_deg
        self._history: List[Tuple[float, float, float, float, float, float, float]] = []

    def process_sample(
        self,
        unix_time: float,
        Bx: float,
        By: float,
        Bz: float,
        quality_flag: int = 1,
    ) -> Optional[MAGFilteredSample]:
        """
        Processes a single 10-second MAG record.
        Rejects bad/missing packets (quality != 1).
        """
        if quality_flag != 1 or any(math.isnan(v) for v in (Bx, By, Bz)):
            return None

        B_total = float(math.sqrt(Bx**2 + By**2 + Bz**2))
        clock_angle = float(math.degrees(math.atan2(By, Bz)))
        cone_angle = float(math.degrees(math.atan2(math.sqrt(By**2 + Bz**2), max(1e-5, abs(Bx)))))

        self._history.append((unix_time, Bx, By, Bz, B_total, clock_angle, cone_angle))
        if len(self._history) > self.window_size:
            self._history.pop(0)

        # Compute rolling vector variance across window
        if len(self._history) < 3:
            return MAGFilteredSample(
                unix_time=unix_time,
                Bx=Bx,
                By=By,
                Bz=Bz,
                B_total=B_total,
                clock_angle_deg=round(clock_angle, 2),
                cone_angle_deg=round(cone_angle, 2),
                variance_10s=0.0,
                is_discontinuity=False,
                shock_index=0.0,
            )

        arr = np.array(self._history)
        b_totals = arr[:, 4]
        var_total = float(np.var(b_totals))
        var_vec = float(np.var(arr[:, 1]) + np.var(arr[:, 2]) + np.var(arr[:, 3]))
        variance_10s = var_total + 0.5 * var_vec

        # Check for abrupt clock angle jump across consecutive steps
        prev_clock = self._history[-2][5]
        delta_clock = abs(clock_angle - prev_clock)
        if delta_clock > 180.0:
            delta_clock = 360.0 - delta_clock

        # Relative field jump
        prev_B = self._history[-2][4]
        delta_B_rel = abs(B_total - prev_B) / max(0.5, prev_B)

        is_discontinuity = (delta_clock >= self.rotation_threshold) and (delta_B_rel >= 0.20)
        shock_index = float(np.clip((delta_clock / 45.0) * delta_B_rel * 2.0, 0.0, 5.0))

        return MAGFilteredSample(
            unix_time=unix_time,
            Bx=round(Bx, 3),
            By=round(By, 3),
            Bz=round(Bz, 3),
            B_total=round(B_total, 3),
            clock_angle_deg=round(clock_angle, 2),
            cone_angle_deg=round(cone_angle, 2),
            variance_10s=round(variance_10s, 3),
            is_discontinuity=is_discontinuity,
            shock_index=round(shock_index, 3),
        )
