"""
Unit tests for AdityaScan PlasmaBetaCalculator.

Validates:
  - Thermal vs magnetic pressure balance under typical quiet solar wind conditions.
  - CME / Magnetic cloud signature identification (low beta, high |B|).
  - Alfvén speed and Mach number calculations.
  - Invalid and edge-case inputs (NaN, zero, negative density/field).
"""

import math
import pytest
from pipeline.physics.plasma_beta_calculator import PlasmaBetaCalculator, PlasmaMHDState


def test_quiet_solar_wind_regime():
    calc = PlasmaBetaCalculator()
    # Typical 1 AU conditions: n_p = 5 cm^-3, v = 400 km/s, T_p = 1e5 K, B = 5 nT
    state = calc.compute(
        unix_time=1730592000.0,
        proton_density_cm3=5.0,
        bulk_speed_kms=400.0,
        proton_temp_k=100000.0,
        b_total_nt=5.0,
    )
    assert state is not None
    assert isinstance(state, PlasmaMHDState)
    # P_th = 5e6 * 1.38e-23 * 1e5 ~ 6.9e-12 Pa
    # P_mag = (5e-9)^2 / (2 * 4pi*1e-7) ~ 9.95e-12 Pa
    # beta ~ 0.69
    assert 0.5 < state.plasma_beta < 1.0
    # rho = 5e6 * 1.67e-27 ~ 8.36e-21 kg/m^3
    # v_A = 5e-9 / sqrt(4pi*1e-7 * 8.36e-21) ~ 48.8 km/s
    assert 40.0 < state.alfven_speed_kms < 60.0
    # Mach number ~ 400 / 49 ~ 8.1
    assert state.alfvenic_mach > 1.0
    assert state.is_super_alfvenic is True
    assert state.is_magnetic_cloud is False


def test_interplanetary_cme_magnetic_cloud():
    calc = PlasmaBetaCalculator()
    # ICME shock / flux rope: high field (25 nT), compressed density (20 cm^-3), cold protons (30,000 K)
    state = calc.compute(
        unix_time=1730595600.0,
        proton_density_cm3=20.0,
        bulk_speed_kms=750.0,
        proton_temp_k=30000.0,
        b_total_nt=25.0,
    )
    assert state is not None
    # In magnetic clouds, magnetic pressure strongly dominates: beta < 0.15
    assert state.plasma_beta < 0.15
    assert state.b_total_nt == 25.0
    assert state.is_magnetic_cloud is True
    assert state.alfven_speed_kms > 100.0


def test_invalid_and_degenerate_inputs():
    calc = PlasmaBetaCalculator()
    # Zero or negative density
    assert calc.compute(1730592000.0, -1.0, 400.0, 100000.0, 5.0) is None
    assert calc.compute(1730592000.0, 0.0, 400.0, 100000.0, 5.0) is None
    # Zero magnetic field
    assert calc.compute(1730592000.0, 5.0, 400.0, 100000.0, 0.0) is None
    # NaN inputs
    assert calc.compute(1730592000.0, float("nan"), 400.0, 100000.0, 5.0) is None
