"""
Unit test suite for AdityaScan TelemetryQCFilter.

Validates:
  - MAG vector physical bounds and total field computation.
  - ASPEX-SWIS proton density and velocity bounds check.
  - SXR single-cadence non-physical spike detection and occultation tagging.
  - Bitmask flag assignments.
"""

import math
import pytest
from pipeline.physics.telemetry_qc_filter import (
    TelemetryQCFilter,
    QC_FLAG_CLEAN,
    QC_FLAG_OCCULTATION,
    QC_FLAG_SPIKE,
    QC_FLAG_OUT_OF_BOUNDS,
    QC_FLAG_CORRUPT,
)


def test_mag_vector_qc():
    qc = TelemetryQCFilter()

    # Valid quiet field: B = (2, -3, 4) -> |B| = sqrt(4+9+16) = sqrt(29) ~ 5.38 nT
    res = qc.validate_mag_vector(2.0, -3.0, 4.0)
    assert res.is_valid is True
    assert res.qc_flag == QC_FLAG_CLEAN
    assert res.sanitized_value == pytest.approx(5.39, abs=0.05)

    # Corrupt NaN input
    res_nan = qc.validate_mag_vector(float("nan"), 0.0, 0.0)
    assert res_nan.is_valid is False
    assert res_nan.qc_flag == QC_FLAG_CORRUPT

    # Non-physical huge field (>250 nT)
    res_oob = qc.validate_mag_vector(200.0, 200.0, 200.0)
    assert res_oob.is_valid is False
    assert res_oob.qc_flag == QC_FLAG_OUT_OF_BOUNDS


def test_plasma_swis_qc():
    qc = TelemetryQCFilter()

    # Valid nominal solar wind: 5.0 cm^-3, 420 km/s, 1e5 K
    res = qc.validate_solar_wind_plasma(5.0, 420.0, 100000.0)
    assert res.is_valid is True
    assert res.qc_flag == QC_FLAG_CLEAN

    # Impossible zero velocity
    res_zero_v = qc.validate_solar_wind_plasma(5.0, 0.0, 100000.0)
    assert res_zero_v.is_valid is False
    assert res_zero_v.qc_flag == QC_FLAG_OUT_OF_BOUNDS


def test_sxr_flux_spikes_and_occultation():
    qc = TelemetryQCFilter(max_sxr_rate_ratio=20.0)

    # Nominal baseline: 1e-6 W/m^2 (C1.0 flare level)
    res1 = qc.validate_sxr_flux(1.0e-6)
    assert res1.is_valid is True

    # Realistic escalation to 5e-6 W/m^2 (5x increase)
    res2 = qc.validate_sxr_flux(5.0e-6)
    assert res2.is_valid is True

    # Spurious instant 100x spike (corrupted sensor readout)
    res3 = qc.validate_sxr_flux(5.0e-4)
    assert res3.is_valid is False
    assert res3.qc_flag == QC_FLAG_SPIKE

    # Eclipse flag
    res_eclipse = qc.validate_sxr_flux(1.0e-6, is_eclipse_flag=True)
    assert res_eclipse.is_valid is False
    assert res_eclipse.qc_flag == QC_FLAG_OCCULTATION
