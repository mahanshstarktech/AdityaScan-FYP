"""
Autonomous Telemetry Quality Control and Bad-Packet Filter Daemon for AdityaScan.

Enforces physical bound invariants and telemetry sanity rules based on:
  - CCSDS 131.0-B-3 Space Data Link Telemetry Standards
  - ISRO Aditya-L1 Payload Specifications (SoLEXS, HEL1OS, MAG, ASPEX-SWIS)

Applies bitmask quality flags:
  - 0x00: CLEAN
  - 0x01: OCCULTATION_OR_CALIBRATION
  - 0x02: SPIKE_RATE_EXCEEDED
  - 0x04: PHYSICAL_BOUND_VIOLATION
  - 0x08: CHECKSUM_OR_FRAME_CORRUPT
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

QC_FLAG_CLEAN: int = 0x00
QC_FLAG_OCCULTATION: int = 0x01
QC_FLAG_SPIKE: int = 0x02
QC_FLAG_OUT_OF_BOUNDS: int = 0x04
QC_FLAG_CORRUPT: int = 0x08

# Physical Solar & Interplanetary Bounds at L1
MAG_B_MIN_NT: float = 0.05
MAG_B_MAX_NT: float = 250.0
SWIS_DENSITY_MIN_CM3: float = 0.05
SWIS_DENSITY_MAX_CM3: float = 300.0
SWIS_VELOCITY_MIN_KMS: float = 180.0
SWIS_VELOCITY_MAX_KMS: float = 2200.0
SXR_FLUX_MIN_WM2: float = 1e-9
SXR_FLUX_MAX_WM2: float = 1e-2


@dataclass
class QCPacketResult:
    is_valid: bool
    qc_flag: int
    rejection_reason: Optional[str] = None
    sanitized_value: Optional[float] = None


class TelemetryQCFilter:
    """
    Validates high-cadence satellite packets prior to neural network ingestion.
    """

    def __init__(self, max_sxr_rate_ratio: float = 50.0) -> None:
        self.max_sxr_rate_ratio = max_sxr_rate_ratio
        self._prev_sxr_flux: Optional[float] = None

    def validate_mag_vector(
        self,
        bx: float,
        by: float,
        bz: float,
    ) -> QCPacketResult:
        """Validates interplanetary magnetic field vector components and magnitude."""
        if any(math.isnan(v) or math.isinf(v) for v in (bx, by, bz)):
            return QCPacketResult(False, QC_FLAG_CORRUPT, "NaN or Inf in magnetic vector")

        b_total = math.sqrt(bx ** 2 + by ** 2 + bz ** 2)
        if b_total < MAG_B_MIN_NT or b_total > MAG_B_MAX_NT:
            return QCPacketResult(
                False,
                QC_FLAG_OUT_OF_BOUNDS,
                f"Total |B|={b_total:.2f} nT outside [{MAG_B_MIN_NT}, {MAG_B_MAX_NT}]",
            )

        return QCPacketResult(True, QC_FLAG_CLEAN, sanitized_value=round(b_total, 2))

    def validate_solar_wind_plasma(
        self,
        density_cm3: float,
        bulk_speed_kms: float,
        proton_temp_k: float,
    ) -> QCPacketResult:
        """Validates ASPEX-SWIS solar wind density and velocity parameters."""
        if any(math.isnan(v) or math.isinf(v) for v in (density_cm3, bulk_speed_kms, proton_temp_k)):
            return QCPacketResult(False, QC_FLAG_CORRUPT, "Corrupt plasma telemetry values")

        if density_cm3 < SWIS_DENSITY_MIN_CM3 or density_cm3 > SWIS_DENSITY_MAX_CM3:
            return QCPacketResult(
                False,
                QC_FLAG_OUT_OF_BOUNDS,
                f"Proton density {density_cm3:.1f} cm^-3 out of range",
            )

        if bulk_speed_kms < SWIS_VELOCITY_MIN_KMS or bulk_speed_kms > SWIS_VELOCITY_MAX_KMS:
            return QCPacketResult(
                False,
                QC_FLAG_OUT_OF_BOUNDS,
                f"Bulk speed {bulk_speed_kms:.1f} km/s out of range",
            )

        return QCPacketResult(True, QC_FLAG_CLEAN, sanitized_value=round(density_cm3, 2))

    def validate_sxr_flux(
        self,
        flux_wm2: float,
        is_eclipse_flag: bool = False,
    ) -> QCPacketResult:
        """Validates soft X-ray spectrometer flux and checks for spurious non-physical spikes."""
        if math.isnan(flux_wm2) or math.isinf(flux_wm2):
            return QCPacketResult(False, QC_FLAG_CORRUPT, "NaN SXR flux")

        if is_eclipse_flag:
            return QCPacketResult(False, QC_FLAG_OCCULTATION, "Spacecraft solar occultation")

        if flux_wm2 < SXR_FLUX_MIN_WM2 or flux_wm2 > SXR_FLUX_MAX_WM2:
            return QCPacketResult(
                False,
                QC_FLAG_OUT_OF_BOUNDS,
                f"SXR flux {flux_wm2:.2e} outside valid physical limits",
            )

        # Check single-cadence non-physical spike
        if self._prev_sxr_flux is not None and self._prev_sxr_flux > 0:
            ratio = flux_wm2 / self._prev_sxr_flux
            if ratio > self.max_sxr_rate_ratio or ratio < (1.0 / self.max_sxr_rate_ratio):
                # Rate of change exceeds theoretical thermal limit of coronal plasma heating
                return QCPacketResult(
                    False,
                    QC_FLAG_SPIKE,
                    f"Spike rate ratio {ratio:.1f} exceeds threshold {self.max_sxr_rate_ratio}",
                )

        self._prev_sxr_flux = flux_wm2
        return QCPacketResult(True, QC_FLAG_CLEAN, sanitized_value=flux_wm2)
