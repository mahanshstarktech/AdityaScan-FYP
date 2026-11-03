"""
Solar Wind Plasma Beta and Alfvén Speed Calculator for AdityaScan.

Grounded in:
  - ASPEX-SWIS User Manual, PRL/ISRO (2024)
  - MAG User Manual, URSC/ISRO (2024)

Computes fundamental magnetohydrodynamic (MHD) quantities from joint in-situ telemetry:
  1. Plasma Beta (β_p):
       β_p = (2 * μ_0 * n_p * k_B * T_p) / |B|^2
     Distinguishes magnetically dominated regimes (β < 1) from kinetic plasma streams (β > 1).
  2. Alfvén Speed (v_A):
       v_A = |B| / sqrt(μ_0 * ρ) = |B| / sqrt(μ_0 * m_p * n_p)  [km/s]
  3. Alfvénic Mach Number (M_A):
       M_A = v_bulk / v_A
     Values of M_A > 1 indicate super-Alfvénic solar wind flow upstream of Earth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# Physical constants (SI units)
MU_0 = 4.0 * math.pi * 1e-7      # H / m
K_B = 1.380649e-23               # J / K
M_P = 1.67262192e-27             # kg (proton mass)


@dataclass
class PlasmaMHDState:
    unix_time: float
    proton_density_cm3: float
    bulk_speed_kms: float
    proton_temp_k: float
    b_total_nt: float
    plasma_beta: float
    alfven_speed_kms: float
    alfvenic_mach: float
    is_super_alfvenic: bool
    is_magnetic_cloud: bool


class PlasmaBetaCalculator:
    """
    Computes real-time MHD parameters by coupling Aditya-L1 MAG and SWIS streams.
    """

    def compute(
        self,
        unix_time: float,
        proton_density_cm3: float,  # n_p in cm^-3
        bulk_speed_kms: float,       # v_bulk in km/s
        proton_temp_k: float,        # T_p in Kelvin
        b_total_nt: float,           # |B| in nT
    ) -> Optional[PlasmaMHDState]:
        if (
            proton_density_cm3 <= 0
            or b_total_nt <= 0
            or proton_temp_k <= 0
            or bulk_speed_kms <= 0
            or any(math.isnan(v) for v in (proton_density_cm3, bulk_speed_kms, proton_temp_k, b_total_nt))
        ):
            return None

        # Convert to SI units
        n_si = proton_density_cm3 * 1e6   # m^-3
        B_si = b_total_nt * 1e-9           # Tesla
        v_si = bulk_speed_kms * 1e3        # m/s

        # 1. Thermal pressure P_th = n_p * k_B * T_p
        p_th = n_si * K_B * proton_temp_k

        # 2. Magnetic pressure P_mag = B^2 / (2 * μ_0)
        p_mag = (B_si ** 2) / (2.0 * MU_0)

        # 3. Plasma Beta β = P_th / P_mag
        beta = p_th / max(1e-20, p_mag)

        # 4. Mass density ρ = n_p * m_p
        rho = n_si * M_P

        # 5. Alfvén speed v_A = B / sqrt(μ_0 * ρ) in m/s -> km/s
        v_A_ms = B_si / math.sqrt(MU_0 * rho)
        v_A_kms = v_A_ms / 1e3

        # 6. Alfvénic Mach number M_A = v_bulk / v_A
        mach_A = bulk_speed_kms / max(1e-3, v_A_kms)

        # Magnetic cloud criteria: low beta (< 0.1) and elevated |B| (> 10 nT)
        is_mag_cloud = (beta < 0.15) and (b_total_nt > 10.0)

        return PlasmaMHDState(
            unix_time=unix_time,
            proton_density_cm3=round(proton_density_cm3, 2),
            bulk_speed_kms=round(bulk_speed_kms, 1),
            proton_temp_k=round(proton_temp_k, 0),
            b_total_nt=round(b_total_nt, 2),
            plasma_beta=round(beta, 4),
            alfven_speed_kms=round(v_A_kms, 1),
            alfvenic_mach=round(mach_A, 2),
            is_super_alfvenic=(mach_A > 1.0),
            is_magnetic_cloud=is_mag_cloud,
        )
