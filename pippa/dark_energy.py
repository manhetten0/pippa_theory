"""Dark-energy scale audit for the four-sector Pippa spectrum.

The module asks a narrow question: can the minimal ``Z2 x Z2`` mixing matrix
contain one Hubble-light homogeneous mode while its other modes remain stable?
It does not identify that mode with observed dark energy by assumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import sector_spectrum


H0_PLANCK_KM_S_MPC: float = 67.4
OMEGA_M_PLANCK: float = 0.315

M_PER_MPC: float = 3.085677581491367e22
M_PER_KPC: float = 3.085677581491367e19
G_SI: float = 6.67430e-11
C_SI: float = 299_792_458.0
HBAR_SI: float = 1.054571817e-34
EV_J: float = 1.602176634e-19


@dataclass(frozen=True)
class DarkEnergyModeAudit:
    """Physical scales and tuning of the symmetric one-light-mode solution."""

    hubble_mass_eV: float
    dark_energy_density_j_m3: float
    vacuum_energy_scale_eV: float
    potential_amplitude_scale_eV: float
    mixing_scale_eV: float
    light_mode_mass_eV: float
    heavy_mode_mass_eV: float
    heavy_to_light_hierarchy: float
    cancellation_fraction: float
    pseudo_goldstone_decay_constant_eV: float
    decay_constant_over_reduced_planck: float
    linear_amplitude_portal_limit: float
    compact_phase_portal_preserves_symmetry: bool
    graph_laplacian_protected: bool
    target_is_unique_light_mode: bool
    relativistic_tachyon_free: bool


def hubble_rate_s_inverse(
    H0_km_s_mpc: float = H0_PLANCK_KM_S_MPC,
) -> float:
    """Convert a Hubble constant in km/s/Mpc to inverse seconds."""
    if not math.isfinite(H0_km_s_mpc) or H0_km_s_mpc <= 0.0:
        raise ValueError("H0_km_s_mpc must be finite and positive")
    return H0_km_s_mpc * 1000.0 / M_PER_MPC


def hubble_mass_eV(H0_km_s_mpc: float = H0_PLANCK_KM_S_MPC) -> float:
    """Return the Hubble energy ``hbar H0`` in eV."""
    return HBAR_SI * hubble_rate_s_inverse(H0_km_s_mpc) / EV_J


def dark_energy_density_j_m3(
    H0_km_s_mpc: float = H0_PLANCK_KM_S_MPC,
    omega_m: float = OMEGA_M_PLANCK,
) -> float:
    """Return ``Omega_DE rho_crit c^2`` for a flat reference cosmology."""
    if not math.isfinite(omega_m) or not 0.0 <= omega_m < 1.0:
        raise ValueError("omega_m must be finite and lie in [0, 1)")
    H0 = hubble_rate_s_inverse(H0_km_s_mpc)
    critical_mass_density = 3.0 * H0 * H0 / (8.0 * math.pi * G_SI)
    return (1.0 - omega_m) * critical_mass_density * C_SI * C_SI


def vacuum_energy_scale_eV(
    H0_km_s_mpc: float = H0_PLANCK_KM_S_MPC,
    omega_m: float = OMEGA_M_PLANCK,
) -> float:
    """Return ``rho_DE^(1/4)`` in natural energy units."""
    density = dark_energy_density_j_m3(H0_km_s_mpc, omega_m)
    energy_joule = (density * (HBAR_SI * C_SI) ** 3) ** 0.25
    return energy_joule / EV_J


def inverse_length_scale_eV(length_kpc: float) -> float:
    """Return ``hbar c/L`` in eV for a length given in kpc."""
    if not math.isfinite(length_kpc) or length_kpc <= 0.0:
        raise ValueError("length_kpc must be finite and positive")
    return HBAR_SI * C_SI / (length_kpc * M_PER_KPC * EV_J)


def reduced_planck_energy_eV() -> float:
    """Return the reduced Planck energy ``sqrt(hbar c^5/(8 pi G))``."""
    return math.sqrt(HBAR_SI * C_SI**5 / (8.0 * math.pi * G_SI)) / EV_J


def portal_coupling_for_mass_protection(
    target_mass_eV: float,
    higgs_vev_gev: float = 246.0,
) -> float:
    r"""Return ``2 m_target^2/v^2`` for an unprotected linear amplitude.

    This bound does not apply to a Goldstone phase of a nonzero condensate:
    ``|H|^2 |Xi|^2`` preserves the common phase symmetry in that realization.
    """
    if not math.isfinite(target_mass_eV) or target_mass_eV < 0.0:
        raise ValueError("target_mass_eV must be finite and non-negative")
    if not math.isfinite(higgs_vev_gev) or higgs_vev_gev <= 0.0:
        raise ValueError("higgs_vev_gev must be finite and positive")
    target_mass_gev = target_mass_eV * 1.0e-9
    return 2.0 * target_mass_gev**2 / higgs_vev_gev**2


def compact_phase_portal_invariance_error(
    amplitudes: tuple[float, float, float, float] | list[float],
    phases: tuple[float, float, float, float] | list[float],
    common_shift: float,
) -> float:
    r"""Test common-phase invariance of ``sum_g |Xi_g|^2``."""
    radial = np.asarray(amplitudes, dtype=float)
    theta = np.asarray(phases, dtype=float)
    if radial.shape != (4,) or theta.shape != (4,):
        raise ValueError("amplitudes and phases must contain four values")
    if not np.all(np.isfinite(radial)) or not np.all(np.isfinite(theta)):
        raise ValueError("amplitudes and phases must be finite")
    if np.any(radial < 0.0) or not math.isfinite(common_shift):
        raise ValueError("amplitudes must be non-negative and shift finite")
    fields = radial * np.exp(1j * theta)
    shifted = radial * np.exp(1j * (theta + common_shift))
    base_density = float(np.sum(np.abs(fields) ** 2))
    shifted_density = float(np.sum(np.abs(shifted) ** 2))
    return abs(shifted_density - base_density) / max(
        abs(base_density),
        np.finfo(float).tiny,
    )


def audit_symmetric_dark_energy_mode(
    mixing_scale_eV: float | None = None,
    *,
    H0_km_s_mpc: float = H0_PLANCK_KM_S_MPC,
    omega_m: float = OMEGA_M_PLANCK,
) -> DarkEnergyModeAudit:
    r"""Audit ``a=b=c=-mu^2`` with one symmetric Hubble-light mode.

    Setting ``m0^2=3 mu^2+m_DE^2`` gives

    ``m_symmetric^2=m_DE^2`` and
    ``m_hidden^2=4 mu^2+m_DE^2``.
    """
    light_mass = hubble_mass_eV(H0_km_s_mpc)
    vacuum_scale = vacuum_energy_scale_eV(H0_km_s_mpc, omega_m)
    mixing_scale = vacuum_scale if mixing_scale_eV is None else mixing_scale_eV
    if not math.isfinite(mixing_scale) or mixing_scale <= 0.0:
        raise ValueError("mixing_scale_eV must be finite and positive")

    mu_squared = mixing_scale * mixing_scale
    target_squared = light_mass * light_mass
    tuned = sector_spectrum.tune_character_mode(
        neg_coupling=-mu_squared,
        mir_coupling=-mu_squared,
        negmir_coupling=-mu_squared,
        target_mode="symmetric",
        target_mass_squared=target_squared,
    )
    heavy_squared = tuned.mode_masses_squared[1]
    heavy_mass = math.sqrt(max(heavy_squared, 0.0))
    graph_laplacian = sector_spectrum.sector_graph_laplacian(
        mu_squared,
        mu_squared,
        mu_squared,
    )
    protected = bool(
        np.allclose(
            graph_laplacian + target_squared * np.eye(4),
            tuned.bare_mass_squared * np.eye(4)
            + sector_spectrum.z2x2_mixing_matrix(
                -mu_squared,
                -mu_squared,
                -mu_squared,
            ),
            rtol=1.0e-12,
            atol=max(target_squared, np.finfo(float).tiny),
        )
    )
    # If the field is frozen near the top of 1-cos(phi/f), its present energy
    # is twice the cosine amplitude.  Normalize that top, not the amplitude,
    # to rho_DE so that height, curvature and FRW evolution use one convention.
    potential_amplitude_scale = vacuum_scale / 2.0**0.25
    decay_constant = (
        potential_amplitude_scale * potential_amplitude_scale / light_mass
    )
    reduced_planck = reduced_planck_energy_eV()

    return DarkEnergyModeAudit(
        hubble_mass_eV=light_mass,
        dark_energy_density_j_m3=dark_energy_density_j_m3(
            H0_km_s_mpc,
            omega_m,
        ),
        vacuum_energy_scale_eV=vacuum_scale,
        potential_amplitude_scale_eV=potential_amplitude_scale,
        mixing_scale_eV=mixing_scale,
        light_mode_mass_eV=light_mass,
        heavy_mode_mass_eV=heavy_mass,
        heavy_to_light_hierarchy=heavy_mass / light_mass,
        cancellation_fraction=tuned.cancellation_fraction,
        pseudo_goldstone_decay_constant_eV=decay_constant,
        decay_constant_over_reduced_planck=decay_constant / reduced_planck,
        linear_amplitude_portal_limit=portal_coupling_for_mass_protection(
            light_mass
        ),
        compact_phase_portal_preserves_symmetry=(
            compact_phase_portal_invariance_error(
                amplitudes=[0.8, 1.0, 1.2, 1.4],
                phases=[0.1, -0.3, 0.7, 1.1],
                common_shift=0.83,
            )
            < 1.0e-15
        ),
        graph_laplacian_protected=protected,
        target_is_unique_light_mode=tuned.target_is_unique_light_mode,
        relativistic_tachyon_free=tuned.relativistic_tachyon_free,
    )
