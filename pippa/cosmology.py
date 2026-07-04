"""Cosmological / inflationary predictions of Pippa (out-of-sample tests).

These formulas were NOT used when fitting the SM coupling constants, so
they are genuine out-of-sample predictions and the most honest test of
whether D = 4/pi encodes physics or is numerology.

Predictions (README sections 19.2, H.4.3):
- Spectral index:   n_s = 1 - 2/(D * N_e)
- Tensor-to-scalar: r   = 12/(D^2 * N_e^2)
- Non-Gaussianity:  f_NL ~ 1/D
- DM/baryon ratio:  Omega_DM/Omega_b = eps^(-D),  eps = D - 1

Reference data: Planck 2018 + BICEP/Keck (used only as comparison, not
as inputs to the formulas).

No external dependencies (stdlib only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import constants

# --- Default model assumption --------------------------------------------

#: Number of e-folds of inflation (standard assumption ~50-60).
N_EFOLDS_DEFAULT: float = 60.0


# --- Pippa predictions ----------------------------------------------------


def spectral_index(n_efolds: float = N_EFOLDS_DEFAULT, D: float = constants.D) -> float:
    """Scalar spectral index n_s = 1 - 2/(D * N_e)."""
    return 1.0 - 2.0 / (D * n_efolds)


def tensor_to_scalar(n_efolds: float = N_EFOLDS_DEFAULT, D: float = constants.D) -> float:
    """Tensor-to-scalar ratio r = 12/(D^2 * N_e^2)."""
    return 12.0 / (D**2 * n_efolds**2)


def non_gaussianity(D: float = constants.D) -> float:
    """Local non-Gaussianity amplitude f_NL ~ 1/D."""
    return 1.0 / D


def dm_to_baryon_ratio(
    D: float = constants.D,
    alpha_A: float | None = None,
    alpha_B: float | None = None,
) -> float:
    """
    Baseline:
        Omega_DM/Omega_b = (D - 1)^(-D)

    Alpha-corrected variant:
        Omega_DM/Omega_b = (D - 1)^(-D) * sqrt((1 + 2 alpha_A + alpha_B)/(1 + alpha_A + alpha_B))

    This keeps the alpha-correction tied to the same SU(2)/U(1) structure
    already used in the electroweak sector, without introducing a new
    free parameter.
    """
    eps = D - 1.0
    ratio = eps ** (-D)

    if alpha_A is None or alpha_B is None:
        return ratio

    denom_su2 = 1.0 + alpha_A + alpha_B
    denom_u1 = 1.0 + 2.0 * alpha_A + alpha_B
    if denom_su2 <= 0.0 or denom_u1 <= 0.0:
        raise ValueError("Invalid alpha correction: non-positive denominator")

    return ratio * math.sqrt(denom_u1 / denom_su2)


# --- Reference values (Planck 2018 / BICEP-Keck 2021) --------------------


@dataclass(frozen=True)
class CosmoObservations:
    """Reference cosmological measurements (comparison only)."""

    n_s: float = 0.9649
    n_s_err: float = 0.0042

    # r: only an upper limit exists (BICEP/Keck 2021): r < 0.036 (95% CL).
    r_upper_95: float = 0.036

    # f_NL local (Planck 2018): -0.9 +/- 5.1.
    f_NL: float = -0.9
    f_NL_err: float = 5.1

    # Omega_DM/Omega_b (Planck 2018): 0.2645/0.04930 = 5.366 +/- ~0.07.
    dm_baryon: float = 5.366
    dm_baryon_err: float = 0.07


OBS = CosmoObservations()
