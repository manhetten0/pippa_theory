from __future__ import annotations

import math

from . import constants

N_EFOLDS_DEFAULT: float = 60.0

def spectral_index(n_efolds: float = N_EFOLDS_DEFAULT, D: float = constants.D) -> float:
    return 1.0 - 2.0 / (D * n_efolds)

def tensor_to_scalar(n_efolds: float = N_EFOLDS_DEFAULT, D: float = constants.D) -> float:
    return 12.0 / (D**2 * n_efolds**2)

def non_gaussianity(D: float = constants.D) -> float:
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
