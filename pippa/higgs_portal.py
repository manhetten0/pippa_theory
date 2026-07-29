"""A falsifiable four-mode Higgs-portal test for the Pippa information sector.

The minimal benchmark assumes four stable complex SM-singlet scalars, one for
each ``Z2 x Z2`` character mode, with

    L_portal = -lambda_HI (H^dagger H) sum_chi |Xi_chi|^2.

After electroweak symmetry breaking this shifts every mode mass squared by
``lambda_HI v^2 / 2`` and gives the trilinear coupling
``-lambda_HI v h |Xi_chi|^2``.  An open mode contributes

    Gamma(h -> Xi Xi*) = lambda_HI^2 v^2 /(16 pi m_h)
                         sqrt(1 - 4 m_Xi^2/m_h^2).

The module deliberately distinguishes an experimental bound from a Pippa
prediction.  Without a theory-fixed portal coupling and four physical mode
masses, the audit status is ``BLOCKED`` rather than a fitted success.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from . import constants
from . import sector_spectrum


SM_HIGGS_TOTAL_WIDTH_GEV: float = 4.07e-3
SM_HIGGS_INVISIBLE_BRANCHING_FRACTION: float = 1.0e-3
ATLAS_HIGGS_INVISIBLE_BR_LIMIT: float = 0.107


class PortalAuditStatus(str, Enum):
    """Scientific status of one Higgs-portal comparison."""

    BLOCKED = "BLOCKED"
    BOUND_ONLY = "BOUND_ONLY"
    COMPATIBLE = "COMPATIBLE"
    EXCLUDED = "EXCLUDED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class HiggsPortalAudit:
    """Prediction-readiness and collider result for the minimal portal."""

    status: PortalAuditStatus
    missing_inputs: tuple[str, ...]
    portal_coupling: float | None
    mode_masses_gev: tuple[float, float, float, float] | None
    invisible_width_gev: float | None
    invisible_branching_fraction: float | None
    observed_limit: float
    parameters_fixed_by_theory: bool
    detail: str


def higgs_portal_mode_masses(
    bare_mass_squared_gev2: float,
    neg_mixing_gev2: float,
    mir_mixing_gev2: float,
    negmir_mixing_gev2: float,
    portal_coupling: float,
    higgs_vev_gev: float = constants.EXP.higgs_vev_GeV,
) -> tuple[float, float, float, float]:
    r"""Return the four physical masses around the zero-``Xi`` vacuum.

    The order is ``symmetric, Neg-character, Mir-character,
    NegMir-character``.  A negative result for any mass squared means that the
    assumed zero-information-field vacuum is unstable and needs a different
    symmetry-breaking analysis.
    """
    values = (
        bare_mass_squared_gev2,
        neg_mixing_gev2,
        mir_mixing_gev2,
        negmir_mixing_gev2,
        portal_coupling,
        higgs_vev_gev,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("portal mass parameters must be finite")
    if higgs_vev_gev <= 0.0:
        raise ValueError("higgs_vev_gev must be positive")

    common_mass_squared = (
        bare_mass_squared_gev2 + 0.5 * portal_coupling * higgs_vev_gev**2
    )
    mixing_eigenvalues = sector_spectrum.analytic_mixing_eigenvalues(
        neg_mixing_gev2,
        mir_mixing_gev2,
        negmir_mixing_gev2,
    )
    masses_squared = common_mass_squared + mixing_eigenvalues
    if np.any(masses_squared < -1.0e-12):
        raise ValueError("a Pippa character mode is tachyonic around Xi=0")
    masses_squared = np.clip(masses_squared, 0.0, None)
    return tuple(float(value) for value in np.sqrt(masses_squared))


def complex_scalar_higgs_width(
    scalar_mass_gev: float,
    portal_coupling: float,
    higgs_mass_gev: float = constants.EXP.m_H_GeV,
    higgs_vev_gev: float = constants.EXP.higgs_vev_GeV,
) -> float:
    r"""Return ``Gamma(h -> Xi Xi*)`` for one stable complex scalar mode."""
    values = scalar_mass_gev, portal_coupling, higgs_mass_gev, higgs_vev_gev
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Higgs decay inputs must be finite")
    if scalar_mass_gev < 0.0:
        raise ValueError("scalar_mass_gev cannot be negative")
    if higgs_mass_gev <= 0.0 or higgs_vev_gev <= 0.0:
        raise ValueError("Higgs mass and vev must be positive")
    if 2.0 * scalar_mass_gev >= higgs_mass_gev:
        return 0.0

    phase_space = math.sqrt(1.0 - 4.0 * scalar_mass_gev**2 / higgs_mass_gev**2)
    return (
        portal_coupling**2
        * higgs_vev_gev**2
        / (16.0 * math.pi * higgs_mass_gev)
        * phase_space
    )


def total_invisible_width(
    mode_masses_gev: Sequence[float],
    portal_coupling: float,
    higgs_mass_gev: float = constants.EXP.m_H_GeV,
    higgs_vev_gev: float = constants.EXP.higgs_vev_GeV,
) -> float:
    """Sum invisible widths over all open complex information modes."""
    masses = tuple(float(mass) for mass in mode_masses_gev)
    if not masses:
        raise ValueError("at least one information mode is required")
    return float(
        sum(
            complex_scalar_higgs_width(
                mass,
                portal_coupling,
                higgs_mass_gev=higgs_mass_gev,
                higgs_vev_gev=higgs_vev_gev,
            )
            for mass in masses
        )
    )


def invisible_branching_fraction(
    invisible_width_gev: float,
    sm_higgs_width_gev: float = SM_HIGGS_TOTAL_WIDTH_GEV,
    sm_invisible_branching_fraction: float = SM_HIGGS_INVISIBLE_BRANCHING_FRACTION,
) -> float:
    """Return the total SM plus new invisible Higgs branching fraction."""
    if invisible_width_gev < 0.0 or not math.isfinite(invisible_width_gev):
        raise ValueError("invisible_width_gev must be finite and non-negative")
    if sm_higgs_width_gev <= 0.0 or not math.isfinite(sm_higgs_width_gev):
        raise ValueError("sm_higgs_width_gev must be finite and positive")
    if not 0.0 <= sm_invisible_branching_fraction < 1.0:
        raise ValueError("sm_invisible_branching_fraction must lie in [0, 1)")
    sm_invisible_width = sm_higgs_width_gev * sm_invisible_branching_fraction
    return (sm_invisible_width + invisible_width_gev) / (
        sm_higgs_width_gev + invisible_width_gev
    )


def maximum_invisible_width(
    branching_limit: float = ATLAS_HIGGS_INVISIBLE_BR_LIMIT,
    sm_higgs_width_gev: float = SM_HIGGS_TOTAL_WIDTH_GEV,
    sm_invisible_branching_fraction: float = SM_HIGGS_INVISIBLE_BRANCHING_FRACTION,
) -> float:
    """Convert the total branching limit into a new-physics width limit."""
    if not 0.0 < branching_limit < 1.0:
        raise ValueError("branching_limit must lie strictly between zero and one")
    if sm_higgs_width_gev <= 0.0 or not math.isfinite(sm_higgs_width_gev):
        raise ValueError("sm_higgs_width_gev must be finite and positive")
    if not 0.0 <= sm_invisible_branching_fraction < branching_limit:
        raise ValueError("the SM invisible branching fraction must be below the limit")
    numerator = branching_limit - sm_invisible_branching_fraction
    return numerator / (1.0 - branching_limit) * sm_higgs_width_gev


def portal_coupling_upper_bound(
    mode_masses_gev: Sequence[float],
    branching_limit: float = ATLAS_HIGGS_INVISIBLE_BR_LIMIT,
    sm_higgs_width_gev: float = SM_HIGGS_TOTAL_WIDTH_GEV,
    higgs_mass_gev: float = constants.EXP.m_H_GeV,
    higgs_vev_gev: float = constants.EXP.higgs_vev_GeV,
) -> float:
    """Return the largest ``|lambda_HI|`` allowed for fixed physical masses."""
    unit_width = total_invisible_width(
        mode_masses_gev,
        portal_coupling=1.0,
        higgs_mass_gev=higgs_mass_gev,
        higgs_vev_gev=higgs_vev_gev,
    )
    if unit_width == 0.0:
        return float("inf")
    return math.sqrt(
        maximum_invisible_width(branching_limit, sm_higgs_width_gev) / unit_width
    )


def audit_higgs_portal_prediction(
    portal_coupling: float | None = None,
    mode_masses_gev: Sequence[float] | None = None,
    modes_are_stable_and_invisible: bool | None = None,
    parameters_fixed_by_theory: bool = False,
    observed_limit: float = ATLAS_HIGGS_INVISIBLE_BR_LIMIT,
) -> HiggsPortalAudit:
    """Audit whether Pippa makes, rather than fits, a collider prediction."""
    missing: list[str] = []
    if portal_coupling is None:
        missing.append("portal_coupling")
    if mode_masses_gev is None:
        missing.append("four_physical_mode_masses")
    if modes_are_stable_and_invisible is None:
        missing.append("mode_stability_and_detector_invisibility")
    if missing:
        return HiggsPortalAudit(
            status=PortalAuditStatus.BLOCKED,
            missing_inputs=tuple(missing),
            portal_coupling=portal_coupling,
            mode_masses_gev=None,
            invisible_width_gev=None,
            invisible_branching_fraction=None,
            observed_limit=observed_limit,
            parameters_fixed_by_theory=parameters_fixed_by_theory,
            detail="The current theory does not yet determine all collider inputs.",
        )

    masses = tuple(float(mass) for mass in mode_masses_gev)
    if len(masses) != 4:
        raise ValueError("the minimal Pippa portal requires four character-mode masses")
    if not modes_are_stable_and_invisible:
        return HiggsPortalAudit(
            status=PortalAuditStatus.NOT_APPLICABLE,
            missing_inputs=(),
            portal_coupling=float(portal_coupling),
            mode_masses_gev=masses,
            invisible_width_gev=None,
            invisible_branching_fraction=None,
            observed_limit=observed_limit,
            parameters_fixed_by_theory=parameters_fixed_by_theory,
            detail="The direct invisible-Higgs limit does not apply to visible or unstable modes.",
        )

    invisible_width = total_invisible_width(masses, float(portal_coupling))
    branching_fraction = invisible_branching_fraction(invisible_width)
    compatible = branching_fraction <= observed_limit
    if parameters_fixed_by_theory:
        status = (
            PortalAuditStatus.COMPATIBLE if compatible else PortalAuditStatus.EXCLUDED
        )
        detail = (
            "The fixed prediction is below the observed limit."
            if compatible
            else "The fixed prediction exceeds the observed limit."
        )
    else:
        status = PortalAuditStatus.BOUND_ONLY
        relation = "below" if compatible else "above"
        detail = (
            f"This freely chosen parameter point lies {relation} the limit; "
            "it is a constraint, not a Pippa prediction."
        )

    return HiggsPortalAudit(
        status=status,
        missing_inputs=(),
        portal_coupling=float(portal_coupling),
        mode_masses_gev=masses,
        invisible_width_gev=invisible_width,
        invisible_branching_fraction=branching_fraction,
        observed_limit=observed_limit,
        parameters_fixed_by_theory=parameters_fixed_by_theory,
        detail=detail,
    )
