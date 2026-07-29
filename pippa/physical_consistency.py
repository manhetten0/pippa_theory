r"""Structural and empirical-readiness checks for the Pippa sector model.

Passing a check here means only that the stated minimal model satisfies that
specific condition.  ``BLOCKED`` is used deliberately when the theory has not
defined the dimensional coupling needed to compare with an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from . import constants
from . import sector_spectrum


# Published benchmark constraints used only as readiness thresholds.
GW_SPEED_RELATIVE_BOUND: float = 1.0e-15
MICROSCOPE_EOTVOS_BOUND: float = 2.7e-15


class AuditStatus(str, Enum):
    """Outcome of one sharply stated consistency check."""

    PASS = "PASS"
    FAIL = "FAIL"
    CONDITIONAL = "CONDITIONAL"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class AuditFinding:
    """One result in a physical consistency audit."""

    name: str
    status: AuditStatus
    detail: str


@dataclass(frozen=True)
class PhysicalConsistencyAudit:
    """Collection of structural checks for one set of model assumptions."""

    findings: tuple[AuditFinding, ...]

    def finding(self, name: str) -> AuditFinding:
        """Return a finding by its stable name."""
        for finding in self.findings:
            if finding.name == name:
                return finding
        raise KeyError(name)

    def counts(self) -> dict[AuditStatus, int]:
        """Count findings by status."""
        return {
            status: sum(finding.status is status for finding in self.findings)
            for status in AuditStatus
        }


def fractional_relativistic_omega(
    momentum: float,
    mass_squared: float = 0.0,
    kappa: float = 1.0,
    alpha: float = constants.D,
) -> float:
    r"""Return ``omega=sqrt(m^2+kappa |k|^alpha)``."""
    omega_squared = sector_spectrum.free_inverse(
        momentum,
        mass_squared,
        kappa=kappa,
        alpha=alpha,
    )
    if omega_squared < 0.0:
        raise ValueError("omega squared is negative")
    return math.sqrt(omega_squared)


def fractional_group_velocity(
    momentum: float,
    mass_squared: float = 0.0,
    kappa: float = 1.0,
    alpha: float = constants.D,
) -> float:
    r"""Return ``d omega / d k`` for positive momentum.

    For the massless Pippa exponent ``alpha=4/pi<2``, the group velocity of a
    pure spatial fractional dispersion diverges as ``k -> 0``.  This proves
    Lorentz violation but, by itself, is not a complete microcausality test.
    """
    if momentum < 0.0:
        raise ValueError("momentum must be non-negative")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if momentum == 0.0:
        if mass_squared == 0.0:
            if alpha < 2.0:
                return float("inf")
            if alpha == 2.0:
                return math.sqrt(kappa)
            return 0.0
        if alpha < 1.0:
            return float("inf")
        if alpha == 1.0:
            return kappa / (2.0 * math.sqrt(mass_squared))
        return 0.0

    omega = fractional_relativistic_omega(
        momentum,
        mass_squared=mass_squared,
        kappa=kappa,
        alpha=alpha,
    )
    return kappa * alpha * momentum ** (alpha - 1.0) / (2.0 * omega)


def spatial_fractional_dispersion_is_lorentz_invariant(
    alpha: float,
    kappa: float = 1.0,
    light_speed: float = 1.0,
    tolerance: float = 1.0e-12,
) -> bool:
    r"""Check whether ``omega^2=m^2+kappa |k|^alpha`` has SR form."""
    return abs(alpha - 2.0) <= tolerance and abs(kappa - light_speed**2) <= tolerance


def modified_wave_group_velocity(
    momentum: float,
    fractional_coefficient: float,
    alpha: float = constants.D,
    light_speed: float = 1.0,
) -> float:
    r"""Group velocity for ``omega^2=c^2 k^2 + beta k^alpha``."""
    if momentum <= 0.0:
        raise ValueError("momentum must be positive")
    omega_squared = (
        light_speed**2 * momentum**2
        + fractional_coefficient * momentum**alpha
    )
    if omega_squared <= 0.0:
        raise ValueError("modified omega squared must be positive")
    derivative = (
        2.0 * light_speed**2 * momentum
        + fractional_coefficient * alpha * momentum ** (alpha - 1.0)
    )
    return derivative / (2.0 * math.sqrt(omega_squared))


def weak_gw_fractional_ratio_bound(
    alpha: float = constants.D,
    relative_speed_bound: float = GW_SPEED_RELATIVE_BOUND,
) -> float:
    r"""Bound ``|beta k^(alpha-2)|`` from a weak-dispersion speed limit.

    To first order, ``|delta v|/c = |alpha-1| |beta k^(alpha-2)| / 2``.
    """
    if relative_speed_bound < 0.0:
        raise ValueError("relative_speed_bound must be non-negative")
    coefficient = abs(alpha - 1.0)
    if coefficient == 0.0:
        return float("inf")
    return 2.0 * relative_speed_bound / coefficient


def visible_spectral_data(
    mass_squared: float,
    neg_coupling: float,
    mir_coupling: float,
    negmir_coupling: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the four pole masses squared and their visible spectral weights."""
    masses_squared = mass_squared + sector_spectrum.analytic_mixing_eigenvalues(
        neg_coupling,
        mir_coupling,
        negmir_coupling,
    )
    weights = np.full(4, 0.25)
    return masses_squared, weights


def lorentz_safe_sector_frequencies(
    momentum: float,
    mass_squared: float,
    neg_coupling: float,
    mir_coupling: float,
    negmir_coupling: float,
    light_speed: float = 1.0,
) -> np.ndarray:
    r"""Return ``omega_chi=sqrt(c^2 k^2+m_chi^2)`` for internal-only mixing.

    Here ``D`` may shape the internal couplings, but it does not replace the
    physical spacetime power ``k^2``.  This is the minimal Lorentz-safe route.
    """
    if momentum < 0.0:
        raise ValueError("momentum must be non-negative")
    masses_squared, _ = visible_spectral_data(
        mass_squared,
        neg_coupling,
        mir_coupling,
        negmir_coupling,
    )
    omega_squared = light_speed**2 * momentum**2 + masses_squared
    if np.any(omega_squared < 0.0):
        raise ValueError("a sector mode has negative omega squared")
    return np.sqrt(omega_squared)


def lorentz_safe_sector_group_velocities(
    momentum: float,
    mass_squared: float,
    neg_coupling: float,
    mir_coupling: float,
    negmir_coupling: float,
    light_speed: float = 1.0,
) -> np.ndarray:
    """Return subluminal group velocities for the internal-only construction."""
    frequencies = lorentz_safe_sector_frequencies(
        momentum,
        mass_squared,
        neg_coupling,
        mir_coupling,
        negmir_coupling,
        light_speed=light_speed,
    )
    return light_speed**2 * momentum / frequencies


def minimal_quadratic_model_is_ghost_free(
    positive_time_kinetic: bool = True,
) -> bool:
    """Check pole residues for the minimal orthogonal sector mixing."""
    _, weights = visible_spectral_data(1.0, 0.1, -0.04, 0.02)
    return positive_time_kinetic and bool(np.all(weights > 0.0))


def decoupling_relative_response_error(
    inverse_free_value: float,
    neg_coupling: float,
    mir_coupling: float,
    negmir_coupling: float,
) -> float:
    """Return the fractional visible-response change caused by ``M``."""
    if inverse_free_value <= 0.0:
        raise ValueError("inverse_free_value must be positive")
    free_response = 1.0 / inverse_free_value
    coupled_response = sector_spectrum.visible_propagator(
        inverse_free_value,
        neg_coupling,
        mir_coupling,
        negmir_coupling,
    )
    return abs(coupled_response / free_response - 1.0)


def equivalence_principle_status(
    predicted_eotvos_parameter: float | None,
    bound: float = MICROSCOPE_EOTVOS_BOUND,
) -> AuditFinding:
    """Compare a predicted composition dependence with MICROSCOPE readiness."""
    if predicted_eotvos_parameter is None:
        return AuditFinding(
            "weak_equivalence_principle",
            AuditStatus.BLOCKED,
            "No composition-dependent matter coupling has been derived.",
        )
    if abs(predicted_eotvos_parameter) <= bound:
        return AuditFinding(
            "weak_equivalence_principle",
            AuditStatus.PASS,
            f"|eta|={abs(predicted_eotvos_parameter):.3e} <= {bound:.3e}.",
        )
    return AuditFinding(
        "weak_equivalence_principle",
        AuditStatus.FAIL,
        f"|eta|={abs(predicted_eotvos_parameter):.3e} > {bound:.3e}.",
    )


def audit_minimal_model(
    mass_squared: float,
    neg_coupling: float,
    mir_coupling: float,
    negmir_coupling: float,
    kappa: float = 1.0,
    alpha: float = constants.D,
    self_coupling: float = 0.0,
    fractional_domain: str = "unspecified",
    bridge_dynamics_defined: bool = False,
    matter_mapping_defined: bool = False,
    predicted_eotvos_parameter: float | None = None,
    modifies_graviton_dispersion: bool | None = None,
) -> PhysicalConsistencyAudit:
    """Run an audit without treating missing physical definitions as passes."""
    if fractional_domain not in {"physical_space", "internal_space", "unspecified"}:
        raise ValueError("invalid fractional_domain")

    stability = sector_spectrum.analyze_stability(
        mass_squared,
        neg_coupling,
        mir_coupling,
        negmir_coupling,
        kappa=kappa,
        alpha=alpha,
        self_coupling=self_coupling,
    )
    findings: list[AuditFinding] = [
        AuditFinding(
            "sector_unitarity",
            AuditStatus.PASS,
            "The minimal first-order sector operator is Hermitian.",
        ),
        AuditFinding(
            "quadratic_boundedness",
            AuditStatus.PASS if stability.quadratic_hamiltonian_bounded else AuditStatus.FAIL,
            f"minimum quadratic eigenvalue={stability.minimum_relativistic_omega_squared:.6g}.",
        ),
        AuditFinding(
            "full_boundedness",
            AuditStatus.PASS if stability.full_hamiltonian_bounded else AuditStatus.FAIL,
            "Includes the sign of the quartic self-coupling and UV kinetic term.",
        ),
        AuditFinding(
            "relativistic_tachyons",
            AuditStatus.PASS if stability.relativistic_tachyon_free else AuditStatus.FAIL,
            "No negative omega^2 modes." if stability.relativistic_tachyon_free else (
                "Negative omega^2 modes: " + ", ".join(stability.unstable_modes)
            ),
        ),
        AuditFinding(
            "quadratic_ghosts",
            AuditStatus.PASS,
            "All four visible spectral weights are +1/4 for a positive time kinetic term.",
        ),
    ]

    if fractional_domain == "physical_space":
        lorentz_ok = spatial_fractional_dispersion_is_lorentz_invariant(alpha, kappa)
        findings.append(
            AuditFinding(
                "lorentz_invariance",
                AuditStatus.PASS if lorentz_ok else AuditStatus.FAIL,
                "Physical-space dispersion is Lorentz invariant." if lorentz_ok else (
                    f"Spatial |k|^{alpha:.6g} dispersion is not Lorentz invariant."
                ),
            )
        )
        findings.append(
            AuditFinding(
                "microcausality",
                AuditStatus.BLOCKED,
                "Requires a retarded Green-function analysis of the nonlocal kernel.",
            )
        )
    elif fractional_domain == "internal_space":
        findings.extend(
            [
                AuditFinding(
                    "lorentz_invariance",
                    AuditStatus.CONDITIONAL,
                    "Safe only if the four-dimensional spacetime kinetic term remains standard.",
                ),
                AuditFinding(
                    "microcausality",
                    AuditStatus.CONDITIONAL,
                    "Internal nonlocality is harmless only if it does not connect spacelike events.",
                ),
            ]
        )
    else:
        findings.extend(
            [
                AuditFinding(
                    "lorentz_invariance",
                    AuditStatus.BLOCKED,
                    "The theory must specify whether the fractional coordinate is physical or internal.",
                ),
                AuditFinding(
                    "microcausality",
                    AuditStatus.BLOCKED,
                    "The spacetime support of the inter-sector kernel is undefined.",
                ),
            ]
        )

    findings.append(
        AuditFinding(
            "bridge_field_dynamics",
            AuditStatus.PASS if bridge_dynamics_defined else AuditStatus.BLOCKED,
            "Bridge kinetic term and potential are defined." if bridge_dynamics_defined else (
                "Gauge covariance is kinematic; bridge kinetic term and potential are missing."
            ),
        )
    )
    findings.append(equivalence_principle_status(predicted_eotvos_parameter))

    if matter_mapping_defined:
        fifth_force_status = AuditStatus.CONDITIONAL
        fifth_force_detail = "Matter coupling exists, but its range and strength still need comparison."
    else:
        fifth_force_status = AuditStatus.BLOCKED
        fifth_force_detail = "No dimensional coupling between sector fields and matter is defined."
    findings.extend(
        [
            AuditFinding("fifth_force", fifth_force_status, fifth_force_detail),
            AuditFinding(
                "solar_system_gravity",
                AuditStatus.BLOCKED,
                "No covariant metric field equations or PPN parameters have been derived.",
            ),
        ]
    )

    if modifies_graviton_dispersion is False:
        findings.append(
            AuditFinding(
                "gravitational_wave_speed",
                AuditStatus.CONDITIONAL,
                "Passes only by assuming M does not modify the tensor kinetic operator.",
            )
        )
    elif modifies_graviton_dispersion is True:
        findings.append(
            AuditFinding(
                "gravitational_wave_speed",
                AuditStatus.FAIL if alpha != 2.0 else AuditStatus.CONDITIONAL,
                "A fractional tensor dispersion requires a coefficient below the GW speed bound.",
            )
        )
    else:
        findings.append(
            AuditFinding(
                "gravitational_wave_speed",
                AuditStatus.BLOCKED,
                "The theory does not state whether M modifies gravitational-wave propagation.",
            )
        )

    return PhysicalConsistencyAudit(tuple(findings))
