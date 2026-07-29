r"""Covariant conservation checks for an information sector sourced by A.

For real information fields ``phi_g`` with

``L_I = -1/2 (nabla phi)^2 - U(phi, A)``,

the diffeomorphism Noether identity is

``nabla_mu T_I^(mu nu) = E_g nabla^nu phi_g - U_A nabla^nu A``.

On the information-field equations of motion, a varying source therefore
produces an exchange current.  Total conservation requires the matter sector
to receive the opposite current.  Treating ``A`` as an external profile while
keeping matter separately conserved is generically inconsistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physical_consistency import AuditFinding, AuditStatus


def _real_vector(values: np.ndarray | list[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return array


def _symmetric_matrix(values: np.ndarray, size: int, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(f"{name} must be symmetric to define a real action")
    return matrix


def trace_information_source(
    energy_density: float,
    pressure: float,
    density_scale: float = 1.0,
) -> float:
    r"""Return the candidate source ``A=-T/rho_*= (rho-3p)/rho_*``."""
    if density_scale <= 0.0:
        raise ValueError("density_scale must be positive")
    return (energy_density - 3.0 * pressure) / density_scale


def quadratic_mixing_potential(
    fields: np.ndarray | list[float],
    mixing: np.ndarray,
    source_factor: float = 1.0,
) -> float:
    r"""Return ``U_M=1/2 f(A) phi^T M phi``."""
    phi = _real_vector(fields, "fields")
    matrix = _symmetric_matrix(mixing, phi.size, "mixing")
    return float(0.5 * source_factor * phi @ matrix @ phi)


def quadratic_source_derivative(
    fields: np.ndarray | list[float],
    mixing: np.ndarray,
    source_factor_derivative: float = 1.0,
) -> float:
    r"""Return ``partial U_M / partial A`` for ``U_M=f(A) phi^T M phi/2``."""
    phi = _real_vector(fields, "fields")
    matrix = _symmetric_matrix(mixing, phi.size, "mixing")
    return float(0.5 * source_factor_derivative * phi @ matrix @ phi)


def information_stress_divergence(
    eom_residuals: np.ndarray | list[float],
    field_gradients: np.ndarray,
    source_gradient: np.ndarray | list[float],
    potential_source_derivative: float,
) -> np.ndarray:
    r"""Evaluate ``E_g grad phi_g - U_A grad A`` from the Noether identity."""
    residuals = _real_vector(eom_residuals, "eom_residuals")
    gradients = np.asarray(field_gradients, dtype=float)
    source_grad = _real_vector(source_gradient, "source_gradient")
    if gradients.ndim != 2 or gradients.shape[0] != residuals.size:
        raise ValueError("field_gradients must have one row per field")
    if gradients.shape[1] != source_grad.size:
        raise ValueError("field and source gradients must share spacetime dimension")
    return residuals @ gradients - potential_source_derivative * source_grad


def required_matter_exchange_current(
    information_divergence: np.ndarray | list[float],
) -> np.ndarray:
    """Return the opposite current required by total conservation."""
    return -_real_vector(information_divergence, "information_divergence")


def total_conservation_residual(
    matter_divergence: np.ndarray | list[float],
    information_divergence: np.ndarray | list[float],
    bridge_divergence: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Return ``nabla(T_matter + T_information + T_bridge)``."""
    matter = _real_vector(matter_divergence, "matter_divergence")
    information = _real_vector(information_divergence, "information_divergence")
    if matter.shape != information.shape:
        raise ValueError("matter and information divergences must agree")
    total = matter + information
    if bridge_divergence is not None:
        bridge = _real_vector(bridge_divergence, "bridge_divergence")
        if bridge.shape != total.shape:
            raise ValueError("bridge divergence must share spacetime dimension")
        total = total + bridge
    return total


def flrw_continuity_residual(
    energy_density: float,
    pressure: float,
    density_time_derivative: float,
    hubble_rate: float,
    exchange_rate: float = 0.0,
) -> float:
    r"""Return ``dot rho + 3H(rho+p) - Q`` for one cosmological sector."""
    return density_time_derivative + 3.0 * hubble_rate * (
        energy_density + pressure
    ) - exchange_rate


def required_flrw_pressure(
    energy_density: float,
    density_time_derivative: float,
    hubble_rate: float,
    exchange_rate: float = 0.0,
) -> float:
    """Return the pressure required by the sector continuity equation."""
    if hubble_rate == 0.0:
        raise ValueError("hubble_rate must be non-zero")
    return (
        exchange_rate - density_time_derivative
    ) / (3.0 * hubble_rate) - energy_density


def effective_w_for_matter_power(source_power: float) -> float:
    r"""Return ``w=n-1`` when ``rho_M proportional A^n`` and dust ``A~a^-3``."""
    return source_power - 1.0


@dataclass(frozen=True)
class ConservationAudit:
    """Noether and stress-tensor status for a proposed source construction."""

    findings: tuple[AuditFinding, ...]

    def finding(self, name: str) -> AuditFinding:
        """Return one conservation finding by name."""
        for finding in self.findings:
            if finding.name == name:
                return finding
        raise KeyError(name)


def audit_conservation(
    source_role: str,
    kernel_domain: str = "internal_local_x",
    bridges_dynamic: bool = False,
    source_is_constant: bool = False,
) -> ConservationAudit:
    """Audit whether the proposed construction has a conserved total tensor.

    ``source_role`` is one of ``external``, ``dynamic`` or ``composite_varied``.
    The last option means that ``A`` is built from matter and the full action is
    varied, including the implicit matter dependence of ``A``.
    """
    if source_role not in {"external", "dynamic", "composite_varied"}:
        raise ValueError("invalid source_role")
    if kernel_domain not in {
        "internal_local_x",
        "covariant_spacetime_nonlocal",
        "unspecified",
    }:
        raise ValueError("invalid kernel_domain")

    findings: list[AuditFinding] = []
    if source_role == "external" and not source_is_constant:
        findings.extend(
            [
                AuditFinding(
                    "total_stress_conservation",
                    AuditStatus.FAIL,
                    "A varying external A injects momentum without a reciprocal equation.",
                ),
                AuditFinding(
                    "matter_separate_conservation",
                    AuditStatus.PASS,
                    "Matter may be conserved separately, but then the full source is inconsistent.",
                ),
                AuditFinding(
                    "information_separate_conservation",
                    AuditStatus.FAIL,
                    "On shell, divergence equals -U_A grad A.",
                ),
            ]
        )
    elif source_role == "external":
        findings.extend(
            [
                AuditFinding(
                    "total_stress_conservation",
                    AuditStatus.CONDITIONAL,
                    "A constant external source does not exchange momentum in this background.",
                ),
                AuditFinding(
                    "matter_separate_conservation",
                    AuditStatus.PASS,
                    "No source gradient drives exchange.",
                ),
                AuditFinding(
                    "information_separate_conservation",
                    AuditStatus.PASS,
                    "No source gradient drives exchange.",
                ),
            ]
        )
    else:
        findings.extend(
            [
                AuditFinding(
                    "total_stress_conservation",
                    AuditStatus.PASS,
                    "Diffeomorphism invariance conserves the sum on all field equations.",
                ),
                AuditFinding(
                    "matter_separate_conservation",
                    AuditStatus.FAIL,
                    "Matter exchanges energy-momentum with the information sector.",
                ),
                AuditFinding(
                    "information_separate_conservation",
                    AuditStatus.FAIL,
                    "Information stress is not separately conserved when grad A is non-zero.",
                ),
                AuditFinding(
                    "exchange_current_cancellation",
                    AuditStatus.PASS,
                    "The two sector currents cancel in the total tensor.",
                ),
            ]
        )

    if kernel_domain == "internal_local_x":
        findings.append(
            AuditFinding(
                "kernel_covariance",
                AuditStatus.PASS,
                "The kernel is nonlocal only internally and local at spacetime x.",
            )
        )
    elif kernel_domain == "covariant_spacetime_nonlocal":
        findings.append(
            AuditFinding(
                "kernel_covariance",
                AuditStatus.CONDITIONAL,
                "Requires a covariant bi-scalar kernel, measure and causal prescription.",
            )
        )
    else:
        findings.append(
            AuditFinding(
                "kernel_covariance",
                AuditStatus.BLOCKED,
                "The spacetime transformation law and support of K(x,y) are undefined.",
            )
        )

    findings.append(
        AuditFinding(
            "bridge_stress_tensor",
            AuditStatus.PASS if bridges_dynamic else AuditStatus.BLOCKED,
            "Dynamic bridges contribute their Noether stress tensor." if bridges_dynamic else (
                "A spacetime-dependent fixed bridge can carry unaccounted momentum."
            ),
        )
    )
    return ConservationAudit(tuple(findings))
