"""Scale-covariance audit for the internal fractional Pippa sector.

The checks in this module are necessary tests for a conformal limit, not a
proof of conformal field theory.  They test dilations of the one-dimensional
fractional lattice currently implemented in :mod:`pippa.covariant_action`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import constants
from . import covariant_action


@dataclass(frozen=True)
class ScaleCovarianceAudit:
    """Numerical diagnostics for the candidate scale-invariant limit."""

    fractional_order: float
    internal_dimension: float
    field_scaling_dimension: float
    pippa_profile_exponent: float
    exponent_mismatch: float
    kernel_scaling_error: float
    kinetic_action_scaling_error: float
    covariant_softening_scaling_error: float
    fixed_softening_scaling_error: float
    massless_inverse_scaling_error: float
    unit_mass_inverse_scaling_error: float

    @property
    def necessary_scale_tests_pass(self) -> bool:
        """Return whether all scale-free dilation checks pass numerically."""
        tolerance = 1.0e-12
        return (
            self.exponent_mismatch < tolerance
            and self.kernel_scaling_error < tolerance
            and self.kinetic_action_scaling_error < tolerance
            and self.covariant_softening_scaling_error < tolerance
            and self.massless_inverse_scaling_error < tolerance
        )


def fractional_field_scaling_dimension(
    internal_dimension: float = 1.0,
    fractional_order: float = constants.D,
) -> float:
    r"""Return the Gaussian field dimension ``Delta=(d-alpha)/2``."""
    if not math.isfinite(internal_dimension) or internal_dimension <= 0.0:
        raise ValueError("internal_dimension must be finite and positive")
    if not math.isfinite(fractional_order) or fractional_order <= 0.0:
        raise ValueError("fractional_order must be finite and positive")
    return 0.5 * (internal_dimension - fractional_order)


def pippa_profile_scaling_exponent(D: float = constants.D) -> float:
    """Return the exponent used by ``A_M proportional M_bar^((1-D)/2)``."""
    if not math.isfinite(D) or D <= 0.0:
        raise ValueError("D must be finite and positive")
    return 0.5 * (1.0 - D)


def _relative_matrix_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    return float(np.linalg.norm(actual - expected) / denominator)


def fractional_kernel_scaling_error(
    dilation: float,
    *,
    n_points: int = 16,
    spacing: float = 0.4,
    fractional_order: float = constants.D,
) -> float:
    r"""Test ``K(s x,s y)=s^(-(1+alpha)) K(x,y)`` on the lattice."""
    if not math.isfinite(dilation) or dilation <= 0.0:
        raise ValueError("dilation must be finite and positive")
    base = covariant_action.fractional_kernel_1d(
        n_points,
        alpha=fractional_order,
        spacing=spacing,
    )
    scaled = covariant_action.fractional_kernel_1d(
        n_points,
        alpha=fractional_order,
        spacing=dilation * spacing,
    )
    expected = dilation ** (-(1.0 + fractional_order)) * base
    return _relative_matrix_error(scaled, expected)


def fractional_kinetic_scaling_error(
    dilation: float,
    *,
    n_points: int = 16,
    spacing: float = 0.4,
    fractional_order: float = constants.D,
) -> float:
    """Test dilation invariance of the massless fractional quadratic action."""
    if not math.isfinite(dilation) or dilation <= 0.0:
        raise ValueError("dilation must be finite and positive")

    phase = 2.0 * math.pi * np.arange(n_points, dtype=float) / n_points
    field = np.cos(phase) + 0.35j * np.sin(2.0 * phase)
    transport = covariant_action.identity_transport(n_points)
    base_kernel = covariant_action.fractional_kernel_1d(
        n_points,
        alpha=fractional_order,
        spacing=spacing,
    )
    base_energy = covariant_action.fractional_kinetic_energy(
        field,
        base_kernel,
        transport,
        measure=spacing,
    )

    field_dimension = fractional_field_scaling_dimension(
        internal_dimension=1.0,
        fractional_order=fractional_order,
    )
    scaled_field = dilation ** (-field_dimension) * field
    scaled_kernel = covariant_action.fractional_kernel_1d(
        n_points,
        alpha=fractional_order,
        spacing=dilation * spacing,
    )
    scaled_energy = covariant_action.fractional_kinetic_energy(
        scaled_field,
        scaled_kernel,
        transport,
        measure=dilation * spacing,
    )
    return abs(scaled_energy - base_energy) / max(
        abs(base_energy),
        np.finfo(float).tiny,
    )


def intersector_kernel_scaling_error(
    dilation: float,
    *,
    scale_softening: bool,
    n_points: int = 16,
    spacing: float = 0.4,
    softening: float = 0.2,
    fractional_order: float = constants.D,
) -> float:
    """Test an inter-sector kernel with a scaled or fixed UV softening length."""
    if not math.isfinite(dilation) or dilation <= 0.0:
        raise ValueError("dilation must be finite and positive")
    base = covariant_action.intersector_kernel_1d(
        n_points,
        alpha=fractional_order,
        spacing=spacing,
        softening=softening,
    )
    scaled = covariant_action.intersector_kernel_1d(
        n_points,
        alpha=fractional_order,
        spacing=dilation * spacing,
        softening=dilation * softening if scale_softening else softening,
    )
    expected = dilation ** (-(1.0 + fractional_order)) * base
    return _relative_matrix_error(scaled, expected)


def inverse_propagator_scaling_error(
    momentum: float,
    dilation: float,
    *,
    mass_squared: float = 0.0,
    mode_shift: float = 0.0,
    kappa: float = 1.0,
    fractional_order: float = constants.D,
) -> float:
    r"""Test homogeneity of ``q(k)=m^2+lambda+kappa |k|^alpha``."""
    values = (
        momentum,
        dilation,
        mass_squared,
        mode_shift,
        kappa,
        fractional_order,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all parameters must be finite")
    if momentum < 0.0 or dilation <= 0.0:
        raise ValueError("momentum must be non-negative and dilation positive")
    if kappa < 0.0 or fractional_order <= 0.0:
        raise ValueError("kappa must be non-negative and fractional_order positive")

    deformation = mass_squared + mode_shift
    base = deformation + kappa * momentum**fractional_order
    scaled = deformation + kappa * (dilation * momentum) ** fractional_order
    expected = dilation**fractional_order * base
    denominator = max(abs(expected), np.finfo(float).tiny)
    return float(abs(scaled - expected) / denominator)


def audit_current_scale_limit(dilation: float = 2.0) -> ScaleCovarianceAudit:
    """Run the complete dilation audit for the current Pippa defaults."""
    field_dimension = fractional_field_scaling_dimension()
    profile_exponent = pippa_profile_scaling_exponent()
    return ScaleCovarianceAudit(
        fractional_order=constants.D,
        internal_dimension=1.0,
        field_scaling_dimension=field_dimension,
        pippa_profile_exponent=profile_exponent,
        exponent_mismatch=abs(field_dimension - profile_exponent),
        kernel_scaling_error=fractional_kernel_scaling_error(dilation),
        kinetic_action_scaling_error=fractional_kinetic_scaling_error(dilation),
        covariant_softening_scaling_error=intersector_kernel_scaling_error(
            dilation,
            scale_softening=True,
        ),
        fixed_softening_scaling_error=intersector_kernel_scaling_error(
            dilation,
            scale_softening=False,
        ),
        massless_inverse_scaling_error=inverse_propagator_scaling_error(
            momentum=1.0,
            dilation=dilation,
        ),
        unit_mass_inverse_scaling_error=inverse_propagator_scaling_error(
            momentum=1.0,
            dilation=dilation,
            mass_squared=1.0,
        ),
    )
