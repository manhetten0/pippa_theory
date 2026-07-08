r"""Gauge-covariant effective action for the Pippa information field.

This module is deliberately modest: it is not a complete TOE Lagrangian.
It implements the first consistency layer an effective Pippa action must
pass:

* the fractional spatial operator is made gauge-covariant by parallel
  transporters between lattice points;
* the bilocal inter-quadrant operator ``M`` uses the same transporters;
* conservative action terms are kept separate from dissipative
  information-unpacking terms.

The lattice formula is the discrete analogue of

    psi^\dagger (-D_mu D^mu)^(D/2) psi

with a long-range fractional kernel.  For U(1), a transporter U_ij maps the
field at point j into the gauge frame at point i.  Under a local phase
rotation g_i = exp(i q chi_i):

    psi_i -> g_i psi_i
    U_ij  -> g_i U_ij g_j^*

the covariant fractional Laplacian and M operator transform like psi, while
their quadratic action terms remain invariant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants


def _complex_vector(values: np.ndarray | list[complex]) -> np.ndarray:
    """Return a one-dimensional complex vector."""
    arr = np.asarray(values, dtype=np.complex128)
    if arr.ndim != 1:
        raise ValueError("field must be a one-dimensional array")
    return arr


def _real_vector(values: np.ndarray | list[float], name: str) -> np.ndarray:
    """Return a one-dimensional real vector."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return arr


def _matrix(values: np.ndarray, name: str, dtype: type) -> np.ndarray:
    """Return a square matrix with the requested dtype."""
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return arr


def _validate_lattice_objects(
    field: np.ndarray | list[complex],
    kernel: np.ndarray,
    transport: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize field, kernel and transport arrays."""
    psi = _complex_vector(field)
    k = _matrix(kernel, "kernel", float)
    u = _matrix(transport, "transport", np.complex128)
    if k.shape != u.shape or k.shape[0] != psi.size:
        raise ValueError("field, kernel and transport dimensions must agree")
    return psi, k, u


def fractional_kernel_1d(
    n_points: int,
    alpha: float = constants.D,
    spacing: float = 1.0,
    periodic: bool = True,
) -> np.ndarray:
    """Build a one-dimensional long-range fractional kernel.

    For a 1D lattice, the kernel of ``(-Delta)^(alpha/2)`` scales as
    ``1 / |x-y|^(1+alpha)``.  The diagonal is zero because the diagonal
    contribution is supplied by the ``psi_i - U_ij psi_j`` difference.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")

    kernel = np.zeros((n_points, n_points), dtype=float)
    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                continue
            steps = abs(i - j)
            if periodic:
                steps = min(steps, n_points - steps)
            distance = spacing * steps
            kernel[i, j] = 1.0 / distance ** (1.0 + alpha)
    return kernel


def identity_transport(n_points: int) -> np.ndarray:
    """Return trivial U(1) transporters ``U_ij = 1``."""
    if n_points < 1:
        raise ValueError("n_points must be positive")
    return np.ones((n_points, n_points), dtype=np.complex128)


def gauge_phase(phases: np.ndarray | list[float], charge: float = 1.0) -> np.ndarray:
    """Return local U(1) phases ``g_i = exp(i q chi_i)``."""
    chi = _real_vector(phases, "phases")
    return np.exp(1j * charge * chi)


def gauge_transform_field(
    field: np.ndarray | list[complex],
    phases: np.ndarray | list[float],
    charge: float = 1.0,
) -> np.ndarray:
    """Apply a local U(1) gauge transformation to a field."""
    psi = _complex_vector(field)
    g = gauge_phase(phases, charge=charge)
    if g.size != psi.size:
        raise ValueError("field and phases dimensions must agree")
    return g * psi


def gauge_transform_transport(
    transport: np.ndarray,
    phases: np.ndarray | list[float],
    charge: float = 1.0,
) -> np.ndarray:
    """Apply ``U_ij -> g_i U_ij g_j^*`` to transporters."""
    u = _matrix(transport, "transport", np.complex128)
    g = gauge_phase(phases, charge=charge)
    if g.size != u.shape[0]:
        raise ValueError("transport and phases dimensions must agree")
    return g[:, None] * u * np.conjugate(g[None, :])


def covariant_fractional_laplacian(
    field: np.ndarray | list[complex],
    kernel: np.ndarray,
    transport: np.ndarray,
) -> np.ndarray:
    """Gauge-covariant discrete fractional Laplacian.

    ``L_i = sum_j K_ij (psi_i - U_ij psi_j)``.
    """
    psi, k, u = _validate_lattice_objects(field, kernel, transport)
    transported = u * psi[None, :]
    return np.sum(k * (psi[:, None] - transported), axis=1)


def covariant_m_operator(
    field: np.ndarray | list[complex],
    kernel: np.ndarray,
    transport: np.ndarray,
) -> np.ndarray:
    """Gauge-covariant bilocal inter-quadrant operator.

    ``M_i = sum_j K_ij U_ij psi_j``.
    """
    psi, k, u = _validate_lattice_objects(field, kernel, transport)
    return np.sum(k * u * psi[None, :], axis=1)


def fractional_kinetic_energy(
    field: np.ndarray | list[complex],
    kernel: np.ndarray,
    transport: np.ndarray,
    kappa: float = 1.0,
    measure: float = 1.0,
) -> float:
    """Positive gauge-invariant fractional kinetic energy."""
    psi, k, u = _validate_lattice_objects(field, kernel, transport)
    if kappa < 0.0:
        raise ValueError("kappa must be non-negative")
    if measure <= 0.0:
        raise ValueError("measure must be positive")
    diff = psi[:, None] - u * psi[None, :]
    value = 0.5 * kappa * measure * measure * np.sum(k * np.abs(diff) ** 2)
    return float(np.real_if_close(value))


def bilocal_coupling_energy(
    field: np.ndarray | list[complex],
    kernel: np.ndarray,
    transport: np.ndarray,
    coupling: float = 1.0,
    measure: float = 1.0,
) -> float:
    """Gauge-invariant quadratic energy from the bilocal ``M`` operator."""
    psi, k, u = _validate_lattice_objects(field, kernel, transport)
    if measure <= 0.0:
        raise ValueError("measure must be positive")
    m_psi = covariant_m_operator(psi, k, u)
    value = -0.5 * coupling * measure * measure * np.vdot(psi, m_psi)
    return float(np.real_if_close(value))


def self_interaction_energy(
    field: np.ndarray | list[complex],
    self_coupling: float,
    rho0: float,
    measure: float = 1.0,
) -> float:
    """Gauge-invariant ``lambda/4 (|psi|^2 - rho0)^2`` energy."""
    psi = _complex_vector(field)
    if self_coupling < 0.0:
        raise ValueError("self_coupling must be non-negative")
    if rho0 < 0.0:
        raise ValueError("rho0 must be non-negative")
    if measure <= 0.0:
        raise ValueError("measure must be positive")
    density = np.abs(psi) ** 2
    value = 0.25 * self_coupling * measure * np.sum((density - rho0) ** 2)
    return float(value)


@dataclass(frozen=True)
class InformationFieldAction:
    """Conservative effective action for the Pippa information field.

    The class collects the spatial terms that can consistently come from a
    gauge-invariant action.  Damping, decoherence and observation sources
    should be modelled by ``RayleighDissipation`` or a fuller open-system
    formalism, not hidden inside this conservative action.
    """

    kernel: np.ndarray
    transport: np.ndarray
    kappa: float = 1.0
    self_coupling: float = 0.0
    rho0: float = 1.0
    m_coupling: float = 0.0
    measure: float = 1.0

    def terms(self, field: np.ndarray | list[complex]) -> dict[str, float]:
        """Return the conservative action/energy contributions."""
        kinetic = fractional_kinetic_energy(
            field,
            self.kernel,
            self.transport,
            kappa=self.kappa,
            measure=self.measure,
        )
        potential = self_interaction_energy(
            field,
            self.self_coupling,
            self.rho0,
            measure=self.measure,
        )
        bilocal = bilocal_coupling_energy(
            field,
            self.kernel,
            self.transport,
            coupling=self.m_coupling,
            measure=self.measure,
        )
        return {
            "fractional_kinetic": kinetic,
            "self_interaction": potential,
            "bilocal_M": bilocal,
        }

    def energy(self, field: np.ndarray | list[complex]) -> float:
        """Return the total conservative energy."""
        return float(sum(self.terms(field).values()))


@dataclass(frozen=True)
class RayleighDissipation:
    """Open-system dissipative sector for macroscopic ``A`` and ``B`` fields."""

    gamma_A: float = 0.0
    gamma_B: float = 0.0
    measure: float = 1.0

    def potential(
        self,
        A: np.ndarray | list[float],
        B: np.ndarray | list[float],
    ) -> float:
        """Rayleigh potential ``R = 1/2 int (gamma_A A^2 + gamma_B B^2)``."""
        a = _real_vector(A, "A")
        b = _real_vector(B, "B")
        if a.shape != b.shape:
            raise ValueError("A and B dimensions must agree")
        if self.gamma_A < 0.0 or self.gamma_B < 0.0:
            raise ValueError("dissipation coefficients must be non-negative")
        if self.measure <= 0.0:
            raise ValueError("measure must be positive")
        value = 0.5 * self.measure * np.sum(
            self.gamma_A * a * a + self.gamma_B * b * b
        )
        return float(value)

    def flow(
        self,
        A: np.ndarray | list[float],
        B: np.ndarray | list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return pure damping terms ``dA/dt`` and ``dB/dt``."""
        a = _real_vector(A, "A")
        b = _real_vector(B, "B")
        if a.shape != b.shape:
            raise ValueError("A and B dimensions must agree")
        self.potential(a, b)
        return -self.gamma_A * a, -self.gamma_B * b

    def entropy_production(
        self,
        A: np.ndarray | list[float],
        B: np.ndarray | list[float],
    ) -> float:
        """Non-negative production rate associated with the damping sector."""
        a = _real_vector(A, "A")
        b = _real_vector(B, "B")
        if a.shape != b.shape:
            raise ValueError("A and B dimensions must agree")
        self.potential(a, b)
        value = self.measure * np.sum(self.gamma_A * a * a + self.gamma_B * b * b)
        return float(value)
