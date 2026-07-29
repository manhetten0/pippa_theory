r"""Gauge-covariant effective action for the Pippa information field.

This module is deliberately modest: it is not a complete TOE Lagrangian.
It implements the first consistency layer an effective Pippa action must
pass:

* the fractional spatial operator is made gauge-covariant by parallel
  transporters between lattice points;
* a genuine inter-sector ``M`` couples independent ``Neg``, ``Mir`` and
  ``NegMir`` fields into the observable ``N`` sector;
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

The older ``covariant_m_operator`` below is an in-sector convolution.  It is
kept as a useful primitive, but when it shares a kernel with the fractional
Laplacian it obeys ``M = s I - L`` and is not an independent physical sector.
Use ``intersector_m_operator`` for the four-sector Pippa construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants


SECTOR_SIGNATURES: dict[str, tuple[int, int]] = {
    "N": (1, 1),
    "Neg": (-1, 1),
    "Mir": (1, -1),
    "NegMir": (-1, -1),
}
HIDDEN_SECTORS: tuple[str, ...] = ("Neg", "Mir", "NegMir")


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


def gauge_transform_intersector_transport(
    transport: np.ndarray,
    target_phases: np.ndarray | list[float],
    source_phases: np.ndarray | list[float],
    charge: float = 1.0,
) -> np.ndarray:
    r"""Transform a transporter connecting two independent sectors.

    For a channel from sector ``a`` into the visible ``N`` sector,

    ``U^(N<-a)_ij -> g^N_i U^(N<-a)_ij (g^a_j)^*``.

    The independent source and target phases are what distinguish this
    object from an ordinary in-sector Wilson line.
    """
    u = _matrix(transport, "transport", np.complex128)
    g_target = gauge_phase(target_phases, charge=charge)
    g_source = gauge_phase(source_phases, charge=charge)
    if g_target.size != u.shape[0] or g_source.size != u.shape[1]:
        raise ValueError("transport, target phases and source phases must agree")
    return g_target[:, None] * u * np.conjugate(g_source[None, :])


def intersector_kernel_1d(
    n_points: int,
    alpha: float = constants.D,
    spacing: float = 1.0,
    source_permutation: np.ndarray | list[int] | None = None,
    softening: float | None = None,
    periodic: bool = True,
) -> np.ndarray:
    r"""Build a softened bilocal kernel ``K(x_i, T x_j)`` for one channel.

    ``source_permutation[j]`` specifies the lattice coordinate of ``T x_j``.
    The identity permutation describes an internal-sector map; a reversed
    permutation describes a spatial reflection.  More general permutations
    can encode a discrete nonlocal projection.

    Unlike a fractional Laplacian kernel, an inter-sector kernel has a finite
    coincident coupling.  ``softening`` supplies its UV length scale.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    if softening is None:
        softening = 0.5 * spacing
    if softening <= 0.0:
        raise ValueError("softening must be positive")

    if source_permutation is None:
        permutation = np.arange(n_points, dtype=int)
    else:
        permutation = np.asarray(source_permutation, dtype=int)
        if permutation.ndim != 1 or permutation.size != n_points:
            raise ValueError("source_permutation must have one entry per point")
        if not np.array_equal(np.sort(permutation), np.arange(n_points)):
            raise ValueError("source_permutation must be a permutation of lattice indices")

    target_positions = spacing * np.arange(n_points, dtype=float)
    source_positions = spacing * permutation.astype(float)
    distance = np.abs(target_positions[:, None] - source_positions[None, :])
    if periodic:
        circumference = spacing * n_points
        distance = np.minimum(distance, circumference - distance)

    exponent = 0.5 * (1.0 + alpha)
    return np.power(distance * distance + softening * softening, -exponent)


@dataclass(frozen=True)
class IntersectorChannel:
    r"""One hidden-sector channel ``a -> N`` in the Pippa ``Z2 x Z2`` model."""

    source_sector: str
    kernel: np.ndarray
    transport: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.source_sector not in HIDDEN_SECTORS:
            raise ValueError(f"source_sector must be one of {HIDDEN_SECTORS}")
        kernel = _matrix(self.kernel, "kernel", float)
        transport = _matrix(self.transport, "transport", np.complex128)
        if kernel.shape != transport.shape:
            raise ValueError("channel kernel and transport dimensions must agree")
        if not np.all(np.isfinite(kernel)) or np.any(kernel < 0.0):
            raise ValueError("channel kernel must be finite and non-negative")
        if not np.isfinite(self.weight):
            raise ValueError("channel weight must be finite")

    def gauge_transformed(
        self,
        target_phases: np.ndarray | list[float],
        source_phases: np.ndarray | list[float],
        charge: float = 1.0,
    ) -> "IntersectorChannel":
        """Return this channel in independently transformed gauge frames."""
        return IntersectorChannel(
            source_sector=self.source_sector,
            kernel=self.kernel,
            transport=gauge_transform_intersector_transport(
                self.transport,
                target_phases,
                source_phases,
                charge=charge,
            ),
            weight=self.weight,
        )


def intersector_m_operator(
    target_field: np.ndarray | list[complex],
    source_fields: dict[str, np.ndarray | list[complex]],
    channels: tuple[IntersectorChannel, ...] | list[IntersectorChannel],
) -> np.ndarray:
    r"""Project independent hidden-sector fields into the visible sector.

    ``M_N,i = sum_a w_a sum_j K^a_ij U^(N<-a)_ij psi^a_j``.

    ``target_field`` fixes the output lattice and its gauge frame; the value
    of the operator comes only from hidden fields.  Consequently this ``M``
    cannot be reconstructed from the visible-sector fractional Laplacian.
    """
    target = _complex_vector(target_field)
    result = np.zeros_like(target)
    if not channels:
        return result

    for channel in channels:
        if channel.source_sector not in source_fields:
            raise KeyError(f"missing field for sector {channel.source_sector}")
        source = _complex_vector(source_fields[channel.source_sector])
        kernel = _matrix(channel.kernel, "kernel", float)
        transport = _matrix(channel.transport, "transport", np.complex128)
        if source.size != target.size or kernel.shape != (target.size, source.size):
            raise ValueError("target, source and channel dimensions must agree")
        result += channel.weight * np.sum(
            kernel * transport * source[None, :],
            axis=1,
        )
    return result


def intersector_coupling_energy(
    target_field: np.ndarray | list[complex],
    source_fields: dict[str, np.ndarray | list[complex]],
    channels: tuple[IntersectorChannel, ...] | list[IntersectorChannel],
    coupling: float = 1.0,
    measure: float = 1.0,
) -> float:
    r"""Gauge-invariant Hermitian energy ``-g Re <psi_N, M_N>``.

    Taking the real part is the discrete form of adding the Hermitian
    conjugate channel to the action.
    """
    target = _complex_vector(target_field)
    if measure <= 0.0:
        raise ValueError("measure must be positive")
    projected = intersector_m_operator(target, source_fields, channels)
    value = -coupling * measure * measure * np.real(np.vdot(target, projected))
    return float(value)


@dataclass(frozen=True)
class IntersectorCoupling:
    """Conservative coupling of the observable field to hidden sectors."""

    channels: tuple[IntersectorChannel, ...]
    coupling: float = 1.0
    measure: float = 1.0

    def operator(
        self,
        target_field: np.ndarray | list[complex],
        source_fields: dict[str, np.ndarray | list[complex]],
    ) -> np.ndarray:
        """Return the hidden-sector projection in the observable frame."""
        return intersector_m_operator(target_field, source_fields, self.channels)

    def energy(
        self,
        target_field: np.ndarray | list[complex],
        source_fields: dict[str, np.ndarray | list[complex]],
    ) -> float:
        """Return the Hermitian inter-sector coupling energy."""
        return intersector_coupling_energy(
            target_field,
            source_fields,
            self.channels,
            coupling=self.coupling,
            measure=self.measure,
        )


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
    """Gauge-covariant bilocal in-sector convolution.

    ``M_i = sum_j K_ij U_ij psi_j``.

    This primitive is not a genuine inter-sector operator when ``kernel`` is
    also used by ``covariant_fractional_laplacian``.  In that case it obeys
    ``M = s I - L``.  Use ``intersector_m_operator`` for independent sectors.
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
