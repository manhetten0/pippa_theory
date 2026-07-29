"""Conservative quantum-field and Standard-Model embedding for Pippa.

This module does not claim to derive quantum mechanics or the Standard Model
from ``D = 4/pi``.  It defines a consistency layer in which:

* physical spacetime keeps the relativistic ``k^2`` dispersion;
* the fractional Pippa operator acts on internal information modes;
* the Schrödinger dispersion is recovered as a low-momentum limit;
* one Standard-Model generation has its usual anomaly-free representations;
* the four Pippa sectors label SM-singlet information amplitudes, not four
  duplicate copies of ordinary particles and antiparticles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import constants


@dataclass(frozen=True)
class QuantumLimitResult:
    """Exact relativistic and approximate nonrelativistic excitation energy."""

    relativistic_energy: float
    exact_excitation_energy: float
    schrodinger_energy: float
    relative_error: float


@dataclass(frozen=True)
class MeasurementResult:
    """Information accounting for one two-qubit measurement strength."""

    strength: float
    state: NDArray[np.complex128]
    system_density_matrix: NDArray[np.complex128]
    record_density_matrix: NDArray[np.complex128]
    full_norm: float
    full_purity: float
    full_entropy_bits: float
    record_information_bits: float
    system_coherence_bits: float
    information_budget_bits: float
    initial_capacity_bits: float
    mutual_information_bits: float


@dataclass(frozen=True)
class GaugeMultiplet:
    """One representation under ``SU(3)c x SU(2)L x U(1)Y``."""

    name: str
    color_dimension: int
    weak_dimension: int
    hypercharge: Fraction
    color_cubic_index: int = 0

    def __post_init__(self) -> None:
        if self.color_dimension not in {1, 3}:
            raise ValueError("minimal SM multiplets have color dimension 1 or 3")
        if self.weak_dimension not in {1, 2}:
            raise ValueError("minimal SM multiplets have weak dimension 1 or 2")
        if self.color_dimension == 1 and self.color_cubic_index != 0:
            raise ValueError("a color singlet has zero SU(3) cubic index")
        if self.color_dimension == 3 and self.color_cubic_index not in {-1, 1}:
            raise ValueError("a color triplet must be fundamental or antifundamental")
        object.__setattr__(self, "hypercharge", Fraction(self.hypercharge))

    @property
    def multiplicity(self) -> int:
        return self.color_dimension * self.weak_dimension

    @property
    def is_sm_singlet(self) -> bool:
        return (
            self.color_dimension == 1
            and self.weak_dimension == 1
            and self.hypercharge == 0
        )


@dataclass(frozen=True)
class GaugeAnomalyReport:
    """Perturbative and global gauge-anomaly coefficients."""

    su3_cubed: Fraction
    su3_squared_u1: Fraction
    su2_squared_u1: Fraction
    u1_cubed: Fraction
    gravity_squared_u1: Fraction
    left_handed_su2_doublets: int

    @property
    def perturbative_anomalies_cancel(self) -> bool:
        return all(
            coefficient == 0
            for coefficient in (
                self.su3_cubed,
                self.su3_squared_u1,
                self.su2_squared_u1,
                self.u1_cubed,
                self.gravity_squared_u1,
            )
        )

    @property
    def global_su2_anomaly_absent(self) -> bool:
        return self.left_handed_su2_doublets % 2 == 0

    @property
    def anomaly_free(self) -> bool:
        return self.perturbative_anomalies_cancel and self.global_su2_anomaly_absent


SM_GENERATION: tuple[GaugeMultiplet, ...] = (
    GaugeMultiplet("Q_L", 3, 2, Fraction(1, 6), color_cubic_index=1),
    GaugeMultiplet("u_R^c", 3, 1, Fraction(-2, 3), color_cubic_index=-1),
    GaugeMultiplet("d_R^c", 3, 1, Fraction(1, 3), color_cubic_index=-1),
    GaugeMultiplet("L_L", 1, 2, Fraction(-1, 2)),
    GaugeMultiplet("e_R^c", 1, 1, Fraction(1, 1)),
)

STERILE_NEUTRINO = GaugeMultiplet("nu_R^c", 1, 1, Fraction(0, 1))
HIGGS_MULTIPLET = GaugeMultiplet("H", 1, 2, Fraction(1, 2))
INFORMATION_MULTIPLET = GaugeMultiplet("Xi", 1, 1, Fraction(0, 1))


def information_density(
    amplitude: ArrayLike,
    density_scale: float = 1.0,
) -> NDArray[np.float64]:
    r"""Return the information intensity ``A = Z_A Xi^dagger Xi``."""
    field = np.asarray(amplitude, dtype=np.complex128)
    if field.size == 0 or not np.isfinite(field).all():
        raise ValueError("information amplitude must be finite and non-empty")
    if density_scale <= 0.0 or not math.isfinite(density_scale):
        raise ValueError("density_scale must be finite and positive")
    return np.asarray(density_scale * np.abs(field) ** 2, dtype=float)


def born_probability_density(
    wavefunction: ArrayLike,
    measure: float | ArrayLike = 1.0,
) -> NDArray[np.float64]:
    r"""Normalize ``|psi|^2`` in the one-particle quantum limit.

    The return value is a probability *density*: multiplying it by ``measure``
    and summing gives one.  This function implements the Born rule; it does not
    claim that the rule has been derived from the information axioms.
    """
    density = information_density(wavefunction)
    weights = np.asarray(measure, dtype=float)
    if weights.ndim == 0:
        if float(weights) <= 0.0 or not math.isfinite(float(weights)):
            raise ValueError("measure must be finite and positive")
        weights = np.full(density.shape, float(weights))
    else:
        if weights.shape != density.shape:
            raise ValueError("measure and wavefunction shapes must agree")
        if not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise ValueError("measure weights must be finite and positive")

    norm = float(np.sum(density * weights))
    if norm <= 0.0:
        raise ValueError("wavefunction must have non-zero norm")
    return density / norm


def shannon_entropy_bits(probabilities: ArrayLike) -> float:
    """Return Shannon entropy in bits for a normalized probability vector."""
    values = np.asarray(probabilities, dtype=float).ravel()
    if values.size < 1 or not np.isfinite(values).all():
        raise ValueError("probabilities must be finite and non-empty")
    if np.any(values < -1.0e-14):
        raise ValueError("probabilities cannot be negative")
    values = np.clip(values, 0.0, None)
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("probabilities must have a positive sum")
    values /= total
    nonzero = values > 0.0
    return -float(np.sum(values[nonzero] * np.log2(values[nonzero])))


def von_neumann_entropy_bits(density_matrix: ArrayLike) -> float:
    """Return von Neumann entropy in bits for a finite density matrix."""
    matrix = np.asarray(density_matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("density_matrix must be square")
    if not np.isfinite(matrix).all():
        raise ValueError("density_matrix must be finite")
    if not np.allclose(matrix, np.conjugate(matrix.T), rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("density_matrix must be Hermitian")
    trace = np.trace(matrix)
    if abs(trace.imag) > 1.0e-12 or trace.real <= 0.0:
        raise ValueError("density_matrix must have a positive real trace")
    normalized = matrix / trace.real
    eigenvalues = np.linalg.eigvalsh(normalized)
    if float(np.min(eigenvalues)) < -1.0e-12:
        raise ValueError("density_matrix must be positive semidefinite")
    return shannon_entropy_bits(np.clip(eigenvalues, 0.0, None))


def controlled_record_unitary(strength: float) -> NDArray[np.complex128]:
    r"""Return a controlled record rotation in ``|00>,|01>,|10>,|11>`` order.

    The system state ``|0>`` leaves the record unchanged.  Conditional on
    system state ``|1>``, the record is rotated as

    ``|0_R> -> cos(theta)|0_R> + sin(theta)|1_R>``.
    """
    if not math.isfinite(strength):
        raise ValueError("measurement strength must be finite")
    cosine = math.cos(strength)
    sine = math.sin(strength)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, cosine, -sine],
            [0.0, 0.0, sine, cosine],
        ],
        dtype=np.complex128,
    )


def simulate_record_creation(
    alpha: complex,
    beta: complex,
    strength: float,
) -> MeasurementResult:
    r"""Entangle a system qubit with a record qubit and account for ``A+B``.

    The input is ``(alpha|0> + beta|1>) tensor |0_R>``.  For strengths from
    zero to ``pi/2``, record information ``A`` grows while relative-entropy
    coherence ``B`` decreases.  Their sum is the initial binary information
    capacity for this ideal closed measurement.
    """
    if not 0.0 <= strength <= math.pi / 2.0:
        raise ValueError("measurement strength must lie between 0 and pi/2")
    amplitudes = np.asarray([alpha, beta], dtype=np.complex128)
    if not np.isfinite(amplitudes).all():
        raise ValueError("qubit amplitudes must be finite")
    input_norm = float(np.sum(np.abs(amplitudes) ** 2))
    if not math.isclose(input_norm, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError("alpha and beta must define a normalized qubit")

    initial_state = np.array([alpha, 0.0, beta, 0.0], dtype=np.complex128)
    state = controlled_record_unitary(strength) @ initial_state
    coefficient_matrix = state.reshape(2, 2)
    system_density = coefficient_matrix @ np.conjugate(coefficient_matrix.T)
    record_density = np.conjugate(coefficient_matrix.T) @ coefficient_matrix
    full_density = np.outer(state, np.conjugate(state))

    full_norm = float(np.real(np.vdot(state, state)))
    full_purity = float(np.real(np.trace(full_density @ full_density)))
    full_entropy = von_neumann_entropy_bits(full_density)
    system_entropy = von_neumann_entropy_bits(system_density)
    record_information = von_neumann_entropy_bits(record_density)
    dephased_system = np.diag(np.diag(system_density))
    system_coherence = von_neumann_entropy_bits(dephased_system) - system_entropy
    system_coherence = max(0.0, system_coherence)
    initial_capacity = shannon_entropy_bits(np.abs(amplitudes) ** 2)
    information_budget = record_information + system_coherence
    mutual_information = system_entropy + record_information - full_entropy

    return MeasurementResult(
        strength=float(strength),
        state=state,
        system_density_matrix=system_density,
        record_density_matrix=record_density,
        full_norm=full_norm,
        full_purity=full_purity,
        full_entropy_bits=full_entropy,
        record_information_bits=record_information,
        system_coherence_bits=system_coherence,
        information_budget_bits=information_budget,
        initial_capacity_bits=initial_capacity,
        mutual_information_bits=mutual_information,
    )


def unpacking_rate_bits_per_time(
    results: tuple[MeasurementResult, ...] | list[MeasurementResult],
    times: ArrayLike,
) -> NDArray[np.float64]:
    r"""Return ``sigma=dA/dt`` for a sampled ideal measurement trajectory."""
    time_values = np.asarray(times, dtype=float).ravel()
    if len(results) != time_values.size or time_values.size < 2:
        raise ValueError("results and times must have the same length >= 2")
    if not np.isfinite(time_values).all() or np.any(np.diff(time_values) <= 0.0):
        raise ValueError("times must be finite and strictly increasing")
    information = np.asarray(
        [result.record_information_bits for result in results],
        dtype=float,
    )
    edge_order = 2 if time_values.size >= 3 else 1
    return np.gradient(information, time_values, edge_order=edge_order)


def gauge_invariant_coherence(
    amplitude: ArrayLike,
    transport: ArrayLike,
) -> NDArray[np.float64]:
    r"""Return link phases ``arg(Xi_i^* U_ij Xi_j)`` as microscopic ``B``.

    The transporter convention is the same as in ``covariant_action``: it maps
    the amplitude at ``j`` into the gauge frame at ``i``.  Consequently these
    phases are invariant under local U(1) frame changes.
    """
    field = np.asarray(amplitude, dtype=np.complex128)
    links = np.asarray(transport, dtype=np.complex128)
    if field.ndim != 1 or field.size < 1:
        raise ValueError("amplitude must be a non-empty one-dimensional field")
    if links.shape != (field.size, field.size):
        raise ValueError("transport must have one link for every field pair")
    if not np.isfinite(field).all() or not np.isfinite(links).all():
        raise ValueError("amplitude and transport must be finite")
    correlator = np.conjugate(field[:, None]) * links * field[None, :]
    return np.angle(correlator)


def internal_fractional_mass_squared(
    internal_mode_norm: float,
    internal_scale: float,
    alpha: float = constants.D,
) -> float:
    r"""Return ``Lambda_I^2 |q|^alpha`` for an internal Pippa mode.

    The result enters the four-dimensional mass squared.  It does not replace
    the physical spacetime momentum term ``|k|^2``.
    """
    if internal_mode_norm < 0.0 or not math.isfinite(internal_mode_norm):
        raise ValueError("internal_mode_norm must be finite and non-negative")
    if internal_scale < 0.0 or not math.isfinite(internal_scale):
        raise ValueError("internal_scale must be finite and non-negative")
    if alpha <= 0.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    return internal_scale**2 * internal_mode_norm**alpha


def relativistic_mode_energy(
    momentum: float,
    rest_mass: float,
    internal_mass_squared: float = 0.0,
) -> float:
    r"""Return ``omega=sqrt(k^2 + m^2 + lambda_internal)`` in ``c=1`` units."""
    if momentum < 0.0 or not math.isfinite(momentum):
        raise ValueError("momentum must be finite and non-negative")
    if rest_mass < 0.0 or not math.isfinite(rest_mass):
        raise ValueError("rest_mass must be finite and non-negative")
    if internal_mass_squared < 0.0 or not math.isfinite(internal_mass_squared):
        raise ValueError("internal_mass_squared must be finite and non-negative")
    return math.sqrt(momentum**2 + rest_mass**2 + internal_mass_squared)


def mass_shell_invariant(
    momentum: float,
    rest_mass: float,
    internal_mass_squared: float = 0.0,
) -> float:
    r"""Return the Lorentz-invariant combination ``omega^2-|k|^2``."""
    energy = relativistic_mode_energy(momentum, rest_mass, internal_mass_squared)
    return energy**2 - momentum**2


def nonrelativistic_quantum_limit(
    momentum: float,
    rest_mass: float,
    internal_mass_squared: float = 0.0,
) -> QuantumLimitResult:
    r"""Compare the relativistic excitation with its Schrödinger limit.

    For ``k^2 + lambda_internal << m^2``:

    ``sqrt(m^2+k^2+lambda)-m ~= (k^2+lambda)/(2m)``.
    """
    if rest_mass <= 0.0:
        raise ValueError("a positive rest_mass is required for the nonrelativistic limit")
    energy = relativistic_mode_energy(momentum, rest_mass, internal_mass_squared)
    exact = energy - rest_mass
    schrodinger = (momentum**2 + internal_mass_squared) / (2.0 * rest_mass)
    relative_error = 0.0 if exact == 0.0 else abs(schrodinger - exact) / exact
    return QuantumLimitResult(energy, exact, schrodinger, relative_error)


def anomaly_coefficients(
    multiplets: tuple[GaugeMultiplet, ...] | list[GaugeMultiplet],
) -> GaugeAnomalyReport:
    """Calculate anomalies for a list of left-handed Weyl representations."""
    su3_cubed = sum(
        multiplet.weak_dimension * multiplet.color_cubic_index
        for multiplet in multiplets
    )
    su3_squared_u1 = sum(
        multiplet.weak_dimension * Fraction(1, 2) * multiplet.hypercharge
        for multiplet in multiplets
        if multiplet.color_dimension == 3
    )
    su2_squared_u1 = sum(
        multiplet.color_dimension * Fraction(1, 2) * multiplet.hypercharge
        for multiplet in multiplets
        if multiplet.weak_dimension == 2
    )
    u1_cubed = sum(
        multiplet.multiplicity * multiplet.hypercharge**3
        for multiplet in multiplets
    )
    gravity_squared_u1 = sum(
        multiplet.multiplicity * multiplet.hypercharge
        for multiplet in multiplets
    )
    doublets = sum(
        multiplet.color_dimension
        for multiplet in multiplets
        if multiplet.weak_dimension == 2
    )
    return GaugeAnomalyReport(
        su3_cubed=Fraction(su3_cubed),
        su3_squared_u1=Fraction(su3_squared_u1),
        su2_squared_u1=Fraction(su2_squared_u1),
        u1_cubed=Fraction(u1_cubed),
        gravity_squared_u1=Fraction(gravity_squared_u1),
        left_handed_su2_doublets=doublets,
    )


def electric_charges(multiplet: GaugeMultiplet) -> tuple[Fraction, ...]:
    r"""Return component charges from ``Q=T3+Y``."""
    if multiplet.weak_dimension == 1:
        return (multiplet.hypercharge,)
    return (
        Fraction(1, 2) + multiplet.hypercharge,
        Fraction(-1, 2) + multiplet.hypercharge,
    )


def canonical_gauge_coupling(coupling: float, kinetic_coefficient: float) -> float:
    r"""Canonicalize ``-kappa F^2/4 + g J.A`` to ``g/sqrt(kappa)``.

    A common positive gauge-kinetic coefficient is therefore a field
    normalization, not a separate observable prediction by itself.
    """
    if not math.isfinite(coupling):
        raise ValueError("coupling must be finite")
    if kinetic_coefficient <= 0.0 or not math.isfinite(kinetic_coefficient):
        raise ValueError("kinetic_coefficient must be finite and positive")
    return coupling / math.sqrt(kinetic_coefficient)


def minimal_m_operator_accepts(multiplet: GaugeMultiplet) -> bool:
    """Whether the minimal bridge may act without extra SM gauge transporters."""
    return multiplet.is_sm_singlet
