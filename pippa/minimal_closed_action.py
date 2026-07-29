r"""Minimal covariant low-energy closure of the four-sector phase theory.

The module implements the four-dimensional phase action

.. math::

    S = \int d^4x \sqrt{-g}\left[
        {M_*^2 \over 2}R + \mathcal L_{\rm SM}
        - {f^2 \over 2}\sum_g \nabla_\mu\theta_g\nabla^\mu\theta_g
        - V_{\rm lock} - V_{\rm DE}
    \right],

where

.. math::

    V_{\rm lock} =
        \sum_{g<h}\kappa_{gh}[1-\cos(\theta_g-\theta_h)],
    \qquad
    V_{\rm DE} = \Lambda^4[1-\cos(q_0)],
    \qquad
    q_0={1\over2}\sum_g\theta_g.

The locking term has an exact global common-phase symmetry.  Its Hessian is
the four-sector graph Laplacian, so it gives masses only to the three relative
character modes.  ``V_DE`` softly breaks that symmetry and gives the common
mode a technically protected mass.

Ordinary matter is minimally coupled to the same metric and has no explicit
``A -> M`` source in this conservative closure.  This avoids an unbalanced
exchange current and a tree-level fifth force, but it also means that the
baryon-to-information matching is not derived here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import sector_spectrum


COMMON_CHARACTER = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=float)


def _four_vector(
    values: np.ndarray | list[float] | tuple[float, ...],
    name: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (4,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain four finite values")
    return vector


@dataclass(frozen=True)
class PotentialTerms:
    """The relative-locking and common-mode contributions to the potential."""

    locking: float
    dark_energy: float

    @property
    def total(self) -> float:
        return self.locking + self.dark_energy


@dataclass(frozen=True)
class PhaseModeSpectrum:
    """Canonical masses in finite-group character order."""

    mode_names: tuple[str, str, str, str]
    masses_squared: tuple[float, float, float, float]
    common_mode_protected_when_unbroken: bool
    relative_modes_non_tachyonic: bool


@dataclass(frozen=True)
class StressEnergyState:
    """Local orthonormal stress tensor of a spherically symmetric phase state."""

    time_kinetic: float
    radial_gradient: float
    potential: float
    energy_density: float
    radial_pressure: float
    tangential_pressure: float

    @property
    def isotropic_average_pressure(self) -> float:
        return (self.radial_pressure + 2.0 * self.tangential_pressure) / 3.0

    @property
    def equation_of_state(self) -> float:
        if self.energy_density == 0.0:
            return math.nan
        return self.isotropic_average_pressure / self.energy_density

    @property
    def anisotropic_stress(self) -> float:
        return self.radial_pressure - self.tangential_pressure


@dataclass(frozen=True)
class WeakFieldSources:
    r"""Effective sources for ``Psi``, nonrelativistic motion and lensing.

    For

    ``ds^2=-(1+2 Phi)dt^2+(1-2 Psi)dx^2``

    in the static linearized limit,

    ``laplacian(Psi) = 4 pi G rho``,
    ``laplacian(Phi) = 4 pi G (rho+p_r+2p_t)``.

    The lensing potential is ``(Phi+Psi)/2``.
    """

    curvature_density: float
    dynamical_density: float
    lensing_density: float
    pressure_trace: float
    anisotropic_stress: float


@dataclass(frozen=True)
class SphericalSourceMasses:
    """Enclosed weak-field source masses for a spherical profile."""

    radius: np.ndarray
    curvature_mass: np.ndarray
    dynamical_mass: np.ndarray
    lensing_mass: np.ndarray


@dataclass(frozen=True)
class HarmonicModeAverage:
    """WKB average of one small-amplitude relative character mode."""

    stress: StressEnergyState
    weak_field: WeakFieldSources
    gradient_to_mass_ratio_squared: float
    oscillations_per_hubble_time: float


@dataclass(frozen=True)
class MinimalPhaseAction:
    """Parameters and derived quantities of the minimal phase action."""

    decay_constant: float
    dark_energy_amplitude: float
    neg_locking_amplitude: float
    mir_locking_amplitude: float
    negmir_locking_amplitude: float

    def __post_init__(self) -> None:
        values = (
            self.decay_constant,
            self.dark_energy_amplitude,
            self.neg_locking_amplitude,
            self.mir_locking_amplitude,
            self.negmir_locking_amplitude,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("action parameters must be finite")
        if self.decay_constant <= 0.0:
            raise ValueError("decay_constant must be positive")
        if any(value < 0.0 for value in values[1:]):
            raise ValueError("potential amplitudes must be non-negative")

    @property
    def locking_amplitudes(self) -> tuple[float, float, float]:
        return (
            self.neg_locking_amplitude,
            self.mir_locking_amplitude,
            self.negmir_locking_amplitude,
        )

    def common_coordinate(
        self,
        phases: np.ndarray | list[float] | tuple[float, ...],
    ) -> float:
        """Return the normalized common character ``q0=sum(theta_g)/2``."""
        theta = _four_vector(phases, "phases")
        return float(COMMON_CHARACTER @ theta)

    def potential_terms(
        self,
        phases: np.ndarray | list[float] | tuple[float, ...],
    ) -> PotentialTerms:
        """Evaluate ``V_lock`` and ``V_DE``."""
        theta = _four_vector(phases, "phases")
        locking = sector_spectrum.phase_locking_energy(
            theta,
            *self.locking_amplitudes,
        )
        common = self.common_coordinate(theta)
        dark_energy = self.dark_energy_amplitude * (1.0 - math.cos(common))
        return PotentialTerms(locking=locking, dark_energy=dark_energy)

    def aligned_potential_hessian(self) -> np.ndarray:
        """Return the phase Hessian at the aligned minimum."""
        graph_hessian = sector_spectrum.sector_graph_laplacian(
            *self.locking_amplitudes
        )
        common_hessian = self.dark_energy_amplitude * np.outer(
            COMMON_CHARACTER,
            COMMON_CHARACTER,
        )
        return graph_hessian + common_hessian

    def canonical_mass_squared_matrix(self) -> np.ndarray:
        """Return the Hessian with canonically normalized fields ``f theta``."""
        return self.aligned_potential_hessian() / self.decay_constant**2

    def mode_spectrum(self) -> PhaseModeSpectrum:
        """Diagonalize the mass matrix in the ``Z2 x Z2`` character basis."""
        transform = sector_spectrum.character_transform()
        character_matrix = (
            transform @ self.canonical_mass_squared_matrix() @ transform.T
        )
        off_diagonal = character_matrix - np.diag(np.diag(character_matrix))
        if not np.allclose(off_diagonal, 0.0, rtol=1.0e-11, atol=1.0e-13):
            raise RuntimeError("the phase Hessian is not character diagonal")
        masses = np.diag(character_matrix)
        tolerance = 1.0e-12 * max(float(np.max(np.abs(masses))), 1.0)
        return PhaseModeSpectrum(
            mode_names=sector_spectrum.CHARACTER_MODES,
            masses_squared=tuple(float(value) for value in masses),
            common_mode_protected_when_unbroken=bool(
                self.dark_energy_amplitude == 0.0
                and abs(float(masses[0])) <= tolerance
            ),
            relative_modes_non_tachyonic=bool(
                np.min(masses[1:]) >= -tolerance
            ),
        )

    def stress_energy_state(
        self,
        phases: np.ndarray | list[float] | tuple[float, ...],
        time_derivatives: np.ndarray | list[float] | tuple[float, ...],
        radial_gradients: np.ndarray | list[float] | tuple[float, ...],
    ) -> StressEnergyState:
        r"""Return ``rho``, ``p_r`` and ``p_t`` in a local orthonormal frame."""
        theta = _four_vector(phases, "phases")
        theta_dot = _four_vector(time_derivatives, "time_derivatives")
        theta_radial = _four_vector(radial_gradients, "radial_gradients")
        time_kinetic = 0.5 * self.decay_constant**2 * float(
            theta_dot @ theta_dot
        )
        radial_gradient = 0.5 * self.decay_constant**2 * float(
            theta_radial @ theta_radial
        )
        potential = self.potential_terms(theta).total
        energy_density = time_kinetic + radial_gradient + potential
        radial_pressure = time_kinetic + radial_gradient - potential
        tangential_pressure = time_kinetic - radial_gradient - potential
        return StressEnergyState(
            time_kinetic=time_kinetic,
            radial_gradient=radial_gradient,
            potential=potential,
            energy_density=energy_density,
            radial_pressure=radial_pressure,
            tangential_pressure=tangential_pressure,
        )


def weak_field_sources(
    energy_density: float,
    radial_pressure: float,
    tangential_pressure: float,
) -> WeakFieldSources:
    """Return the linearized-GR source densities for one local state."""
    values = (energy_density, radial_pressure, tangential_pressure)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("stress-energy components must be finite")
    pressure_trace = radial_pressure + 2.0 * tangential_pressure
    return WeakFieldSources(
        curvature_density=energy_density,
        dynamical_density=energy_density + pressure_trace,
        lensing_density=energy_density + 0.5 * pressure_trace,
        pressure_trace=pressure_trace,
        anisotropic_stress=radial_pressure - tangential_pressure,
    )


def weak_field_sources_from_state(state: StressEnergyState) -> WeakFieldSources:
    """Return weak-field sources derived from a phase stress state."""
    return weak_field_sources(
        state.energy_density,
        state.radial_pressure,
        state.tangential_pressure,
    )


def averaged_harmonic_mode(
    mode_amplitude: float,
    radial_amplitude_gradient: float,
    mode_mass: float,
    decay_constant: float,
    hubble_rate: float,
) -> HarmonicModeAverage:
    r"""Average ``q=Q(r) cos(mt)`` over its rapid oscillations.

    In the quadratic regime,

    ``<K_t>=<V>=f^2 m^2 Q^2/4`` and
    ``<K_r>=f^2 (dQ/dr)^2/4``.

    The cold limit requires both ``m/H >> 1`` and
    ``|dQ/dr|^2/(m^2 Q^2) << 1``.
    """
    values = (
        mode_amplitude,
        radial_amplitude_gradient,
        mode_mass,
        decay_constant,
        hubble_rate,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("harmonic-mode parameters must be finite")
    if mode_amplitude == 0.0:
        raise ValueError("mode_amplitude must be non-zero")
    if mode_mass <= 0.0 or decay_constant <= 0.0 or hubble_rate <= 0.0:
        raise ValueError("mode_mass, decay_constant and hubble_rate must be positive")

    time_kinetic = (
        0.25 * decay_constant**2 * mode_mass**2 * mode_amplitude**2
    )
    potential = time_kinetic
    radial_gradient = (
        0.25 * decay_constant**2 * radial_amplitude_gradient**2
    )
    stress = StressEnergyState(
        time_kinetic=time_kinetic,
        radial_gradient=radial_gradient,
        potential=potential,
        energy_density=time_kinetic + radial_gradient + potential,
        radial_pressure=time_kinetic + radial_gradient - potential,
        tangential_pressure=time_kinetic - radial_gradient - potential,
    )
    gradient_ratio = (
        radial_amplitude_gradient / (mode_mass * mode_amplitude)
    ) ** 2
    return HarmonicModeAverage(
        stress=stress,
        weak_field=weak_field_sources_from_state(stress),
        gradient_to_mass_ratio_squared=gradient_ratio,
        oscillations_per_hubble_time=mode_mass / hubble_rate,
    )


def spherical_source_masses(
    radius: np.ndarray | list[float],
    energy_density: np.ndarray | list[float],
    radial_pressure: np.ndarray | list[float],
    tangential_pressure: np.ndarray | list[float],
) -> SphericalSourceMasses:
    r"""Integrate ``4 pi int_0^r source(s) s^2 ds`` for each weak-field source."""
    r = np.asarray(radius, dtype=float)
    rho = np.asarray(energy_density, dtype=float)
    pressure_r = np.asarray(radial_pressure, dtype=float)
    pressure_t = np.asarray(tangential_pressure, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("radius must be a one-dimensional grid")
    if rho.shape != r.shape or pressure_r.shape != r.shape or pressure_t.shape != r.shape:
        raise ValueError("all profiles must have the radius shape")
    if not all(
        np.all(np.isfinite(values))
        for values in (r, rho, pressure_r, pressure_t)
    ):
        raise ValueError("profiles must be finite")
    if r[0] < 0.0 or np.any(np.diff(r) <= 0.0):
        raise ValueError("radius must be non-negative and strictly increasing")

    pressure_trace = pressure_r + 2.0 * pressure_t
    source_profiles = (
        rho,
        rho + pressure_trace,
        rho + 0.5 * pressure_trace,
    )
    enclosed: list[np.ndarray] = []
    for source in source_profiles:
        integrand = 4.0 * math.pi * source * r * r
        shell_mass = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r)
        enclosed.append(np.concatenate(([0.0], np.cumsum(shell_mass))))
    return SphericalSourceMasses(
        radius=r.copy(),
        curvature_mass=enclosed[0],
        dynamical_mass=enclosed[1],
        lensing_mass=enclosed[2],
    )


def exterior_circular_speed_squared(
    gravitational_constant: float,
    dynamical_mass: float,
    radius: float,
) -> float:
    """Return ``v_c^2=G M_dyn/r`` outside a spherical source."""
    if gravitational_constant <= 0.0 or radius <= 0.0:
        raise ValueError("gravitational_constant and radius must be positive")
    return gravitational_constant * dynamical_mass / radius


def exterior_light_deflection(
    gravitational_constant: float,
    lensing_mass: float,
    impact_parameter: float,
    light_speed: float,
) -> float:
    """Return ``alpha=4 G M_lens/(c^2 b)`` outside a spherical source."""
    if (
        gravitational_constant <= 0.0
        or impact_parameter <= 0.0
        or light_speed <= 0.0
    ):
        raise ValueError("G, impact_parameter and light_speed must be positive")
    return (
        4.0
        * gravitational_constant
        * lensing_mass
        / (light_speed * light_speed * impact_parameter)
    )
