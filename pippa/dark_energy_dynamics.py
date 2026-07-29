"""FRW evolution of the protected common sector phase.

The common phase is treated as a canonical pseudo-Goldstone field
``phi = f theta`` with

``V(theta) = V_amp [1 - cos(theta)]``.

All densities are normalized to today's critical density and the independent
variable is ``N = ln(a)``.  The scalar contribution is included in the
Friedmann equation rather than evolved on a fixed Lambda-CDM background.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


OMEGA_M0_REFERENCE: float = 0.315
OMEGA_R0_REFERENCE: float = 9.2e-5


@dataclass(frozen=True)
class PseudoGoldstoneCosmology:
    """Parameters fixed in units of the present reduced Planck scale."""

    omega_m0: float
    omega_r0: float
    omega_de_target: float
    potential_amplitude: float
    decay_constant_over_planck: float
    mass_over_hubble_at_top: float
    initial_redshift: float


@dataclass(frozen=True)
class EvolutionPoint:
    """Observable state at one redshift."""

    redshift: float
    angle: float
    field_velocity_dN: float
    hubble_over_h0: float
    omega_phi: float
    density_in_present_critical_units: float
    equation_of_state: float
    deceleration: float


@dataclass(frozen=True)
class PseudoGoldstoneEvolution:
    """Cosmological history for one initial phase."""

    cosmology: PseudoGoldstoneCosmology
    initial_angle: float
    initial_field_velocity_dN: float
    points: tuple[EvolutionPoint, ...]
    solver_success: bool
    solver_message: str

    @property
    def today(self) -> EvolutionPoint:
        return min(self.points, key=lambda point: abs(point.redshift))


@dataclass(frozen=True)
class InitialAngleScan:
    """Measure of the Lambda-like initial-angle interval."""

    cosmology: PseudoGoldstoneCosmology
    angles: tuple[float, ...]
    omega_phi0: tuple[float, ...]
    equation_of_state0: tuple[float, ...]
    accepted: tuple[bool, ...]
    density_relative_tolerance: float
    maximum_equation_of_state: float
    accepted_fraction: float
    minimum_accepted_angle: float | None
    maximum_distance_from_top_degrees: float | None


def top_normalized_cosmology(
    *,
    omega_m0: float = OMEGA_M0_REFERENCE,
    omega_r0: float = OMEGA_R0_REFERENCE,
    mass_over_hubble_at_top: float = 1.0,
    initial_redshift: float = 1000.0,
) -> PseudoGoldstoneCosmology:
    """Fix the minimal model from flatness, the DE height and ``|m|/H0``.

    The potential maximum is normalized to today's target dark-energy density:
    ``2 V_amp = rho_DE``.  Requiring the magnitude of the curvature at that
    maximum to be ``mass_over_hubble_at_top * H0`` then fixes ``f``.
    """
    values = (
        omega_m0,
        omega_r0,
        mass_over_hubble_at_top,
        initial_redshift,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("cosmological parameters must be finite")
    if omega_m0 < 0.0 or omega_r0 < 0.0:
        raise ValueError("density fractions must be non-negative")
    if omega_m0 + omega_r0 >= 1.0:
        raise ValueError("matter and radiation must leave room for dark energy")
    if mass_over_hubble_at_top <= 0.0:
        raise ValueError("mass_over_hubble_at_top must be positive")
    if initial_redshift <= 0.0:
        raise ValueError("initial_redshift must be positive")

    omega_de_target = 1.0 - omega_m0 - omega_r0
    potential_amplitude = 0.5 * omega_de_target
    decay_constant = (
        math.sqrt(3.0 * potential_amplitude) / mass_over_hubble_at_top
    )
    return PseudoGoldstoneCosmology(
        omega_m0=omega_m0,
        omega_r0=omega_r0,
        omega_de_target=omega_de_target,
        potential_amplitude=potential_amplitude,
        decay_constant_over_planck=decay_constant,
        mass_over_hubble_at_top=mass_over_hubble_at_top,
        initial_redshift=initial_redshift,
    )


def _state_quantities(
    e_folds: float,
    state: np.ndarray,
    cosmology: PseudoGoldstoneCosmology,
) -> tuple[float, float, float, float, float, float, float, float]:
    x, velocity = (float(value) for value in state)
    f = cosmology.decay_constant_over_planck
    angle = x / f
    potential = cosmology.potential_amplitude * (1.0 - math.cos(angle))
    background = (
        cosmology.omega_m0 * math.exp(-3.0 * e_folds)
        + cosmology.omega_r0 * math.exp(-4.0 * e_folds)
    )
    denominator = 1.0 - velocity * velocity / 6.0
    if denominator <= 0.0:
        raise RuntimeError("scalar kinetic energy made the Friedmann root invalid")

    hubble_squared = (background + potential) / denominator
    kinetic = hubble_squared * velocity * velocity / 6.0
    density = kinetic + potential
    omega_phi = density / hubble_squared
    equation_of_state = (
        (kinetic - potential) / density if density > 1.0e-300 else math.nan
    )
    hdot_over_h2 = -0.5 * (
        (
            3.0 * cosmology.omega_m0 * math.exp(-3.0 * e_folds)
            + 4.0 * cosmology.omega_r0 * math.exp(-4.0 * e_folds)
        )
        / hubble_squared
        + velocity * velocity
    )
    deceleration = -1.0 - hdot_over_h2
    return (
        angle,
        potential,
        hubble_squared,
        kinetic,
        density,
        omega_phi,
        equation_of_state,
        deceleration,
    )


def _equations(
    e_folds: float,
    state: np.ndarray,
    cosmology: PseudoGoldstoneCosmology,
) -> np.ndarray:
    _, velocity = state
    (
        angle,
        _,
        hubble_squared,
        _,
        _,
        _,
        _,
        deceleration,
    ) = _state_quantities(e_folds, state, cosmology)
    potential_gradient = (
        cosmology.potential_amplitude
        * math.sin(angle)
        / cosmology.decay_constant_over_planck
    )
    hdot_over_h2 = -1.0 - deceleration
    acceleration = -(
        3.0 + hdot_over_h2
    ) * velocity - 3.0 * potential_gradient / hubble_squared
    return np.asarray([velocity, acceleration], dtype=float)


def evolve_common_phase(
    initial_angle: float,
    *,
    initial_field_velocity_dN: float = 0.0,
    cosmology: PseudoGoldstoneCosmology | None = None,
    sample_redshifts: tuple[float, ...] = (1000.0, 1.0, 0.5, 0.0),
    relative_tolerance: float = 1.0e-9,
    absolute_tolerance: float = 1.0e-11,
) -> PseudoGoldstoneEvolution:
    """Integrate one homogeneous common-phase history to the present."""
    cosmology = cosmology or top_normalized_cosmology()
    inputs = (
        initial_angle,
        initial_field_velocity_dN,
        relative_tolerance,
        absolute_tolerance,
    )
    if not all(math.isfinite(value) for value in inputs):
        raise ValueError("initial data and tolerances must be finite")
    if relative_tolerance <= 0.0 or absolute_tolerance <= 0.0:
        raise ValueError("solver tolerances must be positive")
    if abs(initial_field_velocity_dN) >= math.sqrt(6.0):
        raise ValueError("|initial_field_velocity_dN| must be smaller than sqrt(6)")
    if not sample_redshifts:
        raise ValueError("sample_redshifts must not be empty")
    if any(
        not math.isfinite(redshift)
        or redshift < 0.0
        or redshift > cosmology.initial_redshift
        for redshift in sample_redshifts
    ):
        raise ValueError(
            "sample redshifts must lie between zero and initial_redshift"
        )

    initial_e_folds = -math.log1p(cosmology.initial_redshift)
    requested = sorted(set(float(redshift) for redshift in sample_redshifts))
    if 0.0 not in requested:
        requested.append(0.0)
        requested.sort()
    evaluation_e_folds = np.asarray(
        sorted(-math.log1p(redshift) for redshift in requested),
        dtype=float,
    )
    initial_state = np.asarray(
        [
            cosmology.decay_constant_over_planck * initial_angle,
            initial_field_velocity_dN,
        ],
        dtype=float,
    )
    solution = solve_ivp(
        _equations,
        (initial_e_folds, 0.0),
        initial_state,
        args=(cosmology,),
        method="DOP853",
        t_eval=evaluation_e_folds,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    if not solution.success:
        return PseudoGoldstoneEvolution(
            cosmology=cosmology,
            initial_angle=initial_angle,
            initial_field_velocity_dN=initial_field_velocity_dN,
            points=(),
            solver_success=False,
            solver_message=solution.message,
        )

    points: list[EvolutionPoint] = []
    for e_folds, state in zip(solution.t, solution.y.T, strict=True):
        (
            angle,
            _,
            hubble_squared,
            _,
            density,
            omega_phi,
            equation_of_state,
            deceleration,
        ) = _state_quantities(float(e_folds), state, cosmology)
        points.append(
            EvolutionPoint(
                redshift=math.exp(-float(e_folds)) - 1.0,
                angle=angle,
                field_velocity_dN=float(state[1]),
                hubble_over_h0=math.sqrt(hubble_squared),
                omega_phi=omega_phi,
                density_in_present_critical_units=density,
                equation_of_state=equation_of_state,
                deceleration=deceleration,
            )
        )

    return PseudoGoldstoneEvolution(
        cosmology=cosmology,
        initial_angle=initial_angle,
        initial_field_velocity_dN=initial_field_velocity_dN,
        points=tuple(points),
        solver_success=True,
        solver_message=solution.message,
    )


def scan_initial_angles(
    *,
    cosmology: PseudoGoldstoneCosmology | None = None,
    sample_count: int = 181,
    density_relative_tolerance: float = 0.1,
    maximum_equation_of_state: float = -0.95,
) -> InitialAngleScan:
    """Scan a uniform phase measure from the potential minimum to its top.

    An angle is called Lambda-like when its present scalar density matches the
    target within ``density_relative_tolerance`` and ``w_phi(0)`` is no larger
    than ``maximum_equation_of_state``.  These are diagnostics, not fitted
    observational likelihoods.
    """
    cosmology = cosmology or top_normalized_cosmology()
    if sample_count < 3:
        raise ValueError("sample_count must be at least three")
    if (
        not math.isfinite(density_relative_tolerance)
        or not 0.0 < density_relative_tolerance < 1.0
    ):
        raise ValueError("density_relative_tolerance must lie in (0, 1)")
    if (
        not math.isfinite(maximum_equation_of_state)
        or not -1.0 <= maximum_equation_of_state <= 1.0
    ):
        raise ValueError("maximum_equation_of_state must lie in [-1, 1]")

    angles = np.linspace(0.0, math.pi, sample_count)
    omega_phi0: list[float] = []
    equation_of_state0: list[float] = []
    accepted: list[bool] = []
    for angle in angles:
        evolution = evolve_common_phase(
            float(angle),
            cosmology=cosmology,
            sample_redshifts=(0.0,),
        )
        if not evolution.solver_success:
            omega_phi0.append(math.nan)
            equation_of_state0.append(math.nan)
            accepted.append(False)
            continue
        today = evolution.today
        relative_density_error = abs(
            today.density_in_present_critical_units
            - cosmology.omega_de_target
        ) / cosmology.omega_de_target
        is_accepted = (
            relative_density_error <= density_relative_tolerance
            and math.isfinite(today.equation_of_state)
            and today.equation_of_state <= maximum_equation_of_state
        )
        omega_phi0.append(today.density_in_present_critical_units)
        equation_of_state0.append(today.equation_of_state)
        accepted.append(is_accepted)

    accepted_angles = [
        float(angle)
        for angle, is_accepted in zip(angles, accepted, strict=True)
        if is_accepted
    ]
    minimum_accepted = min(accepted_angles) if accepted_angles else None
    distance_from_top = (
        math.degrees(math.pi - minimum_accepted)
        if minimum_accepted is not None
        else None
    )
    return InitialAngleScan(
        cosmology=cosmology,
        angles=tuple(float(angle) for angle in angles),
        omega_phi0=tuple(omega_phi0),
        equation_of_state0=tuple(equation_of_state0),
        accepted=tuple(accepted),
        density_relative_tolerance=density_relative_tolerance,
        maximum_equation_of_state=maximum_equation_of_state,
        accepted_fraction=sum(accepted) / sample_count,
        minimum_accepted_angle=minimum_accepted,
        maximum_distance_from_top_degrees=distance_from_top,
    )
