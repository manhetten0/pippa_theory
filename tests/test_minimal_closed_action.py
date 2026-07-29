import math

import numpy as np
import pytest

from pippa import dark_energy_dynamics
from pippa import minimal_closed_action
from pippa import sector_spectrum


def _action(
    *,
    decay_constant=2.0,
    dark_energy_amplitude=0.2,
    weights=(3.0, 4.0, 5.0),
):
    return minimal_closed_action.MinimalPhaseAction(
        decay_constant=decay_constant,
        dark_energy_amplitude=dark_energy_amplitude,
        neg_locking_amplitude=weights[0],
        mir_locking_amplitude=weights[1],
        negmir_locking_amplitude=weights[2],
    )


def test_potential_hessian_matches_a_direct_finite_difference():
    action = _action()
    step = 1.0e-4
    origin = np.zeros(4)
    numerical = np.empty((4, 4))

    for row in range(4):
        for column in range(4):
            e_row = np.eye(4)[row] * step
            e_column = np.eye(4)[column] * step
            numerical[row, column] = (
                action.potential_terms(origin + e_row + e_column).total
                - action.potential_terms(origin + e_row - e_column).total
                - action.potential_terms(origin - e_row + e_column).total
                + action.potential_terms(origin - e_row - e_column).total
            ) / (4.0 * step * step)

    assert numerical == pytest.approx(
        action.aligned_potential_hessian(),
        rel=2.0e-7,
        abs=2.0e-7,
    )


def test_character_spectrum_has_one_soft_common_and_three_relative_modes():
    action = _action()
    spectrum = action.mode_spectrum()

    assert spectrum.mode_names == sector_spectrum.CHARACTER_MODES
    assert spectrum.masses_squared == pytest.approx((0.05, 4.0, 4.5, 3.5))
    assert spectrum.relative_modes_non_tachyonic
    assert not spectrum.common_mode_protected_when_unbroken

    unbroken = _action(dark_energy_amplitude=0.0).mode_spectrum()
    assert unbroken.masses_squared[0] == pytest.approx(0.0, abs=1.0e-15)
    assert unbroken.common_mode_protected_when_unbroken


def test_locking_potential_keeps_exact_common_shift_symmetry():
    action = _action()
    phases = np.asarray([0.1, -0.3, 0.7, 1.2])
    shifted = phases + 17.4

    base_locking = action.potential_terms(phases).locking
    shifted_locking = action.potential_terms(shifted).locking

    assert shifted_locking == pytest.approx(base_locking, abs=1.0e-13)
    assert action.potential_terms(shifted).dark_energy != pytest.approx(
        action.potential_terms(phases).dark_energy
    )


def test_dark_energy_top_matches_the_frw_normalization():
    cosmology = dark_energy_dynamics.top_normalized_cosmology()
    action = _action(
        decay_constant=cosmology.decay_constant_over_planck,
        dark_energy_amplitude=cosmology.potential_amplitude,
        weights=(0.0, 0.0, 0.0),
    )
    common_top = np.full(4, 0.5 * math.pi)
    state = action.stress_energy_state(
        common_top,
        np.zeros(4),
        np.zeros(4),
    )

    assert action.common_coordinate(common_top) == pytest.approx(math.pi)
    assert state.energy_density == pytest.approx(cosmology.omega_de_target)
    assert state.equation_of_state == pytest.approx(-1.0)


def test_a_small_relative_mode_averages_to_cold_matter():
    action = _action(
        decay_constant=1.0,
        dark_energy_amplitude=0.0,
        weights=(1.0, 1.0, 1.0),
    )
    transform = sector_spectrum.character_transform()
    relative_mass = math.sqrt(action.mode_spectrum().masses_squared[1])
    amplitude = 1.0e-3
    times = np.linspace(
        0.0,
        2.0 * math.pi / relative_mass,
        2001,
        endpoint=False,
    )
    densities = []
    radial_pressures = []
    tangential_pressures = []

    for time in times:
        characters = np.asarray(
            [0.0, amplitude * math.cos(relative_mass * time), 0.0, 0.0]
        )
        character_velocities = np.asarray(
            [
                0.0,
                -amplitude
                * relative_mass
                * math.sin(relative_mass * time),
                0.0,
                0.0,
            ]
        )
        state = action.stress_energy_state(
            transform.T @ characters,
            transform.T @ character_velocities,
            np.zeros(4),
        )
        densities.append(state.energy_density)
        radial_pressures.append(state.radial_pressure)
        tangential_pressures.append(state.tangential_pressure)

    average_density = float(np.mean(densities))
    average_radial_pressure = float(np.mean(radial_pressures))
    average_tangential_pressure = float(np.mean(tangential_pressures))
    sources = minimal_closed_action.weak_field_sources(
        average_density,
        average_radial_pressure,
        average_tangential_pressure,
    )

    assert abs(average_radial_pressure / average_density) < 1.0e-6
    assert abs(average_tangential_pressure / average_density) < 1.0e-6
    assert sources.dynamical_density == pytest.approx(
        average_density,
        rel=1.0e-6,
    )
    assert sources.lensing_density == pytest.approx(
        average_density,
        rel=1.0e-6,
    )


def test_static_phase_profile_is_not_automatically_cold_dark_matter():
    action = _action(
        decay_constant=1.0,
        dark_energy_amplitude=0.0,
        weights=(0.0, 0.0, 0.0),
    )
    state = action.stress_energy_state(
        np.zeros(4),
        np.zeros(4),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    sources = minimal_closed_action.weak_field_sources_from_state(state)

    assert state.energy_density == pytest.approx(0.5)
    assert sources.dynamical_density == pytest.approx(0.0)
    assert sources.lensing_density == pytest.approx(0.25)
    assert sources.anisotropic_stress == pytest.approx(1.0)


def test_wkb_gradient_parameter_controls_dynamics_lensing_mismatch():
    homogeneous = minimal_closed_action.averaged_harmonic_mode(
        mode_amplitude=2.0,
        radial_amplitude_gradient=0.0,
        mode_mass=5.0,
        decay_constant=3.0,
        hubble_rate=0.01,
    )
    mildly_inhomogeneous = minimal_closed_action.averaged_harmonic_mode(
        mode_amplitude=2.0,
        radial_amplitude_gradient=2.0,
        mode_mass=5.0,
        decay_constant=3.0,
        hubble_rate=0.01,
    )

    assert homogeneous.stress.equation_of_state == pytest.approx(0.0)
    assert homogeneous.weak_field.dynamical_density == pytest.approx(
        homogeneous.stress.energy_density
    )
    assert homogeneous.weak_field.lensing_density == pytest.approx(
        homogeneous.stress.energy_density
    )
    assert homogeneous.oscillations_per_hubble_time == pytest.approx(500.0)

    assert mildly_inhomogeneous.gradient_to_mass_ratio_squared == pytest.approx(
        0.04
    )
    assert (
        mildly_inhomogeneous.weak_field.lensing_density
        / mildly_inhomogeneous.weak_field.dynamical_density
    ) == pytest.approx(1.01)


def test_pressureless_spherical_profile_has_one_mass_for_motion_and_lensing():
    radius = np.linspace(0.0, 2.0, 101)
    density = np.exp(-radius)
    pressure = np.zeros_like(radius)

    masses = minimal_closed_action.spherical_source_masses(
        radius,
        density,
        pressure,
        pressure,
    )

    assert masses.dynamical_mass == pytest.approx(masses.curvature_mass)
    assert masses.lensing_mass == pytest.approx(masses.curvature_mass)
    assert minimal_closed_action.exterior_circular_speed_squared(
        1.0,
        masses.dynamical_mass[-1],
        radius[-1],
    ) > 0.0
    assert minimal_closed_action.exterior_light_deflection(
        1.0,
        masses.lensing_mass[-1],
        radius[-1],
        1.0,
    ) > 0.0
