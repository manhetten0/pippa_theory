import math

import numpy as np
import pytest

from pippa import dark_energy_dynamics


def test_top_normalization_fixes_height_mass_and_decay_constant():
    cosmology = dark_energy_dynamics.top_normalized_cosmology()

    assert 2.0 * cosmology.potential_amplitude == pytest.approx(
        cosmology.omega_de_target
    )
    curvature_mass_over_hubble = (
        math.sqrt(3.0 * cosmology.potential_amplitude)
        / cosmology.decay_constant_over_planck
    )
    assert curvature_mass_over_hubble == pytest.approx(1.0)
    assert cosmology.decay_constant_over_planck == pytest.approx(1.01359, rel=1e-4)


def test_exact_potential_top_is_a_flat_lambda_like_solution():
    cosmology = dark_energy_dynamics.top_normalized_cosmology()
    evolution = dark_energy_dynamics.evolve_common_phase(
        math.pi,
        cosmology=cosmology,
    )
    today = evolution.today

    assert evolution.solver_success
    assert today.angle == pytest.approx(math.pi, abs=1.0e-10)
    assert today.hubble_over_h0 == pytest.approx(1.0, abs=1.0e-10)
    assert today.density_in_present_critical_units == pytest.approx(
        cosmology.omega_de_target,
        abs=1.0e-10,
    )
    assert today.omega_phi == pytest.approx(cosmology.omega_de_target)
    assert today.equation_of_state == pytest.approx(-1.0, abs=1.0e-12)
    assert today.deceleration < 0.0


def test_canonical_common_phase_never_becomes_phantom():
    for initial_angle in (0.2, 1.0, 2.0, 2.8, math.pi):
        evolution = dark_energy_dynamics.evolve_common_phase(initial_angle)

        assert evolution.solver_success
        for point in evolution.points:
            if math.isfinite(point.equation_of_state):
                assert -1.0 - 1.0e-12 <= point.equation_of_state <= 1.0 + 1.0e-12


def test_friedmann_constraint_and_scalar_continuity_are_satisfied():
    cosmology = dark_energy_dynamics.top_normalized_cosmology()
    e_folds = np.linspace(-math.log1p(cosmology.initial_redshift), 0.0, 401)
    redshifts = tuple(np.exp(-e_folds) - 1.0)
    evolution = dark_energy_dynamics.evolve_common_phase(
        2.4,
        cosmology=cosmology,
        sample_redshifts=redshifts,
    )

    densities = np.asarray(
        [point.density_in_present_critical_units for point in evolution.points]
    )
    equations_of_state = np.asarray(
        [point.equation_of_state for point in evolution.points]
    )
    for point in evolution.points:
        friedmann_rhs = (
            cosmology.omega_m0 * (1.0 + point.redshift) ** 3
            + cosmology.omega_r0 * (1.0 + point.redshift) ** 4
            + point.density_in_present_critical_units
        )
        assert point.hubble_over_h0**2 == pytest.approx(
            friedmann_rhs,
            rel=2.0e-12,
        )

    density_derivative = np.gradient(densities, e_folds, edge_order=2)
    continuity_rhs = -3.0 * (1.0 + equations_of_state) * densities
    assert density_derivative[2:-2] == pytest.approx(
        continuity_rhs[2:-2],
        rel=1.0e-3,
        abs=5.0e-8,
    )


def test_minimum_has_no_dark_energy():
    evolution = dark_energy_dynamics.evolve_common_phase(0.0)
    today = evolution.today

    assert evolution.solver_success
    assert today.density_in_present_critical_units == pytest.approx(0.0)
    assert today.omega_phi == pytest.approx(0.0)
    assert math.isnan(today.equation_of_state)
    assert today.deceleration > 0.0


def test_initial_angle_scan_exposes_but_does_not_assume_tuning():
    scan = dark_energy_dynamics.scan_initial_angles(sample_count=61)

    assert 0.0 < scan.accepted_fraction < 1.0
    assert scan.minimum_accepted_angle is not None
    assert scan.maximum_distance_from_top_degrees is not None
    assert 0.0 < scan.maximum_distance_from_top_degrees < 90.0
    assert not scan.accepted[0]
    assert scan.accepted[-1]


def test_small_early_velocity_is_hubble_damped():
    cosmology = dark_energy_dynamics.top_normalized_cosmology()
    baseline = dark_energy_dynamics.evolve_common_phase(
        2.8,
        cosmology=cosmology,
        sample_redshifts=(1000.0, 0.0),
    )
    moving = dark_energy_dynamics.evolve_common_phase(
        2.8,
        initial_field_velocity_dN=0.05,
        cosmology=cosmology,
        sample_redshifts=(1000.0, 0.0),
    )

    assert moving.points[0].omega_phi < 5.0e-4
    assert moving.today.density_in_present_critical_units == pytest.approx(
        baseline.today.density_in_present_critical_units,
        rel=0.01,
    )
