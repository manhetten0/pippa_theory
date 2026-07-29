"""Danger-point checks for the minimal Pippa model."""

import math

import numpy as np
import pytest

from pippa import constants
from pippa import physical_consistency as pc


def test_physical_fractional_dispersion_is_not_lorentz_invariant():
    assert not pc.spatial_fractional_dispersion_is_lorentz_invariant(constants.D)
    assert pc.spatial_fractional_dispersion_is_lorentz_invariant(2.0)


def test_massless_pure_fractional_group_velocity_is_not_constant():
    assert pc.fractional_group_velocity(0.0, alpha=constants.D) == float("inf")
    velocity_low = pc.fractional_group_velocity(0.01, alpha=constants.D)
    velocity_high = pc.fractional_group_velocity(1.0, alpha=constants.D)

    assert math.isfinite(velocity_low)
    assert velocity_low > velocity_high
    assert velocity_high == pytest.approx(constants.D / 2.0)


def test_weak_gw_bound_restricts_fractional_dispersion_ratio():
    bound = pc.weak_gw_fractional_ratio_bound(constants.D, 1.0e-15)

    assert bound == pytest.approx(2.0e-15 / (constants.D - 1.0))
    assert bound == pytest.approx(7.319584e-15, rel=1.0e-6)


def test_modified_wave_velocity_agrees_with_weak_expansion():
    momentum = 0.8
    beta = 1.0e-7
    ratio = beta * momentum ** (constants.D - 2.0)
    expected_delta = 0.5 * (constants.D - 1.0) * ratio
    velocity = pc.modified_wave_group_velocity(momentum, beta, constants.D)

    assert velocity - 1.0 == pytest.approx(expected_delta, rel=1.0e-6)


def test_visible_poles_have_positive_spectral_weights():
    masses_squared, weights = pc.visible_spectral_data(0.7, 0.1, -0.04, 0.02)

    assert np.all(masses_squared > 0.0)
    assert np.allclose(weights, 0.25)
    assert np.sum(weights) == pytest.approx(1.0)
    assert pc.minimal_quadratic_model_is_ghost_free()
    assert not pc.minimal_quadratic_model_is_ghost_free(positive_time_kinetic=False)


def test_internal_sector_mixing_keeps_standard_modes_subluminal():
    frequencies = pc.lorentz_safe_sector_frequencies(0.8, 0.7, 0.1, -0.04, 0.02)
    velocities = pc.lorentz_safe_sector_group_velocities(0.8, 0.7, 0.1, -0.04, 0.02)

    assert np.all(frequencies > 0.0)
    assert np.all(velocities >= 0.0)
    assert np.all(velocities < 1.0)


def test_decoupling_limit_is_quadratic_in_small_mixing():
    q = 2.0
    couplings = np.array([1.0e-3, -0.7e-3, 0.4e-3])
    error = pc.decoupling_relative_response_error(q, *couplings)
    half_error = pc.decoupling_relative_response_error(q, *(0.5 * couplings))

    assert error > 0.0
    assert half_error / error == pytest.approx(0.25, rel=2.0e-3)


def test_current_physical_space_interpretation_fails_lorentz_check():
    audit = pc.audit_minimal_model(
        mass_squared=0.3,
        neg_coupling=0.2,
        mir_coupling=0.2,
        negmir_coupling=0.2,
        fractional_domain="physical_space",
    )

    assert audit.finding("sector_unitarity").status is pc.AuditStatus.PASS
    assert audit.finding("quadratic_boundedness").status is pc.AuditStatus.PASS
    assert audit.finding("lorentz_invariance").status is pc.AuditStatus.FAIL
    assert audit.finding("microcausality").status is pc.AuditStatus.BLOCKED


def test_internal_space_interpretation_is_conditional_not_automatically_passed():
    audit = pc.audit_minimal_model(
        mass_squared=0.3,
        neg_coupling=0.2,
        mir_coupling=0.2,
        negmir_coupling=0.2,
        fractional_domain="internal_space",
        modifies_graviton_dispersion=False,
    )

    assert audit.finding("lorentz_invariance").status is pc.AuditStatus.CONDITIONAL
    assert audit.finding("microcausality").status is pc.AuditStatus.CONDITIONAL
    assert audit.finding("gravitational_wave_speed").status is pc.AuditStatus.CONDITIONAL


def test_empirical_gravity_checks_remain_blocked_without_matter_mapping():
    audit = pc.audit_minimal_model(
        mass_squared=0.3,
        neg_coupling=0.2,
        mir_coupling=0.2,
        negmir_coupling=0.2,
        fractional_domain="unspecified",
    )

    assert audit.finding("fifth_force").status is pc.AuditStatus.BLOCKED
    assert audit.finding("weak_equivalence_principle").status is pc.AuditStatus.BLOCKED
    assert audit.finding("solar_system_gravity").status is pc.AuditStatus.BLOCKED
    assert audit.finding("gravitational_wave_speed").status is pc.AuditStatus.BLOCKED


def test_fractional_graviton_dispersion_is_flagged():
    audit = pc.audit_minimal_model(
        mass_squared=0.3,
        neg_coupling=0.2,
        mir_coupling=0.2,
        negmir_coupling=0.2,
        fractional_domain="physical_space",
        modifies_graviton_dispersion=True,
    )

    assert audit.finding("gravitational_wave_speed").status is pc.AuditStatus.FAIL


def test_equivalence_principle_constraint_is_not_silently_assumed():
    blocked = pc.equivalence_principle_status(None)
    passing = pc.equivalence_principle_status(1.0e-15)
    failing = pc.equivalence_principle_status(4.0e-15)

    assert blocked.status is pc.AuditStatus.BLOCKED
    assert passing.status is pc.AuditStatus.PASS
    assert failing.status is pc.AuditStatus.FAIL
