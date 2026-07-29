"""Tests of the covariant conservation identity for A-dependent mixing."""

import numpy as np
import pytest

from pippa import covariant_conservation as cc
from pippa import physical_consistency as pc
from pippa import sector_spectrum as ss


def sample_mixing() -> np.ndarray:
    return ss.z2x2_mixing_matrix(0.2, -0.08, 0.04)


def test_trace_source_distinguishes_dust_and_radiation():
    rho = 3.0

    assert cc.trace_information_source(rho, pressure=0.0) == pytest.approx(rho)
    assert cc.trace_information_source(rho, pressure=rho / 3.0) == pytest.approx(0.0)
    assert cc.trace_information_source(rho, pressure=-rho) == pytest.approx(4.0 * rho)


def test_source_derivative_matches_finite_difference_of_potential():
    fields = np.array([0.7, -0.2, 0.5, 0.1])
    mixing = sample_mixing()
    source = 0.8
    step = 1.0e-7
    derivative = cc.quadratic_source_derivative(fields, mixing)
    finite_difference = (
        cc.quadratic_mixing_potential(fields, mixing, source + step)
        - cc.quadratic_mixing_potential(fields, mixing, source - step)
    ) / (2.0 * step)

    assert derivative == pytest.approx(finite_difference, rel=1.0e-9, abs=1.0e-10)


def test_on_shell_information_divergence_is_the_source_exchange_current():
    fields = np.array([0.7, -0.2, 0.5, 0.1])
    source_derivative = cc.quadratic_source_derivative(fields, sample_mixing())
    source_gradient = np.array([0.3, -0.1, 0.04, 0.2])
    field_gradients = np.arange(16, dtype=float).reshape(4, 4) / 10.0
    divergence = cc.information_stress_divergence(
        eom_residuals=np.zeros(4),
        field_gradients=field_gradients,
        source_gradient=source_gradient,
        potential_source_derivative=source_derivative,
    )

    assert np.allclose(divergence, -source_derivative * source_gradient)
    assert not np.allclose(divergence, 0.0)


def test_reciprocal_matter_current_restores_total_conservation():
    information_divergence = np.array([0.03, -0.01, 0.04, 0.02])
    matter_divergence = cc.required_matter_exchange_current(information_divergence)
    residual = cc.total_conservation_residual(
        matter_divergence,
        information_divergence,
    )

    assert np.allclose(residual, 0.0)


def test_one_way_external_source_violates_total_conservation():
    information_divergence = np.array([0.03, -0.01, 0.04, 0.02])
    separately_conserved_matter = np.zeros(4)

    residual = cc.total_conservation_residual(
        separately_conserved_matter,
        information_divergence,
    )

    assert np.array_equal(residual, information_divergence)
    assert not np.allclose(residual, 0.0)


def test_off_shell_noether_identity_contains_field_equations():
    residuals = np.array([0.1, -0.2])
    gradients = np.array([[0.4, 0.1], [-0.3, 0.7]])
    source_gradient = np.array([0.2, -0.5])
    source_derivative = 0.06
    expected = residuals @ gradients - source_derivative * source_gradient

    assert np.allclose(
        cc.information_stress_divergence(
            residuals,
            gradients,
            source_gradient,
            source_derivative,
        ),
        expected,
    )


def test_required_flrw_pressure_exactly_satisfies_continuity():
    rho = 2.4
    drho_dt = -0.31
    hubble = 0.08
    exchange = 0.025
    pressure = cc.required_flrw_pressure(rho, drho_dt, hubble, exchange)

    assert cc.flrw_continuity_residual(
        rho,
        pressure,
        drho_dt,
        hubble,
        exchange,
    ) == pytest.approx(0.0, abs=1.0e-14)


def test_nonlinear_m_of_dust_is_not_generically_pressureless():
    assert cc.effective_w_for_matter_power(1.0) == pytest.approx(0.0)
    assert cc.effective_w_for_matter_power(0.5) == pytest.approx(-0.5)
    assert cc.effective_w_for_matter_power(2.0) == pytest.approx(1.0)


def test_varying_external_A_is_rejected_by_conservation_audit():
    audit = cc.audit_conservation(source_role="external")

    assert audit.finding("total_stress_conservation").status is pc.AuditStatus.FAIL
    assert audit.finding("information_separate_conservation").status is pc.AuditStatus.FAIL


def test_composite_A_conserves_total_but_not_each_sector_separately():
    audit = cc.audit_conservation(
        source_role="composite_varied",
        kernel_domain="internal_local_x",
        bridges_dynamic=True,
    )

    assert audit.finding("total_stress_conservation").status is pc.AuditStatus.PASS
    assert audit.finding("exchange_current_cancellation").status is pc.AuditStatus.PASS
    assert audit.finding("matter_separate_conservation").status is pc.AuditStatus.FAIL
    assert audit.finding("information_separate_conservation").status is pc.AuditStatus.FAIL
    assert audit.finding("kernel_covariance").status is pc.AuditStatus.PASS
    assert audit.finding("bridge_stress_tensor").status is pc.AuditStatus.PASS


def test_missing_bridge_dynamics_is_not_silently_accepted():
    audit = cc.audit_conservation(
        source_role="dynamic",
        kernel_domain="unspecified",
        bridges_dynamic=False,
    )

    assert audit.finding("kernel_covariance").status is pc.AuditStatus.BLOCKED
    assert audit.finding("bridge_stress_tensor").status is pc.AuditStatus.BLOCKED
