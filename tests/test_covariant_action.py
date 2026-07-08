"""Tests for the gauge-covariant effective Pippa action."""

import numpy as np
import pytest

from pippa import covariant_action as ca


def sample_field(n_points: int = 9) -> np.ndarray:
    """Deterministic complex field used by covariance tests."""
    x = np.linspace(0.0, 1.0, n_points)
    amplitude = 0.8 + 0.3 * np.cos(2.0 * np.pi * x)
    phase = 0.2 + 0.7 * np.sin(2.0 * np.pi * x)
    return amplitude * np.exp(1j * phase)


def sample_phases(n_points: int = 9) -> np.ndarray:
    """Nontrivial local U(1) phase profile."""
    x = np.linspace(0.0, 1.0, n_points)
    return 0.4 * np.sin(2.0 * np.pi * x) + 0.15 * np.cos(5.0 * np.pi * x)


def test_fractional_kernel_has_expected_long_range_shape():
    kernel = ca.fractional_kernel_1d(
        n_points=6,
        alpha=1.25,
        spacing=1.0,
        periodic=False,
    )

    assert np.allclose(kernel, kernel.T)
    assert np.allclose(np.diag(kernel), 0.0)
    assert kernel[0, 1] > kernel[0, 2] > kernel[0, 3] > 0.0


def test_covariant_fractional_laplacian_transforms_like_field():
    psi = sample_field()
    phases = sample_phases(psi.size)
    kernel = ca.fractional_kernel_1d(psi.size)
    transport = ca.identity_transport(psi.size)

    psi_g = ca.gauge_transform_field(psi, phases)
    transport_g = ca.gauge_transform_transport(transport, phases)

    lap = ca.covariant_fractional_laplacian(psi, kernel, transport)
    lap_g = ca.covariant_fractional_laplacian(psi_g, kernel, transport_g)
    expected = ca.gauge_transform_field(lap, phases)

    assert np.allclose(lap_g, expected, rtol=1e-12, atol=1e-12)


def test_bilocal_m_operator_transforms_like_field():
    psi = sample_field()
    phases = sample_phases(psi.size)
    kernel = ca.fractional_kernel_1d(psi.size)
    transport = ca.identity_transport(psi.size)

    psi_g = ca.gauge_transform_field(psi, phases)
    transport_g = ca.gauge_transform_transport(transport, phases)

    m_psi = ca.covariant_m_operator(psi, kernel, transport)
    m_psi_g = ca.covariant_m_operator(psi_g, kernel, transport_g)
    expected = ca.gauge_transform_field(m_psi, phases)

    assert np.allclose(m_psi_g, expected, rtol=1e-12, atol=1e-12)


def test_conservative_action_is_gauge_invariant():
    psi = sample_field()
    phases = sample_phases(psi.size)
    kernel = ca.fractional_kernel_1d(psi.size)
    transport = ca.identity_transport(psi.size)

    action = ca.InformationFieldAction(
        kernel=kernel,
        transport=transport,
        kappa=0.8,
        self_coupling=0.3,
        rho0=0.9,
        m_coupling=0.2,
    )

    psi_g = ca.gauge_transform_field(psi, phases)
    transport_g = ca.gauge_transform_transport(transport, phases)
    action_g = ca.InformationFieldAction(
        kernel=kernel,
        transport=transport_g,
        kappa=action.kappa,
        self_coupling=action.self_coupling,
        rho0=action.rho0,
        m_coupling=action.m_coupling,
    )

    assert action.energy(psi) == pytest.approx(action_g.energy(psi_g), rel=1e-12)
    terms = action.terms(psi)
    terms_g = action_g.terms(psi_g)
    assert terms["fractional_kinetic"] == pytest.approx(
        terms_g["fractional_kinetic"],
        rel=1e-12,
    )
    assert terms["self_interaction"] == pytest.approx(
        terms_g["self_interaction"],
        rel=1e-12,
    )
    assert terms["bilocal_M"] == pytest.approx(terms_g["bilocal_M"], rel=1e-12)


def test_fractional_kinetic_energy_is_positive_and_zero_for_flat_parallel_field():
    psi = np.ones(7, dtype=np.complex128)
    kernel = ca.fractional_kernel_1d(psi.size)
    transport = ca.identity_transport(psi.size)

    assert ca.fractional_kinetic_energy(psi, kernel, transport) == pytest.approx(0.0)

    psi[3] = 1.2 + 0.5j
    assert ca.fractional_kinetic_energy(psi, kernel, transport) > 0.0


def test_rayleigh_dissipation_is_open_sector_and_decreases_norm():
    A = np.array([1.0, -0.4, 0.2, 0.8])
    B = np.array([0.3, 0.1, -0.5, 0.7])
    diss = ca.RayleighDissipation(gamma_A=0.4, gamma_B=0.2)

    dA, dB = diss.flow(A, B)
    norm_derivative = 2.0 * (np.dot(A, dA) + np.dot(B, dB))

    assert diss.potential(A, B) > 0.0
    assert diss.entropy_production(A, B) >= 0.0
    assert norm_derivative < 0.0
