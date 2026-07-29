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


def sample_hidden_fields(n_points: int = 9) -> dict[str, np.ndarray]:
    """Independent fields for the three non-observable Pippa sectors."""
    psi = sample_field(n_points)
    return {
        "Neg": 0.7 * np.roll(psi, 1) * np.exp(0.2j),
        "Mir": 1.1 * np.conjugate(np.roll(psi, -2)),
        "NegMir": 0.4 * np.roll(psi, 3) * np.exp(-0.35j),
    }


def sample_intersector_channels(n_points: int = 9) -> tuple[ca.IntersectorChannel, ...]:
    """Three channels with distinct topology, range and weight."""
    identity = np.arange(n_points)
    reflection = identity[::-1]
    shifted_reflection = np.roll(reflection, 2)
    specifications = (
        ("Neg", identity, ca.constants.D, 0.6),
        ("Mir", reflection, 1.1, -0.35),
        ("NegMir", shifted_reflection, 1.6, 0.2),
    )
    return tuple(
        ca.IntersectorChannel(
            source_sector=sector,
            kernel=ca.intersector_kernel_1d(
                n_points,
                alpha=alpha,
                source_permutation=permutation,
            ),
            transport=ca.identity_transport(n_points),
            weight=weight,
        )
        for sector, permutation, alpha, weight in specifications
    )


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


def test_intersector_operator_is_covariant_under_independent_sector_gauges():
    target = sample_field()
    sources = sample_hidden_fields(target.size)
    channels = sample_intersector_channels(target.size)
    target_phases = sample_phases(target.size)
    source_phases = {
        "Neg": np.roll(target_phases, 1) + 0.1,
        "Mir": -0.6 * target_phases + 0.25,
        "NegMir": np.roll(target_phases, -2) - 0.3,
    }

    projected = ca.intersector_m_operator(target, sources, channels)
    target_g = ca.gauge_transform_field(target, target_phases)
    sources_g = {
        sector: ca.gauge_transform_field(field, source_phases[sector])
        for sector, field in sources.items()
    }
    channels_g = tuple(
        channel.gauge_transformed(
            target_phases,
            source_phases[channel.source_sector],
        )
        for channel in channels
    )
    projected_g = ca.intersector_m_operator(target_g, sources_g, channels_g)
    expected = ca.gauge_transform_field(projected, target_phases)

    assert np.allclose(projected_g, expected, rtol=1e-12, atol=1e-12)


def test_intersector_coupling_energy_is_gauge_invariant():
    target = sample_field()
    sources = sample_hidden_fields(target.size)
    channels = sample_intersector_channels(target.size)
    target_phases = sample_phases(target.size)
    source_phases = {
        sector: np.roll(target_phases, shift)
        for shift, sector in enumerate(ca.HIDDEN_SECTORS, start=1)
    }
    coupling = ca.IntersectorCoupling(channels=channels, coupling=0.23)

    target_g = ca.gauge_transform_field(target, target_phases)
    sources_g = {
        sector: ca.gauge_transform_field(field, source_phases[sector])
        for sector, field in sources.items()
    }
    channels_g = tuple(
        channel.gauge_transformed(
            target_phases,
            source_phases[channel.source_sector],
        )
        for channel in channels
    )
    coupling_g = ca.IntersectorCoupling(channels=channels_g, coupling=0.23)

    assert coupling.energy(target, sources) == pytest.approx(
        coupling_g.energy(target_g, sources_g),
        rel=1e-12,
        abs=1e-12,
    )


def test_intersector_operator_is_independent_of_visible_sector_laplacian():
    target = sample_field()
    kernel = ca.fractional_kernel_1d(target.size)
    transport = ca.identity_transport(target.size)
    channels = sample_intersector_channels(target.size)
    sources_a = sample_hidden_fields(target.size)
    sources_b = {sector: field.copy() for sector, field in sources_a.items()}
    sources_b["Mir"] *= 1.8 * np.exp(0.4j)

    laplacian_a = ca.covariant_fractional_laplacian(target, kernel, transport)
    laplacian_b = ca.covariant_fractional_laplacian(target, kernel, transport)
    projected_a = ca.intersector_m_operator(target, sources_a, channels)
    projected_b = ca.intersector_m_operator(target, sources_b, channels)

    assert np.array_equal(laplacian_a, laplacian_b)
    assert not np.allclose(projected_a, projected_b)


def test_intersector_reflection_changes_the_projection():
    n_points = 9
    target = sample_field(n_points)
    source = np.zeros(n_points, dtype=np.complex128)
    source[1] = 1.0 + 0.3j
    identity_channel = ca.IntersectorChannel(
        source_sector="Neg",
        kernel=ca.intersector_kernel_1d(n_points),
        transport=ca.identity_transport(n_points),
    )
    reflected_channel = ca.IntersectorChannel(
        source_sector="Neg",
        kernel=ca.intersector_kernel_1d(
            n_points,
            source_permutation=np.arange(n_points)[::-1],
        ),
        transport=ca.identity_transport(n_points),
    )

    identity_projection = ca.intersector_m_operator(
        target,
        {"Neg": source},
        [identity_channel],
    )
    reflected_projection = ca.intersector_m_operator(
        target,
        {"Neg": source},
        [reflected_channel],
    )

    assert not np.allclose(identity_projection, reflected_projection)
    assert np.argmax(np.abs(identity_projection)) != np.argmax(np.abs(reflected_projection))


def test_intrasector_m_degeneracy_is_explicit_but_intersector_m_is_not():
    target = sample_field()
    kernel = ca.fractional_kernel_1d(target.size)
    transport = ca.identity_transport(target.size)
    laplacian = ca.covariant_fractional_laplacian(target, kernel, transport)
    intrasector_m = ca.covariant_m_operator(target, kernel, transport)
    row_sum = np.sum(kernel, axis=1)

    assert np.allclose(laplacian, row_sum * target - intrasector_m, atol=1e-12)

    channels = sample_intersector_channels(target.size)
    sources = sample_hidden_fields(target.size)
    intersector_m = ca.intersector_m_operator(target, sources, channels)
    sources["Neg"] = np.zeros_like(sources["Neg"])
    changed_intersector_m = ca.intersector_m_operator(target, sources, channels)

    assert not np.allclose(intersector_m, changed_intersector_m)
