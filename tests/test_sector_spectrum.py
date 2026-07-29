"""Tests for the full Z2 x Z2 Pippa sector spectrum."""

import numpy as np
import pytest

from pippa import sector_spectrum as ss


def test_mixing_matrix_has_the_z2x2_convolution_pattern():
    a, b, c = 0.2, -0.35, 0.08
    mixing = ss.z2x2_mixing_matrix(a, b, c)
    expected = np.array(
        [
            [0.0, a, b, c],
            [a, 0.0, c, b],
            [b, c, 0.0, a],
            [c, b, a, 0.0],
        ]
    )

    assert np.array_equal(mixing, expected)
    assert np.array_equal(mixing, mixing.T)


def test_character_transform_diagonalizes_mixing_analytically():
    couplings = (0.17, -0.09, 0.04)
    mixing = ss.z2x2_mixing_matrix(*couplings)
    fourier = ss.character_transform()
    transformed = fourier @ mixing @ fourier.T
    expected = ss.analytic_mixing_eigenvalues(*couplings)

    assert np.allclose(fourier @ fourier.T, np.eye(4), atol=1e-14)
    assert np.allclose(transformed, np.diag(expected), atol=1e-14)
    assert np.allclose(np.sort(np.linalg.eigvalsh(mixing)), np.sort(expected))


def test_equal_couplings_give_one_plus_three_spectrum():
    weight = 0.23
    eigenvalues = ss.analytic_mixing_eigenvalues(weight, weight, weight)

    assert eigenvalues[0] == pytest.approx(3.0 * weight)
    assert np.allclose(eigenvalues[1:], -weight)


def test_nonzero_zero_diagonal_mixing_is_necessarily_indefinite():
    eigenvalues = ss.analytic_mixing_eigenvalues(0.31, -0.12, 0.07)

    assert np.sum(eigenvalues) == pytest.approx(0.0, abs=1e-15)
    assert np.min(eigenvalues) < 0.0 < np.max(eigenvalues)


def test_mixing_is_equivariant_under_every_z2x2_translation():
    mixing = ss.z2x2_mixing_matrix(0.2, -0.1, 0.06)
    field = np.array([0.7 + 0.2j, -0.3j, 1.1 - 0.4j, -0.2 + 0.8j])

    for sector in ss.SECTOR_ORDER:
        translation = ss.sector_translation_matrix(sector)
        assert np.allclose(translation @ mixing, mixing @ translation)
        assert np.allclose(
            mixing @ (translation @ field),
            translation @ (mixing @ field),
        )


def test_bridge_links_make_independent_sector_gauges_invariant():
    q = 1.4
    couplings = (0.18, -0.07, 0.03)
    field = np.array([0.8 + 0.1j, -0.2 + 0.5j, 0.4 - 0.6j, 0.9 + 0.3j])
    phases = np.array([0.2, -0.5, 0.8, -0.1])
    links = ss.identity_sector_links()
    operator = q * np.eye(4) + ss.linked_mixing_matrix(*couplings, links)

    field_g = ss.gauge_transform_sector_fields(field, phases)
    links_g = ss.gauge_transform_sector_links(links, phases)
    operator_g = q * np.eye(4) + ss.linked_mixing_matrix(*couplings, links_g)
    gauge = np.diag(np.exp(1j * phases))

    assert np.allclose(operator_g, gauge @ operator @ np.conjugate(gauge.T))
    assert ss.quadratic_energy(field, operator) == pytest.approx(
        ss.quadratic_energy(field_g, operator_g),
        rel=1e-12,
        abs=1e-12,
    )


def test_mass_gap_can_make_the_relativistic_completion_tachyon_free():
    report = ss.analyze_stability(
        mass_squared=0.3,
        neg_coupling=0.2,
        mir_coupling=0.2,
        negmir_coupling=0.2,
    )

    assert report.schrodinger_unitary
    assert report.relativistic_tachyon_free
    assert report.quadratic_hamiltonian_bounded
    assert report.full_hamiltonian_bounded
    assert report.minimum_relativistic_omega_squared == pytest.approx(0.1)
    assert report.unstable_modes == ()


def test_massless_intersector_mixing_has_unstable_relativistic_modes():
    report = ss.analyze_stability(
        mass_squared=0.0,
        neg_coupling=0.2,
        mir_coupling=0.2,
        negmir_coupling=0.2,
    )
    stabilized = ss.analyze_stability(
        mass_squared=0.0,
        neg_coupling=0.2,
        mir_coupling=0.2,
        negmir_coupling=0.2,
        self_coupling=0.4,
    )

    assert report.schrodinger_unitary
    assert not report.relativistic_tachyon_free
    assert not report.quadratic_hamiltonian_bounded
    assert not report.full_hamiltonian_bounded
    assert set(report.unstable_modes) == {
        "Neg-character",
        "Mir-character",
        "NegMir-character",
    }
    assert stabilized.full_hamiltonian_bounded
    assert not stabilized.relativistic_tachyon_free


def test_negative_fractional_kinetic_coefficient_is_unbounded_in_the_uv():
    report = ss.analyze_stability(
        mass_squared=2.0,
        neg_coupling=0.01,
        mir_coupling=0.01,
        negmir_coupling=0.01,
        kappa=-0.1,
        self_coupling=1.0,
    )

    assert report.minimum_relativistic_omega_squared == float("-inf")
    assert not report.full_hamiltonian_bounded
    assert not report.relativistic_tachyon_free


def test_visible_propagator_matches_full_matrix_inverse():
    q = 1.7
    couplings = (0.16, -0.08, 0.05)
    operator = q * np.eye(4) + ss.z2x2_mixing_matrix(*couplings)
    expected = np.linalg.inv(operator)[0, 0]

    assert ss.visible_propagator(q, *couplings) == pytest.approx(expected)


def test_visible_self_energy_matches_hidden_sector_schur_complement():
    q = 1.7
    couplings = (0.16, -0.08, 0.05)
    operator = q * np.eye(4) + ss.z2x2_mixing_matrix(*couplings)
    visible_to_hidden = operator[0, 1:]
    hidden = operator[1:, 1:]
    schur_self_energy = visible_to_hidden @ np.linalg.inv(hidden) @ visible_to_hidden

    self_energy = ss.visible_self_energy(q, *couplings)
    effective_inverse = ss.visible_effective_inverse(q, *couplings)

    assert self_energy == pytest.approx(schur_self_energy)
    assert effective_inverse == pytest.approx(q - schur_self_energy)
    assert self_energy > 0.0
    assert effective_inverse < q


def test_stable_static_mixing_always_enhances_visible_response():
    rng = np.random.default_rng(20260718)
    for _ in range(50):
        couplings = rng.uniform(-0.4, 0.4, size=3)
        mixing_eigenvalues = ss.analytic_mixing_eigenvalues(*couplings)
        q = 1.0 + max(0.0, -float(np.min(mixing_eigenvalues)))
        free_response = 1.0 / q
        visible_response = ss.visible_propagator(q, *couplings)

        assert ss.visible_self_energy(q, *couplings) >= -1.0e-12
        assert visible_response >= free_response - 1.0e-12


def test_weak_intersector_coupling_has_the_expected_self_energy():
    q = 2.3
    couplings = np.array([1.2e-4, -0.7e-4, 0.4e-4])
    leading_order = float(np.dot(couplings, couplings) / q)

    assert ss.visible_self_energy(q, *couplings) == pytest.approx(
        leading_order,
        rel=2.0e-4,
    )
    assert ss.visible_self_energy(q, 0.0, 0.0, 0.0) == pytest.approx(0.0)
