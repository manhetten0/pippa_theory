"""Consistency tests for the conservative Pippa quantum/SM embedding."""

from __future__ import annotations

from fractions import Fraction

import pytest

pytest.importorskip("numpy")

import numpy as np

from pippa import covariant_action
from pippa import quantum_sm


def test_A_is_information_intensity_and_born_density_is_normalized():
    amplitude = np.array([1.0 + 1.0j, -2.0j, 0.5 - 0.25j])
    measure = np.array([0.2, 0.3, 0.5])

    density = quantum_sm.information_density(amplitude, density_scale=3.0)
    probability = quantum_sm.born_probability_density(amplitude, measure)

    assert np.all(density >= 0.0)
    assert np.sum(probability * measure) == pytest.approx(1.0)
    assert quantum_sm.born_probability_density(7.0j * amplitude, measure) == pytest.approx(
        probability
    )


def test_microscopic_B_link_phase_is_locally_gauge_invariant():
    amplitude = np.array([1.0 + 0.2j, -0.4 + 0.8j, 0.3 - 0.5j])
    transport = covariant_action.identity_transport(amplitude.size)
    phases = np.array([0.2, -0.7, 1.1])

    reference = quantum_sm.gauge_invariant_coherence(amplitude, transport)
    transformed_amplitude = covariant_action.gauge_transform_field(amplitude, phases)
    transformed_transport = covariant_action.gauge_transform_transport(transport, phases)

    assert quantum_sm.gauge_invariant_coherence(
        transformed_amplitude,
        transformed_transport,
    ) == pytest.approx(reference)


def test_internal_fractionality_preserves_relativistic_mass_shell():
    internal_mass_squared = quantum_sm.internal_fractional_mass_squared(
        internal_mode_norm=2.3,
        internal_scale=0.4,
    )
    expected = 1.7**2 + internal_mass_squared

    invariants = [
        quantum_sm.mass_shell_invariant(momentum, 1.7, internal_mass_squared)
        for momentum in (0.0, 0.2, 1.0, 8.0)
    ]

    assert invariants == pytest.approx([expected] * len(invariants))


def test_schrodinger_energy_is_the_low_momentum_limit():
    result = quantum_sm.nonrelativistic_quantum_limit(
        momentum=0.01,
        rest_mass=1.0,
    )

    assert result.schrodinger_energy == pytest.approx(0.01**2 / 2.0)
    assert result.relative_error < 3.0e-5


def test_one_standard_model_generation_is_exactly_anomaly_free():
    report = quantum_sm.anomaly_coefficients(quantum_sm.SM_GENERATION)

    assert report.su3_cubed == 0
    assert report.su3_squared_u1 == 0
    assert report.su2_squared_u1 == 0
    assert report.u1_cubed == 0
    assert report.gravity_squared_u1 == 0
    assert report.left_handed_su2_doublets == 4
    assert report.anomaly_free


def test_standard_model_electric_charges_follow_Q_equals_T3_plus_Y():
    multiplets = {multiplet.name: multiplet for multiplet in quantum_sm.SM_GENERATION}

    assert quantum_sm.electric_charges(multiplets["Q_L"]) == (
        Fraction(2, 3),
        Fraction(-1, 3),
    )
    assert quantum_sm.electric_charges(multiplets["L_L"]) == (
        Fraction(0, 1),
        Fraction(-1, 1),
    )
    assert quantum_sm.electric_charges(quantum_sm.HIGGS_MULTIPLET) == (
        Fraction(1, 1),
        Fraction(0, 1),
    )


def test_minimal_M_acts_only_on_the_sm_singlet_information_field():
    assert quantum_sm.minimal_m_operator_accepts(quantum_sm.INFORMATION_MULTIPLET)
    assert not quantum_sm.minimal_m_operator_accepts(quantum_sm.SM_GENERATION[0])
    assert not quantum_sm.minimal_m_operator_accepts(quantum_sm.HIGGS_MULTIPLET)


def test_common_gauge_kinetic_factor_is_absorbed_into_the_coupling():
    coupling = 0.65
    kappa = 1.27

    effective = quantum_sm.canonical_gauge_coupling(coupling, kappa)

    assert effective == pytest.approx(coupling / np.sqrt(kappa))
    assert quantum_sm.canonical_gauge_coupling(effective, 1.0) == pytest.approx(effective)


def test_two_qubit_measurement_is_unitary_and_creates_a_perfect_record():
    alpha = 1.0 / np.sqrt(2.0)
    beta = 1.0j / np.sqrt(2.0)
    unitary = quantum_sm.controlled_record_unitary(np.pi / 2.0)

    assert np.conjugate(unitary.T) @ unitary == pytest.approx(np.eye(4))

    result = quantum_sm.simulate_record_creation(alpha, beta, np.pi / 2.0)

    assert result.state == pytest.approx(np.array([alpha, 0.0, 0.0, beta]))
    assert result.full_norm == pytest.approx(1.0)
    assert result.full_purity == pytest.approx(1.0)
    assert result.full_entropy_bits == pytest.approx(0.0, abs=1.0e-14)
    assert result.record_information_bits == pytest.approx(1.0)
    assert result.system_coherence_bits == pytest.approx(0.0, abs=1.0e-14)
    assert result.mutual_information_bits == pytest.approx(2.0)


def test_no_interaction_means_no_record_and_full_local_coherence():
    result = quantum_sm.simulate_record_creation(
        1.0 / np.sqrt(2.0),
        1.0 / np.sqrt(2.0),
        0.0,
    )

    assert result.record_information_bits == pytest.approx(0.0, abs=1.0e-14)
    assert result.system_coherence_bits == pytest.approx(1.0)
    assert result.information_budget_bits == pytest.approx(1.0)


def test_record_information_and_coherence_obey_A_plus_B_budget():
    strengths = np.linspace(0.0, np.pi / 2.0, 101)
    results = [
        quantum_sm.simulate_record_creation(
            1.0 / np.sqrt(2.0),
            np.exp(0.37j) / np.sqrt(2.0),
            strength,
        )
        for strength in strengths
    ]
    recorded = np.array([result.record_information_bits for result in results])
    coherence = np.array([result.system_coherence_bits for result in results])
    budgets = np.array([result.information_budget_bits for result in results])
    rates = quantum_sm.unpacking_rate_bits_per_time(results, strengths)

    assert np.all(np.diff(recorded) >= -1.0e-12)
    assert np.all(np.diff(coherence) <= 1.0e-12)
    assert budgets == pytest.approx(np.ones_like(budgets), abs=1.0e-12)
    assert recorded + coherence == pytest.approx(np.ones_like(recorded), abs=1.0e-12)
    assert np.all(rates >= -1.0e-9)


def test_information_budget_tracks_an_asymmetric_initial_state():
    alpha = np.sqrt(0.8)
    beta = np.sqrt(0.2)
    expected_capacity = quantum_sm.shannon_entropy_bits([0.8, 0.2])

    result = quantum_sm.simulate_record_creation(alpha, beta, np.pi / 3.0)

    assert expected_capacity == pytest.approx(0.7219280948873623)
    assert result.initial_capacity_bits == pytest.approx(expected_capacity)
    assert result.information_budget_bits == pytest.approx(expected_capacity)
