import numpy as np
import pytest

from pippa import dark_energy
from pippa import sector_spectrum


def test_negative_symmetric_mixing_gives_one_light_stable_mode():
    target = 1.0e-10
    report = sector_spectrum.tune_character_mode(
        neg_coupling=-1.0,
        mir_coupling=-1.0,
        negmir_coupling=-1.0,
        target_mode="symmetric",
        target_mass_squared=target,
    )

    assert report.mode_masses_squared == pytest.approx(
        (target, 4.0 + target, 4.0 + target, 4.0 + target)
    )
    assert report.light_modes == ("symmetric",)
    assert report.target_is_unique_light_mode
    assert report.relativistic_tachyon_free


def test_graph_laplacian_generates_the_protected_mass_relation():
    weights = (0.7, 1.1, 1.6)
    laplacian = sector_spectrum.sector_graph_laplacian(*weights)
    expected_eigenvalues = sector_spectrum.analytic_graph_laplacian_eigenvalues(
        *weights
    )

    assert laplacian @ [1.0, 1.0, 1.0, 1.0] == pytest.approx(
        [0.0, 0.0, 0.0, 0.0]
    )
    assert sorted(expected_eigenvalues) == pytest.approx(
        sorted(np.linalg.eigvalsh(laplacian))
    )
    assert expected_eigenvalues[0] == 0.0
    assert min(expected_eigenvalues[1:]) > 0.0


def test_phase_locking_energy_has_an_exact_common_shift_symmetry():
    phases = [0.2, -0.4, 1.1, 0.7]
    base = sector_spectrum.phase_locking_energy(phases, 0.7, 1.1, 1.6)
    shifted = sector_spectrum.phase_locking_energy(
        [phase + 123.4 for phase in phases],
        0.7,
        1.1,
        1.6,
    )

    assert shifted == pytest.approx(base, abs=1.0e-13)


def test_higgs_portal_norm_preserves_the_compact_common_phase():
    residual = dark_energy.compact_phase_portal_invariance_error(
        amplitudes=[0.8, 1.0, 1.2, 1.4],
        phases=[0.1, -0.3, 0.7, 1.1],
        common_shift=13.7,
    )

    assert residual < 1.0e-15


def test_positive_symmetric_mixing_gives_three_light_hidden_modes():
    target = 1.0e-10
    report = sector_spectrum.tune_character_mode(
        neg_coupling=1.0,
        mir_coupling=1.0,
        negmir_coupling=1.0,
        target_mode="Neg-character",
        target_mass_squared=target,
    )

    assert report.mode_masses_squared == pytest.approx(
        (4.0 + target, target, target, target)
    )
    assert report.light_modes == (
        "Neg-character",
        "Mir-character",
        "NegMir-character",
    )
    assert not report.target_is_unique_light_mode
    assert report.relativistic_tachyon_free


def test_planck_scale_solution_is_stable_but_extremely_tuned():
    audit = dark_energy.audit_symmetric_dark_energy_mode()

    assert audit.hubble_mass_eV == pytest.approx(1.43772e-33, rel=1.0e-5)
    assert audit.vacuum_energy_scale_eV == pytest.approx(2.24037e-3, rel=1.0e-5)
    assert audit.potential_amplitude_scale_eV == pytest.approx(
        audit.vacuum_energy_scale_eV / 2.0**0.25
    )
    assert audit.dark_energy_density_j_m3 == pytest.approx(5.25323e-10, rel=1.0e-5)
    assert audit.heavy_mode_mass_eV == pytest.approx(
        2.0 * audit.mixing_scale_eV,
        rel=1.0e-12,
    )
    assert audit.heavy_to_light_hierarchy > 1.0e30
    assert audit.cancellation_fraction < 2.0e-61
    assert audit.graph_laplacian_protected
    assert audit.pseudo_goldstone_decay_constant_eV == pytest.approx(
        2.46858e27,
        rel=1.0e-5,
    )
    assert audit.decay_constant_over_reduced_planck == pytest.approx(
        1.01366,
        rel=1.0e-5,
    )
    assert audit.linear_amplitude_portal_limit < 1.0e-88
    assert audit.compact_phase_portal_preserves_symmetry
    assert audit.target_is_unique_light_mode
    assert audit.relativistic_tachyon_free


def test_galactic_inverse_length_scale_reduces_but_does_not_remove_tuning():
    galactic_scale = dark_energy.inverse_length_scale_eV(1.0)
    audit = dark_energy.audit_symmetric_dark_energy_mode(galactic_scale)

    assert galactic_scale == pytest.approx(6.394e-27, rel=1.0e-3)
    assert 1.0e-15 < audit.cancellation_fraction < 1.0e-13
    assert audit.target_is_unique_light_mode
    assert audit.relativistic_tachyon_free
