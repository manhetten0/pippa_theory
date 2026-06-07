"""Тесты базовых формул теории Pippa."""

import math

import pytest

from pippa import (
    constants,
    dark_matter,
    fractal_dimension,
    gravity,
    particle_physics,
    verification,
)


# --- Фрактальная размерность ----------------------------------------------


def test_analytic_D_equals_4_over_pi():
    assert constants.D == pytest.approx(4.0 / math.pi)


def test_numerical_D_converges_to_analytic():
    # Численный вывод должен совпасть с 4/pi.
    numeric = fractal_dimension.compute_D(n_steps=200_000)
    assert numeric == pytest.approx(4.0 / math.pi, rel=1e-4)


# --- Константы связи и бозоны -------------------------------------------


def test_alpha_EM_value():
    assert particle_physics.alpha_EM() == pytest.approx(0.0072871, rel=1e-4)


def test_alpha_EM_within_1_percent_of_experiment():
    pred = particle_physics.alpha_EM()
    assert abs(pred - constants.EXP.alpha_EM) / constants.EXP.alpha_EM < 0.01


def test_alpha_s_is_16_times_alpha_EM():
    assert particle_physics.alpha_s() == pytest.approx(16.0 * particle_physics.alpha_EM())


def test_sin2_theta_W_value():
    assert particle_physics.sin2_theta_W() == pytest.approx(1.0 / (4.0 + 1.0 / math.pi))
    assert particle_physics.sin2_theta_W() == pytest.approx(0.231572, rel=1e-4)


def test_m_W_over_m_Z_equals_cos_theta_W():
    assert particle_physics.m_W_over_m_Z() == pytest.approx(
        particle_physics.cos_theta_W()
    )
    assert particle_physics.m_W_over_m_Z() == pytest.approx(0.876600, rel=1e-4)


def test_lambda_higgs_value():
    assert particle_physics.lambda_higgs() == pytest.approx(4.0 / math.pi**3)
    assert particle_physics.lambda_higgs() == pytest.approx(0.128987, rel=1e-4)


def test_m_higgs_within_1_percent():
    pred = particle_physics.m_higgs()
    assert abs(pred - constants.EXP.m_H_GeV) / constants.EXP.m_H_GeV < 0.01


# --- Гравитация (Приложение G) ---------------------------------------------


def test_newtonian_limit_recovers_inverse_r():
    ch = gravity.InformationChannel(C0=1.0, alpha=0.0, lam=0.0)
    # При Idot=0 и постоянном C: Phi = -I/(C0 r).
    assert ch.potential(2.0, I=4.0) == pytest.approx(-2.0)


def test_acceleration_follows_inverse_square():
    ch = gravity.InformationChannel(C0=1.0)
    # g = -dPhi/dr = -I/(C0 r^2) для ньютоновского предела.
    g = ch.acceleration(r=2.0, I=4.0)
    assert g == pytest.approx(-4.0 / 4.0, rel=1e-3)


def test_saturation_reduces_capacity():
    ch = gravity.InformationChannel(C0=1.0, alpha=2.0)
    assert ch.capacity(I_dot=0.0) == pytest.approx(1.0)
    assert ch.capacity(I_dot=1.0) == pytest.approx(1.0 / 3.0)


def test_potential_rejects_nonpositive_r():
    ch = gravity.InformationChannel()
    with pytest.raises(ValueError):
        ch.potential(0.0, I=1.0)


# --- Тёмная материя --------------------------------------------------------


def test_scaling_exponent_k():
    assert dark_matter.scaling_exponent_k() == pytest.approx(-0.137, abs=1e-3)


def test_M_operator_at_origin_is_one():
    profile = dark_matter.DarkMatterProfile(rho0=1.0, r0=1.0, A_M=5.0)
    assert profile.M_operator(1e-9) == pytest.approx(1.0, abs=1e-6)


def test_alpha_eff_dwarf_is_2_minus_D():
    # Для карликовых (gamma=0): alpha_eff = 2 - D ~= 0.727.
    profile = dark_matter.DarkMatterProfile(rho0=1.0, r0=1.0, A_M=1.0, gamma=0.0)
    assert profile.alpha_eff(5.0) == pytest.approx(2.0 - constants.D)


def test_density_positive():
    profile = dark_matter.DarkMatterProfile(rho0=1.0, r0=2.0, A_M=3.0, gamma=1.2)
    assert profile.density(1.0) > 0.0


# --- Верификация ----------------------------------------------------------


def test_all_core_predictions_within_2_percent():
    for comp in verification.run_all():
        assert abs(comp.rel_error) < 0.02, comp
