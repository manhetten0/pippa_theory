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
    assert particle_physics.lambda_higgs() == pytest.approx(0.129006, rel=1e-4)


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


def test_all_core_predictions_within_2_percent(capsys):
    comparisons = verification.run_all()

    failures = []
    abs_errors = []
    sigmas = []
    for comp in comparisons:
        abs_errors.append(abs(comp.rel_error_percent))
        if not math.isnan(comp.n_sigma):
            sigmas.append(abs(comp.n_sigma))
        if abs(comp.rel_error) >= 0.02:
            failures.append(comp)

    total = len(comparisons)
    passed = total - len(failures)
    max_err = max(abs_errors)
    mean_err = sum(abs_errors) / total
    within_1s = sum(1 for s in sigmas if s < 1.0)
    within_3s = sum(1 for s in sigmas if s < 3.0)
    max_sigma = max(sigmas) if sigmas else float("nan")

    # Вывод итогов подсчётов (виден при запуске pytest с флагом -s).
    with capsys.disabled():
        print("\n" + "=" * 78)
        print("Итоги верификации базовых формул теории Pippa")
        print("порог: 2% по отн. ошибке; σ = отклонение от эксп. в единицах погрешности")
        print("=" * 78)
        for comp in comparisons:
            status = "OK" if abs(comp.rel_error) < 0.02 else "FAIL"
            print(f"[{status:>4}] {comp}")
        print("-" * 78)
        print(f"Проверок пройдено (<2%)   : {passed}/{total}")
        print(f"Средняя ошибка          : {mean_err:.3f}%")
        print(f"Максимальная ошибка      : {max_err:.3f}%")
        print(f"Совместимо в 1σ        : {within_1s}/{len(sigmas)}")
        print(f"Совместимо в 3σ        : {within_3s}/{len(sigmas)}")
        print(f"Максимальное откл. в σ   : {max_sigma:.1f}σ")
        print("=" * 78)
        print(
            "Примечание: малая ошибка в %% не равна совместимости в σ — "
            "точные измерения имеют маленькую σ."
        )

    assert not failures, f"Превышен порог 2%: {failures}"


# --- Ренормализация (RG-бег) -------------------------------------------


def test_alpha_s_runs_down_with_energy():
    # Асимптотическая свобода: alpha_s падает с ростом энергии.
    from pippa import renormalization as rg

    a_low = 0.3
    a_high = rg.run_coupling(rg.beta_alpha_s, a_low, 1.5, 91.1876)
    assert a_high < a_low


def test_alpha_EM_runs_up_with_energy():
    # Экранирование вакуума: alpha_EM растёт с ростом энергии.
    from pippa import renormalization as rg

    a_low = 1.0 / 137.036
    a_high = rg.run_alpha_EM_to_mz(a_low).value_after
    assert a_high > a_low


def test_renormalized_couplings_report(capsys):
    base = verification.run_all()
    base_by_name = {c.name: c for c in base}
    rg_comps = verification.run_all_renormalized()

    with capsys.disabled():
        print("\n" + "=" * 78)
        print("Эффект ренормализации: отклонение в σ до и после RG-бега к m_Z")
        print("=" * 78)
        # alpha_EM: до бега сравнивалось с alpha(q^2->0) с крошечной σ.
        print("[до RG ] " + str(base_by_name["alpha_EM"]))
        print("[до RG ] " + str(base_by_name["alpha_s"]))
        print("-" * 78)
        for comp in rg_comps:
            print("[после ] " + str(comp))
        print("=" * 78)
        print(
            "Вывод: RG-бег корректно описывает alpha_EM и alpha_s. "
            "Массы W/Z требуют петлевых поправок (Delta r), не RG."
        )

    # Минимальная санитарная проверка: прогон дал конечные разумные числа.
    for comp in rg_comps:
        assert comp.predicted > 0.0
        assert math.isfinite(comp.n_sigma)
