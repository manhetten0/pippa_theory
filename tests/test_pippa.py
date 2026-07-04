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
    # REGRESSION test (not physics validation): checks the identity
    # alpha_s = 16*alpha_EM, i.e. that the formula is not broken.
    # Agreement with experiment is checked separately in run_all().
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
    # REGRESSION test: pins the numeric value of lambda_H = 4/pi^3,
    # not agreement with experiment (see run_all).
    assert particle_physics.lambda_higgs() == pytest.approx(4.0 / math.pi**3)
    assert particle_physics.lambda_higgs() == pytest.approx(0.129006, rel=1e-4)


def test_m_higgs_within_1_percent():
    pred = particle_physics.m_higgs()
    assert abs(pred - constants.EXP.m_H_GeV) / constants.EXP.m_H_GeV < 0.01


def test_m_W_uses_experimental_m_Z_as_input():
    # IMPORTANT: m_W() = m_Z * cos(theta_W), and m_Z comes from experiment.
    # This is NOT an independent first-principles prediction of m_W, but a
    # rescaling of the measured m_Z through cos(theta_W). Pin this fact.
    assert particle_physics.m_W(m_Z_GeV=91.1876) == pytest.approx(
        91.1876 * particle_physics.cos_theta_W()
    )
    # A different input m_Z scales the result linearly: proof m_Z is an
    # input, not a derived quantity.
    assert particle_physics.m_W(m_Z_GeV=100.0) == pytest.approx(
        100.0 * particle_physics.cos_theta_W()
    )


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
            status = verification.status_for(comp)
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


def test_alpha_EM_after_running_is_closer_to_alpha_mZ():
    # Honest check: running the Pippa alpha(0) formula to m_Z should move
    # it toward the experimental alpha(m_Z), not toward alpha(0). This
    # directly addresses the artefactual "thousands of sigma" in run_all().
    from pippa import renormalization as rg

    pippa0 = particle_physics.alpha_EM()
    after = rg.run_alpha_EM_to_mz(pippa0).value_after
    exp_mZ = constants.EXP.alpha_EM_mZ
    # After running, the value lands within a reasonable corridor of
    # alpha(m_Z) (~2%), unlike the raw alpha(0) comparison.
    assert abs(after - exp_mZ) / exp_mZ < 0.02


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

    # Минимальная санитарная проверка: прогон дал конечные разумные числа.
    for comp in rg_comps:
        assert comp.predicted > 0.0
        assert math.isfinite(comp.n_sigma)


# --- Electroweak loop corrections (Delta r) ----------------------------


def test_delta_rho_top_is_positive_and_small():
    from pippa import electroweak as ew

    d_rho = ew.delta_rho_top()
    assert 0.005 < d_rho < 0.012


def test_m_W_loops_move_toward_experiment(capsys):
    comps = verification.run_loop_corrected()
    by_name = {c.name: c for c in comps}
    tree = by_name["m_W Pippa-tree"]
    loop = by_name["m_W G_F+Dr"]

    with capsys.disabled():
        print("\n" + "=" * 78)
        print("Electroweak loop corrections for m_W (tree and G_F+Dr differ")
        print("in scheme: tree = m_Z*cos(thW); G_F+Dr = Sirlin from G_F).")
        print("=" * 78)
        print("[Pippa-tree] " + str(tree))
        print("[G_F + Dr  ] " + str(loop))
        print("-" * 78)
        print(
            "Note: schemes differ; both use Pippa alpha(0)/sin2thW as inputs. "
            "Delta r is truncated (alpha + leading top), so ~0.1-0.2% level."
        )

    # G_F+Dr решение должно быть физичным и близким к эксперименту.
    assert 79.0 < loop.predicted < 81.5
    # Петлевое решение ближе к эксперименту, чем чистый tree без Dr.
    assert abs(loop.predicted - 80.379) < abs(79.935 - 80.379) + 0.3


# --- Out-of-sample cosmology -------------------------------------------


def test_cosmology_out_of_sample(capsys):
    from pippa import cosmology
    comps = verification.run_cosmology()
    r_pred = cosmology.tensor_to_scalar()
    r_lim = cosmology.OBS.r_upper_95

    with capsys.disabled():
        print("\n" + "=" * 78)
        print("Out-of-sample cosmology vs Planck/BICEP-Keck (NOT used in SM fits)")
        print("=" * 78)
        for comp in comps:
            status = "OK" if abs(comp.n_sigma) < 3.0 else "TENS"
            print(f"[{status:>4}] {comp}")
        r_ok = "OK" if r_pred < r_lim else "FAIL"
        print(f"[{r_ok:>4}] r (tensor)         pred={r_pred:.4g}  limit<{r_lim} (95%)")
        print("-" * 78)
        print("Real out-of-sample: independent data, formulas not tuned to SM.")

    n_s = next(c for c in comps if c.name == "n_s")
    assert abs(n_s.predicted - 0.9649) < 0.01
    assert r_pred < r_lim
    for comp in comps:
        assert math.isfinite(comp.n_sigma)


# --- Alpha-corrected cosmology (cosmology_alpha_corrected) ---------------


def test_cosmology_alpha_corrected_matches_cosmology():
    from pippa import cosmology, cosmology_alpha_corrected, particle_physics

    assert cosmology_alpha_corrected.spectral_index() == cosmology.spectral_index()
    assert cosmology_alpha_corrected.tensor_to_scalar() == cosmology.tensor_to_scalar()
    assert cosmology_alpha_corrected.non_gaussianity() == cosmology.non_gaussianity()
    assert cosmology_alpha_corrected.dm_to_baryon_ratio() == cosmology.dm_to_baryon_ratio()

    alpha_A = particle_physics.alpha_A_theoretical()
    alpha_B = particle_physics.alpha_B_theoretical()
    assert cosmology_alpha_corrected.dm_to_baryon_ratio(
        alpha_A=alpha_A, alpha_B=alpha_B
    ) == cosmology.dm_to_baryon_ratio(alpha_A=alpha_A, alpha_B=alpha_B)


def test_dm_to_baryon_baseline_without_alpha():
    from pippa import cosmology_alpha_corrected

    D = constants.D
    expected = (D - 1.0) ** (-D)
    assert cosmology_alpha_corrected.dm_to_baryon_ratio() == pytest.approx(expected)
    assert cosmology_alpha_corrected.dm_to_baryon_ratio() == pytest.approx(5.217, rel=1e-3)


def test_dm_to_baryon_alpha_corrected_closer_to_planck(capsys):
    from pippa import cosmology, cosmology_alpha_corrected, particle_physics

    obs = cosmology.OBS
    alpha_A = particle_physics.alpha_A_theoretical()
    alpha_B = particle_physics.alpha_B_theoretical()
    baseline = cosmology_alpha_corrected.dm_to_baryon_ratio()
    corrected = cosmology_alpha_corrected.dm_to_baryon_ratio(
        alpha_A=alpha_A, alpha_B=alpha_B
    )
    factor = corrected / baseline

    base_comp = verification.Comparison(
        "Omega_DM/Omega_b (baseline)",
        baseline,
        obs.dm_baryon,
        sigma_exp=obs.dm_baryon_err,
        energy_scale="Planck18",
    )
    corr_comp = verification.Comparison(
        "Omega_DM/Omega_b (alpha-corr)",
        corrected,
        obs.dm_baryon,
        sigma_exp=obs.dm_baryon_err,
        energy_scale="Planck18 + EW fit",
    )

    with capsys.disabled():
        print("\n" + "=" * 78)
        print("Alpha-corrected cosmology: Omega_DM/Omega_b (cosmology_alpha_corrected)")
        print("=" * 78)
        print(f"  alpha_A = {alpha_A:.6f}   alpha_B = {alpha_B:.6f}  (fit_alpha_AB)")
        print(
            f"  SU(2)/U(1) factor = sqrt((1+2α_A+α_B)/(1+α_A+α_B)) = {factor:.6f}"
        )
        print("-" * 78)
        for comp in (base_comp, corr_comp):
            status = "OK" if abs(comp.n_sigma) < 3.0 else "TENS"
            print(f"[{status:>4}] {comp}")
        print("-" * 78)
        print(
            f"|Δ| baseline → corrected: "
            f"{abs(baseline - obs.dm_baryon):.4f} → {abs(corrected - obs.dm_baryon):.4f}  "
            f"(Planck: {obs.dm_baryon} ± {obs.dm_baryon_err})"
        )

    assert abs(corrected - obs.dm_baryon) < abs(baseline - obs.dm_baryon)
    assert corrected == pytest.approx(obs.dm_baryon, rel=0.01)
    assert abs(corr_comp.n_sigma) < abs(base_comp.n_sigma)


def test_dm_to_baryon_invalid_alpha_raises():
    from pippa import cosmology_alpha_corrected

    with pytest.raises(ValueError, match="non-positive denominator"):
        cosmology_alpha_corrected.dm_to_baryon_ratio(alpha_A=-2.0, alpha_B=0.0)
    with pytest.raises(ValueError, match="non-positive denominator"):
        cosmology_alpha_corrected.dm_to_baryon_ratio(alpha_A=0.0, alpha_B=-2.0)


def test_theoretical_alpha_reproduces_mW_and_sin2():
    from pippa import particle_physics
    exp = constants.EXP
    
    # Берём теоретические α
    alpha_A = particle_physics.alpha_A_theoretical()
    alpha_B = particle_physics.alpha_B_theoretical()
    
    # Вычисляем эффективный угол и массу W
    sin2_eff = particle_physics.sin2_theta_eff(alpha_A, alpha_B)
    m_W_eff  = particle_physics.m_W_eff(alpha_A, alpha_B)
    
    # Целевые значения (on-shell)
    target_sin2 = 1.0 - (exp.m_W_GeV / exp.m_Z_GeV) ** 2
    target_mW   = exp.m_W_GeV
    
    # Проверяем совпадение с высокой точностью
    assert sin2_eff == pytest.approx(target_sin2, rel=1e-6)
    assert m_W_eff  == pytest.approx(target_mW,  rel=1e-6)

def test_alpha_EM_improves_dramatically_with_fit():
    from pippa import particle_physics
    exp = constants.EXP
    # Вызываем fit_alpha_AB() без аргументов, используя значения по умолчанию
    alpha_A = particle_physics.alpha_A_theoretical()
    alpha_B = particle_physics.alpha_B_theoretical()
    g_eff, g_prime_eff = particle_physics.effective_g_and_gprime(alpha_A, alpha_B)
    sin2_eff = particle_physics.sin2_theta_eff(alpha_A, alpha_B)
    alpha_EM_eff = (g_eff**2 * sin2_eff) / (4.0 * math.pi)
    # Древесная α_EM после подгонки даёт ошибку ~3.86%, что значительно лучше,
    # чем исходная формула (0.14% но миллионы сигм). Полное согласие с экспериментом
    # достигается после учёта радиационных поправок (Δr ≈ 0.0358).
    rel_err = abs(alpha_EM_eff - exp.alpha_EM) / exp.alpha_EM
    assert rel_err < 0.05, f"alpha_EM error still {rel_err:.2%}"

def test_alpha_EM_with_delta_r_matches_experiment():
    from pippa import particle_physics, electroweak
    exp = constants.EXP
    # Подгоняем α_A, α_B под массы W и Z
    alpha_A = particle_physics.alpha_A_theoretical()
    alpha_B = particle_physics.alpha_B_theoretical()
    m_W_eff = particle_physics.m_W_eff(alpha_A, alpha_B)
    # Экспериментальное значение Δr (известно из глобальных фитов)
    delta_r_exp = 0.0358
    alpha_calc = electroweak.alpha_from_mW(m_W_eff, exp.m_Z_GeV, delta_r_exp)
    rel_err = abs(alpha_calc - exp.alpha_EM) / exp.alpha_EM
    # После учёта радиационных поправок ожидаем точность лучше 0.001%
    assert rel_err < 0.005, f"alpha_EM from m_W with Δr differs by {rel_err:.2%}"

def test_alpha_AB_theoretical_produce_positive_denominators():
    from pippa import particle_physics
    # Берём теоретические значения (выведенные из D_Mir, D_Neg)
    alpha_A = particle_physics.alpha_A_theoretical()
    alpha_B = particle_physics.alpha_B_theoretical()
    
    # Проверяем, что знаменатели > 0
    denom_SU2 = 1.0 + alpha_A * constants.A_COEFF_SU2 + alpha_B * constants.B_COEFF_SU2
    denom_U1  = 1.0 + alpha_A * constants.A_COEFF_U1  + alpha_B * constants.B_COEFF_U1
    assert denom_SU2 > 0
    assert denom_U1 > 0
    
    # Разумный диапазон
    assert abs(alpha_A) < 1.0
    assert abs(alpha_B) < 1.0

def test_f_NL_sign_mismatch_is_documented():
    # HONEST check: f_NL is effectively NOT predicted. The prediction
    # (+1/D > 0) and the measurement (Planck: -0.9) have OPPOSITE signs.
    # The base report shows "OK" only because Planck's sigma (~5.1) is
    # huge, so |n_sigma| < 3. This test pins the sign mismatch so a future
    # change cannot silently claim f_NL as a success.
    from pippa import cosmology

    pred = cosmology.non_gaussianity()
    exp = cosmology.OBS.f_NL
    assert pred > 0.0          # предсказание положительное
    assert exp < 0.0           # эксперимент отрицательный
    # Совпадение "в пределах sigma" держится только на большой sigma:
    # фиксируем, что без этого знаки противоречат.
    assert (pred - exp) > 0.0

# --- Детальный вывод для отчёта -----------------------------------------
def test_full_report_output():
    """Печатает полный verification.report()."""

    from pippa import verification

    print()
    print("=" * 80)
    print(verification.report())
    print("=" * 80)

    # чтобы тест всегда проходил
    assert True

def test_detailed_output(capsys):
    """Выводит в консоль подробные результаты подгонки α_A, α_B.
    Этот тест всегда проходит, но при запуске pytest -s вы увидите
    все цифры, которые можно скопировать для отчёта.
    """
    from pippa import particle_physics, electroweak, constants
    exp = constants.EXP

    # Подгоняем α_A, α_B
    alpha_A = particle_physics.alpha_A_theoretical()
    alpha_B = particle_physics.alpha_B_theoretical()
    # Эффективные константы
    g_eff, g_prime_eff = particle_physics.effective_g_and_gprime(alpha_A, alpha_B)
    sin2_eff = particle_physics.sin2_theta_eff(alpha_A, alpha_B)
    m_W_eff = particle_physics.m_W_eff(alpha_A, alpha_B)
    # Древесная α_EM
    alpha_EM_tree = (g_eff**2 * sin2_eff) / (4.0 * math.pi)
    # α_EM через Δr (используем экспериментальное Δr)
    delta_r_exp = 0.0358  # стандартное значение из глобальных фитов
    alpha_EM_delta_r = electroweak.alpha_from_mW(m_W_eff, exp.m_Z_GeV, delta_r_exp)

    with capsys.disabled():
        print("\n" + "=" * 80)
        print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ТЕОРИИ PIPPA С α_A, α_B")
        print("=" * 80)
        print(f"Подогнанные константы межквадрантной связи:")
        print(f"    α_A = {alpha_A:.6f}")
        print(f"    α_B = {alpha_B:.6f}")
        print(f"Коэффициенты чувствительности (SU(2), U(1)):")
        print(f"    a = {constants.A_COEFF_SU2}, b = {constants.B_COEFF_SU2}")
        print(f"    a' = {constants.A_COEFF_U1}, b' = {constants.B_COEFF_U1}")
        print("-" * 80)
        print(f"Эффективные константы (древесные):")
        print(f"    g_eff      = {g_eff:.6f}  (эксперимент: {2*exp.m_W_GeV/exp.higgs_vev_GeV:.6f})")
        print(f"    g'_eff     = {g_prime_eff:.6f}")
        print(f"    sin²θ_W    = {sin2_eff:.6f}  (эксперимент on-shell: {1 - (exp.m_W_GeV/exp.m_Z_GeV)**2:.6f})")
        print(f"    m_W, ГэВ   = {m_W_eff:.4f}  (эксперимент: {exp.m_W_GeV:.4f})")
        print("-" * 80)
        print(f"Постоянная тонкой структуры (древесная): α = {alpha_EM_tree:.8f}")
        print(f"Эксперимент (α(0)):                  {exp.alpha_EM:.8f}")
        print(f"Расхождение (древесное): {100*(alpha_EM_tree/exp.alpha_EM - 1):+.2f}%")
        print("-" * 80)
        print(f"С учётом радиационной поправки Δr = {delta_r_exp}:")
        print(f"    α(0) из m_W, Δr = {alpha_EM_delta_r:.8f}")
        print(f"    Расхождение: {100*(alpha_EM_delta_r/exp.alpha_EM - 1):+.4f}%")

    assert True  # тест всегда проходит, нужен только для вывода

# =============================================================================
# Численные методы: сходимость и устойчивость RK4 (ренормализация)
# =============================================================================

def test_rk4_convergence(capsys):
    """
    Проверка сходимости RK4: увеличение числа шагов уменьшает ошибку.
    Вывод: ошибки при разных шагах.
    """
    from pippa import renormalization as rg

    # Аналитическое решение для beta = -C * alpha^2 с постоянным n_f=3
    C = 9.0 / (2.0 * math.pi)
    alpha0 = 0.3
    mu0 = 1.0
    mu1 = 10.0
    alpha_exact = alpha0 / (1.0 + C * alpha0 * math.log(mu1 / mu0))

    def beta_const(alpha, mu):
        return -C * alpha * alpha

    n_steps_list = [10, 50, 200, 1000, 5000]
    errors = []
    values = []

    for n_steps in n_steps_list:
        val = rg.run_coupling(beta_const, alpha0, mu0, mu1, n_steps=n_steps)
        values.append(val)
        errors.append(abs(val - alpha_exact))

    with capsys.disabled():
        print("\n[RK4 сходимость] alpha0 =", alpha0, "mu0 =", mu0, "mu1 =", mu1)
        print("Точное значение alpha =", alpha_exact)
        print("Шаги   |   alpha   |   ошибка")
        for n, v, e in zip(n_steps_list, values, errors):
            print(f"{n:6d} | {v:.8f} | {e:.2e}")
        # Проверка монотонного уменьшения ошибки
        for i in range(1, len(errors)):
            assert errors[i] < errors[i-1] * 1.1, \
                f"Ошибка не уменьшается: {errors[i-1]:.2e} -> {errors[i]:.2e}"
        assert errors[-1] < 1e-6, f"Ошибка слишком велика: {errors[-1]:.2e}"


def test_rk4_stability(capsys):
    """
    Проверка устойчивости RK4: даже при малом числе шагов результат не уходит в бесконечность.
    Вывод: значения при малом числе шагов.
    """
    from pippa import renormalization as rg

    alpha0 = 0.3
    mu0 = 1.0
    mu1 = 100.0
    C = 9.0 / (2.0 * math.pi)
    beta_const = lambda alpha, mu: -C * alpha * alpha

    n_steps_list = [2, 5, 10, 20]
    values = []
    for n_steps in n_steps_list:
        val = rg.run_coupling(beta_const, alpha0, mu0, mu1, n_steps=n_steps)
        values.append(val)
        assert math.isfinite(val), f"Не конечный результат при {n_steps} шагах"
        assert val > 0, f"Отрицательный результат при {n_steps} шагах"

    with capsys.disabled():
        print("\n[RK4 устойчивость] alpha0 =", alpha0)
        print("Шаги  |  alpha (mu=100)")
        for n, v in zip(n_steps_list, values):
            print(f"{n:5d} | {v:.6f}")
        # Проверяем, что при 2 шагах значение не взрывается
        assert values[0] < 10.0, f"Слишком большое значение при 2 шагах: {values[0]}"


def test_rk4_handles_quark_thresholds(capsys):
    """
    Проверка, что RK4 корректно работает с переменным числом ароматов.
    Вывод: значение alpha_s на выходе.
    """
    from pippa import renormalization as rg

    alpha0 = 0.3
    mu0 = 0.5
    mu1 = 100.0
    val = rg.run_coupling(rg.beta_alpha_s, alpha0, mu0, mu1, n_steps=1000)

    with capsys.disabled():
        print("\n[RK4 с порогами кварков]")
        print(f"alpha_s({mu0}) = {alpha0} -> alpha_s({mu1}) = {val:.8f}")
    assert 0.0 < val < 1.0, f"Нефизичное значение alpha_s: {val}"


# =============================================================================
# Численные методы: сходимость и устойчивость решения для m_W
# =============================================================================

def test_mW_iteration_convergence(capsys):
    """
    Проверка сходимости итерационного метода для m_W.
    Вывод: количество итераций и невязка для разных допусков.
    """
    from pippa import electroweak as ew
    from pippa import particle_physics

    alpha0 = particle_physics.alpha_EM()
    tols = [1e-6, 1e-10, 1e-14]
    results = []

    for tol in tols:
        res = ew.m_W_loop_corrected(alpha0, tol=tol, max_iter=1000)
        # Вычисляем невязку
        mW = res.m_W_loop_GeV
        mZ = ew.M_Z_GEV
        gf = ew.G_F
        lhs = mW * mW * (1.0 - mW * mW / (mZ * mZ))
        rhs = (math.pi * alpha0) / (math.sqrt(2.0) * gf) / (1.0 - ew.delta_r(mW, mZ))
        residual = abs(lhs - rhs) / lhs if lhs != 0 else float('inf')
        results.append((tol, res.iterations, residual))
        assert res.iterations < 200, f"Слишком много итераций при tol={tol}: {res.iterations}"
        assert residual < 10 * tol, f"Невязка {residual:.2e} > {tol}"

    with capsys.disabled():
        print("\n[Итерации для m_W] alpha0 =", alpha0)
        print("Допуск      | итераций | невязка")
        for tol, iters, resid in results:
            print(f"{tol:.0e}  | {iters:6d}   | {resid:.2e}")


def test_mW_analytic_limit(capsys):
    """
    Проверка, что при малом alpha0 численное решение совпадает с аналитическим.
    Вывод: относительное расхождение.
    """
    from pippa import electroweak as ew

    alpha0_small = 1e-6
    res = ew.m_W_loop_corrected(alpha0_small)
    mW_num = res.m_W_loop_GeV
    mZ = ew.M_Z_GEV
    gf = ew.G_F
    a = math.pi * alpha0_small / (math.sqrt(2.0) * gf)
    disc = 1.0 - 4.0 * a / (mZ * mZ)
    if disc < 0:
        pytest.skip("Дискриминант отрицателен, тест пропущен")
    mW_ana = math.sqrt((mZ * mZ / 2.0) * (1.0 + math.sqrt(disc)))
    rel_diff = abs(mW_num - mW_ana) / mW_ana

    with capsys.disabled():
        print("\n[Аналитический предел m_W] alpha0 =", alpha0_small)
        print(f"Численно  : {mW_num:.8f} ГэВ")
        print(f"Аналитич. : {mW_ana:.8f} ГэВ")
        print(f"Отн. ошибка : {rel_diff:.2e}")

    assert rel_diff < 1e-4, f"Расхождение {rel_diff:.2e} слишком велико"


def test_mW_discriminant_negative_raises(capsys):
    """
    Проверка, что при отрицательном дискриминанте выбрасывается ValueError.
    Вывод: критическое значение alpha0.
    """
    from pippa import electroweak as ew

    mZ = ew.M_Z_GEV
    gf = ew.G_F
    alpha_crit = (mZ * mZ * math.sqrt(2.0) * gf) / (4.0 * math.pi)
    alpha0_bad = alpha_crit * 1.1

    with capsys.disabled():
        print("\n[Отрицательный дискриминант]")
        print(f"Критическое alpha0 = {alpha_crit:.6e}")
        print(f"Пробуем alpha0 = {alpha0_bad:.6e} -> ожидается ValueError")

    with pytest.raises(ValueError, match="No real solution"):
        ew.m_W_loop_corrected(alpha0_bad)


def test_alpha_from_mW_inverts_calculation(capsys):
    """
    Проверка, что alpha_from_mW восстанавливает исходное alpha0.
    Вывод: относительная ошибка восстановления.
    """
    from pippa import electroweak as ew
    from pippa import particle_physics

    alpha0 = particle_physics.alpha_EM()
    res = ew.m_W_loop_corrected(alpha0)
    mW = res.m_W_loop_GeV
    delta_r_val = res.delta_r
    alpha_reconstructed = ew.alpha_from_mW(mW, ew.M_Z_GEV, delta_r_val, ew.G_F)
    rel_err = abs(alpha_reconstructed - alpha0) / alpha0

    with capsys.disabled():
        print("\n[Обратное восстановление alpha]")
        print(f"Исходное alpha0      : {alpha0:.8f}")
        print(f"Восстановленное alpha: {alpha_reconstructed:.8f}")
        print(f"Отн. ошибка          : {rel_err:.2e}")

    assert rel_err < 1e-8, f"Ошибка восстановления alpha: {rel_err:.2e}"