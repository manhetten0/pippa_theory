#!/usr/bin/env python3
"""
Проверка универсального уравнения мод в Теории Pippa

Уравнение: P₁ C M + P₂ M₀ = M₀^(1-1/μ) M^(1/μ)
где:
- M - масса моды (МэВ)
- C - когерентность (0-1)
- M₀ = 0.511 МэВ - фундаментальный масштаб
- μ = 1.261 - фрактальная размерность (ранее μ=1.26, теперь μ=1.272?)
- P₁ = 0.423, P₂ = 0.577 (из подгонки)

Проверка для различных частиц.
"""

import math
import numpy as np

# Параметры теории
M0 = 0.511      # МэВ
MU = 1.272      # Фрактальная размерность (обновлено до 1.272)
P1 = 0.423      # Из подгонки
P2 = 0.577      # Из подгонки

# Экспериментальные массы
M_ELECTRON = 0.511
M_MUON = 105.66
M_PROTON = 938.272
M_NEUTRON = 939.565

def calculate_left_side(mass, coherence):
    """Вычисление левой части уравнения: P₁ C M + P₂ M₀"""
    return P1 * coherence * mass + P2 * M0

def calculate_right_side(mass):
    """Вычисление правой части уравнения: M₀^(1-1/μ) M^(1/μ)"""
    return M0**(1 - 1/MU) * mass**(1/MU)

def solve_for_coherence(mass):
    """Решение уравнения относительно когерентности C"""
    # P₁ C M + P₂ M₀ = M₀^(1-1/μ) M^(1/μ)
    # P₁ C M = M₀^(1-1/μ) M^(1/μ) - P₂ M₀
    # C = [M₀^(1-1/μ) M^(1/μ) - P₂ M₀] / (P₁ M)

    right_side = calculate_right_side(mass)
    c = (right_side - P2 * M0) / (P1 * mass)
    return c

def verify_particle(particle_name, mass_exp, coherence_assumed=None):
    """Проверка уравнения для конкретной частицы"""
    print(f"\n=== Проверка для {particle_name} ===")
    print(f"Экспериментальная масса M = {mass_exp} МэВ")

    # Если когерентность задана, проверить уравнение
    if coherence_assumed is not None:
        left_side = calculate_left_side(mass_exp, coherence_assumed)
        right_side = calculate_right_side(mass_exp)

        print(f"Предполагаемая когерентность C = {coherence_assumed}")
        print(f"Левая часть: P₁ C M + P₂ M₀ = {left_side:.6f}")
        print(f"Правая часть: M₀^(1-1/μ) M^(1/μ) = {right_side:.6f}")
        print(f"Расхождение: {abs(left_side - right_side):.8f}")

        return abs(left_side - right_side)
    else:
        # Решить для когерентности
        c_calc = solve_for_coherence(mass_exp)
        print(f"Рассчитанная когерентность C = {c_calc:.6f}")

        return c_calc

def verify_electron():
    """Специальная проверка для электрона (эталон)"""
    print("\n=== Проверка для электрона (эталон) ===")

    # Для электрона C должен быть 1
    c_assumed = 1.0
    left_side = calculate_left_side(M_ELECTRON, c_assumed)
    right_side = calculate_right_side(M_ELECTRON)

    print(f"Предполагаемая когерентность C = {c_assumed}")
    print(f"Левая часть: P₁ C M + P₂ M₀ = {left_side:.6f}")
    print(f"Правая часть: M₀^(1-1/μ) M^(1/μ) = {right_side:.6f}")
    print(f"Расхождение: {abs(left_side - right_side):.8f}")

    # Для электрона правая часть должна быть равна левой при C=1
    # M₀^(1-1/μ) M^(1/μ) при M=M₀ должна давать M₀
    expected_right = M0**(1-1/MU) * M0**(1/MU)
    print(f"Математически: M₀^(1-1/μ) * M₀^(1/μ) = M₀^(1-1/μ + 1/μ) = M₀^1 = {M0}")
    print(f"Ожидаемая правая часть для M=M₀: {expected_right}")
    print(f"Фактическая правая часть: {right_side}")

    return abs(left_side - right_side)

def test_muon_with_calculated_c():
    """Проверка мюона с рассчитанной когерентностью"""
    print("\n=== Проверка мюона с рассчитанной когерентностью ===")

    c_calc = solve_for_coherence(M_MUON)
    print(f"Рассчитанная когерентность для мюона C = {c_calc:.6f}")

    # Проверить уравнение с этой когерентностью
    left_side = calculate_left_side(M_MUON, c_calc)
    right_side = calculate_right_side(M_MUON)

    print(f"Левая часть: {left_side:.6f}")
    print(f"Правая часть: {right_side:.6f}")
    print(f"Расхождение: {abs(left_side - right_side):.8f}")

    return c_calc

def verify_all_particles():
    """Проверка всех основных частиц"""
    print("=== Проверка универсального уравнения мод Теории Pippa ===")

    # Электрон (эталон)
    verify_electron()

    # Мюон
    test_muon_with_calculated_c()

    # Протон
    c_proton = verify_particle("Протон", M_PROTON)

    # Нейтрон
    c_neutron = verify_particle("Нейтрон", M_NEUTRON)

    print("=== Сводная таблица ===")
    print("Частица\t\tМасса (МэВ)\tКогерентность C\tРасхождение")
    print(f"Электрон\t{M_ELECTRON}\t\t1.0\t\t{verify_electron():.8f}")
    print(f"Мюон\t\t{M_MUON}\t\t{test_muon_with_calculated_c():.3f}\t\t-")
    print(f"Протон\t\t{M_PROTON}\t\t{c_proton:.3f}")
    print(f"Нейтрон\t\t{M_NEUTRON}\t\t{c_neutron:.3f}")

def run_comprehensive_test():
    """Комплексная проверка уравнения"""
    print("=== Комплексная проверка универсального уравнения мод ===")

    verify_all_particles()

    # Тестирование с разными значениями μ
    print("=== Тестирование с разными значениями μ ===")
    test_mu_values = [1.26, 1.261, 1.272, 1.28]

    for mu_test in test_mu_values:
        print(f"\nТестирование с μ = {mu_test}")
        # Обновляем глобальную переменную
        global MU
        MU = mu_test

        c_muon = solve_for_coherence(M_MUON)
        print(f"Когерентность мюона при μ={mu_test}: C={c_muon:.6f}")

        # Проверяем уравнение
        left = calculate_left_side(M_MUON, c_muon)
        right = calculate_right_side(M_MUON)
        print(f"Расхождение: {abs(left - right):.8f}")

if __name__ == "__main__":
    run_comprehensive_test()
