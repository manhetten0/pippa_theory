"""Фундаментальные и теоретические константы теории Pippa.

Единственный источник истины для констант, чтобы избежать
дублирования значений между модулями.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Теоретические константы Pippa ---------------------------------------

#: Фрактальная размерность D = 4/pi (Аксиома IV).
#: Аналитическое значение; численный вывод см. fractal_dimension.compute_D().
D: float = 4.0 / math.pi  # ~= 1.2732395447

# --- Фрактальные размерности квадрантов (выведены из α_A, α_B) ---
# Найдены из условия den_SU2 = (D_N/D_Mir)^4, den_U1 = (D_Neg/D_N)^4
# и требования воспроизведения m_W, sin²θ_W.
D_MIR: float = 1.27677   # уточнённое значение
D_NEG: float = 1.285284  # уточнённое значение

#: Число квадрантов фазового отображения (N, Neg, Mir, NegMir).
N_QUADRANTS: int = 4

#: 3D-аналог размерности (средняя L1-норма на единичной 3-сфере),
#: входит в формулу Хиггса и Коиде.
D3: float = 3.0 / 2.0

def alpha_A_from_D(D_Mir: float = D_MIR, D_Neg: float = D_NEG) -> float:
    """Вычисляет α_A из фрактальных размерностей квадрантов."""
    D = 4.0 / math.pi
    den_SU2 = (D / D_Mir) ** 4
    den_U1 = (D_Neg / D) ** 4
    return den_U1 - den_SU2

def alpha_B_from_D(D_Mir: float = D_MIR, D_Neg: float = D_NEG) -> float:
    """Вычисляет α_B из фрактальных размерностей квадрантов."""
    D = 4.0 / math.pi
    den_SU2 = (D / D_Mir) ** 4
    den_U1 = (D_Neg / D) ** 4
    return 2.0 * den_SU2 - den_U1 - 1.0

# Обновляем значения α_A, α_B на теоретические (вместо приблизительных)
alpha_A: float = alpha_A_from_D()
alpha_B: float = alpha_B_from_D()

# Теоретические коэффициенты чувствительности (могут быть выведены)
A_COEFF_SU2: float = 1.0
B_COEFF_SU2: float = 1.0
A_COEFF_U1: float = 2.0   # U(1) вдвое сильнее реагирует на поле A
B_COEFF_U1: float = 1.0

# --- Экспериментальные эталонные значения (PDG 2024) ---------------------


@dataclass(frozen=True)
class Experimental:
    """Эталонные экспериментальные значения для верификации."""

    alpha_EM: float = 1.0 / 137.035999  # постоянная тонкой структуры
    alpha_s: float = 0.1180             # сильная связь на масштабе Z
    sin2_theta_W: float = 0.23122       # угол Вайнберга (MS-bar, Z-полюс)
    m_W_over_m_Z: float = 80.379 / 91.1876

    m_H_GeV: float = 125.20             # масса Хиггса
    lambda_H: float = 0.12951           # квартичная самосвязь Хиггса
    higgs_vev_GeV: float = 246.0        # вакуумное среднее v

    m_e_MeV: float = 0.51099895         # электрон
    m_mu_MeV: float = 105.6583755       # мюон
    m_tau_MeV: float = 1776.86          # тау-лептон

    m_Z_GeV: float = 91.1876
    m_W_GeV: float = 80.379

    # --- Экспериментальные погрешности (1 sigma, PDG 2024) ------------
    # Нужны, чтобы оценивать предсказания в единицах sigma, а не
    # только в процентах.
    # Истинная sigma alpha(0) крошечная (alpha известна до ~1e-10).
    alpha_EM_err: float = 1.1e-12       # настоящая sigma alpha(0), PDG
    # "Order-of-magnitude" sigma для честного сравнения приближённой
    # формулы Pippa (~0.1%), не привязанной к конкретному масштабу.
    alpha_EM_scale_err: float = 1e-5    # масштабная неопределённость формулы
    alpha_s_err: float = 0.0009         # delta alpha_s(M_Z)
    sin2_theta_W_err: float = 0.00004
    m_W_over_m_Z_err: float = 0.00016   # из delta m_W ~ 0.012, delta m_Z ~ 0.0021
    m_H_GeV_err: float = 0.11
    lambda_H_err: float = 0.00023       # производная от delta m_H
    m_W_GeV_err: float = 0.012
    m_Z_GeV_err: float = 0.0021

    # --- Значения на масштабе m_Z (для сравнения после RG-бега) ----
    # alpha_EM(m_Z) в MS-bar ≈ 1/127.95; alpha_s(m_Z) ≈ 0.1180.
    alpha_EM_mZ: float = 1.0 / 127.951
    # Погрешность доминируется неопределённостью адронного вклада
    # Delta_alpha_had: d(alpha)/alpha ~ d(Delta) => sigma ~ 0.00007 * alpha.
    alpha_EM_mZ_err: float = 0.00007 * (1.0 / 127.951)
    alpha_s_mZ: float = 0.1180
    alpha_s_mZ_err: float = 0.0009


#: Глобальный экземпляр эталонных значений.
EXP = Experimental()
