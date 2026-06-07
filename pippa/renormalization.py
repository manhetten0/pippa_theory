"""Однопетлевой RG-бег констант связи (ренормализация).

Цель модуля: честно проверить гипотезу о том, что формулы Pippa задают
константы на некотором масштабе mu, а наблюдаемые значения получаются
ренормгрупповым бегом к масштабу измерения (обычно m_Z).

Реализованы однопетлевые бета-функции Стандартной модели:
- QED: d(alpha_EM)/d ln(mu) = (2/(3pi)) * sum_f N_c Q_f^2 * alpha_EM^2
- QCD: d(alpha_s)/d ln(mu) = -(b0/(2pi)) * alpha_s^2, b0 = 11 - 2/3 n_f

Интегрирование выполняется методом Рунге-Кутты 4-го порядка (RK4)
только на стандартной библиотеке, без внешних зависимостей.

ВАЖНО (область применимости):
- RG-бег корректно описывает alpha_EM и alpha_s.
- Массы W/Z требуют не бега, а электрослабых петлевых поправок
  (Delta r, Delta rho, self-energies) — это другой класс расчёта,
  который здесь НЕ реализован. Поэтому для m_W бег не применяется.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

# --- Опорные масштабы (ГэВ) ----------------------------------------------

#: Масса Z-бозона — стандартный масштаб для сравнения констант связи.
M_Z_GEV: float = 91.1876

#: Условный низкоэнергетический масштаб (томсоновский предел для alpha_EM).
M_LOW_GEV: float = 0.000511  # ~ масса электрона, q^2 -> 0


def _active_quark_flavors(mu_GeV: float) -> int:
    """Число активных кварковых ароматов на масштабе mu (пороговая схема)."""
    if mu_GeV < 1.275:      # ниже c-кварка
        return 3
    if mu_GeV < 4.18:       # ниже b-кварка
        return 4
    if mu_GeV < 173.0:      # ниже t-кварка
        return 5
    return 6


# --- Бета-функции (однопетлевые) -----------------------------------------


def beta_alpha_s(alpha_s: float, mu_GeV: float) -> float:
    """d(alpha_s)/d ln(mu): однопетлевая QCD.

    b0 = 11 - (2/3) n_f. Знак минус => асимптотическая свобода
    (alpha_s падает с ростом энергии).
    """
    n_f = _active_quark_flavors(mu_GeV)
    b0 = 11.0 - (2.0 / 3.0) * n_f
    return -(b0 / (2.0 * math.pi)) * alpha_s**2


def beta_alpha_EM(alpha_EM: float, mu_GeV: float) -> float:
    """d(alpha_EM)/d ln(mu): однопетлевая QED.

    Сумма по заряженным фермионам выше порога mu с цветовым фактором N_c.
    alpha_EM растёт с энергией (экранирование вакуума).
    """
    # (фермион, заряд Q, число цветов N_c, порог в ГэВ)
    fermions = [
        ("e", 1.0, 1, 0.000511),
        ("mu", 1.0, 1, 0.1057),
        ("tau", 1.0, 1, 1.777),
        ("u", 2.0 / 3.0, 3, 0.0022),
        ("c", 2.0 / 3.0, 3, 1.275),
        ("t", 2.0 / 3.0, 3, 173.0),
        ("d", -1.0 / 3.0, 3, 0.0047),
        ("s", -1.0 / 3.0, 3, 0.095),
        ("b", -1.0 / 3.0, 3, 4.18),
    ]
    sum_q2 = sum(
        n_c * q**2 for _name, q, n_c, thr in fermions if mu_GeV >= thr
    )
    return (2.0 / (3.0 * math.pi)) * sum_q2 * alpha_EM**2


# --- RK4-интегратор по ln(mu) --------------------------------------------


def run_coupling(
    beta: Callable[[float, float], float],
    value0: float,
    mu_from_GeV: float,
    mu_to_GeV: float,
    n_steps: int = 2000,
) -> float:
    """Проинтегрировать бета-функцию от mu_from к mu_to (RK4 по t=ln mu).

    beta(value, mu) = d(value)/d ln(mu).
    """
    if mu_from_GeV <= 0.0 or mu_to_GeV <= 0.0:
        raise ValueError("Масштабы mu должны быть положительны.")

    t0 = math.log(mu_from_GeV)
    t1 = math.log(mu_to_GeV)
    h = (t1 - t0) / n_steps

    value = value0
    t = t0
    for _ in range(n_steps):
        mu = math.exp(t)
        k1 = beta(value, mu)
        k2 = beta(value + 0.5 * h * k1, math.exp(t + 0.5 * h))
        k3 = beta(value + 0.5 * h * k2, math.exp(t + 0.5 * h))
        k4 = beta(value + h * k3, math.exp(t + h))
        value += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    return value


# --- Высокоуровневые помощники -------------------------------------------


@dataclass(frozen=True)
class RunResult:
    """Результат прогона одной константы по RG."""

    name: str
    value_before: float
    value_after: float
    mu_from_GeV: float
    mu_to_GeV: float


def run_alpha_s_to_mz(alpha_s_pippa: float, mu_from_GeV: float) -> RunResult:
    """Прогнать alpha_s от масштаба формулы Pippa к m_Z."""
    after = run_coupling(beta_alpha_s, alpha_s_pippa, mu_from_GeV, M_Z_GEV)
    return RunResult("alpha_s", alpha_s_pippa, after, mu_from_GeV, M_Z_GEV)


def run_alpha_EM_to_mz(alpha_EM_pippa: float, mu_from_GeV: float = M_LOW_GEV) -> RunResult:
    """Прогнать alpha_EM от низкого масштаба (формула Pippa) к m_Z."""
    after = run_coupling(beta_alpha_EM, alpha_EM_pippa, mu_from_GeV, M_Z_GEV)
    return RunResult("alpha_EM", alpha_EM_pippa, after, mu_from_GeV, M_Z_GEV)
