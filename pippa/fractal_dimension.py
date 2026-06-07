"""Численный вывод фрактальной размерности D = 4/pi (Аксиома IV).

D - среднее значение L1-нормы единичного евклидова вектора,
усреднённое по всем направлениям:

    D = (1/2pi) * integral_0^{2pi} (|cos t| + |sin t|) dt = 8/(2pi) = 4/pi
"""

from __future__ import annotations

import math

from . import constants


def l1_norm_of_unit_vector(theta: float) -> float:
    """L1-норма единичного вектора (cos t, sin t)."""
    return abs(math.cos(theta)) + abs(math.sin(theta))


def compute_D(n_steps: int = 1_000_000) -> float:
    """Численно вычислить D методом трапеций по углу.

    Args:
        n_steps: число шагов разбиения интервала [0, 2pi].

    Returns:
        Численная оценка D, должна сходиться к 4/pi.
    """
    two_pi = 2.0 * math.pi
    dt = two_pi / n_steps
    total = 0.0
    for i in range(n_steps):
        t0 = i * dt
        t1 = t0 + dt
        total += 0.5 * (l1_norm_of_unit_vector(t0) + l1_norm_of_unit_vector(t1)) * dt
    return total / two_pi


def analytic_D() -> float:
    """Аналитическое значение D = 4/pi."""
    return constants.D
