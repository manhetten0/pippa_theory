"""Профиль тёмной материи и закон масштабирования (теория Pippa).

Базовые формулы из раздела 13.1 и Приложения G.10:

    rho_DM(r) = rho0 * (r0/r)^alpha_eff * M[A](r)
    alpha_eff(r) = (2 - D) + gamma * (1 - exp(-r/r0))
    M[A](r) = g_M * [1 + A_M * (1 - exp(-r/r0))]
    A_M propto M_bar^k,  k = (1 - D)/2 ~= -0.137

``g_M`` is the overall inter-sector projection strength.  Consequently the
physical no-projection limit is exactly ``g_M = 0 -> rho_DM = 0``.  ``A_M``
only changes the radial shape after the projection has been switched on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import constants


def scaling_exponent_k() -> float:
    """Теоретический показатель закона масштабирования: k = (1 - D)/2."""
    return (1.0 - constants.D) / 2.0


def A_M_scaling(M_bar: float, amplitude: float = 106.6) -> float:
    """Амплитуда межквадрантного вклада: A_M = amplitude * M_bar^k."""
    return amplitude * M_bar ** scaling_exponent_k()


@dataclass
class DarkMatterProfile:
    """Параметрический профиль плотности тёмной материи Pippa.

    Attributes:
        rho0: масштаб плотности.
        r0: характерный радиус.
        A_M: амплитуда оператора межквадрантной связи M[A].
        gamma: параметр адиабатического сжатия
               (0 для карликовых, >0 для спиральных галактик).
        intersector_strength: общая сила проекции g_M; при g_M=0 вклад
                              тёмной материи обязан исчезнуть.
    """

    rho0: float
    r0: float
    A_M: float
    gamma: float = 0.0
    intersector_strength: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.intersector_strength):
            raise ValueError("intersector_strength должно быть конечным")
        if self.intersector_strength < 0.0:
            raise ValueError("intersector_strength не может быть отрицательным")

    def alpha_eff(self, r: float) -> float:
        """Эффективный показатель степени alpha_eff(r)."""
        return (2.0 - constants.D) + self.gamma * (1.0 - math.exp(-r / self.r0))

    def M_operator(self, r: float) -> float:
        """Return ``g_M * [1 + A_M*(1 - exp(-r/r0))]``."""
        shape = 1.0 + self.A_M * (1.0 - math.exp(-r / self.r0))
        return self.intersector_strength * shape

    def density(self, r: float) -> float:
        """Плотность тёмной материи rho_DM(r)."""
        if r <= 0.0:
            raise ValueError("r должно быть положительным")
        return (
            self.rho0
            * (self.r0 / r) ** self.alpha_eff(r)
            * self.M_operator(r)
        )
