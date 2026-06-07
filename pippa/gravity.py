"""Эмерджентная гравитация из насыщения информационного канала.

Реализует базовые формулы Приложения G:
- эффективный потенциал Phi(r) = -(I + lambda*Idot) / (C(r) * r)
- насыщение канала C_eff(r) = C0 / (1 + alpha * Idot(r))
- гравитационное ускорение g = -dPhi/dr
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InformationChannel:
    """Модель информационного канала с конечной пропускной способностью.

    Attributes:
        C0: базовая пропускная способность канала.
        alpha: коэффициент насыщения канала.
        lam: универсальная безразмерная константа связи (lambda).
    """

    C0: float = 1.0
    alpha: float = 0.0
    lam: float = 0.0

    def capacity(self, I_dot: float) -> float:
        """Эффективная пропускная способность C_eff = C0 / (1 + alpha*Idot)."""
        return self.C0 / (1.0 + self.alpha * I_dot)

    def potential(self, r: float, I: float, I_dot: float = 0.0) -> float:
        """Эффективный гравитационный потенциал Phi(r).

        Phi(r) = -(I + lambda*Idot) / (C(r) * r)
        """
        if r <= 0.0:
            raise ValueError("r должно быть положительным")
        c = self.capacity(I_dot)
        return -(I + self.lam * I_dot) / (c * r)

    def acceleration(
        self, r: float, I: float, I_dot: float = 0.0, dr: float = 1e-6
    ) -> float:
        """Гравитационное ускорение g = -dPhi/dr (численная производная).

        Возвращает радиальную компоненту (отрицательна = притяжение).
        """
        phi_plus = self.potential(r + dr, I, I_dot)
        phi_minus = self.potential(r - dr, I, I_dot)
        return -(phi_plus - phi_minus) / (2.0 * dr)

    def newtonian_limit(self, r: float, I: float) -> float:
        """Ньютоновский предел (Idot -> 0, постоянное C): Phi = -I/(C0 r)."""
        return self.potential(r, I, I_dot=0.0)
