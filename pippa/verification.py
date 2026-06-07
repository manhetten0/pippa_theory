"""Верификация предсказаний теории Pippa против эксперимента (PDG 2024).

Сравнивает базовые формулы с эталонными значениями и считает
относительное расхождение.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import constants, particle_physics, fractal_dimension


@dataclass(frozen=True)
class Comparison:
    """Результат сравнения одного предсказания с экспериментом."""

    name: str
    predicted: float
    experimental: float
    sigma_exp: float = 0.0
    energy_scale: str = "-"

    @property
    def rel_error(self) -> float:
        """Относительное расхождение (доля)."""
        return (self.predicted - self.experimental) / self.experimental

    @property
    def rel_error_percent(self) -> float:
        """Относительное расхождение в процентах."""
        return 100.0 * self.rel_error

    @property
    def n_sigma(self) -> float:
        """Отклонение предсказания от эксперимента в единицах sigma.

        Это честнее процентов: показывает, совместимо ли предсказание
        с измерением в пределах его погрешности.
        """
        if self.sigma_exp <= 0.0:
            return float("nan")
        return (self.predicted - self.experimental) / self.sigma_exp

    def __str__(self) -> str:
        sig = f"{self.n_sigma:+.1f}σ" if self.sigma_exp > 0.0 else "   n/a"
        return (
            f"{self.name:<18} pred={self.predicted:.6g}  "
            f"exp={self.experimental:.6g}  err={self.rel_error_percent:+.2f}%  "
            f"{sig:>7}  @{self.energy_scale}"
        )


def run_all() -> list[Comparison]:
    """Выполнить все базовые проверки и вернуть список сравнений."""
    exp = constants.EXP
    return [
        Comparison(
            "D (4/pi)", fractal_dimension.analytic_D(), 4.0 / 3.14159265358979,
            sigma_exp=0.0, energy_scale="-",
        ),
        Comparison(
            "alpha_EM", particle_physics.alpha_EM(), exp.alpha_EM,
            sigma_exp=exp.alpha_EM_err, energy_scale="q^2->0",
        ),
        Comparison(
            "alpha_s", particle_physics.alpha_s(), exp.alpha_s,
            sigma_exp=exp.alpha_s_err, energy_scale="M_Z",
        ),
        Comparison(
            "sin^2 theta_W", particle_physics.sin2_theta_W(), exp.sin2_theta_W,
            sigma_exp=exp.sin2_theta_W_err, energy_scale="M_Z (MS-bar)",
        ),
        Comparison(
            "m_W/m_Z", particle_physics.m_W_over_m_Z(), exp.m_W_over_m_Z,
            sigma_exp=exp.m_W_over_m_Z_err, energy_scale="on-shell",
        ),
        Comparison(
            "lambda_H", particle_physics.lambda_higgs(), exp.lambda_H,
            sigma_exp=exp.lambda_H_err, energy_scale="m_H",
        ),
        Comparison(
            "m_H (GeV)", particle_physics.m_higgs(), exp.m_H_GeV,
            sigma_exp=exp.m_H_GeV_err, energy_scale="on-shell",
        ),
        Comparison(
            "m_W (GeV)", particle_physics.m_W(), exp.m_W_GeV,
            sigma_exp=exp.m_W_GeV_err, energy_scale="on-shell",
        ),
    ]


def report() -> str:
    """Сформировать текстовый отчёт по всем проверкам."""
    lines = ["Верификация базовых формул теории Pippa (PDG 2024)", "=" * 60]
    lines.extend(str(c) for c in run_all())
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
