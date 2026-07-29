#!/usr/bin/env python3
"""Простая toy-модель Вселенной с красивой визуализацией.

Идея:
- Берём плоскую FRW-вселенную с радиацией, барионами, тёмной материей и
  тёмной энергией.
- Интегрируем уравнение расширения в безразмерном времени:
      da/dτ = a * E(a)
  где E(a) = H(a)/H0 = sqrt(Ωr/a^4 + Ωm/a^3 + ΩΛ).
- Отдельно показываем:
  1) рост масштаба a(τ),
  2) эволюцию долей компонент,
  3) H(a) и параметр замедления q(a).

Скрипт самодостаточный: только numpy + matplotlib.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CosmoParams:
    """Параметры toy-космологии."""

    omega_r0: float = 9.0e-5
    omega_b0: float = 0.049
    omega_dm0: float = 0.264
    omega_l0: float = 0.68791  # выбрано так, чтобы сумма была ~1

    @property
    def omega_m0(self) -> float:
        return self.omega_b0 + self.omega_dm0

    @property
    def omega_tot0(self) -> float:
        return self.omega_r0 + self.omega_m0 + self.omega_l0


def e_of_a(a: np.ndarray, p: CosmoParams) -> np.ndarray:
    """Безразмерный Хаббл-фактор E(a) = H(a)/H0."""
    a = np.asarray(a, dtype=float)
    return np.sqrt(
        p.omega_r0 / a**4 + p.omega_m0 / a**3 + p.omega_l0
    )


def rhs(a: float, p: CosmoParams) -> float:
    """da/dτ = a * E(a)."""
    return float(a * e_of_a(np.array([a]), p)[0])


def integrate_scale_factor(
    p: CosmoParams,
    a0: float = 1.0e-4,
    tau_end: float = 18.0,
    n_steps: int = 6000,
) -> tuple[np.ndarray, np.ndarray]:
    """Интегрирует a(τ) методом RK4."""
    tau = np.linspace(0.0, tau_end, n_steps + 1)
    dt = tau[1] - tau[0]
    a = np.empty_like(tau)
    a[0] = a0

    for i in range(n_steps):
        ai = a[i]
        k1 = rhs(ai, p)
        k2 = rhs(ai + 0.5 * dt * k1, p)
        k3 = rhs(ai + 0.5 * dt * k2, p)
        k4 = rhs(ai + dt * k3, p)
        a[i + 1] = ai + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Чуть-чуть страхуемся от численной деградации.
        if a[i + 1] <= 0.0 or not np.isfinite(a[i + 1]):
            a[i + 1] = a[i]

    return tau, a


def fractional_densities(a: np.ndarray, p: CosmoParams) -> dict[str, np.ndarray]:
    """Доли компонент Ω_i(a) / ΣΩ_j(a)."""
    a = np.asarray(a, dtype=float)
    Er2 = p.omega_r0 / a**4
    Eb2 = p.omega_b0 / a**3
    Edm2 = p.omega_dm0 / a**3
    El2 = np.full_like(a, p.omega_l0)
    total = Er2 + Eb2 + Edm2 + El2
    return {
        "radiation": Er2 / total,
        "baryons": Eb2 / total,
        "dark_matter": Edm2 / total,
        "dark_energy": El2 / total,
    }


def deceleration_parameter(a: np.ndarray, p: CosmoParams) -> np.ndarray:
    """q(a) = 1/2 Ω_m(a) + Ω_r(a) - Ω_Λ(a) для плоской toy-модели."""
    fracs = fractional_densities(a, p)
    omega_m = fracs["baryons"] + fracs["dark_matter"]
    omega_r = fracs["radiation"]
    omega_l = fracs["dark_energy"]
    return 0.5 * omega_m + omega_r - omega_l


def make_figure(
    tau: np.ndarray,
    a: np.ndarray,
    p: CosmoParams,
    out_path: Path,
) -> None:
    """Строит и сохраняет визуализацию."""
    fracs = fractional_densities(a, p)
    E = e_of_a(a, p)
    q = deceleration_parameter(a, p)

    # Берём логарифмическую ось по a для компонент.
    idx = np.argsort(a)
    a_sorted = a[idx]
    E_sorted = E[idx]
    q_sorted = q[idx]
    fr = {k: v[idx] for k, v in fracs.items()}

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(tau, a)
    ax1.set_yscale("log")
    ax1.set_xlabel("безразмерное время τ")
    ax1.set_ylabel("масштабный фактор a(τ)")
    ax1.set_title("Рост масштаба Вселенной")
    ax1.grid(True, which="both", alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(a_sorted, fr["radiation"], label="радиация")
    ax2.plot(a_sorted, fr["baryons"], label="барионы")
    ax2.plot(a_sorted, fr["dark_matter"], label="тёмная материя")
    ax2.plot(a_sorted, fr["dark_energy"], label="тёмная энергия")
    ax2.set_xscale("log")
    ax2.set_xlabel("масштабный фактор a")
    ax2.set_ylabel("доля компоненты")
    ax2.set_ylim(0.0, 1.02)
    ax2.set_title("Смена доминирующей компоненты")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(frameon=False, fontsize=9)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(a_sorted, E_sorted, label="H(a)/H0")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("масштабный фактор a")
    ax3.set_ylabel("H(a)/H0")
    ax3.grid(True, which="both", alpha=0.25)

    ax3b = ax3.twinx()
    ax3b.plot(a_sorted, q_sorted, label="q(a)", linestyle="--")
    ax3b.set_ylabel("q(a)")

    # Общая легенда для двух осей.
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="best")
    ax3.set_title("Разгон расширения: Hubble-фактор и q(a)")

    total0 = p.omega_tot0
    title = (
        "Toy-модель Вселенной: FRW + радиация + материя + тёмная энергия\n"
        f"Ωr0={p.omega_r0:.2e}, Ωb0={p.omega_b0:.3f}, Ωdm0={p.omega_dm0:.3f}, "
        f"ΩΛ0={p.omega_l0:.5f}, сумма={total0:.5f}"
    )
    fig.suptitle(title, fontsize=12.5, y=0.975)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple toy universe model with plots.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("toy_universe.png"),
        help="куда сохранить картинку",
    )
    parser.add_argument("--a0", type=float, default=1.0e-4, help="начальный масштаб")
    parser.add_argument("--tau-end", type=float, default=18.0, help="конец времени")
    parser.add_argument("--steps", type=int, default=6000, help="число шагов интегрирования")
    args = parser.parse_args()

    p = CosmoParams()
    tau, a = integrate_scale_factor(
        p=p,
        a0=args.a0,
        tau_end=args.tau_end,
        n_steps=args.steps,
    )
    make_figure(tau, a, p, args.out)

    print(f"Saved figure to: {args.out.resolve()}")
    print(f"Final scale factor: a(tau_end) = {a[-1]:.6g}")
    print(f"Model parameters sum to: {p.omega_tot0:.6f}")


if __name__ == "__main__":
    main()
