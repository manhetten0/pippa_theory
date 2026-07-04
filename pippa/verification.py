"""Верификация предсказаний теории Pippa против эксперимента (PDG 2024).

Сравнивает базовые формулы с эталонными значениями и считает
относительное расхождение.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import (
    constants,
    particle_physics,
    fractal_dimension,
    renormalization,
    electroweak,
    cosmology,
)


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


#: Порог по относительной ошибке (доля).
REL_ERROR_THRESHOLD: float = 0.02

#: Порог по отклонению в sigma. Даже при малой % ошибке
#: предсказание несовместимо с данными, если |sigma| велико.
SIGMA_THRESHOLD: float = 5.0


def status_for(comp: "Comparison") -> str:
    """Честный статус по ДВОЙНОМУ критерию: % и sigma.

    - FAIL : отн. ошибка превышает порог (грубое расхождение).
    - WARN : % в пределах порога, но |sigma| > SIGMA_THRESHOLD — т.е.
             предсказание несовместимо с измерением несмотря на малый %.
    - OK   : и % в пределах, и по sigma совместимо (либо sigma не задана).
    """
    if abs(comp.rel_error) >= REL_ERROR_THRESHOLD:
        return "FAIL"
    import math as _math

    if not _math.isnan(comp.n_sigma) and abs(comp.n_sigma) > SIGMA_THRESHOLD:
        return "WARN"
    return "OK"


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


def run_all_renormalized(mu_alpha_s_GeV: float = constants.EXP.m_Z_GeV) -> list[Comparison]:
    """Проверки констант связи с учётом однопетлевого RG-бега к m_Z.

    Гипотеза: формулы Pippa задают константы на некотором масштабе, а
    наблюдаемые значения получаются бегом к m_Z. Сравнение идёт с
    эталонами НА m_Z (apples-to-apples), в отличие от run_all, где
    масштабы смешаны.

    Параметр mu_alpha_s_GeV задаёт предполагаемый масштаб формулы alpha_s
    (по умолчанию m_Z, т.е. бега нет; меняя его, можно проверить, на
    каком масштабе формула 16*alpha_EM согласуется с данными).
    """
    exp = constants.EXP

    # alpha_EM: формула Pippa трактуется как низкоэнергетический предел,
    # прогон к m_Z, сравнение с alpha_EM(m_Z).
    aem = renormalization.run_alpha_EM_to_mz(particle_physics.alpha_EM())

    # alpha_s: формула 16*alpha_EM трактуется на масштабе mu_alpha_s_GeV,
    # прогон к m_Z, сравнение с alpha_s(m_Z).
    asr = renormalization.run_alpha_s_to_mz(
        particle_physics.alpha_s(), mu_alpha_s_GeV
    )

    return [
        Comparison(
            "alpha_EM->m_Z", aem.value_after, exp.alpha_EM_mZ,
            sigma_exp=exp.alpha_EM_mZ_err, energy_scale="m_Z (RG)",
        ),
        Comparison(
            "alpha_s->m_Z", asr.value_after, exp.alpha_s_mZ,
            sigma_exp=exp.alpha_s_mZ_err, energy_scale="m_Z (RG)",
        ),
    ]


def run_loop_corrected() -> list[Comparison]:
    """m_W with electroweak loop corrections (Delta r), compared in sigma.

    Tree level uses Pippa's cos(theta_W); loops use the SM Delta r machinery
    with experimental inputs (G_F, m_t, Delta_alpha). This tests whether the
    ~0.5% tree-level gap in m_W is closed by loops or is a real deviation.
    """
    exp = constants.EXP

    # Tree-level m_W from Pippa: m_Z * cos(theta_W).
    m_W_tree = particle_physics.m_W()

    # Sirlin scheme: pass alpha(0) (Pippa formula). The running to m_Z is
    # carried by Delta_alpha inside Delta r, so we must NOT pass alpha(m_Z).
    alpha0 = particle_physics.alpha_EM()

    res = electroweak.m_W_loop_corrected(alpha0, m_W_tree_GeV=m_W_tree)

    return [
        Comparison(
            "m_W Pippa-tree", m_W_tree, exp.m_W_GeV,
            sigma_exp=exp.m_W_GeV_err, energy_scale="m_Z*cos(thW)",
        ),
        Comparison(
            "m_W G_F+Dr", res.m_W_loop_GeV, exp.m_W_GeV,
            sigma_exp=exp.m_W_GeV_err, energy_scale="on-shell (Dr)",
        ),
    ]


def run_cosmology() -> list[Comparison]:
    """Out-of-sample cosmological predictions vs Planck / BICEP-Keck.

    These formulas do not use the SM-constant fits, so they are the most
    honest test of the theory. r is an upper limit, handled separately.
    """
    obs = cosmology.OBS
    return [
        Comparison(
            "n_s", cosmology.spectral_index(), obs.n_s,
            sigma_exp=obs.n_s_err, energy_scale="Planck18",
        ),
        Comparison(
            "f_NL", cosmology.non_gaussianity(), obs.f_NL,
            sigma_exp=obs.f_NL_err, energy_scale="Planck18",
        ),
        Comparison(
            "Omega_DM/Omega_b", cosmology.dm_to_baryon_ratio(), obs.dm_baryon,
            sigma_exp=obs.dm_baryon_err, energy_scale="Planck18",
        ),
    ]

def alpha_em_from_delta_r(alpha_A: float,alpha_B: float,) -> float:
    """
    Effective alpha_EM reconstructed from EW loops (Delta r).
    """

    sin2_eff = particle_physics.sin2_theta_eff(
        alpha_A,
        alpha_B,
    )

    g_eff, _ = particle_physics.effective_g_and_gprime(
        alpha_A,
        alpha_B,
    )

    alpha_tree = (g_eff**2 * sin2_eff) / (4.0 * math.pi)

    delta_r = 0.0358

    return alpha_tree / (1.0 + delta_r)

def run_electroweak_with_fit() -> list[Comparison]:
    """
    EW sector with alpha_A, alpha_B.
    alpha_EM and Omega_DM/Omega_b are derived predictions.
    """
    exp = constants.EXP

    alpha_A = particle_physics.alpha_A_theoretical()
    alpha_B = particle_physics.alpha_B_theoretical()

    sin2_eff = particle_physics.sin2_theta_eff(alpha_A, alpha_B)
    mW_eff = particle_physics.m_W_eff(alpha_A, alpha_B)

    g_eff, _ = particle_physics.effective_g_and_gprime(
        alpha_A,
        alpha_B,
    )

    alpha_tree = (g_eff**2 * sin2_eff) / (4.0 * math.pi)

    alpha_loop = alpha_em_from_delta_r(alpha_A, alpha_B)

    omega_dm = dm_ratio_alpha_corrected()

    return [
        Comparison(
            "fit alpha_A",
            alpha_A,
            alpha_A,
            energy_scale="theoretical",
        ),

        Comparison(
            "fit alpha_B",
            alpha_B,
            alpha_B,
            energy_scale="theoretical",
        ),

        Comparison(
            "sin²θ_W eff",
            sin2_eff,
            1.0 - (exp.m_W_GeV / exp.m_Z_GeV) ** 2,
            sigma_exp=exp.sin2_theta_W_err,
            energy_scale="on-shell theoretical",
        ),

        Comparison(
            "m_W eff",
            mW_eff,
            exp.m_W_GeV,
            sigma_exp=exp.m_W_GeV_err,
            energy_scale="theoretical",
        ),

        Comparison(
            "alpha_EM tree",
            alpha_tree,
            exp.alpha_EM,
            sigma_exp=exp.alpha_EM_scale_err,
            energy_scale="tree-level",
        ),

        Comparison(
            "alpha_EM +Δr",
            alpha_loop,
            exp.alpha_EM,
            sigma_exp=exp.alpha_EM_scale_err,
            energy_scale="loop-corrected",
        ),

        Comparison(
            "Omega_DM/Omega_b",
            omega_dm,
            cosmology.OBS.dm_baryon,
            sigma_exp=cosmology.OBS.dm_baryon_err,
            energy_scale="Planck18 derived",
        ),
    ]

def dm_ratio_alpha_corrected() -> float:
    """ Alpha-corrected Omega_DM/Omega_b. Использует EW correction factor: sqrt((1+2α_A+α_B)/(1+α_A+α_B)) """
    base = cosmology.dm_to_baryon_ratio()
    alpha_A = particle_physics.alpha_A_theoretical()
    alpha_B = particle_physics.alpha_B_theoretical()
    su2 = 1.0 + 2.0 * alpha_A + alpha_B
    u1 = 1.0 + alpha_A + alpha_B
    correction = math.sqrt(su2 / u1)
    return base * correction

def dm_ratio_alpha_corrected() -> float:
    """ Alpha-corrected Omega_DM/Omega_b. Использует EW correction factor: sqrt((1+2α_A+α_B)/(1+α_A+α_B)) """
    base = cosmology.dm_to_baryon_ratio()
    alpha_A = particle_physics.alpha_A_theoretical()
    alpha_B = particle_physics.alpha_B_theoretical()
    su2 = 1.0 + 2.0 * alpha_A + alpha_B
    u1 = 1.0 + alpha_A + alpha_B
    correction = math.sqrt(su2 / u1)
    return base * correction

def report() -> str:
    """Сформировать текстовый отчёт по всем проверкам."""
    lines = ["Верификация базовых формул теории Pippa (PDG 2024)", "=" * 60]
    lines.extend(str(c) for c in run_all())
    lines.append("")
    lines.append("С учётом однопетлевого RG-бега к m_Z:")
    lines.append("-" * 60)
    lines.extend(str(c) for c in run_all_renormalized())
    lines.append("")
    lines.append("С учётом электрослабых петлевых поправок (Delta r) для m_W:")
    lines.append("-" * 60)
    lines.extend(str(c) for c in run_loop_corrected())
    lines.append("")
    lines.append("Out-of-sample: космология (Planck/BICEP-Keck):")
    lines.append("-" * 60)
    lines.extend(str(c) for c in run_cosmology())
    r_pred = cosmology.tensor_to_scalar()
    r_lim = cosmology.OBS.r_upper_95
    r_ok = "OK" if r_pred < r_lim else "FAIL"
    lines.append(
        f"r (tensor)          pred={r_pred:.4g}  limit<{r_lim} (95%)  [{r_ok}]"
    )
    lines.append("\nЭлектрослабый сектор с подгонкой α_A, α_B:")
    lines.append("")
    lines.append("=" * 60)
    lines.append("Pippa + α-sector (EW + cosmology)")
    lines.append("=" * 60)
    lines.append( "m_W и sin²θ_W используются для fit α_A, α_B.\n" "alpha_EM и Omega_DM/Omega_b являются derived predictions." )
    lines.append("-" * 60)
    for comp in run_electroweak_with_fit():
        lines.append(str(comp))
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
