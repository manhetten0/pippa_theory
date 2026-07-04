"""Константы связи и массы частиц из первых принципов (теория Pippa).

Базовые (точные) формулы из раздела 11.3.1:
- постоянная тонкой структуры alpha_EM
- сильная константа alpha_s
- угол Вайнберга sin^2(theta_W)
- отношение масс m_W/m_Z
- бозон Хиггса (lambda_H, m_H)

Феноменологические степенные формулы (массы лептонов с параметром
beta, нейтрино) помечены теорией как НЕТОЧНЫЕ и вынесены в класс
Phenomenology, чтобы не смешивать их с базовым ядром.
"""

from __future__ import annotations

import math

from . import constants


# --- Базовые (точные) формулы --------------------------------------------

def alpha_A_theoretical() -> float:
    return constants.alpha_A_from_D()

def alpha_B_theoretical() -> float:
    return constants.alpha_B_from_D()


def alpha_EM() -> float:
    """Постоянная тонкой структуры: [2 ln2 - ln3] / (4 pi^2)."""
    return (2.0 * math.log(2.0) - math.log(3.0)) / (4.0 * math.pi**2)


def alpha_s() -> float:
    """Сильная константа: 2 * dim(SU(3)) * alpha_EM = 16 * alpha_EM."""
    dim_su3 = 8
    return 2.0 * dim_su3 * alpha_EM()


def sin2_theta_W() -> float:
    """Угол Вайнберга: 1 / (N_q + D/N_q) = 1 / (4 + 1/pi)."""
    n_q = constants.N_QUADRANTS
    return 1.0 / (n_q + constants.D / n_q)


def cos_theta_W() -> float:
    """cos(theta_W) = sqrt(1 - sin^2 theta_W) = m_W/m_Z."""
    return math.sqrt(1.0 - sin2_theta_W())


def m_W_over_m_Z() -> float:
    """Отношение масс W/Z из первых принципов = cos(theta_W)."""
    return cos_theta_W()


def lambda_higgs() -> float:
    """Квартичная самосвязь Хиггса: (D/2)^3 / 2 = 4/pi^3."""
    return (constants.D / 2.0) ** 3 / 2.0


def m_higgs(vev_GeV: float = constants.EXP.higgs_vev_GeV) -> float:
    """Масса Хиггса: m_H = v * sqrt(2 lambda_H) = v * (2/pi)^(3/2)."""
    return vev_GeV * math.sqrt(2.0 * lambda_higgs())


def m_W(m_Z_GeV: float = constants.EXP.m_Z_GeV) -> float:
    """Масса W бозона из m_Z и cos(theta_W)."""
    return m_Z_GeV * cos_theta_W()


# --- Феноменологические (НЕТОЧНЫЕ) степенные формулы ----------------------


class Phenomenology:
    """Степенные/экспоненциальные формулы, помеченные теорией как
    неточные феноменологические приближения (раздел 11.3.1).

    Не входят в проверяемое ядро. Используют подгоночный параметр beta.
    """

    BETA: float = 0.11  # подгоночный параметр информационной модуляции

    @classmethod
    def H_eff(cls, n: int, alpha: float | None = None) -> float:
        """H_eff,n = [ln(1/alpha) / (n sqrt(D))] * [ln(1+n)]^beta."""
        a = alpha if alpha is not None else alpha_EM()
        return (math.log(1.0 / a) / (n * math.sqrt(constants.D))) * (
            math.log(1.0 + n) ** cls.BETA
        )

    @classmethod
    def lepton_mass_MeV(cls, n: int, m_e_MeV: float = constants.EXP.m_e_MeV) -> float:
        """Масса лептона поколения n (накопительная H_eff). НЕТОЧНО."""
        h_cumulative = sum(cls.H_eff(k) for k in range(1, n + 1))
        return m_e_MeV * math.exp(constants.D * h_cumulative)


def bare_g_and_gprime(v: float = constants.EXP.higgs_vev_GeV,
                      m_Z: float = constants.EXP.m_Z_GeV,
                      sin2_theta_bare: float | None = None) -> tuple[float, float]:
    """Голые константы связи из геометрии Pippa (без α_A,α_B)."""
    if sin2_theta_bare is None:
        sin2_theta_bare = sin2_theta_W()  # из геометрии 1/(4+1/π)
    cos_theta_bare = math.sqrt(1.0 - sin2_theta_bare)
    m_W_bare = m_Z * cos_theta_bare
    g_bare = 2.0 * m_W_bare / v
    g_prime_bare = g_bare * math.tan(math.asin(math.sqrt(sin2_theta_bare)))
    return g_bare, g_prime_bare

def effective_g_and_gprime(
    alpha_A: float | None = None,
    alpha_B: float | None = None,
    v: float = constants.EXP.higgs_vev_GeV,
    m_Z: float = constants.EXP.m_Z_GeV
) -> tuple[float, float]:
    if alpha_A is None:
        alpha_A = alpha_A_theoretical()
    if alpha_B is None:
        alpha_B = alpha_B_theoretical()

    """Эффективные константы связи после учёта межквадрантной связи."""
    g_bare, g_prime_bare = bare_g_and_gprime(v, m_Z)
    # Коэффициенты из constants
    a = constants.A_COEFF_SU2
    b = constants.B_COEFF_SU2
    a_prime = constants.A_COEFF_U1
    b_prime = constants.B_COEFF_U1
    denom_SU2 = 1.0 + alpha_A * a + alpha_B * b
    denom_U1 = 1.0 + alpha_A * a_prime + alpha_B * b_prime
    g_eff = g_bare / math.sqrt(denom_SU2) if denom_SU2 > 0 else float('nan')
    g_prime_eff = g_prime_bare / math.sqrt(denom_U1) if denom_U1 > 0 else float('nan')
    return g_eff, g_prime_eff

def sin2_theta_eff(alpha_A: float = constants.alpha_A,
                   alpha_B: float = constants.alpha_B,
                   v: float = constants.EXP.higgs_vev_GeV,
                   m_Z: float = constants.EXP.m_Z_GeV) -> float:
    """Эффективный угол смешивания (on-shell) с учётом поправок."""
    g_eff, g_prime_eff = effective_g_and_gprime(alpha_A, alpha_B, v, m_Z)
    return g_prime_eff**2 / (g_eff**2 + g_prime_eff**2)

def m_W_eff(alpha_A: float = constants.alpha_A,
            alpha_B: float = constants.alpha_B,
            v: float = constants.EXP.higgs_vev_GeV) -> float:
    """Эффективная масса W-бозона (древесная) с учётом поправок."""
    g_eff, _ = effective_g_and_gprime(alpha_A, alpha_B, v)
    return g_eff * v / 2.0
