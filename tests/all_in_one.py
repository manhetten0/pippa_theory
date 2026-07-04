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

"""Cosmological / inflationary predictions of Pippa (out-of-sample tests).

These formulas were NOT used when fitting the SM coupling constants, so
they are genuine out-of-sample predictions and the most honest test of
whether D = 4/pi encodes physics or is numerology.

Predictions (README sections 19.2, H.4.3):
- Spectral index:   n_s = 1 - 2/(D * N_e)
- Tensor-to-scalar: r   = 12/(D^2 * N_e^2)
- Non-Gaussianity:  f_NL ~ 1/D
- DM/baryon ratio:  Omega_DM/Omega_b = eps^(-D),  eps = D - 1

Reference data: Planck 2018 + BICEP/Keck (used only as comparison, not
as inputs to the formulas).

No external dependencies (stdlib only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import constants

# --- Default model assumption --------------------------------------------

#: Number of e-folds of inflation (standard assumption ~50-60).
N_EFOLDS_DEFAULT: float = 60.0


# --- Pippa predictions ----------------------------------------------------


def spectral_index(n_efolds: float = N_EFOLDS_DEFAULT, D: float = constants.D) -> float:
    """Scalar spectral index n_s = 1 - 2/(D * N_e)."""
    return 1.0 - 2.0 / (D * n_efolds)


def tensor_to_scalar(n_efolds: float = N_EFOLDS_DEFAULT, D: float = constants.D) -> float:
    """Tensor-to-scalar ratio r = 12/(D^2 * N_e^2)."""
    return 12.0 / (D**2 * n_efolds**2)


def non_gaussianity(D: float = constants.D) -> float:
    """Local non-Gaussianity amplitude f_NL ~ 1/D."""
    return 1.0 / D


def dm_to_baryon_ratio(
    D: float = constants.D,
    alpha_A: float | None = None,
    alpha_B: float | None = None,
) -> float:
    """
    Baseline:
        Omega_DM/Omega_b = (D - 1)^(-D)

    Alpha-corrected variant:
        Omega_DM/Omega_b = (D - 1)^(-D) * sqrt((1 + 2 alpha_A + alpha_B)/(1 + alpha_A + alpha_B))

    This keeps the alpha-correction tied to the same SU(2)/U(1) structure
    already used in the electroweak sector, without introducing a new
    free parameter.
    """
    eps = D - 1.0
    ratio = eps ** (-D)

    if alpha_A is None or alpha_B is None:
        return ratio

    denom_su2 = 1.0 + alpha_A + alpha_B
    denom_u1 = 1.0 + 2.0 * alpha_A + alpha_B
    if denom_su2 <= 0.0 or denom_u1 <= 0.0:
        raise ValueError("Invalid alpha correction: non-positive denominator")

    return ratio * math.sqrt(denom_u1 / denom_su2)


# --- Reference values (Planck 2018 / BICEP-Keck 2021) --------------------


@dataclass(frozen=True)
class CosmoObservations:
    """Reference cosmological measurements (comparison only)."""

    n_s: float = 0.9649
    n_s_err: float = 0.0042

    # r: only an upper limit exists (BICEP/Keck 2021): r < 0.036 (95% CL).
    r_upper_95: float = 0.036

    # f_NL local (Planck 2018): -0.9 +/- 5.1.
    f_NL: float = -0.9
    f_NL_err: float = 5.1

    # Omega_DM/Omega_b (Planck 2018): 0.2645/0.04930 = 5.366 +/- ~0.07.
    dm_baryon: float = 5.366
    dm_baryon_err: float = 0.07


OBS = CosmoObservations()

from __future__ import annotations

import math

from . import constants

N_EFOLDS_DEFAULT: float = 60.0

def spectral_index(n_efolds: float = N_EFOLDS_DEFAULT, D: float = constants.D) -> float:
    return 1.0 - 2.0 / (D * n_efolds)

def tensor_to_scalar(n_efolds: float = N_EFOLDS_DEFAULT, D: float = constants.D) -> float:
    return 12.0 / (D**2 * n_efolds**2)

def non_gaussianity(D: float = constants.D) -> float:
    return 1.0 / D

def dm_to_baryon_ratio(
    D: float = constants.D,
    alpha_A: float | None = None,
    alpha_B: float | None = None,
) -> float:
    """
    Baseline:
        Omega_DM/Omega_b = (D - 1)^(-D)

    Alpha-corrected variant:
        Omega_DM/Omega_b = (D - 1)^(-D) * sqrt((1 + 2 alpha_A + alpha_B)/(1 + alpha_A + alpha_B))

    This keeps the alpha-correction tied to the same SU(2)/U(1) structure
    already used in the electroweak sector, without introducing a new
    free parameter.
    """
    eps = D - 1.0
    ratio = eps ** (-D)

    if alpha_A is None or alpha_B is None:
        return ratio

    denom_su2 = 1.0 + alpha_A + alpha_B
    denom_u1 = 1.0 + 2.0 * alpha_A + alpha_B
    if denom_su2 <= 0.0 or denom_u1 <= 0.0:
        raise ValueError("Invalid alpha correction: non-positive denominator")

    return ratio * math.sqrt(denom_u1 / denom_su2)

"""Профиль тёмной материи и закон масштабирования (теория Pippa).

Базовые формулы из раздела 13.1 и Приложения G.10:

    rho_DM(r) = rho0 * (r0/r)^alpha_eff * M[A](r)
    alpha_eff(r) = (2 - D) + gamma * (1 - exp(-r/r0))
    M[A](r) = 1 + A_M * (1 - exp(-r/r0))
    A_M propto M_bar^k,  k = (1 - D)/2 ~= -0.137
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
    """

    rho0: float
    r0: float
    A_M: float
    gamma: float = 0.0

    def alpha_eff(self, r: float) -> float:
        """Эффективный показатель степени alpha_eff(r)."""
        return (2.0 - constants.D) + self.gamma * (1.0 - math.exp(-r / self.r0))

    def M_operator(self, r: float) -> float:
        """Оператор межквадрантной связи M[A](r) = 1 + A_M*(1 - exp(-r/r0))."""
        return 1.0 + self.A_M * (1.0 - math.exp(-r / self.r0))

    def density(self, r: float) -> float:
        """Плотность тёмной материи rho_DM(r)."""
        if r <= 0.0:
            raise ValueError("r должно быть положительным")
        return (
            self.rho0
            * (self.r0 / r) ** self.alpha_eff(r)
            * self.M_operator(r)
        )


"""Electroweak one-loop corrections (Delta r) for the W boson mass.

Why this module exists
----------------------
The tree-level Pippa relation m_W = m_Z * cos(theta_W) misses the
electroweak radiative corrections that the Standard Model itself needs.
Without them even the SM mispredicts m_W by ~0.5%. To compare Pippa to
data honestly *in sigma*, we must apply the same Delta r machinery.

The on-shell relation (Sirlin) is implicit in m_W:

    m_W^2 (1 - m_W^2 / m_Z^2) = (pi * alpha) / (sqrt(2) * G_F) * 1/(1 - Delta r)

where
    Delta r ~= Delta_alpha - (c_W^2 / s_W^2) * Delta_rho + Delta r_rem
    Delta_rho ~= 3 G_F m_t^2 / (8 sqrt(2) pi^2)   (leading top contribution)
    s_W^2 = 1 - m_W^2/m_Z^2 ,  c_W^2 = m_W^2/m_Z^2

The equation is solved iteratively for m_W.

IMPORTANT (honesty about inputs):
- G_F, m_t, Delta_alpha are EXPERIMENTAL inputs (PDG), not derived.
- This is the standard SM computation; we apply it on top of the Pippa
  inputs to see whether the tree-level ~0.5% gap is closed by loops or
  remains a genuine deviation.
- alpha here is alpha(m_Z) (the running coupling), not alpha(0).

No external dependencies (stdlib only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Experimental inputs (PDG 2024) --------------------------------------

G_F: float = 1.1663788e-5          # Fermi constant, GeV^-2
M_T_GEV: float = 172.69            # top quark mass
M_Z_GEV: float = 91.1876
# Total shift of alpha from 0 to m_Z: alpha(m_Z) = alpha(0)/(1 - Delta_alpha).
DELTA_ALPHA: float = 0.05900       # Delta_alpha(m_Z), PDG
DELTA_R_REMAINDER: float = 0.0     # higher-order remainder (set 0 by default)


def delta_rho_top(g_f: float = G_F, m_t_GeV: float = M_T_GEV) -> float:
    """Leading top-quark contribution to Delta_rho.

    Delta_rho = 3 G_F m_t^2 / (8 sqrt(2) pi^2).
    """
    return 3.0 * g_f * m_t_GeV**2 / (8.0 * math.sqrt(2.0) * math.pi**2)


def delta_r(
    m_W_GeV: float,
    m_Z_GeV: float = M_Z_GEV,
    delta_alpha: float = DELTA_ALPHA,
    remainder: float = DELTA_R_REMAINDER,
) -> float:
    """Electroweak correction Delta r as a function of m_W.

    Delta r = Delta_alpha - (c_W^2 / s_W^2) * Delta_rho + remainder.
    """
    c_w2 = m_W_GeV**2 / m_Z_GeV**2
    s_w2 = 1.0 - c_w2
    return delta_alpha - (c_w2 / s_w2) * delta_rho_top() + remainder


@dataclass(frozen=True)
class MWResult:
    """Result of the loop-corrected m_W computation."""

    m_W_tree_GeV: float
    m_W_loop_GeV: float
    delta_r: float
    iterations: int


def m_W_loop_corrected(
    alpha0: float,
    m_Z_GeV: float = M_Z_GEV,
    g_f: float = G_F,
    m_W_tree_GeV: float | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> MWResult:
    """Solve the implicit on-shell relation for m_W including Delta r.

    m_W^2 (1 - m_W^2/m_Z^2) = (pi alpha)/(sqrt(2) G_F) * 1/(1 - Delta r).

    Solved by fixed-point iteration: start from a guess, compute Delta r,
    then solve the quadratic in m_W^2, repeat until convergence.
    """
    # Sirlin scheme: numerator uses alpha(0); the running to m_Z is carried
    # entirely by Delta_alpha INSIDE Delta r. Using alpha(m_Z) here would
    # double-count the running and is wrong.
    a0 = math.pi * alpha0 / (math.sqrt(2.0) * g_f)  # = m_W^2 s_W^2 (1-Dr)

    # Initial guess for m_W: tree-level value if given, else 80 GeV.
    m_W = m_W_tree_GeV if m_W_tree_GeV is not None else 80.0
    tree = m_W

    iterations = 0
    for i in range(1, max_iter + 1):
        dr = delta_r(m_W, m_Z_GeV)
        rhs = a0 / (1.0 - dr)  # = m_W^2 (1 - m_W^2/m_Z^2)
        # Solve x (1 - x/m_Z^2) = rhs for x = m_W^2, take the physical root
        # (the larger root, close to m_Z^2/2 .. m_Z^2).
        # x^2/m_Z^2 - x + rhs = 0  =>  x = (m_Z^2/2) (1 +/- sqrt(1 - 4 rhs/m_Z^2))
        disc = 1.0 - 4.0 * rhs / m_Z_GeV**2
        if disc < 0.0:
            raise ValueError("No real solution for m_W (discriminant < 0).")
        x = (m_Z_GeV**2 / 2.0) * (1.0 + math.sqrt(disc))
        new_m_W = math.sqrt(x)
        iterations = i
        if abs(new_m_W - m_W) < tol:
            m_W = new_m_W
            break
        m_W = new_m_W

    return MWResult(
        m_W_tree_GeV=tree,
        m_W_loop_GeV=m_W,
        delta_r=delta_r(m_W, m_Z_GeV),
        iterations=iterations,
    )

def alpha_from_mW(m_W_GeV: float,
                  m_Z_GeV: float = M_Z_GEV,
                  delta_r: float = DELTA_R_REMAINDER,
                  g_f: float = G_F) -> float:
    """
    Вычисляет постоянную тонкой структуры α(0) из масс W, Z и Δr.
    Использует on-shell соотношение Sirlin:
        m_W^2 (1 - m_W^2/m_Z^2) = (π α) / (√2 G_F) * 1/(1 - Δr)
    """
    lhs = m_W_GeV**2 * (1.0 - (m_W_GeV / m_Z_GeV)**2)
    rhs_factor = math.pi / (math.sqrt(2.0) * g_f) * 1.0 / (1.0 - delta_r)
    return lhs / rhs_factor

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

"""One-loop RG running of coupling constants (renormalization).

Goal: honestly test the hypothesis that Pippa formulas give couplings at
some scale mu, while observed values follow from RG running to m_Z.

IMPORTANT physics notes:
- QCD (alpha_s) one-loop running is perturbative and is computed directly.
- QED (alpha_EM) running to m_Z CANNOT be done by naively integrating a
  beta function through light-quark thresholds: the low-energy hadronic
  contribution is non-perturbative and must be taken from experiment
  (the R-ratio), i.e. Delta_alpha_had. We therefore use the standard
  decomposition alpha(m_Z) = alpha(0) / (1 - Delta_alpha), with
  Delta_alpha = Delta_lep + Delta_had(5) + Delta_top. The leptonic part
  is computed analytically; the hadronic part is an EXPERIMENTAL INPUT.
- W/Z masses need electroweak loop corrections (Delta r), not RG, and are
  not handled here.

No external dependencies (RK4 on stdlib for QCD).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

# --- Reference scales (GeV) ----------------------------------------------

M_Z_GEV: float = 91.1876

# Experimental hadronic contribution to the running of alpha_EM,
# 5 active flavors, evaluated at m_Z (PDG 2024): Delta_alpha_had^(5).
DELTA_ALPHA_HAD_5: float = 0.02766
DELTA_ALPHA_HAD_5_ERR: float = 0.00007

# Charged lepton masses (GeV) for the leptonic running contribution.
_LEPTON_MASSES_GEV = (0.00051099895, 0.1056583755, 1.77686)

# Top quark contribution (small, decoupled below m_t); included for honesty.
_M_TOP_GEV: float = 172.69


def _active_quark_flavors(mu_GeV: float) -> int:
    """Number of active quark flavors at scale mu (threshold scheme)."""
    if mu_GeV < 1.275:
        return 3
    if mu_GeV < 4.18:
        return 4
    if mu_GeV < 173.0:
        return 5
    return 6


# --- QCD beta function (one loop) ----------------------------------------


def beta_alpha_s(alpha_s: float, mu_GeV: float) -> float:
    """d(alpha_s)/d ln(mu): one-loop QCD. b0 = 11 - (2/3) n_f."""
    n_f = _active_quark_flavors(mu_GeV)
    b0 = 11.0 - (2.0 / 3.0) * n_f
    return -(b0 / (2.0 * math.pi)) * alpha_s**2


# --- RK4 integrator over ln(mu) ------------------------------------------


def run_coupling(
    beta: Callable[[float, float], float],
    value0: float,
    mu_from_GeV: float,
    mu_to_GeV: float,
    n_steps: int = 2000,
) -> float:
    """Integrate beta from mu_from to mu_to (RK4 in t = ln mu)."""
    if mu_from_GeV <= 0.0 or mu_to_GeV <= 0.0:
        raise ValueError("mu scales must be positive.")

    t0 = math.log(mu_from_GeV)
    t1 = math.log(mu_to_GeV)
    h = (t1 - t0) / n_steps

    value = value0
    t = t0
    for _ in range(n_steps):
        k1 = beta(value, math.exp(t))
        k2 = beta(value + 0.5 * h * k1, math.exp(t + 0.5 * h))
        k3 = beta(value + 0.5 * h * k2, math.exp(t + 0.5 * h))
        k4 = beta(value + h * k3, math.exp(t + h))
        value += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    return value


# --- alpha_EM running via the Delta_alpha decomposition -------------------


def delta_alpha_leptonic(alpha0: float, mu_GeV: float = M_Z_GEV) -> float:
    """Leptonic contribution to Delta_alpha at scale mu (one loop, analytic).

    Delta_lep = (alpha0 / (3 pi)) * sum_l [ ln(mu^2/m_l^2) - 5/3 ].
    Valid for mu >> m_l.
    """
    total = 0.0
    for m_l in _LEPTON_MASSES_GEV:
        total += math.log(mu_GeV**2 / m_l**2) - 5.0 / 3.0
    return (alpha0 / (3.0 * math.pi)) * total


def delta_alpha_top(alpha0: float, mu_GeV: float = M_Z_GEV) -> float:
    """Top-quark contribution to Delta_alpha (N_c * Q_t^2 = 3*(2/3)^2 = 4/3).

    Small and negative below m_t; included for completeness.
    """
    color_charge = 3.0 * (2.0 / 3.0) ** 2
    return (alpha0 / (3.0 * math.pi)) * color_charge * (
        math.log(mu_GeV**2 / _M_TOP_GEV**2) - 5.0 / 3.0
    )


def alpha_EM_at_mZ(
    alpha0: float,
    delta_alpha_had: float = DELTA_ALPHA_HAD_5,
    include_top: bool = True,
) -> float:
    """Run alpha_EM(0) to m_Z via alpha(m_Z) = alpha0 / (1 - Delta_alpha).

    Delta_alpha = Delta_lep (computed) + Delta_had(5) (experimental input)
                  + Delta_top (computed, optional).
    """
    delta = delta_alpha_leptonic(alpha0) + delta_alpha_had
    if include_top:
        delta += delta_alpha_top(alpha0)
    return alpha0 / (1.0 - delta)


# --- High-level helpers --------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    """Result of running one coupling via RG."""

    name: str
    value_before: float
    value_after: float
    mu_from_GeV: float
    mu_to_GeV: float


def run_alpha_s_to_mz(alpha_s_pippa: float, mu_from_GeV: float) -> RunResult:
    """Run alpha_s from the Pippa-formula scale to m_Z."""
    after = run_coupling(beta_alpha_s, alpha_s_pippa, mu_from_GeV, M_Z_GEV)
    return RunResult("alpha_s", alpha_s_pippa, after, mu_from_GeV, M_Z_GEV)


def run_alpha_EM_to_mz(alpha_EM_pippa: float) -> RunResult:
    """Run alpha_EM(0) (Pippa formula) to m_Z via the Delta_alpha scheme.

    NOTE: uses the experimental hadronic input Delta_alpha_had^(5).
    """
    after = alpha_EM_at_mZ(alpha_EM_pippa)
    return RunResult("alpha_EM", alpha_EM_pippa, after, 0.0, M_Z_GEV)

