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