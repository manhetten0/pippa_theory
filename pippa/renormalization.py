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
