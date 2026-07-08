import numpy as np


# ============================================================
# Pippa local/global dimension tests
# ============================================================

def boosted_vector(theta, beta):
    """
    Lorentz-boosted phase vector.
    """

    gamma = 1.0 / np.sqrt(1.0 - beta**2)

    a = gamma * (np.cos(theta) - beta * np.sin(theta))
    b = gamma * (np.sin(theta) - beta * np.cos(theta))

    return a, b


def global_dimension(beta, n=200000):
    """
    Global ergodic dimension:
        <L1> / sqrt(<L2^2>)
    over full phase cycle.
    """

    theta = np.linspace(0, 2 * np.pi, n)

    a, b = boosted_vector(theta, beta)

    L1 = np.abs(a) + np.abs(b)
    L2_sq = a**2 + b**2

    return np.mean(L1) / np.sqrt(np.mean(L2_sq))


def local_dimension(theta0, dtheta):
    """
    Local tangent-space dimension using infinitesimal
    phase displacement.

    D_local = ||dv||_1 / ||dv||_2
    """

    v1 = np.array([
        np.cos(theta0),
        np.sin(theta0),
    ])

    v2 = np.array([
        np.cos(theta0 + dtheta),
        np.sin(theta0 + dtheta),
    ])

    dv = v2 - v1

    L1 = np.sum(np.abs(dv))
    L2 = np.sqrt(np.sum(dv**2))

    return L1 / L2


def effective_dimension(theta_max, n=100000):
    """
    Average local dimension over finite phase arc.
    """

    theta = np.linspace(0, theta_max, n)

    D = np.abs(np.sin(theta)) + np.abs(np.cos(theta))

    return np.mean(D)


# ============================================================
# TESTS
# ============================================================

def test_global_dimension_rest_frame():
    """
    Rest-frame global dimension should equal 4/pi.
    """

    D = global_dimension(beta=0.0)

    assert abs(D - 4 / np.pi) < 1e-3


def test_global_dimension_invariant_under_boost():
    """
    Global dimension should remain Lorentz invariant.
    """

    D0 = global_dimension(beta=0.0)
    D1 = global_dimension(beta=0.5)
    D2 = global_dimension(beta=0.9)

    assert abs(D0 - D1) < 1e-3
    assert abs(D1 - D2) < 1e-3


def test_local_dimension_approaches_one():
    """
    Tangent-space geometry near axis becomes Euclidean.
    """

    D_local = local_dimension(
        theta0=0.0,
        dtheta=1e-8,
    )

    assert abs(D_local - 1.0) < 1e-6


def test_local_dimension_at_diagonal():
    """
    Near diagonal direction local dimension approaches sqrt(2).
    """

    D_local = local_dimension(
        theta0=np.pi / 4,
        dtheta=1e-8,
    )

    assert abs(D_local - np.sqrt(2)) < 1e-6


def test_global_dimension_between_limits():
    """
    Global dimension should lie between
    Euclidean and taxicab limits.
    """

    D_global = global_dimension(beta=0.3)

    assert 1.0 < D_global < np.sqrt(2)


def test_effective_dimension_low_arc():
    """
    Small phase arcs should look nearly Euclidean.
    """

    D_eff = effective_dimension(theta_max=1e-3)

    assert abs(D_eff - 1.0) < 1e-3


def test_effective_dimension_full_cycle():
    """
    Full ergodic averaging restores 4/pi.
    """

    theta = np.linspace(0, 2 * np.pi, 200000)

    D = np.mean(
        np.abs(np.sin(theta)) +
        np.abs(np.cos(theta))
    )

    assert abs(D - 4 / np.pi) < 1e-3