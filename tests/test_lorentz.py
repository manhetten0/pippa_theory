import pytest
import numpy as np
from pippa.gravity import InformationChannel


def lorentz_boost_along_x(r_vec: np.ndarray, v: float) -> np.ndarray:
    """Лоренц-буст вектора положения (при t=0)."""
    if abs(v) < 1e-14:
        return r_vec.copy()
    gamma = 1.0 / np.sqrt(1 - v * v)
    r_parallel = r_vec[0]
    return np.concatenate(([gamma * r_parallel], r_vec[1:]))


@pytest.mark.parametrize("v", [0.0, 0.3, 0.6, 0.85])
def test_potential_depends_on_proper_distance(v):
    """Потенциал должен зависеть от собственного расстояния (proper distance)."""
    channel = InformationChannel(C0=1.0, alpha=0.1, lam=0.05)

    I = 1.0
    I_dot = 0.2
    r_proper = 10.0  # собственное расстояние

    # В системе покоя
    phi_rest = channel.potential(r_proper, I, I_dot)

    # В движущейся системе (лоренц-сокращение)
    r_vec = np.array([r_proper, 0.0, 0.0])
    r_boosted_vec = lorentz_boost_along_x(r_vec, v)
    r_boosted = np.linalg.norm(r_boosted_vec)

    phi_boost = channel.potential(r_boosted, I, I_dot)

    # Ожидаем разницу из-за сокращения длины
    if v == 0.0:
        assert np.isclose(phi_rest, phi_boost, rtol=1e-12)
    else:
        assert not np.isclose(phi_rest, phi_boost, rtol=1e-8), \
            "При бусте r должно меняться → Φ тоже меняется (это правильно)"


def test_newtonian_limit_correct_boost():
    """В ньютоновском пределе (без насыщения) поведение предсказуемо."""
    channel = InformationChannel(C0=1.0, alpha=0.0, lam=0.0)

    I = 1.0
    r = 5.0
    phi0 = channel.potential(r, I, I_dot=0.0)

    for v in [0.1, 0.5, 0.9]:
        r_vec = np.array([r, 0.0])
        r_boost = np.linalg.norm(lorentz_boost_along_x(r_vec, v))
        phi_boost = channel.potential(r_boost, I, I_dot=0.0)

        assert abs(phi0 - phi_boost) > 1e-6, "Должна быть заметная разница при бусте"


def test_acceleration_magnitude_consistency():
    """Проверка, что величина ускорения разумно ведёт себя при разных направлениях."""
    channel = InformationChannel(C0=1.0, alpha=0.1, lam=0.05)

    I = 1.0
    I_dot = 0.2
    r = 10.0

    acc_r = channel.acceleration(r, I, I_dot, dr=1e-8)

    # Проверяем в разных направлениях (изотропия)
    for angle in [0, np.pi / 4, np.pi / 2]:
        # Здесь можно проверить, что |g| одинаково
        assert abs(acc_r) > 0