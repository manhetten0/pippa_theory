"""Tests for the structural-complexity definition and SPARC pilot."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

import numpy as np

from pippa import sparc
from pippa import structural_complexity as complexity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPARC_ARCHIVE = PROJECT_ROOT / sparc.SPARC_ARCHIVE_NAME


def _synthetic_fields(size: int = 64) -> dict[str, np.ndarray]:
    y, x = np.mgrid[:size, :size]

    def gaussian(cx: float, cy: float, sigma: float, amplitude: float = 1.0):
        radius_squared = (x - cx) ** 2 + (y - cy) ** 2
        return amplitude * np.exp(-radius_squared / (2.0 * sigma * sigma))

    hierarchy = (
        gaussian(32, 32, 14, 0.6)
        + gaussian(20, 22, 5, 1.2)
        + gaussian(43, 21, 4, 1.0)
        + gaussian(24, 43, 3, 0.9)
        + gaussian(42, 43, 2, 0.7)
        + gaussian(19, 20, 1, 1.5)
        + gaussian(44, 22, 1, 1.4)
    )
    point = np.zeros((size, size))
    point[12, 27] = 1.0
    return {
        "uniform": np.ones((size, size)),
        "point": point,
        "simple": gaussian(32, 32, 8),
        "hierarchy": hierarchy,
        "noise": np.random.default_rng(42).exponential(size=(size, size)),
    }


def test_spectral_complexity_has_both_simple_endpoints():
    fields = _synthetic_fields()

    uniform = complexity.spectral_complexity(fields["uniform"])
    point = complexity.spectral_complexity(fields["point"])

    assert uniform.entropy == pytest.approx(0.0, abs=1.0e-14)
    assert uniform.disequilibrium == pytest.approx(1.0)
    assert uniform.value == pytest.approx(0.0, abs=1.0e-14)

    assert point.entropy == pytest.approx(1.0)
    assert point.disequilibrium == pytest.approx(0.0, abs=1.0e-14)
    assert point.value == pytest.approx(0.0, abs=1.0e-14)


def test_multiscale_hierarchy_is_more_complex_than_simple_shape_and_noise():
    fields = _synthetic_fields()
    scores = {
        name: complexity.multiscale_complexity(field).value
        for name, field in fields.items()
    }

    assert scores["uniform"] == pytest.approx(0.0, abs=1.0e-14)
    assert scores["simple"] < scores["hierarchy"]
    assert scores["noise"] < scores["hierarchy"]


def test_fourier_complexity_ignores_total_amplitude_and_translation():
    field = _synthetic_fields()["hierarchy"]
    reference = complexity.multiscale_complexity(field).value

    assert complexity.multiscale_complexity(37.0 * field).value == pytest.approx(
        reference,
        rel=1.0e-12,
        abs=1.0e-14,
    )
    assert complexity.multiscale_complexity(np.roll(field, (7, -11), axis=(0, 1))).value == pytest.approx(
        reference,
        rel=1.0e-12,
        abs=1.0e-14,
    )


def test_radial_complexity_uses_shape_not_profile_normalization():
    radius = np.array([0.1, 0.3, 0.8, 1.7, 3.0, 5.0])
    profile = np.exp(-radius / 1.2) * (1.0 + 0.2 * np.cos(4.0 * radius))

    reference = complexity.radial_profile_complexity(radius, profile).value
    rescaled = complexity.radial_profile_complexity(radius, 1.0e9 * profile).value

    assert 0.0 <= reference <= 1.0
    assert rescaled == pytest.approx(reference, rel=1.0e-12, abs=1.0e-14)


def test_rotmod_archive_exposes_surface_brightness_columns():
    if not SPARC_ARCHIVE.exists():
        pytest.skip(f"SPARC archive not found: {SPARC_ARCHIVE}")

    curve = sparc.load_galaxy(SPARC_ARCHIVE, "NGC3198")

    assert curve.surface_brightness_disk is not None
    assert curve.surface_brightness_bulge is not None
    assert curve.surface_brightness_disk.shape == curve.radius_kpc.shape
    assert curve.surface_brightness_bulge.shape == curve.radius_kpc.shape
    assert np.isfinite(curve.surface_brightness_disk).all()
    assert complexity.stellar_radial_complexity(curve).value > 0.0


def test_sparc_radial_complexity_pilot_is_finite(capsys):
    if not SPARC_ARCHIVE.exists():
        pytest.skip(f"SPARC archive not found: {SPARC_ARCHIVE}")

    result = complexity.evaluate_sparc_complexity(SPARC_ARCHIVE)

    with capsys.disabled():
        print("\nSPARC one-dimensional structural-complexity pilot")
        print(
            f"galaxies={result.n_galaxies}, "
            f"C=[{result.complexity_min:.5f}, {result.complexity_median:.5f}, "
            f"{result.complexity_max:.5f}]"
        )
        print(
            f"held-out RMSE: baseline={result.baseline_rmse_dex:.6f} dex, "
            f"with complexity={result.complexity_rmse_dex:.6f} dex, "
            f"relative improvement={result.relative_improvement:.3%}"
        )

    assert result.n_galaxies == 175
    assert 0.0 <= result.complexity_min <= result.complexity_median
    assert result.complexity_median <= result.complexity_max <= 1.0
    assert math.isfinite(result.baseline_rmse_dex)
    assert math.isfinite(result.complexity_rmse_dex)
    assert math.isfinite(result.relative_improvement)
