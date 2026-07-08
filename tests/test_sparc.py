"""SPARC validation tests for the Pippa dark-matter profile."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

import numpy as np

from pippa import constants
from pippa import sparc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPARC_ARCHIVE = PROJECT_ROOT / sparc.SPARC_ARCHIVE_NAME


def require_sparc_archive() -> Path:
    if not SPARC_ARCHIVE.exists():
        pytest.skip(f"SPARC archive not found: {SPARC_ARCHIVE}")
    return SPARC_ARCHIVE


def test_sparc_archive_contains_full_rotmod_catalog():
    archive = require_sparc_archive()
    galaxies = sparc.list_galaxies(archive)

    assert len(galaxies) == 175
    for galaxy in sparc.REFERENCE_GALAXIES:
        assert galaxy in galaxies


def test_sparc_theory_constraints_are_fixed_by_D():
    archive = require_sparc_archive()
    massive = sparc.load_galaxy(archive, "NGC2841")
    dwarf = sparc.load_galaxy(archive, "DDO064")

    assert sparc.pippa_scaling_exponent() == pytest.approx((1.0 - constants.D) / 2.0)
    assert sparc.classify_galaxy(massive) == "bulge"
    assert sparc.classify_galaxy(dwarf) == "gas-dwarf"
    assert sparc.pippa_gamma(massive) == pytest.approx(1.39)
    assert sparc.pippa_gamma(dwarf) == pytest.approx(0.0)

    # k < 0, so lower baryonic mass gets a larger inter-quadrant amplitude.
    assert sparc.pippa_A_M(dwarf) > sparc.pippa_A_M(massive)


def test_sparc_full_catalog_compares_pippa_nfw_and_mond(capsys):
    archive = require_sparc_archive()
    rows = sparc.fit_catalog(archive)

    free = np.array([row.pippa_free.chi2_dof for row in rows])
    universal = np.array([row.pippa_universal.chi2_dof for row in rows])
    nfw = np.array([row.nfw.chi2_dof for row in rows])
    mond_fixed = np.array([row.mond_fixed_a0.chi2_dof for row in rows])
    mond_free = np.array([row.mond_free_a0.chi2_dof for row in rows])

    with capsys.disabled():
        print("\n" + "=" * 78)
        print("SPARC full-catalog validation: weighted chi2/dof for 175 galaxies")
        print("MOND-fixed uses a0=1.2e-10 m/s^2; MOND-free also fits a0 per galaxy.")
        print("=" * 78)
        print("Model          pars  median   mean    chi2<2  chi2<5  wins vs NFW/MONDfree")
        print("-" * 78)
        for label, n_parameters, values, wins_nfw, wins_mond in (
            ("Pippa-free", "5", free, int(np.sum(free < nfw)), int(np.sum(free < mond_free))),
            (
                "Pippa-univ",
                "3",
                universal,
                int(np.sum(universal < nfw)),
                int(np.sum(universal < mond_free)),
            ),
            ("NFW", "3", nfw, 0, int(np.sum(nfw < mond_free))),
            ("MOND-fixed", "1-2", mond_fixed, int(np.sum(mond_fixed < nfw)), 0),
            ("MOND-free", "2-3", mond_free, int(np.sum(mond_free < nfw)), 0),
        ):
            print(
                f"{label:<12} {n_parameters:>4} "
                f"{float(np.median(values)):7.3f} "
                f"{float(np.mean(values)):7.3f} "
                f"{float(np.mean(values < 2.0)):7.1%} "
                f"{float(np.mean(values < 5.0)):7.1%} "
                f"{wins_nfw:5d}/{wins_mond:<5d}"
            )

        print("-" * 78)
        print("Type       N    median free/univ/NFW/MONDfree    univ<M  free<M")
        for galaxy_type in ("disk", "bulge", "gas-dwarf"):
            mask = np.array([row.galaxy_type == galaxy_type for row in rows])
            print(
                f"{galaxy_type:<10} {int(mask.sum()):3d}  "
                f"{float(np.median(free[mask])):6.3f} / "
                f"{float(np.median(universal[mask])):6.3f} / "
                f"{float(np.median(nfw[mask])):6.3f} / "
                f"{float(np.median(mond_free[mask])):6.3f}        "
                f"{int(np.sum(universal[mask] < mond_free[mask])):3d}    "
                f"{int(np.sum(free[mask] < mond_free[mask])):3d}"
            )

        print("-" * 78)
        worst = sorted(rows, key=lambda row: row.pippa_universal.chi2_dof, reverse=True)[:5]
        print("Worst universal fits:")
        for row in worst:
            print(
                f"  {row.galaxy:<10} {row.galaxy_type:<10} "
                f"free={row.pippa_free.chi2_dof:.3f} "
                f"univ={row.pippa_universal.chi2_dof:.3f} "
                f"nfw={row.nfw.chi2_dof:.3f} "
                f"mondfree={row.mond_free_a0.chi2_dof:.3f}"
            )
        print("=" * 78)

    assert len(rows) == 175
    assert min(row.n_points for row in rows) >= sparc.MIN_VALID_ROTATION_POINTS
    assert all(math.isfinite(value) and value > 0.0 for value in free)
    assert all(math.isfinite(value) and value > 0.0 for value in universal)
    assert all(math.isfinite(value) and value > 0.0 for value in nfw)
    assert all(math.isfinite(value) and value > 0.0 for value in mond_fixed)
    assert all(math.isfinite(value) and value > 0.0 for value in mond_free)

    assert float(np.median(free)) < 0.6
    assert float(np.median(universal)) < 1.0
    assert float(np.mean(free < 2.0)) >= 0.80
    assert float(np.mean(universal < 2.0)) >= 0.70

    # The universal theory constraints should not erase the free-profile fit.
    assert float(np.median(universal - free)) < 0.25

    # Same full SPARC catalog, same weighted metric: Pippa beats NFW overall.
    assert int(np.sum(free < nfw)) >= 130
    assert int(np.sum(universal < nfw)) >= 95
    assert float(np.median(universal)) < float(np.median(nfw))

    # MOND is an important baseline; the per-galaxy-a0 variant is the fairer
    # comparison against 3-parameter halo fits.
    assert float(np.median(mond_fixed)) < 4.0
    assert float(np.median(mond_free)) < 1.3
    assert int(np.sum(free < mond_free)) >= 100
    assert int(np.sum(universal < mond_free)) >= 100
