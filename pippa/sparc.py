"""SPARC rotation-curve validation helpers for the Pippa DM profile.

This module keeps the observational test separate from the lightweight
core formulas.  It uses the local ``Rotmod_LTG.zip`` archive and never
downloads data during tests.
"""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import constants


G_KPC_KMS2_PER_MSUN: float = 4.30091e-6
SPARC_ARCHIVE_NAME: str = "Rotmod_LTG.zip"
MIN_VALID_ROTATION_POINTS: int = 4
MOND_A0_M_PER_S2: float = 1.2e-10
M_PER_KPC: float = 3.085677581491367e19

PIPPA_AMPLITUDE: float = 106.6
FIDUCIAL_MASS_TO_LIGHT: float = 0.5

THEORY_GAMMA_BY_TYPE: dict[str, float] = {
    "disk": 1.61,
    "bulge": 1.39,
    "gas-dwarf": 0.0,
}

REFERENCE_GALAXY_TYPES: dict[str, str] = {
    "NGC3198": "disk",
    "NGC2841": "bulge",
    "NGC3109": "gas-dwarf",
    "DDO064": "gas-dwarf",
    "NGC0891": "bulge",
}

REFERENCE_GALAXIES: tuple[str, ...] = tuple(REFERENCE_GALAXY_TYPES)


@dataclass(frozen=True)
class GalaxyRotationCurve:
    """One SPARC ``*_rotmod.dat`` rotation curve."""

    name: str
    radius_kpc: np.ndarray
    v_obs_kms: np.ndarray
    err_v_kms: np.ndarray
    v_gas_kms: np.ndarray
    v_disk_kms: np.ndarray
    v_bulge_kms: np.ndarray

    @property
    def n_points(self) -> int:
        return int(self.radius_kpc.size)


@dataclass(frozen=True)
class FitResult:
    """Weighted fit result for a rotation-curve model."""

    galaxy: str
    model: str
    chi2_dof: float
    n_points: int
    n_parameters: int
    parameters: dict[str, float]


@dataclass(frozen=True)
class CatalogFit:
    """All model fits for one SPARC galaxy."""

    galaxy: str
    galaxy_type: str
    n_points: int
    pippa_free: FitResult
    pippa_universal: FitResult
    nfw: FitResult
    mond_fixed_a0: FitResult
    mond_free_a0: FitResult


def default_archive_path(project_root: str | Path | None = None) -> Path:
    """Return the expected local SPARC archive path."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    return root / SPARC_ARCHIVE_NAME


def list_galaxies(archive_path: str | Path) -> list[str]:
    """List galaxy names available in a local SPARC rotmod archive."""
    with zipfile.ZipFile(archive_path) as archive:
        return sorted(
            name.removesuffix("_rotmod.dat")
            for name in archive.namelist()
            if name.endswith("_rotmod.dat")
        )


def load_galaxy(archive_path: str | Path, galaxy: str) -> GalaxyRotationCurve:
    """Load one galaxy rotation curve from a local SPARC archive."""
    target = f"{galaxy}_rotmod.dat"
    with zipfile.ZipFile(archive_path) as archive:
        if target not in archive.namelist():
            raise KeyError(f"{target} not found in {archive_path}")

        rows: list[list[float]] = []
        for line in archive.read(target).decode("utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            rows.append([float(value) for value in parts[:6]])

    if not rows:
        raise ValueError(f"{target} contains no rotation-curve rows")

    data = np.asarray(rows, dtype=float)
    data = data[np.argsort(data[:, 0])]
    valid = (
        (data[:, 0] > 0.0)
        & np.isfinite(data[:, :6]).all(axis=1)
        & (data[:, 2] > 0.0)
    )
    data = data[valid]
    if data.shape[0] < MIN_VALID_ROTATION_POINTS:
        raise ValueError(f"{target} has too few valid points: {data.shape[0]}")

    return GalaxyRotationCurve(
        name=galaxy,
        radius_kpc=data[:, 0],
        v_obs_kms=data[:, 1],
        err_v_kms=data[:, 2],
        v_gas_kms=data[:, 3],
        v_disk_kms=data[:, 4],
        v_bulge_kms=data[:, 5],
    )


def signed_square(values: np.ndarray) -> np.ndarray:
    """SPARC convention: gas may contribute with a signed velocity squared."""
    return np.sign(values) * values * values


def baryonic_velocity_squared(
    curve: GalaxyRotationCurve,
    mass_to_light: float,
) -> np.ndarray:
    """Baryonic contribution with one shared stellar mass-to-light factor."""
    return (
        signed_square(curve.v_gas_kms)
        + mass_to_light * signed_square(curve.v_disk_kms)
        + mass_to_light * signed_square(curve.v_bulge_kms)
    )


def baryonic_velocity_squared_split(
    curve: GalaxyRotationCurve,
    disk_mass_to_light: float,
    bulge_mass_to_light: float,
) -> np.ndarray:
    """Baryonic contribution with separate disk and bulge M/L factors."""
    return (
        signed_square(curve.v_gas_kms)
        + disk_mass_to_light * signed_square(curve.v_disk_kms)
        + bulge_mass_to_light * signed_square(curve.v_bulge_kms)
    )


def pippa_scaling_exponent(D: float = constants.D) -> float:
    """Theory exponent for ``A_M proportional M_bar^k``."""
    return (1.0 - D) / 2.0


def classify_galaxy(curve: GalaxyRotationCurve) -> str:
    """Classify a curve into the theory's disk/bulge/gas-dwarf buckets."""
    if curve.name in REFERENCE_GALAXY_TYPES:
        return REFERENCE_GALAXY_TYPES[curve.name]

    v_bulge = float(np.max(np.abs(curve.v_bulge_kms)))
    v_disk = float(np.max(np.abs(curve.v_disk_kms)))
    v_obs = float(np.max(curve.v_obs_kms))

    if v_bulge > 0.25 * max(v_disk, 1.0):
        return "bulge"
    if v_obs < 85.0:
        return "gas-dwarf"
    return "disk"


def baryonic_mass_proxy(
    curve: GalaxyRotationCurve,
    mass_to_light: float = FIDUCIAL_MASS_TO_LIGHT,
) -> float:
    """Estimate a baryonic mass scale from the observed baryonic curve.

    The rotmod archive does not include tabulated total baryonic masses, so
    the validation uses the enclosed dynamical proxy ``M ~ r V_bar^2 / G``.
    This preserves the theory's fixed exponent while keeping the test fully
    reproducible from the SPARC archive alone.
    """
    v2_bar = np.clip(baryonic_velocity_squared(curve, mass_to_light), 0.0, None)
    m_proxy = np.max(curve.radius_kpc * v2_bar / G_KPC_KMS2_PER_MSUN)
    return max(float(m_proxy), 1.0e6)


def pippa_A_M(
    curve: GalaxyRotationCurve,
    amplitude: float = PIPPA_AMPLITUDE,
) -> float:
    """Theory amplitude ``A_M = 106.6 * M_bar^((1-D)/2)``."""
    return amplitude * baryonic_mass_proxy(curve) ** pippa_scaling_exponent()


def pippa_gamma(curve: GalaxyRotationCurve) -> float:
    """Theory gamma fixed by the morphology bucket."""
    return THEORY_GAMMA_BY_TYPE[classify_galaxy(curve)]


def pippa_density(
    radius_kpc: np.ndarray,
    rho0_msun_kpc3: float,
    r0_kpc: float,
    A_M: float,
    gamma: float,
    D: float = constants.D,
) -> np.ndarray:
    """Pippa halo density from Appendix G.10."""
    r = np.clip(np.asarray(radius_kpc, dtype=float), 1.0e-4, None)
    r0 = max(float(r0_kpc), 1.0e-4)
    alpha_eff = (2.0 - D) + gamma * (1.0 - np.exp(-r / r0))
    M_operator = 1.0 + A_M * (1.0 - np.exp(-r / r0))
    return rho0_msun_kpc3 * np.power(r0 / r, alpha_eff) * M_operator


def pippa_halo_velocity(
    radius_kpc: np.ndarray,
    log10_rho0: float,
    r0_kpc: float,
    A_M: float,
    gamma: float,
    n_grid: int = 384,
) -> np.ndarray:
    """Circular velocity from the spherical Pippa halo density."""
    r = np.asarray(radius_kpc, dtype=float)
    r_max = max(float(np.max(r)), 1.0e-2)
    grid = np.geomspace(1.0e-4, r_max, n_grid)
    rho = pippa_density(grid, 10.0**log10_rho0, r0_kpc, A_M, gamma)
    integrand = 4.0 * math.pi * rho * grid * grid
    mass = np.zeros_like(grid)
    mass[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid))
    enclosed = np.interp(r, grid, mass)
    v2 = G_KPC_KMS2_PER_MSUN * enclosed / r
    return np.sqrt(np.clip(v2, 0.0, None))


def pippa_total_velocity(
    curve: GalaxyRotationCurve,
    mass_to_light: float,
    log10_rho0: float,
    r0_kpc: float,
    A_M: float,
    gamma: float,
) -> np.ndarray:
    """Full rotation curve: baryons plus Pippa halo."""
    halo = pippa_halo_velocity(curve.radius_kpc, log10_rho0, r0_kpc, A_M, gamma)
    v2_total = baryonic_velocity_squared(curve, mass_to_light) + halo * halo
    return np.sqrt(np.clip(v2_total, 1.0e-9, None))


def nfw_halo_velocity(
    radius_kpc: np.ndarray,
    log10_rho0: float,
    rs_kpc: float,
) -> np.ndarray:
    """NFW comparison halo velocity with the same SPARC units."""
    r = np.asarray(radius_kpc, dtype=float)
    rho0 = 10.0**log10_rho0
    rs = max(float(rs_kpc), 1.0e-4)
    x = np.clip(r / rs, 1.0e-8, None)
    mass = 4.0 * math.pi * rho0 * rs**3 * (np.log1p(x) - x / (1.0 + x))
    return np.sqrt(np.clip(G_KPC_KMS2_PER_MSUN * mass / r, 0.0, None))


def nfw_total_velocity(
    curve: GalaxyRotationCurve,
    mass_to_light: float,
    log10_rho0: float,
    rs_kpc: float,
) -> np.ndarray:
    """Full rotation curve: baryons plus NFW halo."""
    halo = nfw_halo_velocity(curve.radius_kpc, log10_rho0, rs_kpc)
    v2_total = baryonic_velocity_squared(curve, mass_to_light) + halo * halo
    return np.sqrt(np.clip(v2_total, 1.0e-9, None))


def acceleration_m_per_s2_to_km2_s2_per_kpc(acceleration: float) -> float:
    """Convert acceleration to ``(km/s)^2/kpc``."""
    return acceleration * M_PER_KPC / 1.0e6


def mond_total_velocity(
    curve: GalaxyRotationCurve,
    disk_mass_to_light: float,
    bulge_mass_to_light: float | None = None,
    a0_m_per_s2: float = MOND_A0_M_PER_S2,
) -> np.ndarray:
    """MOND/RAR rotation curve using the McGaugh-style interpolation.

    Formula:
        g_obs = g_bar / (1 - exp(-sqrt(g_bar / a0)))

    ``a0`` is fixed by default.  Disk and bulge mass-to-light ratios may be
    fitted separately; when no bulge value is given, the disk value is reused.
    """
    if bulge_mass_to_light is None:
        bulge_mass_to_light = disk_mass_to_light

    v2_bar = np.clip(
        baryonic_velocity_squared_split(curve, disk_mass_to_light, bulge_mass_to_light),
        0.0,
        None,
    )
    g_bar = v2_bar / curve.radius_kpc
    a0 = acceleration_m_per_s2_to_km2_s2_per_kpc(a0_m_per_s2)

    g_obs = np.zeros_like(g_bar)
    positive = g_bar > 0.0
    x = np.sqrt(g_bar[positive] / a0)
    denom = 1.0 - np.exp(-x)
    g_obs[positive] = g_bar[positive] / np.clip(denom, 1.0e-12, None)
    return np.sqrt(np.clip(g_obs * curve.radius_kpc, 1.0e-9, None))


def _fit_mond(
    curve: GalaxyRotationCurve,
    fit_a0: bool,
    max_nfev: int = 600,
) -> FitResult:
    has_bulge = float(np.max(np.abs(curve.v_bulge_kms))) > 1.0e-9
    a0_default = acceleration_m_per_s2_to_km2_s2_per_kpc(MOND_A0_M_PER_S2)

    def decode(x: np.ndarray) -> tuple[float, float, float]:
        if has_bulge:
            disk_mtl = float(x[0])
            bulge_mtl = float(x[1])
            log10_a0 = float(x[2]) if fit_a0 else math.log10(a0_default)
        else:
            disk_mtl = float(x[0])
            bulge_mtl = 0.0
            log10_a0 = float(x[1]) if fit_a0 else math.log10(a0_default)
        return disk_mtl, bulge_mtl, 10.0**log10_a0

    if has_bulge and fit_a0:
        x0 = np.array([0.5, 0.7, math.log10(a0_default)])
        lower = np.array([0.05, 0.05, 2.0])
        upper = np.array([2.5, 2.5, 5.5])
    elif has_bulge:
        x0 = np.array([0.5, 0.7])
        lower = np.array([0.05, 0.05])
        upper = np.array([2.5, 2.5])
    elif fit_a0:
        x0 = np.array([0.5, math.log10(a0_default)])
        lower = np.array([0.05, 2.0])
        upper = np.array([2.5, 5.5])
    else:
        x0 = np.array([0.5])
        lower = np.array([0.05])
        upper = np.array([2.5])

    def residual(x: np.ndarray) -> np.ndarray:
        disk_mtl, bulge_mtl, a0 = decode(x)
        a0_m_per_s2 = a0 * 1.0e6 / M_PER_KPC
        pred = mond_total_velocity(curve, disk_mtl, bulge_mtl, a0_m_per_s2)
        return (pred - curve.v_obs_kms) / np.maximum(curve.err_v_kms, 1.0)

    result = least_squares(
        residual,
        x0=x0,
        bounds=(lower, upper),
        max_nfev=max_nfev,
        xtol=1.0e-6,
        ftol=1.0e-6,
        gtol=1.0e-6,
    )
    disk_mtl, bulge_mtl, a0 = decode(result.x)
    a0_m_per_s2 = a0 * 1.0e6 / M_PER_KPC
    pred = mond_total_velocity(curve, disk_mtl, bulge_mtl, a0_m_per_s2)
    n_parameters = int(result.x.size)
    model_name = "MOND-free-a0" if fit_a0 else "MOND-fixed-a0"
    return FitResult(
        galaxy=curve.name,
        model=model_name,
        chi2_dof=chi2_dof(curve, pred, n_parameters=n_parameters),
        n_points=curve.n_points,
        n_parameters=n_parameters,
        parameters={
            "disk_mass_to_light": disk_mtl,
            "bulge_mass_to_light": bulge_mtl,
            "a0_m_per_s2": a0_m_per_s2,
        },
    )


def chi2_dof(
    curve: GalaxyRotationCurve,
    predicted_kms: np.ndarray,
    n_parameters: int,
) -> float:
    """Weighted chi-square per degree of freedom."""
    sigma = np.maximum(curve.err_v_kms, 1.0)
    residual = (predicted_kms - curve.v_obs_kms) / sigma
    dof = max(curve.n_points - n_parameters, 1)
    return float(np.sum(residual * residual) / dof)


def fit_pippa_free(curve: GalaxyRotationCurve, max_nfev: int = 600) -> FitResult:
    """Fit the free 5-parameter Pippa profile."""
    gamma0 = pippa_gamma(curve)

    def residual(x: np.ndarray) -> np.ndarray:
        pred = pippa_total_velocity(curve, x[0], x[1], x[2], x[3], x[4])
        return (pred - curve.v_obs_kms) / np.maximum(curve.err_v_kms, 1.0)

    result = least_squares(
        residual,
        x0=np.array([0.5, 7.2, 5.0, 1.0, gamma0]),
        bounds=(
            np.array([0.05, 4.0, 0.05, 0.0, -0.5]),
            np.array([2.5, 11.0, 80.0, 8.0, 3.0]),
        ),
        max_nfev=max_nfev,
        xtol=1.0e-6,
        ftol=1.0e-6,
        gtol=1.0e-6,
    )
    pred = pippa_total_velocity(curve, *result.x)
    return FitResult(
        galaxy=curve.name,
        model="Pippa-free",
        chi2_dof=chi2_dof(curve, pred, n_parameters=5),
        n_points=curve.n_points,
        n_parameters=5,
        parameters={
            "mass_to_light": float(result.x[0]),
            "log10_rho0": float(result.x[1]),
            "r0_kpc": float(result.x[2]),
            "A_M": float(result.x[3]),
            "gamma": float(result.x[4]),
        },
    )


def fit_pippa_universal(curve: GalaxyRotationCurve, max_nfev: int = 600) -> FitResult:
    """Fit Pippa with ``A_M`` and ``gamma`` fixed by the theory relations."""
    A_M = pippa_A_M(curve)
    gamma = pippa_gamma(curve)

    def residual(x: np.ndarray) -> np.ndarray:
        pred = pippa_total_velocity(curve, x[0], x[1], x[2], A_M, gamma)
        return (pred - curve.v_obs_kms) / np.maximum(curve.err_v_kms, 1.0)

    result = least_squares(
        residual,
        x0=np.array([0.5, 7.2, 5.0]),
        bounds=(
            np.array([0.05, 4.0, 0.05]),
            np.array([2.5, 11.0, 80.0]),
        ),
        max_nfev=max_nfev,
        xtol=1.0e-6,
        ftol=1.0e-6,
        gtol=1.0e-6,
    )
    pred = pippa_total_velocity(curve, result.x[0], result.x[1], result.x[2], A_M, gamma)
    return FitResult(
        galaxy=curve.name,
        model="Pippa-universal",
        chi2_dof=chi2_dof(curve, pred, n_parameters=3),
        n_points=curve.n_points,
        n_parameters=3,
        parameters={
            "mass_to_light": float(result.x[0]),
            "log10_rho0": float(result.x[1]),
            "r0_kpc": float(result.x[2]),
            "A_M": float(A_M),
            "gamma": float(gamma),
        },
    )


def fit_nfw(curve: GalaxyRotationCurve, max_nfev: int = 600) -> FitResult:
    """Fit a same-data NFW comparison model."""
    def residual(x: np.ndarray) -> np.ndarray:
        pred = nfw_total_velocity(curve, x[0], x[1], x[2])
        return (pred - curve.v_obs_kms) / np.maximum(curve.err_v_kms, 1.0)

    result = least_squares(
        residual,
        x0=np.array([0.5, 7.2, 5.0]),
        bounds=(
            np.array([0.05, 4.0, 0.05]),
            np.array([2.5, 11.0, 150.0]),
        ),
        max_nfev=max_nfev,
        xtol=1.0e-6,
        ftol=1.0e-6,
        gtol=1.0e-6,
    )
    pred = nfw_total_velocity(curve, *result.x)
    return FitResult(
        galaxy=curve.name,
        model="NFW",
        chi2_dof=chi2_dof(curve, pred, n_parameters=3),
        n_points=curve.n_points,
        n_parameters=3,
        parameters={
            "mass_to_light": float(result.x[0]),
            "log10_rho0": float(result.x[1]),
            "rs_kpc": float(result.x[2]),
        },
    )


def fit_mond(curve: GalaxyRotationCurve, max_nfev: int = 600) -> FitResult:
    """Fit MOND/RAR with fixed ``a0`` and stellar M/L parameter(s)."""
    return _fit_mond(curve, fit_a0=False, max_nfev=max_nfev)


def fit_mond_free_a0(curve: GalaxyRotationCurve, max_nfev: int = 600) -> FitResult:
    """Fit MOND/RAR with per-galaxy ``a0`` plus stellar M/L parameter(s)."""
    return _fit_mond(curve, fit_a0=True, max_nfev=max_nfev)


def fit_mond_fixed_a0(curve: GalaxyRotationCurve, max_nfev: int = 600) -> FitResult:
    """Explicit alias for the fixed-a0 MOND baseline."""
    return fit_mond(curve, max_nfev=max_nfev)


def fit_reference_sample(archive_path: str | Path) -> list[tuple[FitResult, FitResult, FitResult]]:
    """Fit the README representative SPARC galaxies."""
    rows: list[tuple[FitResult, FitResult, FitResult]] = []
    for galaxy in REFERENCE_GALAXIES:
        curve = load_galaxy(archive_path, galaxy)
        rows.append((fit_pippa_free(curve), fit_pippa_universal(curve), fit_nfw(curve)))
    return rows


def fit_catalog(
    archive_path: str | Path,
    galaxies: list[str] | tuple[str, ...] | None = None,
) -> list[CatalogFit]:
    """Fit Pippa and NFW models for a SPARC galaxy list.

    When ``galaxies`` is omitted, the full local rotmod archive is used.
    """
    names = list(galaxies) if galaxies is not None else list_galaxies(archive_path)
    rows: list[CatalogFit] = []
    for galaxy in names:
        curve = load_galaxy(archive_path, galaxy)
        rows.append(
            CatalogFit(
                galaxy=galaxy,
                galaxy_type=classify_galaxy(curve),
                n_points=curve.n_points,
                pippa_free=fit_pippa_free(curve),
                pippa_universal=fit_pippa_universal(curve),
                nfw=fit_nfw(curve),
                mond_fixed_a0=fit_mond_fixed_a0(curve),
                mond_free_a0=fit_mond_free_a0(curve),
            )
        )
    return rows
