"""Scale-aware structural complexity and a pilot SPARC diagnostic.

The metric is intentionally separate from mass or field amplitude.  It uses
the normalized power carried by spatial modes and combines normalized spectral
entropy with disequilibrium.  Both a perfectly ordered field and a completely
flat mode spectrum therefore have zero complexity.

``evaluate_sparc_complexity`` is only a one-dimensional pilot test.  The local
SPARC rotmod archive contains radial surface-brightness profiles, not galaxy
images, so it cannot measure bars, spiral arms, clumps, or other 2-D structure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import dctn
from scipy.ndimage import gaussian_filter


SpectralBasis = Literal["fourier", "cosine"]
DEFAULT_SCALE_FRACTIONS: tuple[float, ...] = (
    0.0,
    1.0 / 64.0,
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
)


@dataclass(frozen=True)
class ComplexityComponents:
    """Entropy, disequilibrium, and their product for one scale."""

    entropy: float
    disequilibrium: float
    value: float


@dataclass(frozen=True)
class MultiscaleComplexity:
    """Complexity values measured after smoothing at several scales."""

    scale_fractions: tuple[float, ...]
    per_scale: tuple[ComplexityComponents, ...]

    @property
    def value(self) -> float:
        """Mean complexity over all requested scales."""
        return float(np.mean([component.value for component in self.per_scale]))


@dataclass(frozen=True)
class SparcComplexityResult:
    """Grouped cross-validation result for the radial SPARC pilot test."""

    n_galaxies: int
    complexity_min: float
    complexity_median: float
    complexity_max: float
    baseline_rmse_dex: float
    complexity_rmse_dex: float
    relative_improvement: float
    permutation_p_value: float | None


class _SurfaceBrightnessCurve(Protocol):
    name: str
    radius_kpc: NDArray[np.float64]
    surface_brightness_disk: NDArray[np.float64] | None
    surface_brightness_bulge: NDArray[np.float64] | None


@dataclass(frozen=True)
class _GalaxySample:
    name: str
    complexity: float
    log10_mass_proxy: float
    log10_g_bar: NDArray[np.float64]
    log10_g_obs: NDArray[np.float64]
    radius_fraction: NDArray[np.float64]


def _as_field(field: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(field, dtype=float)
    if values.ndim == 0 or values.size < 2:
        raise ValueError("field must contain at least two samples")
    if not np.isfinite(values).all():
        raise ValueError("field must contain only finite values")
    if float(np.sum(values * values)) <= 0.0:
        raise ValueError("field must have non-zero power")
    return values


def mode_probabilities(
    field: ArrayLike,
    basis: SpectralBasis = "fourier",
) -> NDArray[np.float64]:
    """Return normalized power fractions ``p_n`` of the field modes.

    Fourier modes are suitable for periodic maps and preserve translation
    invariance.  Cosine modes are suitable for non-periodic radial profiles.
    """
    values = _as_field(field)
    if basis == "fourier":
        modes = np.fft.fftn(values, norm="ortho")
    elif basis == "cosine":
        modes = dctn(values, norm="ortho")
    else:
        raise ValueError(f"unsupported spectral basis: {basis}")

    power = np.abs(modes).ravel() ** 2
    return np.asarray(power / np.sum(power), dtype=float)


def complexity_from_probabilities(probabilities: ArrayLike) -> ComplexityComponents:
    """Compute normalized entropy ``H``, disequilibrium ``Q``, and ``H*Q``."""
    probabilities_array = np.asarray(probabilities, dtype=float).ravel()
    if probabilities_array.size < 2:
        raise ValueError("at least two mode probabilities are required")
    if not np.isfinite(probabilities_array).all():
        raise ValueError("mode probabilities must be finite")
    if np.any(probabilities_array < -1.0e-15):
        raise ValueError("mode probabilities cannot be negative")

    probabilities_array = np.clip(probabilities_array, 0.0, None)
    total = float(np.sum(probabilities_array))
    if total <= 0.0:
        raise ValueError("mode probabilities must have a positive sum")
    probabilities_array /= total

    n_modes = probabilities_array.size
    nonzero = probabilities_array > 0.0
    entropy = -float(
        np.sum(probabilities_array[nonzero] * np.log(probabilities_array[nonzero]))
    ) / math.log(n_modes)
    uniform = 1.0 / n_modes
    disequilibrium = (
        n_modes
        / (n_modes - 1.0)
        * float(np.sum((probabilities_array - uniform) ** 2))
    )

    entropy = float(np.clip(entropy, 0.0, 1.0))
    disequilibrium = float(np.clip(disequilibrium, 0.0, 1.0))
    return ComplexityComponents(
        entropy=entropy,
        disequilibrium=disequilibrium,
        value=entropy * disequilibrium,
    )


def spectral_complexity(
    field: ArrayLike,
    basis: SpectralBasis = "fourier",
) -> ComplexityComponents:
    """Measure structural complexity at the field's native resolution."""
    return complexity_from_probabilities(mode_probabilities(field, basis=basis))


def multiscale_complexity(
    field: ArrayLike,
    scale_fractions: Sequence[float] = DEFAULT_SCALE_FRACTIONS,
    basis: SpectralBasis = "fourier",
) -> MultiscaleComplexity:
    """Average spectral complexity after smoothing over several scales.

    Scale values are fractions of the shortest non-singleton grid dimension,
    which makes the result less sensitive to the chosen pixel resolution.
    """
    values = _as_field(field)
    scales = tuple(float(scale) for scale in scale_fractions)
    if not scales:
        raise ValueError("at least one smoothing scale is required")
    if any(not math.isfinite(scale) or scale < 0.0 for scale in scales):
        raise ValueError("smoothing scale fractions must be finite and non-negative")

    non_singleton_sizes = [size for size in values.shape if size > 1]
    characteristic_size = float(min(non_singleton_sizes))
    boundary_mode = "wrap" if basis == "fourier" else "reflect"

    components: list[ComplexityComponents] = []
    for scale in scales:
        sigma = scale * characteristic_size
        smoothed = (
            values
            if sigma == 0.0
            else gaussian_filter(values, sigma=sigma, mode=boundary_mode)
        )
        components.append(spectral_complexity(smoothed, basis=basis))

    return MultiscaleComplexity(scales, tuple(components))


def resample_radial_profile(
    radius: ArrayLike,
    profile: ArrayLike,
    n_points: int = 128,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate an irregular radial profile onto a uniform radial grid."""
    radius_array = np.asarray(radius, dtype=float).ravel()
    profile_array = np.asarray(profile, dtype=float).ravel()
    if radius_array.size != profile_array.size or radius_array.size < 2:
        raise ValueError("radius and profile must have the same length >= 2")
    if n_points < 8:
        raise ValueError("n_points must be at least 8")
    if not np.isfinite(radius_array).all() or not np.isfinite(profile_array).all():
        raise ValueError("radius and profile must be finite")
    if np.any(radius_array < 0.0):
        raise ValueError("radius cannot be negative")

    order = np.argsort(radius_array)
    sorted_radius = radius_array[order]
    sorted_profile = profile_array[order]
    unique_radius, inverse = np.unique(sorted_radius, return_inverse=True)
    if unique_radius.size < 2:
        raise ValueError("radial profile needs at least two distinct radii")

    profile_sum = np.bincount(inverse, weights=sorted_profile)
    profile_count = np.bincount(inverse)
    unique_profile = profile_sum / profile_count
    grid = np.linspace(unique_radius[0], unique_radius[-1], n_points)
    return grid, np.interp(grid, unique_radius, unique_profile)


def radial_profile_complexity(
    radius: ArrayLike,
    profile: ArrayLike,
    n_points: int = 128,
    scale_fractions: Sequence[float] = DEFAULT_SCALE_FRACTIONS,
) -> MultiscaleComplexity:
    """Measure a non-periodic radial profile using cosine modes."""
    _, sampled_profile = resample_radial_profile(radius, profile, n_points=n_points)
    return multiscale_complexity(
        sampled_profile,
        scale_fractions=scale_fractions,
        basis="cosine",
    )


def stellar_radial_complexity(
    curve: _SurfaceBrightnessCurve,
    n_points: int = 128,
) -> MultiscaleComplexity:
    """Measure the 1-D stellar-light complexity available in SPARC rotmod."""
    disk = curve.surface_brightness_disk
    if disk is None:
        raise ValueError(f"{curve.name} has no disk surface-brightness profile")
    bulge = curve.surface_brightness_bulge
    bulge_values = np.zeros_like(disk) if bulge is None else bulge
    total_profile = np.clip(disk, 0.0, None) + np.clip(bulge_values, 0.0, None)
    return radial_profile_complexity(curve.radius_kpc, total_profile, n_points=n_points)


def _load_sparc_samples(archive_path: str | Path) -> list[_GalaxySample]:
    from . import sparc

    samples: list[_GalaxySample] = []
    for name in sparc.list_galaxies(archive_path):
        curve = sparc.load_galaxy(archive_path, name)
        complexity = stellar_radial_complexity(curve).value
        v_bar_squared = sparc.baryonic_velocity_squared(
            curve,
            sparc.FIDUCIAL_MASS_TO_LIGHT,
        )
        valid = (
            (curve.radius_kpc > 0.0)
            & (curve.v_obs_kms > 0.0)
            & (v_bar_squared > 0.0)
            & np.isfinite(v_bar_squared)
        )
        if int(np.sum(valid)) < sparc.MIN_VALID_ROTATION_POINTS:
            continue

        radius = curve.radius_kpc[valid]
        samples.append(
            _GalaxySample(
                name=name,
                complexity=complexity,
                log10_mass_proxy=math.log10(sparc.baryonic_mass_proxy(curve)),
                log10_g_bar=np.log10(v_bar_squared[valid] / radius),
                log10_g_obs=np.log10(curve.v_obs_kms[valid] ** 2 / radius),
                radius_fraction=radius / float(np.max(radius)),
            )
        )
    return samples


def _design_matrix(
    sample: _GalaxySample,
    include_complexity: bool,
    complexity: float,
) -> NDArray[np.float64]:
    x = sample.log10_g_bar
    columns = [
        x,
        x * x,
        x * x * x,
        np.full(x.size, sample.log10_mass_proxy),
        sample.radius_fraction,
    ]
    if include_complexity:
        columns.extend((np.full(x.size, complexity), complexity * x))
    return np.column_stack(columns)


def _cross_validated_rmse(
    samples: Sequence[_GalaxySample],
    folds: NDArray[np.int64],
    include_complexity: bool,
    complexities: NDArray[np.float64],
) -> float:
    galaxy_errors: list[float] = []
    for fold in range(int(np.max(folds)) + 1):
        training_indices = np.flatnonzero(folds != fold)
        test_indices = np.flatnonzero(folds == fold)

        training_design = np.vstack(
            [
                _design_matrix(samples[index], include_complexity, complexities[index])
                for index in training_indices
            ]
        )
        training_target = np.concatenate(
            [samples[index].log10_g_obs for index in training_indices]
        )
        weights = np.concatenate(
            [
                np.full(samples[index].log10_g_obs.size, 1.0 / samples[index].log10_g_obs.size)
                for index in training_indices
            ]
        )

        weight_sum = float(np.sum(weights))
        mean = np.sum(weights[:, None] * training_design, axis=0) / weight_sum
        variance = (
            np.sum(weights[:, None] * (training_design - mean) ** 2, axis=0)
            / weight_sum
        )
        scale = np.sqrt(variance)
        scale[scale < 1.0e-12] = 1.0
        standardized = (training_design - mean) / scale
        standardized = np.column_stack((np.ones(standardized.shape[0]), standardized))

        gram = standardized.T @ (weights[:, None] * standardized)
        ridge = np.eye(gram.shape[0]) * 1.0e-8
        ridge[0, 0] = 0.0
        coefficients = np.linalg.solve(
            gram + ridge,
            standardized.T @ (weights * training_target),
        )

        for index in test_indices:
            test_design = _design_matrix(
                samples[index],
                include_complexity,
                complexities[index],
            )
            test_design = (test_design - mean) / scale
            test_design = np.column_stack((np.ones(test_design.shape[0]), test_design))
            residual = test_design @ coefficients - samples[index].log10_g_obs
            galaxy_errors.append(float(np.sqrt(np.mean(residual * residual))))

    return float(np.mean(galaxy_errors))


def evaluate_sparc_complexity(
    archive_path: str | Path,
    n_folds: int = 5,
    n_permutations: int = 0,
    seed: int = 12345,
) -> SparcComplexityResult:
    """Test whether radial complexity improves held-out gravity prediction.

    The baseline predicts local observed acceleration from baryonic
    acceleration (cubic), a baryonic mass proxy, and normalized radius.  The
    extended model adds galaxy complexity and its interaction with baryonic
    acceleration.  Folds are grouped by galaxy, so no points from a test
    galaxy enter its training set.  A positive ``relative_improvement`` is the
    outcome expected if the available complexity proxy adds predictive power.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_permutations < 0:
        raise ValueError("n_permutations cannot be negative")

    samples = _load_sparc_samples(archive_path)
    if len(samples) < n_folds:
        raise ValueError("not enough valid galaxies for grouped cross-validation")

    complexities = np.asarray([sample.complexity for sample in samples], dtype=float)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    folds = np.empty(len(samples), dtype=np.int64)
    folds[order] = np.arange(len(samples)) % n_folds

    baseline_rmse = _cross_validated_rmse(
        samples,
        folds,
        include_complexity=False,
        complexities=complexities,
    )
    complexity_rmse = _cross_validated_rmse(
        samples,
        folds,
        include_complexity=True,
        complexities=complexities,
    )
    relative_improvement = (baseline_rmse - complexity_rmse) / baseline_rmse

    permutation_p_value: float | None = None
    if n_permutations:
        null_improvements: list[float] = []
        for _ in range(n_permutations):
            shuffled = rng.permutation(complexities)
            shuffled_rmse = _cross_validated_rmse(
                samples,
                folds,
                include_complexity=True,
                complexities=shuffled,
            )
            null_improvements.append((baseline_rmse - shuffled_rmse) / baseline_rmse)
        at_least_observed = sum(
            improvement >= relative_improvement for improvement in null_improvements
        )
        permutation_p_value = (at_least_observed + 1.0) / (n_permutations + 1.0)

    return SparcComplexityResult(
        n_galaxies=len(samples),
        complexity_min=float(np.min(complexities)),
        complexity_median=float(np.median(complexities)),
        complexity_max=float(np.max(complexities)),
        baseline_rmse_dex=baseline_rmse,
        complexity_rmse_dex=complexity_rmse,
        relative_improvement=relative_improvement,
        permutation_p_value=permutation_p_value,
    )
