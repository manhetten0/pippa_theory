r"""Spectrum and stability of the minimal four-sector Pippa model.

The sectors are the elements of ``Z2 x Z2`` in the order
``N, Neg, Mir, NegMir``.  Translation-invariant mixing depends only on the
group difference between source and target sectors and is therefore fixed by
three real couplings.  The finite group Fourier transform diagonalizes it
exactly.

The module separates two notions of stability:

* the first-order-in-time Pippa/Schrodinger dynamics has real frequencies
  whenever its quadratic operator is Hermitian;
* a hypothetical relativistic completion has ``omega^2`` equal to the same
  spatial eigenvalues and is tachyon-free only when all of them are
  non-negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants


SECTOR_ORDER: tuple[str, ...] = ("N", "Neg", "Mir", "NegMir")
SECTOR_BITS: dict[str, tuple[int, int]] = {
    "N": (0, 0),
    "Neg": (1, 0),
    "Mir": (0, 1),
    "NegMir": (1, 1),
}
CHARACTER_MODES: tuple[str, ...] = (
    "symmetric",
    "Neg-character",
    "Mir-character",
    "NegMir-character",
)


def compose_sectors(left: str, right: str) -> str:
    """Return the ``Z2 x Z2`` product of two named sectors."""
    try:
        left_bits = SECTOR_BITS[left]
        right_bits = SECTOR_BITS[right]
    except KeyError as exc:
        raise ValueError(f"unknown sector: {exc.args[0]}") from exc
    result = (left_bits[0] ^ right_bits[0], left_bits[1] ^ right_bits[1])
    return next(name for name, bits in SECTOR_BITS.items() if bits == result)


def sector_translation_matrix(element: str) -> np.ndarray:
    r"""Return the regular-representation matrix ``(P_h psi)_g=psi_(g+h)``."""
    if element not in SECTOR_BITS:
        raise ValueError(f"unknown sector: {element}")
    matrix = np.zeros((4, 4), dtype=float)
    for target_index, target in enumerate(SECTOR_ORDER):
        source = compose_sectors(target, element)
        matrix[target_index, SECTOR_ORDER.index(source)] = 1.0
    return matrix


def z2x2_mixing_matrix(
    neg_coupling: float,
    mir_coupling: float,
    negmir_coupling: float,
) -> np.ndarray:
    r"""Return the most general real group-convolution mixing without self term."""
    return (
        neg_coupling * sector_translation_matrix("Neg")
        + mir_coupling * sector_translation_matrix("Mir")
        + negmir_coupling * sector_translation_matrix("NegMir")
    )


def sector_graph_laplacian(
    neg_weight: float,
    mir_weight: float,
    negmir_weight: float,
) -> np.ndarray:
    r"""Return the weighted Laplacian of the four-sector Cayley graph.

    Positive weights define a pairwise-difference energy.  The result is

    ``L = (w_Neg+w_Mir+w_NegMir) I
          - w_Neg P_Neg - w_Mir P_Mir - w_NegMir P_NegMir``.

    Every row sums to zero, so the common sector mode is protected by the
    simultaneous shift ``psi_g -> psi_g + constant``.
    """
    weights = (neg_weight, mir_weight, negmir_weight)
    if not all(np.isfinite(weight) for weight in weights):
        raise ValueError("graph weights must be finite")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("graph weights must be non-negative")
    degree = float(sum(weights))
    adjacency = z2x2_mixing_matrix(*weights)
    return degree * np.eye(4) - adjacency


def analytic_graph_laplacian_eigenvalues(
    neg_weight: float,
    mir_weight: float,
    negmir_weight: float,
) -> np.ndarray:
    """Return graph-Laplacian eigenvalues in ``CHARACTER_MODES`` order."""
    weights = (neg_weight, mir_weight, negmir_weight)
    if not all(np.isfinite(weight) for weight in weights):
        raise ValueError("graph weights must be finite")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("graph weights must be non-negative")
    return np.array(
        [
            0.0,
            2.0 * (neg_weight + negmir_weight),
            2.0 * (mir_weight + negmir_weight),
            2.0 * (neg_weight + mir_weight),
        ],
        dtype=float,
    )


def phase_locking_energy(
    phases: np.ndarray | list[float],
    neg_weight: float,
    mir_weight: float,
    negmir_weight: float,
) -> float:
    r"""Return ``sum_(g<h) w_gh [1-cos(theta_g-theta_h)]``.

    The periodic potential is exactly invariant under a common phase shift.
    Its Hessian at an aligned configuration is the sector graph Laplacian.
    """
    theta = np.asarray(phases, dtype=float)
    if theta.shape != (4,) or not np.all(np.isfinite(theta)):
        raise ValueError("phases must contain four finite values")
    laplacian = sector_graph_laplacian(
        neg_weight,
        mir_weight,
        negmir_weight,
    )
    adjacency = np.diag(np.diag(laplacian)) - laplacian
    energy = 0.0
    for left in range(4):
        for right in range(left + 1, 4):
            energy += adjacency[left, right] * (
                1.0 - np.cos(theta[left] - theta[right])
            )
    return float(energy)


def character_transform() -> np.ndarray:
    """Return the orthonormal finite Fourier transform of ``Z2 x Z2``."""
    return 0.5 * np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ]
    )


def analytic_mixing_eigenvalues(
    neg_coupling: float,
    mir_coupling: float,
    negmir_coupling: float,
) -> np.ndarray:
    """Return mixing eigenvalues in ``CHARACTER_MODES`` order."""
    a = float(neg_coupling)
    b = float(mir_coupling)
    c = float(negmir_coupling)
    return np.array(
        [
            a + b + c,
            -a + b - c,
            a - b - c,
            -a - b + c,
        ],
        dtype=float,
    )


def free_inverse(
    momentum: float | np.ndarray,
    mass_squared: float,
    kappa: float = 1.0,
    alpha: float = constants.D,
) -> float | np.ndarray:
    r"""Return ``q(k)=m^2+kappa |k|^alpha`` for the spatial operator."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    k = np.asarray(momentum, dtype=float)
    value = mass_squared + kappa * np.abs(k) ** alpha
    if value.ndim == 0:
        return float(value)
    return value


def quadratic_sector_operator(
    momentum: float,
    mass_squared: float,
    neg_coupling: float,
    mir_coupling: float,
    negmir_coupling: float,
    kappa: float = 1.0,
    alpha: float = constants.D,
) -> np.ndarray:
    """Return the four-sector quadratic operator at one momentum."""
    diagonal = free_inverse(momentum, mass_squared, kappa=kappa, alpha=alpha)
    return float(diagonal) * np.eye(4) + z2x2_mixing_matrix(
        neg_coupling,
        mir_coupling,
        negmir_coupling,
    )


def mode_spectrum(
    momentum: float | np.ndarray,
    mass_squared: float,
    neg_coupling: float,
    mir_coupling: float,
    negmir_coupling: float,
    kappa: float = 1.0,
    alpha: float = constants.D,
) -> np.ndarray:
    r"""Return mode energies, or ``omega^2`` for a relativistic completion."""
    q = np.atleast_1d(free_inverse(momentum, mass_squared, kappa, alpha))
    eigenvalues = analytic_mixing_eigenvalues(
        neg_coupling,
        mir_coupling,
        negmir_coupling,
    )
    spectrum = q[:, None] + eigenvalues[None, :]
    if np.asarray(momentum).ndim == 0:
        return spectrum[0]
    return spectrum


def identity_sector_links() -> np.ndarray:
    """Return the trivial link background between all sector gauge frames."""
    return np.ones((4, 4), dtype=np.complex128)


def gauge_transform_sector_fields(
    fields: np.ndarray | list[complex],
    phases: np.ndarray | list[float],
    charge: float = 1.0,
) -> np.ndarray:
    """Apply independent U(1) phases to the four sector amplitudes."""
    psi = np.asarray(fields, dtype=np.complex128)
    chi = np.asarray(phases, dtype=float)
    if psi.shape != (4,) or chi.shape != (4,):
        raise ValueError("fields and phases must contain four sector values")
    return np.exp(1j * charge * chi) * psi


def gauge_transform_sector_links(
    links: np.ndarray,
    phases: np.ndarray | list[float],
    charge: float = 1.0,
) -> np.ndarray:
    r"""Apply ``U_gh -> g_g U_gh g_h^*`` to sector bridge variables."""
    u = np.asarray(links, dtype=np.complex128)
    chi = np.asarray(phases, dtype=float)
    if u.shape != (4, 4) or chi.shape != (4,):
        raise ValueError("links must be 4x4 and phases must contain four values")
    phases_vector = np.exp(1j * charge * chi)
    return phases_vector[:, None] * u * np.conjugate(phases_vector[None, :])


def linked_mixing_matrix(
    neg_coupling: float,
    mir_coupling: float,
    negmir_coupling: float,
    links: np.ndarray,
) -> np.ndarray:
    """Dress the group-convolution weights with inter-sector gauge links."""
    u = np.asarray(links, dtype=np.complex128)
    if u.shape != (4, 4):
        raise ValueError("links must be a 4x4 matrix")
    return z2x2_mixing_matrix(
        neg_coupling,
        mir_coupling,
        negmir_coupling,
    ) * u


def quadratic_energy(fields: np.ndarray | list[complex], operator: np.ndarray) -> float:
    """Return ``psi^dagger Q psi`` for a Hermitian sector operator."""
    psi = np.asarray(fields, dtype=np.complex128)
    matrix = np.asarray(operator, dtype=np.complex128)
    if psi.shape != (4,) or matrix.shape != (4, 4):
        raise ValueError("fields must have length 4 and operator must be 4x4")
    if not np.allclose(matrix, np.conjugate(matrix.T), rtol=1e-12, atol=1e-12):
        raise ValueError("quadratic operator must be Hermitian")
    return float(np.real(np.vdot(psi, matrix @ psi)))


@dataclass(frozen=True)
class StabilityReport:
    """Stability conditions for the minimal four-sector model."""

    mixing_eigenvalues: tuple[float, float, float, float]
    zero_momentum_spectrum: tuple[float, float, float, float]
    minimum_relativistic_omega_squared: float
    schrodinger_unitary: bool
    relativistic_tachyon_free: bool
    quadratic_hamiltonian_bounded: bool
    full_hamiltonian_bounded: bool
    unstable_modes: tuple[str, ...]


@dataclass(frozen=True)
class TunedModeReport:
    """Spectrum after tuning one character mode to a target mass squared."""

    target_mode: str
    target_mass_squared: float
    bare_mass_squared: float
    mixing_eigenvalues: tuple[float, float, float, float]
    mode_masses_squared: tuple[float, float, float, float]
    light_modes: tuple[str, ...]
    target_is_unique_light_mode: bool
    relativistic_tachyon_free: bool
    cancellation_fraction: float


def tune_character_mode(
    neg_coupling: float,
    mir_coupling: float,
    negmir_coupling: float,
    target_mode: str,
    target_mass_squared: float,
    tolerance: float = 1.0e-12,
) -> TunedModeReport:
    """Choose ``m0^2`` so one character mode has a requested physical mass.

    Physical mode masses are evaluated as differences of mixing eigenvalues,
    ``m_chi^2 = m_target^2 + lambda_chi - lambda_target``.  This avoids losing
    an extremely small target mass to floating-point cancellation.
    """
    values = (
        neg_coupling,
        mir_coupling,
        negmir_coupling,
        target_mass_squared,
        tolerance,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("couplings, target mass and tolerance must be finite")
    if target_mode not in CHARACTER_MODES:
        raise ValueError(f"target_mode must be one of {CHARACTER_MODES}")
    if target_mass_squared < 0.0:
        raise ValueError("target_mass_squared must be non-negative")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    eigenvalues = analytic_mixing_eigenvalues(
        neg_coupling,
        mir_coupling,
        negmir_coupling,
    )
    target_index = CHARACTER_MODES.index(target_mode)
    target_eigenvalue = float(eigenvalues[target_index])
    bare_mass_squared = target_mass_squared - target_eigenvalue
    masses_squared = target_mass_squared + eigenvalues - target_eigenvalue

    mixing_scale = max(
        float(np.max(np.abs(eigenvalues))),
        np.finfo(float).tiny,
    )
    degeneracy_tolerance = tolerance * mixing_scale
    light_indices = np.flatnonzero(
        np.abs(eigenvalues - target_eigenvalue) <= degeneracy_tolerance
    )
    light_modes = tuple(CHARACTER_MODES[int(index)] for index in light_indices)
    stability_tolerance = tolerance * max(
        float(np.max(np.abs(masses_squared))),
        np.finfo(float).tiny,
    )
    tachyon_free = bool(np.min(masses_squared) >= -stability_tolerance)

    cancellation_scale = max(
        abs(target_eigenvalue),
        abs(bare_mass_squared),
        np.finfo(float).tiny,
    )
    cancellation_fraction = target_mass_squared / cancellation_scale

    return TunedModeReport(
        target_mode=target_mode,
        target_mass_squared=float(target_mass_squared),
        bare_mass_squared=float(bare_mass_squared),
        mixing_eigenvalues=tuple(float(value) for value in eigenvalues),
        mode_masses_squared=tuple(float(value) for value in masses_squared),
        light_modes=light_modes,
        target_is_unique_light_mode=(
            len(light_modes) == 1 and light_modes[0] == target_mode
        ),
        relativistic_tachyon_free=tachyon_free,
        cancellation_fraction=float(cancellation_fraction),
    )


def analyze_stability(
    mass_squared: float,
    neg_coupling: float,
    mir_coupling: float,
    negmir_coupling: float,
    kappa: float = 1.0,
    alpha: float = constants.D,
    self_coupling: float = 0.0,
    tolerance: float = 1.0e-12,
) -> StabilityReport:
    r"""Analyze boundedness and linear stability around the zero-field state.

    A positive quartic self-coupling bounds the amplitude even when a negative
    quadratic mode triggers symmetry breaking.  It does not make the origin
    tachyon-free.  A negative ``kappa`` is unstable at arbitrarily high
    momentum and cannot be repaired by a local quartic term.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    eigenvalues = analytic_mixing_eigenvalues(
        neg_coupling,
        mir_coupling,
        negmir_coupling,
    )
    zero_spectrum = mass_squared + eigenvalues

    if kappa < 0.0:
        minimum = float("-inf")
        tachyon_free = False
        quadratic_bounded = False
        full_bounded = False
        unstable_modes = CHARACTER_MODES
    else:
        minimum = float(np.min(zero_spectrum))
        unstable_modes = tuple(
            mode
            for mode, value in zip(CHARACTER_MODES, zero_spectrum, strict=True)
            if value < -tolerance
        )
        tachyon_free = not unstable_modes
        quadratic_bounded = minimum >= -tolerance
        full_bounded = self_coupling > 0.0 or (
            self_coupling >= 0.0 and quadratic_bounded
        )
        if self_coupling < 0.0:
            full_bounded = False

    return StabilityReport(
        mixing_eigenvalues=tuple(float(value) for value in eigenvalues),
        zero_momentum_spectrum=tuple(float(value) for value in zero_spectrum),
        minimum_relativistic_omega_squared=minimum,
        schrodinger_unitary=True,
        relativistic_tachyon_free=tachyon_free,
        quadratic_hamiltonian_bounded=quadratic_bounded,
        full_hamiltonian_bounded=full_bounded,
        unstable_modes=unstable_modes,
    )


def visible_propagator(
    inverse_free_value: float,
    neg_coupling: float,
    mir_coupling: float,
    negmir_coupling: float,
    pole_tolerance: float = 1.0e-12,
) -> float:
    r"""Return the exact visible component ``(Q^-1)_NN``.

    The visible basis vector has equal overlap ``1/2`` with all four
    character modes, hence ``G_NN = 1/4 sum_chi 1/(q+lambda_chi)``.
    """
    denominators = inverse_free_value + analytic_mixing_eigenvalues(
        neg_coupling,
        mir_coupling,
        negmir_coupling,
    )
    if np.any(np.abs(denominators) <= pole_tolerance):
        raise ValueError("visible propagator is evaluated at a sector-mode pole")
    return float(0.25 * np.sum(1.0 / denominators))


def visible_effective_inverse(
    inverse_free_value: float,
    neg_coupling: float,
    mir_coupling: float,
    negmir_coupling: float,
) -> float:
    """Return the exact inverse propagator after hidden sectors are removed."""
    propagator = visible_propagator(
        inverse_free_value,
        neg_coupling,
        mir_coupling,
        negmir_coupling,
    )
    if abs(propagator) <= 1.0e-15:
        raise ValueError("visible propagator vanishes")
    return 1.0 / propagator


def visible_self_energy(
    inverse_free_value: float,
    neg_coupling: float,
    mir_coupling: float,
    negmir_coupling: float,
) -> float:
    r"""Return ``Sigma=q-Q_eff`` generated by the three hidden sectors."""
    return inverse_free_value - visible_effective_inverse(
        inverse_free_value,
        neg_coupling,
        mir_coupling,
        negmir_coupling,
    )
