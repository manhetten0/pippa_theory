"""Collider-readiness tests for the four-mode Pippa Higgs portal."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("numpy")

import numpy as np

from pippa import constants
from pippa import higgs_portal
from pippa import sector_spectrum


def test_portal_mass_spectrum_uses_all_four_z2x2_character_modes():
    bare_mass_squared = 900.0
    portal_coupling = 0.002
    mixings = (12.0, -5.0, 3.0)
    common = bare_mass_squared + 0.5 * portal_coupling * constants.EXP.higgs_vev_GeV**2
    expected_squared = common + sector_spectrum.analytic_mixing_eigenvalues(*mixings)

    masses = higgs_portal.higgs_portal_mode_masses(
        bare_mass_squared,
        *mixings,
        portal_coupling,
    )

    assert np.square(masses) == pytest.approx(expected_squared)


def test_higgs_decay_closes_at_the_pair_production_threshold():
    half_higgs_mass = constants.EXP.m_H_GeV / 2.0

    assert higgs_portal.complex_scalar_higgs_width(
        half_higgs_mass,
        0.01,
    ) == pytest.approx(0.0)
    assert higgs_portal.complex_scalar_higgs_width(
        half_higgs_mass + 1.0,
        0.01,
    ) == pytest.approx(0.0)
    assert higgs_portal.complex_scalar_higgs_width(
        half_higgs_mass - 1.0,
        0.01,
    ) > 0.0


def test_coupling_bound_saturates_the_atlas_branching_limit():
    light_modes = (1.0, 1.0, 1.0, 1.0)
    coupling_limit = higgs_portal.portal_coupling_upper_bound(light_modes)
    width = higgs_portal.total_invisible_width(light_modes, coupling_limit)
    branching = higgs_portal.invisible_branching_fraction(width)

    assert coupling_limit == pytest.approx(0.00356, rel=0.01)
    assert branching == pytest.approx(
        higgs_portal.ATLAS_HIGGS_INVISIBLE_BR_LIMIT,
        rel=1.0e-12,
    )


def test_current_pippa_portal_prediction_is_blocked_not_fitted():
    audit = higgs_portal.audit_higgs_portal_prediction()

    assert audit.status is higgs_portal.PortalAuditStatus.BLOCKED
    assert set(audit.missing_inputs) == {
        "portal_coupling",
        "four_physical_mode_masses",
        "mode_stability_and_detector_invisibility",
    }
    assert audit.invisible_branching_fraction is None


def test_free_parameter_point_is_only_a_bound_not_a_prediction():
    audit = higgs_portal.audit_higgs_portal_prediction(
        portal_coupling=0.001,
        mode_masses_gev=(10.0, 20.0, 30.0, 40.0),
        modes_are_stable_and_invisible=True,
        parameters_fixed_by_theory=False,
    )

    assert audit.status is higgs_portal.PortalAuditStatus.BOUND_ONLY
    assert audit.invisible_branching_fraction is not None


def test_identifying_portal_coupling_with_alpha_A_would_exclude_light_modes():
    audit = higgs_portal.audit_higgs_portal_prediction(
        portal_coupling=abs(constants.alpha_A),
        mode_masses_gev=(1.0, 1.0, 1.0, 1.0),
        modes_are_stable_and_invisible=True,
        parameters_fixed_by_theory=True,
    )

    assert audit.status is higgs_portal.PortalAuditStatus.EXCLUDED
    assert audit.invisible_branching_fraction is not None
    assert audit.invisible_branching_fraction > 0.9


def test_heavy_character_modes_have_no_invisible_higgs_width():
    heavy = constants.EXP.m_H_GeV
    audit = higgs_portal.audit_higgs_portal_prediction(
        portal_coupling=1.0,
        mode_masses_gev=(heavy, heavy, heavy, heavy),
        modes_are_stable_and_invisible=True,
        parameters_fixed_by_theory=True,
    )

    assert audit.status is higgs_portal.PortalAuditStatus.COMPATIBLE
    assert audit.invisible_width_gev == pytest.approx(0.0)
    assert audit.invisible_branching_fraction == pytest.approx(
        higgs_portal.SM_HIGGS_INVISIBLE_BRANCHING_FRACTION
    )
    assert math.isinf(higgs_portal.portal_coupling_upper_bound((heavy,) * 4))
