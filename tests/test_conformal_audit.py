import pytest

from pippa import conformal_audit


def test_fractional_field_dimension_matches_pippa_profile_exponent():
    audit = conformal_audit.audit_current_scale_limit()

    assert audit.field_scaling_dimension == pytest.approx(-0.13661977236758135)
    assert audit.field_scaling_dimension == pytest.approx(
        audit.pippa_profile_exponent,
        abs=1.0e-15,
    )


@pytest.mark.parametrize("dilation", [0.5, 2.0, 3.0])
def test_scale_free_fractional_action_passes_dilation_tests(dilation):
    audit = conformal_audit.audit_current_scale_limit(dilation)

    assert audit.kernel_scaling_error < 1.0e-12
    assert audit.kinetic_action_scaling_error < 1.0e-12
    assert audit.covariant_softening_scaling_error < 1.0e-12
    assert audit.massless_inverse_scaling_error < 1.0e-12
    assert audit.necessary_scale_tests_pass


def test_fixed_dimensionful_scales_break_dilation_covariance():
    audit = conformal_audit.audit_current_scale_limit(dilation=2.0)

    assert audit.fixed_softening_scaling_error > 1.0e-2
    assert audit.unit_mass_inverse_scaling_error > 1.0e-2


def test_intersector_mode_shift_breaks_inverse_propagator_homogeneity():
    residual = conformal_audit.inverse_propagator_scaling_error(
        momentum=1.0,
        dilation=2.0,
        mode_shift=0.4,
    )

    assert residual > 0.0
