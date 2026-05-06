import jax
import jax.numpy as jnp
import pytest

from src.distribution import PointMass, UniformDistribution
from src.sequence import VDCSchedule
from src.stepsize import (
    ConstantSchedule,
    EGScheduleFrom,
    EGStepsizeSpec,
    IIDSchedule,
    StepsizeSpec,
)
from test._helpers import assert_allclose

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using `field\\(init=False\\)`:UserWarning"
)


def test_stepsize_spec_requires_jax_scalar():
    with pytest.raises(Exception, match="failed typechecking"):
        StepsizeSpec(0.5)


def test_eg_stepsize_spec_from_stepsize_uses_equal_steps_by_default():
    spec = EGStepsizeSpec.from_stepsize(StepsizeSpec(jnp.array(0.5)))

    assert_allclose(spec.extrapolation_stepsize, 0.5)
    assert_allclose(spec.descent_stepsize, 0.5)


def test_eg_stepsize_spec_from_stepsize_applies_ratio():
    spec = EGStepsizeSpec.from_stepsize(
        StepsizeSpec(jnp.array(0.5)),
        ratio=4.0,
    )

    assert_allclose(spec.extrapolation_stepsize, 0.25)
    assert_allclose(spec.descent_stepsize, 1.0)


def test_constant_schedule_repeats_stepsize():
    specs = ConstantSchedule(0.125)(4)

    assert specs.stepsize.shape == (4,)
    assert_allclose(specs.stepsize, [0.125, 0.125, 0.125, 0.125])


def test_iid_schedule_samples_distribution_with_key():
    specs = IIDSchedule(PointMass(0.7))(3, key=jax.random.key(0))

    assert specs.stepsize.shape == (3,)
    assert_allclose(specs.stepsize, [0.7, 0.7, 0.7])


def test_eg_schedule_from_wraps_inner_schedule_with_ratio():
    specs = EGScheduleFrom(ConstantSchedule(0.5), ratio=4.0)(3)

    assert_allclose(specs.extrapolation_stepsize, [0.25, 0.25, 0.25])
    assert_allclose(specs.descent_stepsize, [1.0, 1.0, 1.0])


def test_vdc_schedule_maps_low_discrepancy_sequence_through_distribution():
    specs = VDCSchedule(UniformDistribution(-1.0, 1.0))(4)

    assert_allclose(specs.stepsize, [-1.0, 0.0, -0.5, 0.5])
