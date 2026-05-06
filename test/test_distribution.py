import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.distribution import (
    ArcsineDistribution,
    LinearTransform,
    Mixture,
    ParetoDistribution,
    PointMass,
    Reciprocal,
    UniformDistribution,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using `field\\(init=False\\)`:UserWarning"
)


def assert_allclose(actual, expected):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def test_point_mass_cdf_generate_and_sample():
    dist = PointMass(value=2.5)

    assert_allclose(dist.cdf(jnp.array(2.49)), 0.0)
    assert_allclose(dist.cdf(jnp.array(2.5)), 1.0)
    assert_allclose(dist.cdf(jnp.array(3.0)), 1.0)
    assert_allclose(dist.generate(jnp.array(0.0)), 2.5)
    assert_allclose(dist.generate(jnp.array(1.0)), 2.5)

    samples = dist.sample(jax.random.key(0), (2, 3))
    assert samples.shape == (2, 3)
    assert_allclose(samples, jnp.full((2, 3), 2.5))


def test_uniform_distribution_cdf_generate_and_vector_helper():
    dist = UniformDistribution(low=-2.0, high=6.0)

    assert_allclose(dist.cdf(jnp.array(-3.0)), 0.0)
    assert_allclose(dist.cdf(jnp.array(-2.0)), 0.0)
    assert_allclose(dist.cdf(jnp.array(2.0)), 0.5)
    assert_allclose(dist.cdf(jnp.array(6.0)), 1.0)
    assert_allclose(dist.cdf(jnp.array(7.0)), 1.0)
    assert_allclose(dist.generate(jnp.array(0.25)), 0.0)

    us = jnp.array([[0.0, 0.25], [0.5, 1.0]])
    assert_allclose(dist._generate(us), [[-2.0, 0.0], [2.0, 6.0]])


def test_pareto_distribution_cdf_and_generate():
    dist = ParetoDistribution(scale=2.0, shape=3.0)

    assert_allclose(dist.cdf(jnp.array(1.0)), 0.0)
    assert_allclose(dist.cdf(jnp.array(2.0)), 0.0)
    assert_allclose(dist.cdf(jnp.array(4.0)), 1.0 - (2.0 / 4.0) ** 3)
    assert_allclose(dist.generate(jnp.array(0.0)), 2.0)
    assert_allclose(dist.generate(jnp.array(0.5)), 2.0 * 0.5 ** (-1.0 / 3.0))


def test_arcsine_distribution_clips_cdf_and_generates_quantiles():
    dist = ArcsineDistribution()

    assert_allclose(dist.cdf(jnp.array(-1.0)), 0.0)
    assert_allclose(dist.cdf(jnp.array(0.25)), 1.0 / 3.0)
    assert_allclose(dist.cdf(jnp.array(0.5)), 0.5)
    assert_allclose(dist.cdf(jnp.array(2.0)), 1.0)
    assert_allclose(dist.generate(jnp.array(0.0)), 0.0)
    assert_allclose(dist.generate(jnp.array(0.5)), 0.5)
    assert_allclose(dist.generate(jnp.array(1.0)), 1.0)


def test_mixture_normalizes_weights_and_combines_cdfs():
    dist = Mixture(
        components=(PointMass(0.0), UniformDistribution(0.0, 2.0)),
        weights=[1.0, 3.0],
    )

    assert_allclose(dist.weights, [0.25, 0.75])
    assert_allclose(dist.acc_weights, [0.25, 1.0])
    assert_allclose(dist.cdf(jnp.array(-1.0)), 0.0)
    assert_allclose(dist.cdf(jnp.array(0.0)), 0.25)
    assert_allclose(dist.cdf(jnp.array(1.0)), 0.25 + 0.75 * 0.5)
    assert_allclose(dist.cdf(jnp.array(3.0)), 1.0)


def test_mixture_generate_uses_selected_component_local_uniform():
    dist = Mixture(
        components=(
            UniformDistribution(0.0, 10.0),
            UniformDistribution(100.0, 200.0),
        ),
        weights=[1.0, 3.0],
    )

    assert_allclose(dist.generate(jnp.array(0.0)), 0.0)
    assert_allclose(dist.generate(jnp.array(0.125)), 5.0)
    assert_allclose(dist.generate(jnp.array(0.25)), 100.0)
    assert_allclose(dist.generate(jnp.array(0.625)), 150.0)


def test_mixture_rejects_component_weight_length_mismatch():
    with pytest.raises(ValueError, match="Number of components 1"):
        Mixture(components=(PointMass(0.0),), weights=[1.0, 3.0])


def test_linear_transform_applies_to_cdf_and_generate():
    dist = LinearTransform(
        inner=UniformDistribution(0.0, 1.0),
        scale=4.0,
        shift=-1.0,
    )

    assert_allclose(dist.cdf(jnp.array(-2.0)), 0.0)
    assert_allclose(dist.cdf(jnp.array(1.0)), 0.5)
    assert_allclose(dist.cdf(jnp.array(4.0)), 1.0)
    assert_allclose(dist.generate(jnp.array(0.25)), 0.0)


def test_reciprocal_distribution_cdf_and_generate():
    dist = Reciprocal(inner=UniformDistribution(1.0, 2.0))

    assert_allclose(dist.cdf(jnp.array(0.25)), 0.0)
    assert_allclose(dist.cdf(jnp.array(0.5)), 0.0)
    assert_allclose(dist.cdf(jnp.array(2.0 / 3.0)), 0.5)
    assert_allclose(dist.cdf(jnp.array(1.0)), 1.0)
    assert_allclose(dist.generate(jnp.array(0.0)), 0.5)
    assert_allclose(dist.generate(jnp.array(0.5)), 2.0 / 3.0)
    assert_allclose(dist.generate(jnp.array(1.0)), 1.0)
