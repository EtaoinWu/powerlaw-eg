import abc

import jax
import jax.numpy as jnp
import typinox as tpx
from jaxtyping import Array, Float, Key, Scalar

from .utils import C, normalize


class DistributionBase(tpx.TypedModule):
    """Scalar distribution sampled by inverse transform.

    Subclasses define a CDF and a quantile map from a scalar uniform
    variate. The base class adds JAX RNG handling and shape broadcasting.
    """

    @abc.abstractmethod
    def cdf(self, x: Float[Scalar, ""]) -> Float[Scalar, ""]:
        """Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : float
            Scalar evaluation point.

        Returns
        -------
        float
            Probability mass at or below ``x``.
        """
        pass

    @abc.abstractmethod
    def generate(self, u: Float[Scalar, ""]) -> Float[Scalar, ""]:
        """Map a unit-uniform variate to a sample.

        Parameters
        ----------
        u : float
            Scalar value in ``[0, 1]``.

        Returns
        -------
        float
            Quantile corresponding to ``u``.
        """
        pass

    def _generate(self, u: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
        """Broadcast ``generate`` over an arbitrary array shape.

        Parameters
        ----------
        u
            Array of unit-uniform variates.

        Returns
        -------
        Array
            Samples with the same shape as ``u``.
        """
        if u.ndim == 0:
            return self.generate(u)
        else:
            return jax.vmap(self._generate)(u)

    def sample(
        self, key: Key[Scalar, ""], shape: tuple[int, ...]
    ) -> Float[Array, " *"]:
        """Draw samples with JAX randomness.

        Parameters
        ----------
        key
            JAX random key.
        shape
            Desired sample shape.

        Returns
        -------
        Array of shape ``shape``
            Samples from the distribution.
        """
        u = jax.random.uniform(key, shape=shape)
        return self._generate(u)


class PointMass(DistributionBase):
    """Degenerate distribution concentrated at one scalar."""

    value: Float[Scalar, ""] = tpx.field(default=0.0, converter=C)

    def cdf(self, x: Float[Scalar, ""]) -> Float[Scalar, ""]:
        return jnp.where(x < self.value, 0.0, 1.0)

    def generate(self, u: Float[Scalar, ""]) -> Float[Scalar, ""]:
        return self.value


class UniformDistribution(DistributionBase):
    """Uniform distribution on a finite interval."""

    low: Float[Scalar, ""] = tpx.field(default=0.0, converter=C)
    high: Float[Scalar, ""] = tpx.field(default=1.0, converter=C)

    def cdf(self, x: Float[Scalar, ""]) -> Float[Scalar, ""]:
        return jnp.clip((x - self.low) / (self.high - self.low), 0.0, 1.0)

    def generate(self, u: Float[Scalar, ""]) -> Float[Scalar, ""]:
        return self.low + u * (self.high - self.low)


class ParetoDistribution(DistributionBase):
    """
    Pareto distribution with support ``[scale, inf)`` and power-law tail
    decay parameter ``shape``.
    """

    scale: Float[Scalar, ""] = tpx.field(default=1.0, converter=C)
    shape: Float[Scalar, ""] = tpx.field(default=1.0, converter=C)

    def cdf(self, x: Float[Scalar, ""]) -> Float[Scalar, ""]:
        return jnp.where(
            x < self.scale,
            0.0,
            1.0 - (self.scale / x) ** self.shape,
        )

    def generate(self, u: Float[Scalar, ""]) -> Float[Scalar, ""]:
        return self.scale * (1.0 - u) ** (-1.0 / self.shape)


class ArcsineDistribution(DistributionBase):
    """Arcsine distribution on ``[0, 1]``."""

    def cdf(self, x: Float[Scalar, ""]) -> Float[Scalar, ""]:
        x = jnp.clip(x, 0.0, 1.0)
        return (2.0 / jnp.pi) * jnp.arcsin(jnp.sqrt(x))

    def generate(self, u: Float[Scalar, ""]) -> Float[Scalar, ""]:
        return jnp.sin((jnp.pi / 2.0) * u) ** 2


class Mixture(DistributionBase):
    """Finite mixture of scalar distributions."""

    components: tuple[DistributionBase, ...]
    weights: Float[Array, " n_component"] = tpx.field(converter=jnp.array)
    acc_weights: Float[Array, " n_component"] = tpx.field(default=None)

    def __post_init__(self):
        # post init happens before converter
        self.weights = normalize(jnp.array(self.weights))
        self.acc_weights = jnp.cumsum(self.weights)

    def __validate__(self):
        if len(self.components) != self.weights.shape[0]:
            raise ValueError(
                f"Number of components {len(self.components)} does not match number of weights {self.weights.shape[0]}."
            )

    def cdf(self, x: Float[Scalar, ""]) -> Float[Scalar, ""]:
        return jnp.sum(
            self.weights * jnp.array([comp.cdf(x) for comp in self.components]),
            axis=0,
        )

    def generate(self, u: Float[Scalar, ""]) -> Float[Scalar, ""]:
        component = jnp.argmax(
            self.acc_weights > u,
            axis=0,
        )
        # Rescale global ``u`` to each component interval; only the selected
        # component value is returned, but all values stay JAX-traceable.
        us = (u - (self.acc_weights - self.weights)) / self.weights
        us = jnp.clip(us, 0.0, 1.0)
        xs = jnp.array([d.generate(u) for d, u in zip(self.components, us)])
        return xs[component]


class LinearTransform(DistributionBase):
    """Distribution of ``shift + scale * X``, where X ~ inner."""

    inner: DistributionBase
    scale: Float[Scalar, ""] = tpx.field(default=1.0, converter=C)
    shift: Float[Scalar, ""] = tpx.field(default=0.0, converter=C)

    def cdf(self, x: Float[Scalar, ""]) -> Float[Scalar, ""]:
        return self.inner.cdf((x - self.shift) / self.scale)

    def generate(self, u: Float[Scalar, ""]) -> Float[Scalar, ""]:
        return self.shift + self.scale * self.inner.generate(u)


class Reciprocal(DistributionBase):
    """Distribution of ``1 / X`` for positive-support ``X``."""

    inner: DistributionBase

    def cdf(self, x: Float[Scalar, ""]) -> Float[Scalar, ""]:
        return 1 - self.inner.cdf(1.0 / x)

    def generate(self, u: Float[Scalar, ""]) -> Float[Scalar, ""]:
        return 1.0 / self.inner.generate(1.0 - u)
