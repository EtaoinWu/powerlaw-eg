import abc
import functools as ft

import jax
import jax.numpy as jnp
import typinox as tpx
from jaxtyping import Float, Key, Scalar

from .distribution import DistributionBase
from .types import Vmapped
from .utils import C


class StepsizeSpec(tpx.TypedModule):
    """Single stepsize used by one-step descent methods."""

    stepsize: Float[Scalar, ""]


class EGStepsizeSpec(tpx.TypedModule):
    """Pair of stepsizes used by extragradient methods."""

    extrapolation_stepsize: Float[Scalar, ""]
    descent_stepsize: Float[Scalar, ""]

    @classmethod
    def from_stepsize(
        cls, spec: StepsizeSpec, ratio: Float[Scalar, ""] | float | None = None
    ):
        """Build an extragradient pair from one base stepsize.

        Parameters
        ----------
        spec
            Base stepsize specification.
        ratio
            Desired ratio ``descent_stepsize / extrapolation_stepsize``.
            If ``None``, both EG steps use ``spec.stepsize``.

        Returns
        -------
        EGStepsizeSpec
            Extrapolation and descent stepsizes with product equal to
            ``spec.stepsize ** 2`` when ``ratio`` is provided.
        """
        if ratio is None:
            return cls(
                extrapolation_stepsize=spec.stepsize,
                descent_stepsize=spec.stepsize,
            )
        else:
            return cls(
                extrapolation_stepsize=spec.stepsize / jnp.sqrt(ratio),
                descent_stepsize=spec.stepsize * jnp.sqrt(ratio),
            )


class ScheduleBase[T](tpx.TypedModule):
    """Base class for (potentially stochastic) stepsize schedules.

    Generic parameters
    ------------------
    T
        Type of stepsize specification.
    """

    @abc.abstractmethod
    def __call__(
        self, n_iter: int, *, key: Key[Scalar, ""]
    ) -> Vmapped[T, " {n_iter}"]:
        """Generate ``n_iter`` stepsize specifications.

        Parameters
        ----------
        n_iter
            Number of iterations.
        key
            JAX random key for stochastic schedules.

        Returns
        -------
        T[n_iter]
            Batched schedule specifications.
        """
        pass


class DeterministicScheduleBase[T](ScheduleBase[T]):
    """Base class for schedules that do not use randomness."""

    @abc.abstractmethod
    def __call__(
        self, n_iter: int, *, key: Key[Scalar, ""] | None = None
    ) -> Vmapped[T, " {n_iter}"]:
        pass


class ConstantSchedule(DeterministicScheduleBase[StepsizeSpec]):
    """Schedule that repeats one scalar stepsize."""

    stepsize: Float[Scalar, ""] = tpx.field(converter=C)

    def __call__(
        self, n_iter: int, *, key: Key[Scalar, ""] | None = None
    ) -> Vmapped[StepsizeSpec, " {n_iter}"]:
        """Repeat ``stepsize`` for ``n_iter`` iterations."""
        return jax.vmap(lambda _: StepsizeSpec(self.stepsize))(
            jnp.arange(n_iter)
        )


class IIDSchedule(ScheduleBase[StepsizeSpec]):
    """Schedule that samples stepsizes i.i.d. from a distribution."""

    distribution: DistributionBase

    def __call__(
        self, n_iter: int, *, key: Key[Scalar, ""]
    ) -> Vmapped[StepsizeSpec, " {n_iter}"]:
        """Draw ``n_iter`` independent stepsize specifications."""
        distribution: DistributionBase = self.distribution
        samples = distribution.sample(shape=(n_iter,), key=key)
        return jax.vmap(StepsizeSpec)(samples)


class EGScheduleFrom(ScheduleBase[EGStepsizeSpec]):
    """Convert a one-step schedule into an extragradient schedule."""

    inner: ScheduleBase[StepsizeSpec]
    ratio: Float[Scalar, ""] | None = tpx.field(
        default=None, converter=lambda x: None if x is None else jnp.float_(x)
    )

    def __call__(
        self, n_iter: int, *, key: Key[Scalar, ""] | None = None
    ) -> Vmapped[EGStepsizeSpec, " {n_iter}"]:
        """Generate EG stepsize pairs from the wrapped schedule."""
        stepsizes = self.inner(n_iter, key=key)  # type: ignore[arg-type]
        return jax.vmap(
            ft.partial(EGStepsizeSpec.from_stepsize, ratio=self.ratio)
        )(stepsizes)
