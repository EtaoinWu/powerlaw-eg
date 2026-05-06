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
    stepsize: Float[Scalar, ""]


class EGStepsizeSpec(tpx.TypedModule):
    extrapolation_stepsize: Float[Scalar, ""]
    descent_stepsize: Float[Scalar, ""]

    @classmethod
    def from_stepsize(
        cls, spec: StepsizeSpec, ratio: Float[Scalar, ""] | float | None = None
    ):
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
    """
    Generates a sequence of (potentially stochastic) stepsizes.
    """

    @abc.abstractmethod
    def __call__(
        self, n_iter: int, *, key: Key[Scalar, ""]
    ) -> Vmapped[T, " {n_iter}"]:
        pass


class DeterministicScheduleBase[T](ScheduleBase[T]):
    """
    Generates a sequence of deterministic stepsizes.
    """

    @abc.abstractmethod
    def __call__(
        self, n_iter: int, *, key: Key[Scalar, ""] | None = None
    ) -> Vmapped[T, " {n_iter}"]:
        pass


class ConstantSchedule(DeterministicScheduleBase[StepsizeSpec]):
    """
    A schedule that always uses the same stepsize.
    """

    stepsize: Float[Scalar, ""] = tpx.field(converter=C)

    def __call__(
        self, n_iter: int, *, key: Key[Scalar, ""] | None = None
    ) -> Vmapped[StepsizeSpec, " {n_iter}"]:
        return jax.vmap(lambda _: StepsizeSpec(self.stepsize))(
            jnp.arange(n_iter)
        )


class IIDSchedule(ScheduleBase[StepsizeSpec]):
    """
    A schedule that samples stepsizes i.i.d. from a distribution.
    """

    distribution: DistributionBase

    def __call__(
        self, n_iter: int, *, key: Key[Scalar, ""]
    ) -> Vmapped[StepsizeSpec, " {n_iter}"]:
        distribution: DistributionBase = self.distribution
        samples = distribution.sample(shape=(n_iter,), key=key)
        return jax.vmap(StepsizeSpec)(samples)


class EGScheduleFrom(ScheduleBase[EGStepsizeSpec]):
    """
    A schedule wrapper that uses same stepsizes for
    both steps in EG.
    """

    inner: ScheduleBase[StepsizeSpec]
    ratio: Float[Scalar, ""] | None = tpx.field(
        default=None, converter=lambda x: None if x is None else jnp.float_(x)
    )

    def __call__(
        self, n_iter: int, *, key: Key[Scalar, ""] | None = None
    ) -> Vmapped[EGStepsizeSpec, " {n_iter}"]:
        stepsizes = self.inner(n_iter, key=key)
        return jax.vmap(
            ft.partial(EGStepsizeSpec.from_stepsize, ratio=self.ratio)
        )(stepsizes)
