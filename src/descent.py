import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import typinox as tpx
from jaxtyping import Float, Integer, Key, Scalar, ScalarLike

from .stepsize import (
    DeterministicScheduleBase,
    EGStepsizeSpec,
    ScheduleBase,
    StepsizeSpec,
)
from .types import DifferentiableGameBase, StrategyProfile, Vmapped
from .utils import typed


@typed
def gradient_update(params, grads, stepsize: Float[ScalarLike, ""] | None = None):
    if stepsize is None:

        def _update(param, grad):
            return param - grad
    else:

        def _update(param, grad):
            return param - stepsize * grad

    return jt.map(_update, params, grads)


@typed
def anchoring(params, bases, alpha: Float[ScalarLike, ""]):
    def _anchor(param, base):
        return (1 - alpha) * param + alpha * base

    return jt.map(_anchor, params, bases)


@typed
def tree_norm(tree) -> Float[Scalar, ""]:
    leaves = jt.leaves(tree)
    return jnp.sqrt(
        sum(
            jnp.sum(jnp.square(leaf))
            for leaf in leaves
            if eqx.is_inexact_array(leaf)
        )
    )


class DescentOutput(tpx.TypedModule):
    trajectory: StrategyProfile
    gradient_norm: Float[Scalar, ""]


@typed
def gradient_descent(
    game: DifferentiableGameBase,
    init_params: StrategyProfile,
    schedule: ScheduleBase[StepsizeSpec]
    | DeterministicScheduleBase[StepsizeSpec],
    n_iter: int,
    key: Key[Scalar, ""] | None = None,
) -> Vmapped[DescentOutput, " {n_iter}"]:
    specs = schedule(n_iter, key=key)  # type: ignore

    @typed
    def step(
        params: StrategyProfile, spec: StepsizeSpec
    ) -> tuple[StrategyProfile, DescentOutput]:
        grad = game.grad(params)
        output = DescentOutput(
            trajectory=params,
            gradient_norm=tree_norm(grad),
        )

        new_params = gradient_update(params, grad, stepsize=spec.stepsize)
        return new_params, output

    _, outputs = jax.lax.scan(step, init_params, specs, length=n_iter)
    return outputs


@typed
def extragradient(
    game: DifferentiableGameBase,
    init_params: StrategyProfile,
    schedule: ScheduleBase[EGStepsizeSpec]
    | DeterministicScheduleBase[EGStepsizeSpec],
    n_iter: int,
    key: Key[Scalar, ""] | None = None,
) -> Vmapped[DescentOutput, " {n_iter}"]:
    specs = schedule(n_iter, key=key)  # type: ignore

    @typed
    def step(
        params: StrategyProfile, spec: EGStepsizeSpec
    ) -> tuple[StrategyProfile, DescentOutput]:
        grad = game.grad(params)
        output = DescentOutput(
            trajectory=params,
            gradient_norm=tree_norm(grad),
        )

        e_params = gradient_update(
            params, grad, stepsize=spec.extrapolation_stepsize
        )
        e_grad = game.grad(e_params)
        new_params = gradient_update(
            params, e_grad, stepsize=spec.descent_stepsize
        )
        return new_params, output

    _, outputs = jax.lax.scan(step, init_params, specs, length=n_iter)
    return outputs


@typed
def anchored_extragradient(
    game: DifferentiableGameBase,
    init_params: StrategyProfile,
    schedule: ScheduleBase[EGStepsizeSpec]
    | DeterministicScheduleBase[EGStepsizeSpec],
    n_iter: int,
    key: Key[Scalar, ""] | None = None,
) -> Vmapped[DescentOutput, " {n_iter}"]:
    specs = schedule(n_iter, key=key)  # type: ignore

    @typed
    def step(
        params: StrategyProfile,
        indexed_spec: tuple[Integer[Scalar, ""], EGStepsizeSpec],
    ) -> tuple[StrategyProfile, DescentOutput]:
        idx, spec = indexed_spec
        grad = game.grad(params)
        output = DescentOutput(
            trajectory=params,
            gradient_norm=tree_norm(grad),
        )

        basepoint = anchoring(params, init_params, alpha=1.0 / (idx + 2))
        e_params = gradient_update(
            basepoint, grad, stepsize=spec.extrapolation_stepsize
        )
        e_grad = game.grad(e_params)
        new_params = gradient_update(
            basepoint, e_grad, stepsize=spec.descent_stepsize
        )
        return new_params, output

    _, outputs = jax.lax.scan(
        step, init_params, (jnp.arange(n_iter), specs), length=n_iter
    )
    return outputs
