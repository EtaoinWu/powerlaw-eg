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
def gradient_update(
    params, grads, stepsize: Float[ScalarLike, ""] | None = None
):
    """Apply a tree-structured gradient step.

    Parameters
    ----------
    params
        PyTree of parameters.
    grads
        PyTree of gradients with the same structure as ``params``.
    stepsize
        Optional scalar multiplier. If ``None``, a unit step is used.

    Returns
    -------
    PyTree
        Updated parameters ``params - stepsize * grads``.
    """
    if stepsize is None:

        def _update(param, grad):
            return param - grad
    else:

        def _update(param, grad):
            return param - stepsize * grad

    return jt.map(_update, params, grads)


@typed
def anchoring(params, bases, alpha: Float[ScalarLike, ""]):
    """Interpolate a parameter tree toward a base tree.

    Parameters
    ----------
    params
        Current parameter PyTree.
    bases
        Base parameter PyTree.
    alpha
        Weight on ``bases``.

    Returns
    -------
    PyTree
        Convex combination ``(1 - alpha) * params + alpha * bases``.
    """

    def _anchor(param, base):
        return (1 - alpha) * param + alpha * base

    return jt.map(_anchor, params, bases)


@typed
def tree_norm(tree) -> Float[Scalar, ""]:
    """Compute L2 norm over pytree.

    Parameters
    ----------
    tree
        PyTree of arrays.

    Returns
    -------
    Float[Scalar, ""]
        L2 norm of all the arrays concatenated.
    """
    leaves = jt.leaves(tree)
    return jnp.sqrt(
        sum(
            jnp.sum(jnp.square(leaf))
            for leaf in leaves
            if eqx.is_inexact_array(leaf)
        )
    )


class DescentOutput(tpx.TypedModule):
    """Per-iteration state recorded by descent algorithms."""

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
    """Run standard gradient descent.

    Parameters
    ----------
    game
        Differentiable game supplying per-player gradients.
    init_params
        Initial strategy profile.
    schedule
        Stepsize schedule.
    n_iter
        Number of iterations.
    key
        Optional JAX random key for stochastic schedules.

    Returns
    -------
    DescentOutput[n_iter]
        Pre-update trajectory and gradient norm at each iteration.
    """
    specs = schedule(n_iter, key=key)  # type: ignore

    @typed
    def step(
        params: StrategyProfile, spec: StepsizeSpec
    ) -> tuple[StrategyProfile, DescentOutput]:
        grad = game.grad(params)
        # Record pre-update state so trajectory[t] and gradient_norm[t]
        # refer to the same iterate.
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
    """Run the extragradient method.

    Parameters
    ----------
    game
        Differentiable game supplying per-player gradients.
    init_params
        Initial strategy profile.
    schedule
        Schedule of extrapolation and descent stepsizes.
    n_iter
        Number of iterations.
    key
        Optional JAX random key for stochastic schedules.

    Returns
    -------
    DescentOutput[n_iter]
        Pre-update trajectory and gradient norm at each iteration.
    """
    specs = schedule(n_iter, key=key)  # type: ignore

    @typed
    def step(
        params: StrategyProfile, spec: EGStepsizeSpec
    ) -> tuple[StrategyProfile, DescentOutput]:
        grad = game.grad(params)
        # Record at the current point before the extragradient correction.
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
def optimistic_gradient(
    game: DifferentiableGameBase,
    init_params: StrategyProfile,
    schedule: ScheduleBase[EGStepsizeSpec]
    | DeterministicScheduleBase[EGStepsizeSpec],
    n_iter: int,
    key: Key[Scalar, ""] | None = None,
) -> Vmapped[DescentOutput, " {n_iter}"]:
    """Run the optimistic gradient (OG) algorithm.

    Parameters
    ----------
    game
        Differentiable game supplying per-player gradients.
    init_params
        Initial strategy profile.
    schedule
        Schedule of extrapolation and descent stepsizes.
    n_iter
        Number of iterations.
    key
        Optional JAX random key for stochastic schedules.

    Returns
    -------
    DescentOutput[n_iter]
        Pre-update trajectory and gradient norm at each iteration.
    """
    specs = schedule(n_iter, key=key)  # type: ignore

    @typed
    def step(
        state: tuple[StrategyProfile, StrategyProfile], spec: EGStepsizeSpec
    ) -> tuple[tuple[StrategyProfile, StrategyProfile], DescentOutput]:
        params, last_grad = state
        output = DescentOutput(
            trajectory=params,
            gradient_norm=tree_norm(last_grad),
        )

        e_params = gradient_update(
            params, last_grad, stepsize=spec.extrapolation_stepsize
        )
        e_grad = game.grad(e_params)
        new_params = gradient_update(
            params, e_grad, stepsize=spec.descent_stepsize
        )
        new_state = (new_params, e_grad)
        return new_state, output

    _, outputs = jax.lax.scan(
        step, (init_params, game.grad(init_params)), specs, length=n_iter
    )
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
    """Run Extragradient with anchoring.
    See:
        Taeho Yoon and Ernest K. Ryu, “Accelerated algorithms for smooth
        convex-concave minimax problems with O(1/k^2) rate on squared
        gradient norm,” in Proceedings of the 38th International
        Conference on Machine Learning, July 2021, pp. 12098–12109.
        Available: https://proceedings.mlr.press/v139/yoon21d.html


    Parameters
    ----------
    game
        Differentiable game supplying per-player gradients.
    init_params
        Initial strategy profile and anchor point.
    schedule
        Schedule of extrapolation and descent stepsizes.
    n_iter
        Number of iterations.
    key
        Optional JAX random key for stochastic schedules.

    Returns
    -------
    DescentOutput[n_iter]
        Pre-update trajectory and gradient norm at each iteration.
    """
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

        # The first scan index is zero, so idx + 2 gives anchor weights
        # 1/2, 1/3, ...
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


@typed
def anchored_optimistic_gradient(
    game: DifferentiableGameBase,
    init_params: StrategyProfile,
    schedule: ScheduleBase[EGStepsizeSpec]
    | DeterministicScheduleBase[EGStepsizeSpec],
    n_iter: int,
    key: Key[Scalar, ""] | None = None,
) -> Vmapped[DescentOutput, " {n_iter}"]:
    """Run the anchored optimistic gradient (OG) algorithm.

    Parameters
    ----------
    game
        Differentiable game supplying per-player gradients.
    init_params
        Initial strategy profile and anchor point.
    schedule
        Schedule of extrapolation and descent stepsizes.
    n_iter
        Number of iterations.
    key
        Optional JAX random key for stochastic schedules.

    Returns
    -------
    DescentOutput[n_iter]
        Pre-update trajectory and gradient norm at each iteration.
    """
    specs = schedule(n_iter, key=key)  # type: ignore

    @typed
    def step(
        state: tuple[StrategyProfile, StrategyProfile],
        indexed_spec: tuple[Integer[Scalar, ""], EGStepsizeSpec],
    ) -> tuple[tuple[StrategyProfile, StrategyProfile], DescentOutput]:
        params, last_grad = state
        idx, spec = indexed_spec
        output = DescentOutput(
            trajectory=params,
            gradient_norm=tree_norm(last_grad),
        )
        basepoint = anchoring(params, init_params, alpha=1.0 / (idx + 2))

        e_params = gradient_update(
            basepoint, last_grad, stepsize=spec.extrapolation_stepsize
        )
        e_grad = game.grad(e_params)
        new_params = gradient_update(
            basepoint, e_grad, stepsize=spec.descent_stepsize
        )
        new_state = (new_params, e_grad)
        return new_state, output

    _, outputs = jax.lax.scan(
        step,
        (init_params, game.grad(init_params)),
        (jnp.arange(n_iter), specs),
        length=n_iter,
    )
    return outputs
