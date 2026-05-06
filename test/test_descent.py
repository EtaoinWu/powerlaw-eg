import jax.numpy as jnp
import pytest

from src.descent import (
    anchored_extragradient,
    anchoring,
    extragradient,
    gradient_descent,
    gradient_update,
    tree_norm,
)
from src.stepsize import ConstantSchedule, EGScheduleFrom
from src.types import BiaffineGame, SinglePlayerGame
from test._helpers import assert_allclose


def test_gradient_update_subtracts_grads_with_optional_stepsize():
    params = (jnp.array([1.0, 2.0]), {"w": jnp.array([3.0])})
    grads = (jnp.array([0.5, -1.0]), {"w": jnp.array([2.0])})

    unit_step = gradient_update(params, grads)
    scaled_step = gradient_update(params, grads, stepsize=0.25)

    assert_allclose(unit_step[0], [0.5, 3.0])
    assert_allclose(unit_step[1]["w"], [1.0])
    assert_allclose(scaled_step[0], [0.875, 2.25])
    assert_allclose(scaled_step[1]["w"], [2.5])


def test_anchoring_interpolates_params_toward_base():
    params = (jnp.array([2.0, 4.0]),)
    bases = (jnp.array([-2.0, 0.0]),)

    assert_allclose(anchoring(params, bases, alpha=0.25)[0], [1.0, 3.0])


def test_tree_norm_uses_inexact_array_leaves():
    tree = (jnp.array([3.0, 4.0]), jnp.array([12]))

    assert_allclose(tree_norm(tree), 5.0)


@pytest.mark.parametrize(
    ("game", "init", "minimum", "stepsize", "n_iter"),
    [
        (
            SinglePlayerGame(
                lambda x: 0.5 * jnp.sum((x - jnp.array([1.5, -2.0])) ** 2)
            ),
            jnp.array([8.0, 5.0]),
            jnp.array([1.5, -2.0]),
            0.25,
            80,
        ),
        (
            SinglePlayerGame(
                lambda x: (
                    0.5
                    * (
                        4.0 * (x[0] + 1.0) ** 2
                        + 0.5 * (x[1] - 3.0) ** 2
                        + 2.0 * (x[2] - 0.25) ** 2
                    )
                )
            ),
            jnp.array([5.0, -4.0, 2.0]),
            jnp.array([-1.0, 3.0, 0.25]),
            0.2,
            120,
        ),
    ],
)
def test_gradient_descent_converges_to_minimum_for_convex_single_player_game(
    game,
    init,
    minimum,
    stepsize,
    n_iter,
):
    outputs = gradient_descent(
        game=game,
        init_params=(init,),
        schedule=ConstantSchedule(stepsize),
        n_iter=n_iter,
    )

    assert outputs.trajectory[0].shape == (n_iter, *init.shape)
    assert_allclose(outputs.trajectory[0][-1], minimum, atol=1e-4)
    assert_allclose(outputs.gradient_norm[-1], 0.0, atol=1e-4)


@pytest.mark.parametrize(
    ("bilinear", "nash_x", "nash_y", "init_params"),
    [
        (
            jnp.array([[2.0, -1.0], [0.5, 1.5]]),
            jnp.array([0.75, -1.25]),
            jnp.array([1.5, -0.5]),
            (jnp.array([4.0, -3.0]), jnp.array([-2.0, 2.5])),
        ),
        (
            jnp.array([[1.0, -0.2], [0.3, 1.4], [-0.6, 0.8]]),
            jnp.array([0.5, -1.0, 1.25]),
            jnp.array([-0.75, 1.5]),
            (jnp.array([2.0, -3.0, 0.25]), jnp.array([3.0, -1.0])),
        ),
    ],
)
def test_eg_and_anchored_eg_converge_on_same_biaffine_games(
    bilinear,
    nash_x,
    nash_y,
    init_params,
):
    game = BiaffineGame.from_Nash(
        bilinear=bilinear,
        nash_x=nash_x,
        nash_y=nash_y,
        nash_value=jnp.array(0.0),
    )
    schedule = EGScheduleFrom(ConstantSchedule(0.4))

    eg_outputs = extragradient(game, init_params, schedule, n_iter=1000)
    anchored_outputs = anchored_extragradient(
        game,
        init_params,
        schedule,
        n_iter=3000,
    )

    assert_allclose(eg_outputs.gradient_norm[-1], 0.0, atol=1e-5)
    assert_allclose(anchored_outputs.gradient_norm[-1], 0.0, atol=5e-3)
