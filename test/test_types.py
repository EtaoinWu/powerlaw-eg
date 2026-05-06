import jax.numpy as jnp

from src.types import BiaffineGame, SinglePlayerGame
from test._helpers import assert_allclose


def test_single_player_game_autodiffs_value_function():
    game = SinglePlayerGame(
        lambda x: 0.5 * jnp.sum((x - jnp.array([1.0, -2.0])) ** 2)
    )
    strategies = (jnp.array([3.0, 1.0]),)

    assert game.n_player == 1
    assert_allclose(game.value(strategies)[0], 6.5)
    assert_allclose(game.grad(strategies)[0], [2.0, 3.0])
    values, grads = game.value_and_grad(strategies)
    assert_allclose(values[0], 6.5)
    assert_allclose(grads[0], [2.0, 3.0])


def test_single_player_game_uses_explicit_gradient_when_supplied():
    game = SinglePlayerGame(
        f=lambda x: jnp.sum(x**2),
        f_grad=lambda x: 3.0 * x,
    )

    assert_allclose(game.grad((jnp.array([2.0, -1.0]),))[0], [6.0, -3.0])


def test_biaffine_game_value_and_gradient_match_formula():
    game = BiaffineGame(
        bilinear=jnp.array([[2.0, -1.0], [0.5, 1.5]]),
        bias_x=jnp.array([0.25, -0.75]),
        bias_y=jnp.array([1.0, 0.5]),
        constant=jnp.array(-2.0),
    )
    x = jnp.array([1.0, -2.0])
    y = jnp.array([0.5, 3.0])

    loss = x @ game.bilinear @ y + game.bias_x @ x
    loss = loss + game.bias_y @ y + game.constant

    assert game.n_player == 2
    assert_allclose(game.value((x, y)), (loss, -loss))
    assert_allclose(game.grad((x, y))[0], game.bilinear @ y + game.bias_x)
    assert_allclose(game.grad((x, y))[1], -(game.bilinear.T @ x + game.bias_y))


def test_biaffine_game_from_nash_sets_stationary_profile_and_value():
    bilinear = jnp.array([[2.0, -1.0], [0.5, 1.5]])
    nash_x = jnp.array([0.75, -1.25])
    nash_y = jnp.array([1.5, -0.5])
    nash_value = jnp.array(3.0)

    game = BiaffineGame.from_Nash(
        bilinear=bilinear,
        nash_x=nash_x,
        nash_y=nash_y,
        nash_value=nash_value,
    )

    assert_allclose(game.grad((nash_x, nash_y))[0], [0.0, 0.0])
    assert_allclose(game.grad((nash_x, nash_y))[1], [0.0, 0.0])
    assert_allclose(game.value((nash_x, nash_y)), (3.0, -3.0))
