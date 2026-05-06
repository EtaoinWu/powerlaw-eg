import abc

import jax
import typinox as tpx
from beartype.typing import Callable, Self
from jaxtyping import Array, Float, Scalar
from typinox import Vmapped as Vmapped

from .utils import identity


class GameBase[Action](tpx.TypedModule):
    """Base interface for normal-form games over strategy profiles."""

    @property
    @abc.abstractmethod
    def n_player(self) -> int:
        pass

    @abc.abstractmethod
    def value(
        self, strategies: tuple[Action, ...]
    ) -> tuple[Float[Scalar, ""], ...]:
        """Evaluate player objectives at a strategy profile.

        Parameters
        ----------
        strategies
            One action per player.

        Returns
        -------
        tuple[Float[Scalar, ""], ...]
            Scalar objective value for each player.
        """
        pass


class DifferentiableGameBase[Action](GameBase[Action]):
    """Base interface for games whose the loss function is differentiable."""

    @abc.abstractmethod
    def grad(self, strategies: tuple[Action, ...]) -> tuple[Action, ...]:
        """Gradient over all players.

        Parameters
        ----------
        strategies
            One action per player.

        Returns
        -------
        (same pytree as `strategies`)
            Gradient of each player's objective with respect to their own strategy.
        """
        pass

    def value_and_grad(
        self, strategies: tuple[Action, ...]
    ) -> tuple[tuple[Float[Scalar, ""], ...], tuple[Action, ...]]:
        """Evaluate values and gradients at the same profile."""
        return self.value(strategies), self.grad(strategies)


class SinglePlayerGame[Action](DifferentiableGameBase[Action]):
    """Turn an optimization problem into a degenerate game with n=1."""

    f: Callable[[Action], Float[Scalar, ""]] = tpx.field(static=True)
    f_grad: Callable[[Action], Action] = tpx.field(
        default=identity, static=True
    )

    def __post_init__(self):
        if self.f_grad is identity:
            # ``identity`` is a sentinel for "no gradient supplied"; replace
            # it with autodiff after typinox construction.
            self.f_grad = jax.grad(self.f)

    @property
    def n_player(self) -> int:
        return 1

    def value(self, strategies: tuple[Action, ...]) -> tuple[Float[Scalar, ""]]:
        assert len(strategies) == 1
        return (self.f(strategies[0]),)

    def grad(self, strategies: tuple[Action, ...]) -> tuple[Action, ...]:
        assert len(strategies) == 1
        return (self.f_grad(strategies[0]),)


type Strategy = Float[Array, " _"]
type StrategyProfile = tuple[Strategy, ...]


class BiaffineGame(DifferentiableGameBase[Strategy]):
    """Two-player zero-sum biaffine game.

    The first player minimizes
        x^T A y + x^T p + q^T y + c
    and the second player maximizes the same scalar payoff.
    """

    bilinear: Float[Array, "n m"]
    bias_x: Float[Array, " n"]
    bias_y: Float[Array, " m"]
    constant: Float[Scalar, ""]

    @property
    def n_player(self) -> int:
        return 2

    def value(
        self, strategies: tuple[Strategy, ...]
    ) -> tuple[Float[Scalar, ""], ...]:
        assert len(strategies) == self.n_player
        x, y = strategies
        loss = (
            x @ self.bilinear @ y
            + self.bias_x @ x
            + self.bias_y @ y
            + self.constant
        )
        return loss, -loss

    def grad(self, strategies: tuple[Strategy, ...]) -> tuple[Strategy, ...]:
        assert len(strategies) == self.n_player
        x, y = strategies
        grad_x = self.bilinear @ y + self.bias_x
        grad_y = self.bilinear.T @ x + self.bias_y
        # Algorithms subtract returned directions for every player, so the
        # maximizer's gradient is negated to produce ascent in ``y``.
        return grad_x, -grad_y

    @classmethod
    def from_Nash(
        cls,
        bilinear: Float[Array, "n m"],
        nash_x: Float[Array, " n"],
        nash_y: Float[Array, " m"],
        nash_value: Float[Scalar, ""],
    ) -> Self:
        """Construct biases with a specific Nash equilibrium.

        Parameters
        ----------
        bilinear
            Bilinear payoff matrix.
        nash_x
            First player's target stationary strategy.
        nash_y
            Second player's target stationary strategy.
        nash_value
            Payoff value at the target profile.

        Returns
        -------
        BiaffineGame
            Game with zero gradient at ``(nash_x, nash_y)``.
        """
        return cls(
            bilinear=bilinear,
            bias_x=-bilinear @ nash_y,
            bias_y=-bilinear.T @ nash_x,
            constant=nash_value + nash_x @ bilinear @ nash_y,
        )
