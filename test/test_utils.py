import jax.numpy as jnp

from src.utils import (
    C,
    Ci,
    identity,
    normalize,
    tree_concat,
    tree_stack,
    tree_unstack,
)
from test._helpers import assert_allclose


def test_identity_returns_same_object():
    value = object()

    assert identity(value) is value


def test_scalar_converters_create_jax_scalars():
    assert C(1.5).shape == ()
    assert Ci(2).shape == ()
    assert_allclose(C(1.5), 1.5)
    assert_allclose(Ci(2), 2)


def test_normalize_scales_by_sum():
    assert_allclose(normalize(jnp.array([2.0, 3.0, 5.0])), [0.2, 0.3, 0.5])


def test_tree_stack_and_unstack_are_inverse_for_matching_trees():
    trees = [
        (jnp.array([1.0, 2.0]), {"b": jnp.array([3.0])}),
        (jnp.array([4.0, 5.0]), {"b": jnp.array([6.0])}),
    ]

    stacked = tree_stack(trees)

    assert_allclose(stacked[0], [[1.0, 2.0], [4.0, 5.0]])
    assert_allclose(stacked[1]["b"], [[3.0], [6.0]])

    unstacked = list(tree_unstack(stacked))
    assert len(unstacked) == 2
    assert_allclose(unstacked[0][0], trees[0][0])
    assert_allclose(unstacked[0][1]["b"], trees[0][1]["b"])
    assert_allclose(unstacked[1][0], trees[1][0])
    assert_allclose(unstacked[1][1]["b"], trees[1][1]["b"])


def test_tree_concat_concatenates_corresponding_leaves():
    left = (jnp.array([1.0, 2.0]), {"b": jnp.array([[3.0]])})
    right = (jnp.array([4.0]), {"b": jnp.array([[5.0], [6.0]])})

    result = tree_concat(left, right)

    assert_allclose(result[0], [1.0, 2.0, 4.0])
    assert_allclose(result[1]["b"], [[3.0], [5.0], [6.0]])
