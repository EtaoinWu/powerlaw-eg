import jax.numpy as jnp
from beartype import beartype
from jax import (
    tree as jt,
)
from jaxtyping import Float, Integer, Scalar, ScalarLike, jaxtyped

from . import _chore as _chore

typed = jaxtyped(typechecker=beartype)


def identity[T](x: T) -> T:
    return x


@typed
def C(x: Float[ScalarLike, ""]) -> Float[Scalar, ""]:
    """Convert a scalar-like float to a JAX scalar."""
    return jnp.float_(x)


@typed
def Ci(x: Integer[ScalarLike, ""]) -> Integer[Scalar, ""]:
    """Convert a scalar-like integer to a JAX scalar."""
    return jnp.int_(x)


def normalize(x):
    """Normalize an array by its sum.

    Parameters
    ----------
    x
        Array-like weights.

    Returns
    -------
    Array
        ``x / sum(x)``.
    """
    return x / jnp.sum(x)


def tree_stack(trees):
    """Stack matching leaves from a sequence of PyTrees.

    Parameters
    ----------
    trees
        Sequence of PyTrees with identical structure.

    Returns
    -------
    PyTree
        PyTree whose leaves are stacked along a new leading axis.

    Notes
    -----
    For ``([a, b], c)`` and ``([a', b'], c')``, returns
    ``([stack(a, a'), stack(b, b')], stack(c, c'))``.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jt.flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    """The inverse of ``tree_stack``; unstack a PyTree in the leading axis.

    Parameters
    ----------
    tree
        PyTree whose leaves share a leading dimension.

    Yields
    ------
    PyTree
        One unbatched PyTree per leading-axis index.
    """
    leaves, treedef = jt.flatten(tree)
    n_trees = leaves[0].shape[0]
    for i in range(n_trees):
        new_leaves = [leaf[i] for leaf in leaves]
        yield treedef.unflatten(new_leaves)


def tree_concat(*args):
    """Concatenate matching leaves from PyTrees along axis zero.

    Parameters
    ----------
    *args
        PyTrees with identical structure and concatenable leaves.

    Returns
    -------
    PyTree
        PyTree with corresponding leaves concatenated along the leading axis.
    """

    def _concat(*args):
        return jnp.concatenate(args, axis=0)

    return jt.map(_concat, *args)
