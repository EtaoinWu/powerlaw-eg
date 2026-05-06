import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Key, Scalar, ScalarLike

from .utils import typed


@typed
def random_orthogonal(
    key: Key[Scalar, ""], dim: int
) -> Float[Array, "{dim} {dim}"]:
    """Generate a random orthogonal matrix.

    Parameters
    ----------
    key
        JAX random key.
    dim
        Matrix dimension.

    Returns
    -------
    float[dim, dim]
        Orthogonal matrix sampled from Haar measure on ``O(dim)``.
    """
    Z = jr.normal(key, (dim, dim))
    Q, R = jnp.linalg.qr(Z)

    # QR fixes column signs by convention; undo that bias to get Haar measure.
    d = jnp.diagonal(R)
    ph = d / jnp.abs(d)
    return Q * ph[None, :]


@typed
def generate_matrix_svd(
    key: Key[Scalar, ""],
    n: int,
    m: int,
    s_min: Float[ScalarLike, ""],
    s_max: Float[ScalarLike, ""],
) -> Float[Array, "{n} {m}"]:
    """Generate a random matrix with singular values log-uniformly distributed
    in ``[s_min, s_max]``.

    Parameters
    ----------
    key
        JAX random key.
    n
        Number of rows.
    m
        Number of columns.
    s_min
        Lower endpoint for log-uniform singular values.
    s_max
        Upper endpoint for log-uniform singular values.

    Returns
    -------
    float[n, m]
        Matrix whose nonzero singular values lie in
        ``[s_min, s_max]``.
    """
    key_sigmas, key_u, key_v = jr.split(key, 3)

    rank = min(n, m)
    U = random_orthogonal(key_u, n)
    V = random_orthogonal(key_v, m)
    log_sigmas = jr.uniform(
        key_sigmas, (rank,), minval=jnp.log(s_min), maxval=jnp.log(s_max)
    )
    sigmas = jnp.exp(log_sigmas)

    U_ = U[:, :rank]
    V_ = V[:, :rank]

    return U_ @ jnp.diag(sigmas) @ V_.T


@typed
def generate_vector_radius(
    key: Key[Scalar, ""],
    n: int,
    radius: Float[ScalarLike, ""],
) -> Float[Array, " {n}"]:
    """Generate a vector uniformly on a sphere.

    Parameters
    ----------
    key
        JAX random key.
    n
        Vector dimension.
    radius
        Euclidean radius.

    Returns
    -------
    float[n]
        Length-``n`` vector with norm ``radius``.
    """
    v = jr.normal(key, (n,))
    v = v / jnp.linalg.norm(v) * radius
    return v


def serial_map(fn):
    """Map a function serially over leading axes using ``jax.lax.scan``.
    Similar to ``jax.vmap`` and ``jax.pmap``; only works with in_axis=0.

    Parameters
    ----------
    fn
        Function applied to one slice from each input.

    Returns
    -------
    Callable
        Function with the same calling convention as ``jax.vmap(fn)`` but
        sequential execution order.
    """

    def inner(none: None, xs):
        return None, fn(*xs)

    def outer(*xs):
        # ``scan`` avoids materializing all mapped calls at once, which is
        # useful for memory-heavy experiment sweeps.
        _, result = jax.lax.scan(inner, None, xs)
        return result

    return outer
