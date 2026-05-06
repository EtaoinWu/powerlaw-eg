import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Key, Scalar, ScalarLike

from .utils import typed


@typed
def random_orthogonal(key: Key[Scalar, ""], dim: int) -> Float[Array, "{dim} {dim}"]:
    """
    Generate a random orthogonal matrix using the Haar measure.
    """
    Z = jr.normal(key, (dim, dim))
    Q, R = jnp.linalg.qr(Z)

    # Correct the signs of Q to ensure a uniform Haar measure over O(dim)
    d = jnp.diagonal(R)
    ph = d / jnp.abs(d)
    return Q * ph

@typed
def generate_matrix_svd(key: Key[Scalar, ""], n: int, m: int, s_min: Float[ScalarLike, ""], s_max: Float[ScalarLike, ""]) -> Float[Array, "{n} {m}"]:
    """
    Generate a random (n, m) matrix with singular values log-uniformly distributed between s_min and s_max.
    """
    key_sigmas, key_u, key_v = jr.split(key, 3)

    rank = min(n, m)
    U = random_orthogonal(key_u, n)
    V = random_orthogonal(key_v, m)
    log_sigmas = jr.uniform(key_sigmas, (rank,), minval=jnp.log(s_min), maxval=jnp.log(s_max))
    sigmas = jnp.exp(log_sigmas)

    U_ = U[:, :rank]
    V_ = V[:, :rank]

    return U @ jnp.diag(sigmas) @ V_.T

@typed
def generate_vector_radius(key: Key[Scalar, ""], n: int, radius: Float[ScalarLike, ""]) -> Float[Array, " {n}"]:
    """
    Generate a length-n vector uniformly on the sphere with the given radius.
    """
    v = jr.normal(key, (n,))
    v = v / jnp.linalg.norm(v) * radius
    return v

def serial_map(fn):
    def inner(none: None, xs):
        return None, fn(*xs)
    
    def outer(*xs):
        _, result = jax.lax.scan(inner, None, xs)
        return result
    
    return outer