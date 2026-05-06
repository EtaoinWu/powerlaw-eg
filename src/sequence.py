import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, Key, Scalar, UInt32

from .distribution import DistributionBase
from .stepsize import DeterministicScheduleBase, IIDSchedule, StepsizeSpec
from .types import Vmapped
from .utils import typed


@typed
def _reverse_bits_uint32(n: Integer[Scalar, ""]) -> UInt32[Scalar, ""]:
    n_u32 = n.astype(jnp.uint32)
    n_u32 = ((n_u32 >> 1) & 0x55555555) | ((n_u32 & 0x55555555) << 1)
    n_u32 = ((n_u32 >> 2) & 0x33333333) | ((n_u32 & 0x33333333) << 2)
    n_u32 = ((n_u32 >> 4) & 0x0F0F0F0F) | ((n_u32 & 0x0F0F0F0F) << 4)
    n_u32 = ((n_u32 >> 8) & 0x00FF00FF) | ((n_u32 & 0x00FF00FF) << 8)
    n_u32 = ((n_u32 >> 16) & 0x0000FFFF) | ((n_u32 & 0x0000FFFF) << 16)
    return n_u32


POW2_32 = 4294967296.0


@typed
def van_der_corput(n: int) -> Float[Array, " {n}"]:
    indices = jnp.arange(n, dtype=jnp.uint32)
    reversed_bits = jax.vmap(_reverse_bits_uint32)(indices)
    return reversed_bits.astype(jnp.float32) / POW2_32


class VDCSchedule(IIDSchedule, DeterministicScheduleBase[StepsizeSpec]):
    def __call__(
        self, n_iter: int, *, key: Key[Scalar, ""] | None = None
    ) -> Vmapped[StepsizeSpec, " {n_iter}"]:
        distribution: DistributionBase = self.distribution
        u = van_der_corput(n_iter)
        stepsizes = jax.vmap(distribution.generate)(u)
        return jax.vmap(StepsizeSpec)(stepsizes)
