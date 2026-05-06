"""Package-wide JAX runtime configuration."""

import warnings

from jax import config

# Keep test output stable across jaxtyping releases while retaining runtime
# type checks in the public modules.
warnings.filterwarnings("ignore", message="As of jaxtyping version 0.2.24")

config.update("jax_enable_x64", True)
config.update("jax_numpy_rank_promotion", "warn")
