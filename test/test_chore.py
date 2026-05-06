from jax import config

import src


def test_package_import_configures_jax_runtime():
    assert src._chore is not None
    assert config.jax_enable_x64 is True
    assert config.jax_numpy_rank_promotion == "warn"
