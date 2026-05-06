import jax.random as jr
import numpy as np

from src.experiment_util import generate_matrix_svd


def test_generate_matrix_svd_supports_rectangular_shapes():
    matrix = generate_matrix_svd(
        jr.key(0),
        n=3,
        m=5,
        s_min=0.25,
        s_max=2.0,
    )

    singular_values = np.linalg.svd(np.asarray(matrix), compute_uv=False)

    assert matrix.shape == (3, 5)
    assert np.all(singular_values >= 0.25)
    assert np.all(singular_values <= 2.0)
