import numpy as np


def assert_allclose(actual, expected, **kwargs):
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), **kwargs
    )
