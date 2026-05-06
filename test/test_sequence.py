from src.sequence import van_der_corput
from test._helpers import assert_allclose


def test_van_der_corput_starts_with_base_two_low_discrepancy_values():
    assert_allclose(
        van_der_corput(8),
        [0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875],
    )


def test_van_der_corput_empty_sequence_has_requested_shape():
    sequence = van_der_corput(0)
    assert sequence.shape == (0,)

    sequence = van_der_corput(10)
    assert sequence.shape == (10,)
