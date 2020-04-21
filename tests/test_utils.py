import numpy as np
from numpy.testing import assert_array_equal

from masterthesis.utils import rescale_regression_results, round_cefr_score


def test_round_cefr_score():
    assert round_cefr_score('A2') == 'A2'
    assert round_cefr_score('B1/B2') == 'B2'
    assert round_cefr_score('B2/C1') == 'C1'


def test_rescale_regression_results():
    num_class = 7
    y = np.array([1, 2, 6, 4, 0, 2, 1, 5, 1, 0, 4, 4, 6, 2, 5])
    norm_y = y / num_class
    assert_array_equal(rescale_regression_results(norm_y, num_class), y)
