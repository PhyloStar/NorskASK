import keras.backend as K
import numpy as np
from numpy.testing import assert_array_equal
from pytest import approx

from masterthesis.models.utils import ranked_accuracy, ranked_prediction


def test_ranked_prediction():
    y = K.variable(np.array([[0.3, 0.2, 0.1],
                             [0.8, 0.2, 0.1],
                             [0.8, 0.7, 0.2],
                             [0.8, 0.7, 0.6],
                             [0.8, 0.2, 0.7]]))
    pred = K.eval(ranked_prediction(y))
    assert_array_equal(pred, np.array([0, 1, 2, 3, 1]))


def test_ranked_accuracy():
    y_pred = K.variable(np.array([[0.3, 0.2, 0.1],
                                  [0.8, 0.2, 0.1],
                                  [0.8, 0.7, 0.2],
                                  [0.8, 0.7, 0.6],
                                  [0.8, 0.2, 0.7]]))
    y_true = K.variable(np.array([[0, 0, 0],
                                  [1, 0, 0],
                                  [1, 1, 0],
                                  [1, 1, 1],
                                  [1, 0, 0]]))
    success_result = K.eval(ranked_accuracy(y_true, y_pred))
    assert success_result == 1

    y_pred = K.variable(np.array([[0.6, 0.6, 0.1],  # fail here
                                  [0.8, 0.2, 0.1],
                                  [0.8, 0.7, 0.2],
                                  [0.3, 0.7, 0.6],  # and here
                                  [0.8, 0.2, 0.7]]))
    y_true = K.variable(np.array([[0, 0, 0],
                                  [1, 0, 0],
                                  [1, 1, 0],
                                  [1, 1, 1],
                                  [1, 0, 0]]))
    partial_result = K.eval(ranked_accuracy(y_true, y_pred))
    assert partial_result == approx(3 / 5)

    y_pred = K.variable(np.array([[0.8, 0.8, 0.1],
                                  [0.8, 0.8, 0.1],
                                  [0.8, 0.3, 0.2],
                                  [0.8, 0.7, 0.3],
                                  [0.8, 0.7, 0.7]]))
    y_true = K.variable(np.array([[0, 0, 0],
                                  [1, 0, 0],
                                  [1, 1, 0],
                                  [1, 1, 1],
                                  [1, 0, 0]]))
    fail_result = K.eval(ranked_accuracy(y_true, y_pred))
    assert fail_result == 0
