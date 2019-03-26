from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import f1_score

from masterthesis.utils import rescale_regression_results


class F1Metrics(Callback):
    def __init__(self, dev_x, dev_y, weights_path, average='macro'):
        super().__init__()
        if isinstance(dev_y, list):
            self.dev_y = dev_y[0]
        else:
            self.dev_y = dev_y
        self.dev_x = dev_x
        self.weights_path = weights_path
        self.average = average
        if self.dev_y.ndim == 1:
            self.highest_class = self.dev_y.max()
        self.multi = None

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.val_f1s = []
        self.best_f1 = float('-inf')

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        self.multi = self.multi or len(self.model.outputs) > 1
        val_predict = self.model.predict(self.dev_x)
        if self.multi:
            assert isinstance(list, val_predict)
            val_predict = val_predict[0]
        if val_predict.shape[1] == 1:
            # Regression
            val_predict = rescale_regression_results(val_predict, self.highest_class).ravel()
        _val_f1 = f1_metric(self.dev_y, val_predict, self.average)
        self.val_f1s.append(_val_f1)
        logs['val_f1'] = _val_f1
        if _val_f1 > self.best_f1:
            print("Epoch %d: val_f1 improved from %f to %f: saving weights as %s" %
                  (epoch + 1, self.best_f1, _val_f1, self.weights_path))
            self.best_f1 = _val_f1
            self.model.save_weights(self.weights_path)
        else:
            print("Epoch %d: val_f1 did not improve (%f >= %f)" %
                  (epoch + 1, self.best_f1, _val_f1))


def f1_metric(gold, predicted, average='macro'):
    if gold.ndim == 2:
        gold = np.argmax(gold, axis=1)
    if predicted.ndim == 2:
        predicted = np.argmax(predicted, axis=1)
    return f1_score(gold, predicted, average=average)
