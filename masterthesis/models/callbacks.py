from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import f1_score


class F1Metrics(Callback):
    def __init__(self, dev_x, dev_y, weights_path, average='macro'):
        if isinstance(dev_y, list) and len(dev_y) > 1:
            self.dev_y = dev_y[0]
            self.multi = True
        else:
            self.dev_y = dev_y
            self.multi = False
        self.dev_x = dev_x
        self.weights_path = weights_path
        self.average = average

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.val_f1s = []
        self.best_f1 = float('-inf')

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        val_predict = self.model.predict(self.dev_x)
        if self.multi:
            val_predict = val_predict[0]
        _val_f1 = f1_metric(self.dev_y, val_predict, self.average)
        self.val_f1s.append(_val_f1)
        if _val_f1 > self.best_f1:
            print("Epoch %d: val_f1 improved from %f to %f: saving weights as %s" %
                  (epoch + 1, self.best_f1, _val_f1, self.weights_path))
            self.best_f1 = _val_f1
            self.model.save_weights(self.weights_path)
        else:
            print("Epoch %d: val_f1 did not improve (%f >= %f)" %
                  (epoch + 1, self.best_f1, _val_f1))


def f1_metric(gold, predicted, average='macro'):
    gold_id = np.argmax(gold, axis=1)
    pred_id = np.argmax(predicted, axis=1)

    return f1_score(gold_id, pred_id, average=average)
