from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np

from masterthesis.utils import heatmap, round_cefr_score
from masterthesis.utils import safe_plt as plt


def _collapse_array(arr):
    if not isinstance(arr, np.ndarray):
        raise ValueError("Needs a Numpy array to collapse labels like this")
    if not issubclass(arr.dtype.type, np.integer):
        raise ValueError("Array must be of integer type")
    return (arr + 1) // 2


def report(true, pred, labels):
    print(classification_report(true, pred, target_names=labels))
    print('Accuracy: %.3f' % accuracy_score(true, pred))
    print('Macro F1: %.3f' % f1_score(true, pred, average='macro'))
    print('Micro F1: %.3f' % f1_score(true, pred, average='micro'))
    print('Weighted F1: %.3f' % f1_score(true, pred, average='weighted'))
    print("== Confusion matrix ==")
    conf_matrix = confusion_matrix(true, pred)
    print(conf_matrix)

    if len(labels) == 7:
        print("=== Using collapsed labels ===")
        fig, axes = plt.subplots(1, 2)
        try:
            collapsed_true = _collapse_array(true)
            collapsed_pred = _collapse_array(pred)
        except ValueError:
            collapsed_true = [round_cefr_score(s) for s in true]
            collapsed_pred = [round_cefr_score(s) for s in pred]
        collapsed_labels = ['A2', 'B1', 'B2', 'C1']
        print(classification_report(collapsed_true, collapsed_pred, target_names=collapsed_labels))
        print('Accuracy: %.3f' % accuracy_score(collapsed_true, collapsed_pred))
        print("Macro f1: %.3f" % f1_score(collapsed_true, collapsed_pred, average='macro'))
        print("Micro f1: %.3f" % f1_score(collapsed_true, collapsed_pred, average='micro'))
        print("Weighted f1: %.3f" % f1_score(collapsed_true, collapsed_pred, average='weighted'))
        print("== Confusion matrix ==")
        collapsed_conf_matrix = confusion_matrix(collapsed_true, collapsed_pred)
        print(collapsed_conf_matrix)
        heatmap(collapsed_conf_matrix, collapsed_labels, collapsed_labels, axes[1])
        main_axis = axes[0]
    else:
        main_axis = plt.gca()

    heatmap(conf_matrix, labels, labels, main_axis)
    plt.show()
