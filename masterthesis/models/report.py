from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np

from masterthesis.utils import heatmap
from masterthesis.utils import safe_plt as plt


def _collapse_array(arr):
    if isinstance(arr, np.ndarray):
        return (arr + 1) // 2
    return [(s + 1) // 2 for s in arr]


def report(true, pred, labels, ax=None):
    print(classification_report(true, pred, target_names=labels))
    print('Macro F1: %.3f' % f1_score(true, pred, average='macro'))
    print('Micro F1: %.3f' % f1_score(true, pred, average='micro'))
    print('Weighted F1: %.3f' % f1_score(true, pred, average='weighted'))
    print("== Confusion matrix ==")
    conf_matrix = confusion_matrix(true, pred)
    print(conf_matrix)

    heatmap(conf_matrix, labels, labels, ax=ax)
    plt.show()
