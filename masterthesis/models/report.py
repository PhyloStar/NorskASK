from sklearn.metrics import classification_report, confusion_matrix, f1_score

from masterthesis.utils import heatmap


def report(true, pred, labels, normalize: bool = False, ax=None):
    print(classification_report(true, pred, target_names=labels))
    print('Macro F1: %.3f' % f1_score(true, pred, average='macro'))
    print('Micro F1: %.3f' % f1_score(true, pred, average='micro'))
    print('Weighted F1: %.3f' % f1_score(true, pred, average='weighted'))
    print("== Confusion matrix ==")
    conf_matrix = confusion_matrix(true, pred)
    print(conf_matrix)

    heatmap(conf_matrix, labels, labels, normalize=normalize, ax=ax)
