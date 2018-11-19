from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from masterthesis.utils import heatmap
from masterthesis.utils import safe_plt as plt


def report(true, pred, labels):
    print(classification_report(true, pred, target_names=labels))
    print('Accuracy: %.3f' % accuracy_score(true, pred))
    print("== Confusion matrix ==")
    conf_matrix = confusion_matrix(true, pred)
    print(conf_matrix)
    heatmap(conf_matrix, labels, labels)
    plt.show()
