from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from masterthesis.utils import heatmap
from masterthesis.utils import safe_plt as plt


def report(true, pred, labels):
    print(classification_report(true, pred, target_names=labels))
    print("Macro f1: %.3f" % f1_score(true, pred, average='macro'))
    print("Weighted f1: %.3f" % f1_score(true, pred, average='weighted'))
    print('Accuracy: %.3f' % accuracy_score(true, pred))
    print("== Confusion matrix ==")
    conf_matrix = confusion_matrix(true, pred)
    print(conf_matrix)

    if len(labels) == 7:
        print("=== Using collapsed labels ===")
        fig, axes = plt.subplots(1, 2)
        collapsed_true = (true + 1) // 2
        collapsed_pred = (pred + 1) // 2
        collapsed_labels = ['A2', 'B1', 'B2', 'C1']
        print(classification_report(collapsed_true, collapsed_pred, target_names=collapsed_labels))
        print("Macro f1: %.3f" % f1_score(collapsed_true, collapsed_pred, average='macro'))
        print("Weighted f1: %.3f" % f1_score(collapsed_true, collapsed_pred, average='weighted'))
        print('Accuracy: %.3f' % accuracy_score(collapsed_true, collapsed_pred))
        print("== Confusion matrix ==")
        collapsed_conf_matrix = confusion_matrix(collapsed_true, collapsed_pred)
        print(collapsed_conf_matrix)
        heatmap(collapsed_conf_matrix, collapsed_labels, collapsed_labels, axes[1])
        main_axis = axes[0]
    else:
        main_axis = plt.gca()
        
    heatmap(conf_matrix, labels, labels, main_axis)
    plt.show()
