from sklearn.metrics import classification_report, confusion_matrix, f1_score

from masterthesis.utils import AUX_OUTPUT_NAME, heatmap, OUTPUT_NAME, safe_plt as plt


def report(true, pred, labels, normalize: bool = False, ax=None):
    print(classification_report(true, pred, target_names=labels))
    print('Macro F1: %.3f' % f1_score(true, pred, average='macro'))
    print('Micro F1: %.3f' % f1_score(true, pred, average='micro'))
    print('Weighted F1: %.3f' % f1_score(true, pred, average='weighted'))
    print("== Confusion matrix ==")
    conf_matrix = confusion_matrix(true, pred)
    print(conf_matrix)

    heatmap(conf_matrix, labels, labels, normalize=normalize, ax=ax)


def multi_task_report(history, true, pred, labels):
    fig, axes = plt.subplots(2, 2)
    ax1 = plt.subplot(223)
    ax2 = plt.subplot(221, sharex=ax1)
    xs = list(range(1, len(history['loss']) + 1))

    ax1.plot(xs, history['loss'], label='training loss'),
    ax1.plot(xs, history['val_loss'], label='validation loss'),
    ax1.legend()
    ax1.set(xlabel='Epoch', ylabel='Loss')

    acc = OUTPUT_NAME + '_acc'
    val_acc = 'val_' + acc

    aux_acc = AUX_OUTPUT_NAME + '_acc'
    aux_val_acc = 'val_' + aux_acc

    ax2.plot(xs, history[acc], label='CEFR train acc'),
    ax2.plot(xs, history[aux_acc], label='L1 train acc'),
    ax2.plot(xs, history[val_acc], label='CEFR val acc')
    ax2.plot(xs, history[aux_val_acc], label='L1 val acc')
    ax2.legend()
    ax2.set(ylabel='Accuracy')

    ax3 = plt.subplot(122)
    report(true, pred, labels, ax=ax3)
