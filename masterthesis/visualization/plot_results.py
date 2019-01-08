import pickle
import sys

from masterthesis.utils import safe_plt as plt
from masterthesis.models.report import report
from masterthesis.results import Results  # noqa: F401


def print_config(config):
    col_width = max(len(k) for k in config.keys())
    fmt = '{:%d} {}' % col_width
    for key, val in config.items():
        print(fmt.format(key, val))


def main():
    results_path = sys.argv[1]

    results = pickle.load(open(results_path, 'rb'))  # type: Results

    history = results.history
    true = results.true
    pred = results.predictions

    print_config(results.config)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1b = ax1.twinx()
    lines = [
        ax1.plot(history['loss'], 'b-', label='training loss'),
        ax1.plot(history['val_loss'], 'r-', label='validation loss'),
        ax1b.plot(history['acc'], 'c--', label='training accuracy'),
        ax1b.plot(history['val_acc'], 'm--', label='validation accuracy')
    ]
    lines = [line for l in lines for line in l]  # Flatten
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels)
    ax1.set(xlabel='Epoch', ylabel='Loss')
    ax1b.set(ylabel='Accuracy')

    report(true, pred, list('1234567'), ax=ax2)


if __name__ == '__main__':
    main()
