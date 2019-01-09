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

    fig, axes = plt.subplots(2, 2)

    ax1 = plt.subplot(223)
    ax1.plot(history['loss'], label='training loss'),
    ax1.plot(history['val_loss'], label='validation loss'),
    ax1.legend()
    ax1.set(xlabel='Epoch', ylabel='Loss')

    ax2 = plt.subplot(221, sharex=ax1)
    ax2.plot(history['acc'], label='training accuracy'),
    ax2.plot(history['val_acc'], label='validation accuracy')
    ax2.legend()
    ax2.set(ylabel='Accuracy')

    ax3 = plt.subplot(122)
    report(true, pred, list('1234567'), ax=ax3)


if __name__ == '__main__':
    main()
