import argparse
import pickle
from pathlib import Path

from masterthesis.utils import CEFR_LABELS, ROUND_CEFR_LABELS, LANG_LABELS
from masterthesis.utils import safe_plt as plt
from masterthesis.models.report import report
from masterthesis.results import Results  # noqa: F401


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('--nli', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    return parser.parse_args()


def print_config(config):
    col_width = max(len(k) for k in config.keys())
    fmt = '{:%d} {}' % col_width
    for key, val in config.items():
        print(fmt.format(key, val))


def plot_history(history, ax1, ax2):
    ax1.plot(history['loss'], label='training loss'),
    ax1.plot(history['val_loss'], label='validation loss'),
    ax1.legend()
    ax1.set(xlabel='Epoch', ylabel='Loss')

    ax2.plot(history['acc'], label='training accuracy'),
    ax2.plot(history['val_acc'], label='validation accuracy')
    ax2.legend()
    ax2.set(ylabel='Accuracy')


def main():
    args = parse_args()

    results = pickle.load(args.results.open('rb'))  # type: Results

    history = results.history
    true = results.true
    pred = results.predictions

    print_config(results.config)

    fig, axes = plt.subplots(2, 2)
    ax1 = plt.subplot(223)
    ax2 = plt.subplot(221, sharex=ax1)
    plot_history(history, ax1, ax2)

    if args.nli:
        labels = LANG_LABELS
    elif max(true) > 4:
        labels = CEFR_LABELS
    else:
        labels = ROUND_CEFR_LABELS

    ax3 = plt.subplot(122)
    report(true, pred, labels, normalize=args.normalize, ax=ax3)


if __name__ == '__main__':
    main()
