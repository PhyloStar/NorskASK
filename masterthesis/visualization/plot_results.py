import argparse
import logging
from pathlib import Path
import pickle

import numpy as np
from sklearn.metrics import confusion_matrix

from masterthesis.models.report import report
from masterthesis.results import Results  # noqa: F401
from masterthesis.utils import (
    AUX_OUTPUT_NAME,
    CEFR_LABELS,
    heatmap,
    LANG_LABELS,
    OUTPUT_NAME,
    ROUND_CEFR_LABELS,
    safe_plt as plt,
)

logger = logging.getLogger(__name__)
logging.basicConfig()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--nli", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    return parser.parse_args()


def print_config(config):
    col_width = max(len(k) for k in config.keys())
    fmt = "{:%d} {}" % col_width
    for key, val in config.items():
        print(fmt.format(key, val))


def multi_task_plot_history(history, ax1, ax2):
    xs = np.arange(len(history["loss"])) + 1

    ax1.plot(xs, history["loss"], label="train"),
    ax1.plot(xs, history["val_loss"], label="validation"),
    ax1.legend()
    ax1.set(xlabel="Epoch", ylabel="Loss")

    logger.warning(list(history.keys()))

    aux_acc = AUX_OUTPUT_NAME + "_acc"
    key_labels = [(aux_acc, "train L1"), ("val_" + aux_acc, "val. L1")]
    for key, label in key_labels:
        ax2.plot(xs, history[key], label=label),
    if OUTPUT_NAME + "_acc" in history:
        acc = OUTPUT_NAME + "_acc"
        key_labels = [(acc, "train CEFR"), ("val_" + acc, "val. CEFR")]
        for key, label in key_labels:
            ax2.plot(xs, history[key], label=label),
    elif "val_f1" in history:
        label = "val. CEFR F1"
        ax = ax2.twinx()
        ax.plot(xs, history["val_f1"], label=label, secondary_y=True)
        ax.set_ylabel("Macro F1")

    ax2.legend()
    ax2.set(xlabel="Epoch", ylabel="Accuracy")


def plot_history(history, ax1, ax2):
    xs = np.arange(len(history["loss"])) + 1
    ax1.plot(xs, history["loss"], label="training"),
    ax1.plot(xs, history["val_loss"], label="validation"),
    ax1.legend()
    ax1.set(xlabel="Epoch", ylabel="Loss")

    if "acc" in history:
        ax2.plot(xs, history["acc"], label="training"),
        ax2.plot(xs, history["val_acc"], label="validation")
        ax2.legend()
        ax2.set(xlabel="Epoch", ylabel="Accuracy")
    elif "val_f1" in history:
        ax2.plot(xs, history["val_f1"], label="validation", color="C1")
        ax2.legend()
        ax2.set(xlabel="Epoch", ylabel="F1 score")
    else:
        print("Available keys in history:")
        print(list(history.keys()))


def main():
    args = parse_args()

    results = pickle.load(args.results.open("rb"))  # type: Results

    history = results.history
    true = results.true
    pred = results.predictions

    print_config(results.config)

    if args.nli or results.config.get("nli", False):
        labels = LANG_LABELS
    elif max(true) > 4:
        labels = CEFR_LABELS
    else:
        labels = ROUND_CEFR_LABELS

    if history is None:
        report(true, pred, labels, normalize=False)
        heatmap_ax = plt.gca()
    else:
        fig, axes = plt.subplots(2, 2)
        fig.set_size_inches(5, 4)
        plt.tight_layout()
        ax1 = plt.subplot(223)
        ax2 = plt.subplot(221, sharex=ax1)
        if results.config.get("multi", False):
            multi_task_plot_history(history, ax1, ax2)
        else:
            plot_history(history, ax1, ax2)

        ax3 = plt.subplot(222)
        report(true, pred, labels, normalize=False, ax=ax3)
        ax3.set(ylabel="Gold class")
        heatmap_ax = plt.subplot(224)
    conf_matrix = confusion_matrix(true, pred)
    heatmap(conf_matrix, labels, labels, normalize=True, ax=heatmap_ax)
    heatmap_ax.set(xlabel="Predicted class", ylabel="Gold class")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
