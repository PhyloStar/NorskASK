import argparse
from pathlib import Path
import pickle
from typing import Iterable, List

import numpy as np
from sklearn.metrics import classification_report

from masterthesis.utils import CEFR_LABELS, safe_plt as plt
from masterthesis.models.report import report


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    return parser.parse_args()


def get_majority_predictions(files: Iterable[Path]) -> List[int]:
    all_predictions = [pickle.load(f.open('rb')).predictions for f in files]
    num_models = len(all_predictions)

    num_examples = len(all_predictions[0])
    ensemble_predictions = []
    for i in range(num_examples):
        preds = [p[i] for p in all_predictions]
        avg_pred = np.round(np.sum(preds) / num_models)
        ensemble_predictions.append(int(avg_pred))
    return ensemble_predictions


def main():
    args = parse_args()

    true = pickle.load(args.files[0].open('rb')).true
    ensemble_predictions = get_majority_predictions(args.files)

    print(classification_report(true, ensemble_predictions, target_names=CEFR_LABELS))
    report(true, ensemble_predictions, CEFR_LABELS)
    plt.show()


if __name__ == '__main__':
    main()
