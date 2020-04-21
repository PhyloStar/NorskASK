import argparse
import logging
from pathlib import Path
import pickle

from keras.models import load_model, Model
import numpy as np

from masterthesis.features.build_features import words_to_sequences, pos_to_sequences
from masterthesis.models.report import report
from masterthesis.results import save_results
from masterthesis.utils import (
    CEFR_LABELS,
    load_split,
    ROUND_CEFR_LABELS,
    MODEL_DIR,
    rescale_regression_results,
)

logger = logging.getLogger(__name__)
logging.basicConfig()

pos2i_path = MODEL_DIR / "pos2i.pkl"

pos2i = pickle.load(pos2i_path.open("rb"))
(x_pos,) = pos_to_sequences(700, ["test"], pos2i)


def get_input_reps(w2i, multi_input: bool, split="test"):
    (x,) = words_to_sequences(700, [split], w2i)
    if multi_input:
        (x_pos,) = pos_to_sequences(700, [split], pos2i)
        x = [x, x_pos]
    return x


def load_model_and_w2i(model_path: Path):
    model = load_model(str(model_path))

    w2i_path = model_path.parent / (model_path.stem + "_w2i.pkl")
    w2i = pickle.load(w2i_path.open("rb"))
    return model, w2i


def get_predictions(model: Model, x, multi_output: bool) -> np.ndarray:
    """Return only the first set of predictions if multi-task setup."""
    predictions = model.predict(x)
    if multi_output:
        return predictions[0]
    return predictions


def find_model_paths(job_ids):
    model_globs = ["*%s*model.h5" % i for i in job_ids]
    return [f for g in model_globs for f in MODEL_DIR.glob(g)]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--collapsed", action="store_true")
    parser.add_argument("job_ids", type=str, nargs="+")
    parser.add_argument(
        "--debug", dest="loglevel", action="store_const", const=logging.DEBUG
    )
    parser.add_argument(
        "--quiet", dest="loglevel", action="store_const", const=logging.WARN
    )
    parser.set_defaults(loglevel=logging.INFO)
    args = parser.parse_args()
    logging.getLogger(None).setLevel(args.loglevel)
    return args


["26805083", "26805084"]


def main():
    args = parse_args()
    model_paths = find_model_paths(args.job_ids)

    print(model_paths)

    if args.collapsed:
        test_meta = load_split("test", round_cefr=True)
        round_target_scores = np.array(
            [ROUND_CEFR_LABELS.index(c) for c in test_meta["cefr"]], dtype=int
        )
        targets = round_target_scores
        highest_class = 3
        labels = ROUND_CEFR_LABELS
    else:
        test_meta = load_split("test", round_cefr=False)
        target_scores = np.array(
            [CEFR_LABELS.index(c) for c in test_meta["cefr"]], dtype=int
        )
        targets = target_scores
        highest_class = 6
        labels = CEFR_LABELS

    for model_path in model_paths:
        model, w2i = load_model_and_w2i(model_path)

        multi_input = isinstance(model.input, list) and len(model.input) == 2
        multi_output = isinstance(model.outputs, list) and len(model.outputs) > 1

        x = get_input_reps(w2i, multi_input)
        del w2i
        predictions = get_predictions(model, x, multi_output)
        del x
        del model

        # Round to integers and clip to score range
        pred = rescale_regression_results(predictions, highest_class).ravel()
        report(targets, pred, labels)

        name = model_path.stem + "_test_eval"
        save_results(name, {}, {}, targets, pred)


if __name__ == "__main__":
    main()
