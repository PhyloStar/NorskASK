import argparse
from typing import Iterable, Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, LinearSVR

from masterthesis.features.build_features import (
    bag_of_words,
    filename_iter,
    iterate_mixed_pos_docs,
    iterate_pos_docs,
)
from masterthesis.models.report import report
from masterthesis.results import save_results
from masterthesis.utils import DATA_DIR, get_file_name, load_split

conll_folder = DATA_DIR / "conll"


def pos_line_iter(split) -> Iterable[str]:
    for doc in iterate_pos_docs(split):
        yield " ".join(doc)


def mixed_pos_line_iter(split) -> Iterable[str]:
    for doc in iterate_mixed_pos_docs(split):
        yield " ".join(doc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices={"bow", "char", "pos", "mix"}, default="bow")
    parser.add_argument(
        "--algorithm", choices={"logreg", "svc", "svr"}, default="logreg"
    )
    parser.add_argument("--round-cefr", action="store_true")
    parser.add_argument("--nli", action="store_true")
    parser.add_argument("--eval-on-test", action="store_true")
    return parser.parse_args()


def preprocess(
    kind: str,
    max_features: Optional[int],
    train_meta,
    test_meta,
    eval_on_test: bool = False,
):
    if eval_on_test:
        train_split = "train,dev"
        test_split = "test"
    else:
        train_split = "train"
        test_split = "dev"
    if kind == "pos":
        vectorizer = CountVectorizer(
            lowercase=False,
            token_pattern=r"[^\s]+",
            ngram_range=(2, 4),
            max_features=max_features,
        )
        train_x = vectorizer.fit_transform(pos_line_iter(train_split))
        test_x = vectorizer.transform(pos_line_iter(test_split))
        num_features = len(vectorizer.vocabulary_)
    elif kind == "mix":
        vectorizer = CountVectorizer(
            lowercase=False,
            token_pattern=r"[^\s]+",
            ngram_range=(1, 3),
            max_features=max_features,
        )
        train_x = vectorizer.fit_transform(mixed_pos_line_iter(train_split))
        test_x = vectorizer.transform(mixed_pos_line_iter(test_split))
        num_features = len(vectorizer.vocabulary_)
    elif kind == "char":
        train_x, vectorizer = bag_of_words(
            train_split, analyzer="char", ngram_range=(1, 3), max_features=max_features
        )
        test_x = vectorizer.transform(filename_iter(test_meta))
        num_features = len(vectorizer.vocabulary_)
    elif kind == "bow":
        train_x, vectorizer = bag_of_words(
            train_split,
            max_features=max_features,
            token_pattern=r"[^\s]+",
            lowercase=False,
        )
        test_x = vectorizer.transform(filename_iter(test_meta))
        num_features = len(vectorizer.vocabulary_)
    else:
        raise ValueError('Feature type "%s" is not supported' % kind)
    print("Number of features is %d" % len(vectorizer.vocabulary_))
    return train_x, test_x, num_features


def main():
    args = parse_args()
    if args.eval_on_test:
        train_meta = load_split("train,dev", round_cefr=args.round_cefr)
        test_meta = load_split("test", round_cefr=args.round_cefr)
    else:
        train_meta = load_split("train", round_cefr=args.round_cefr)
        test_meta = load_split("dev", round_cefr=args.round_cefr)

    train_x, test_x, num_features = preprocess(
        args.kind, None, train_meta, test_meta, eval_on_test=args.eval_on_test
    )
    print(train_x.shape)
    print(test_x.shape)

    labels = sorted(train_meta.cefr.unique())

    train_y = [labels.index(c) for c in train_meta.cefr]
    test_y = [labels.index(c) for c in test_meta.cefr]
    print(len(train_y))
    print(len(test_y))

    print("Fitting classifier ...")
    if args.algorithm == "logreg":
        clf = LogisticRegression(solver="lbfgs", multi_class="multinomial")
    elif args.algorithm == "svc":
        clf = LinearSVC()
    elif args.algorithm == "svr":
        clf = LinearSVR()

    clf.fit(train_x, train_y)

    predictions = clf.predict(test_x)
    if args.algorithm == "svr":
        predictions = np.clip(np.floor(predictions + 0.5), 0, max(train_y))

    report(test_y, predictions, labels)

    if args.nli:
        name = "linear_%s_nli" % args.algorithm
    else:
        name = "linear_" + args.algorithm
    name = get_file_name(name)
    save_results(name, args.__dict__, None, test_y, predictions)


if __name__ == "__main__":
    main()
