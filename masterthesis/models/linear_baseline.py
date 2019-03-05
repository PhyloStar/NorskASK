import argparse
from typing import Iterable, Optional

from sklearn.feature_extraction.text import CountVectorizer
import sklearn.linear_model

from masterthesis.features.build_features import (
    bag_of_words, filename_iter, iterate_mixed_pos_docs, iterate_pos_docs
)
from masterthesis.models.report import report
from masterthesis.results import save_results
from masterthesis.utils import DATA_DIR, load_split


conll_folder = DATA_DIR / 'conll'


def pos_line_iter(split) -> Iterable[str]:
    for doc in iterate_pos_docs(split):
        yield ' '.join(doc)


def mixed_pos_line_iter(split) -> Iterable[str]:
    for doc in iterate_mixed_pos_docs(split):
        yield ' '.join(doc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices={'word', 'char', 'pos', 'mixed'}, default='bow')
    parser.add_argument('--round-cefr', action='store_true')
    return parser.parse_args()


def preprocess(kind: str, max_features: Optional[int], train_meta, dev_meta):
    if kind == 'pos':
        vectorizer = CountVectorizer(
            lowercase=False, token_pattern=r"[^\s]+",
            ngram_range=(2, 4), max_features=max_features)
        train_x = vectorizer.fit_transform(pos_line_iter('train'))
        dev_x = vectorizer.transform(pos_line_iter('dev'))
        num_features = len(vectorizer.vocabulary_)
    elif kind == 'mixed':
        vectorizer = CountVectorizer(
            lowercase=False, token_pattern=r"[^\s]+",
            ngram_range=(1, 3), max_features=max_features)
        train_x = vectorizer.fit_transform(mixed_pos_line_iter('train'))
        dev_x = vectorizer.transform(mixed_pos_line_iter('dev'))
        num_features = len(vectorizer.vocabulary_)
    elif kind == 'char':
        train_x, vectorizer = bag_of_words(
            'train', analyzer='char', ngram_range=(1, 3), max_features=max_features)
        dev_x = vectorizer.transform(filename_iter(dev_meta))
        num_features = len(vectorizer.vocabulary_)
    elif kind == 'word':
        train_x, vectorizer = bag_of_words(
            'train', max_features=max_features,
            token_pattern=r"[^\s]+", lowercase=False)
        dev_x = vectorizer.transform(filename_iter(dev_meta))
        num_features = len(vectorizer.vocabulary_)
    else:
        raise ValueError('Feature type "%s" is not supported' % kind)
    print("Number of features is %d" % len(vectorizer.vocabulary_))
    return train_x, dev_x, num_features


def main():
    args = parse_args()
    train_meta = load_split('train', round_cefr=args.round_cefr)
    dev_meta = load_split('dev', round_cefr=args.round_cefr)

    train_x, dev_x, num_features = preprocess(args.kind, None, train_meta, dev_meta)

    labels = sorted(train_meta.cefr.unique())

    train_y = [labels.index(c) for c in train_meta.cefr]
    dev_y = [labels.index(c) for c in dev_meta.cefr]

    print("Fitting classifier ...")
    clf = sklearn.linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')
    clf.fit(train_x, train_y)

    predictions = clf.predict(dev_x)
    report(dev_y, predictions, labels)
    save_results('linear_baseline', None, None, dev_y, predictions)


if __name__ == '__main__':
    main()
