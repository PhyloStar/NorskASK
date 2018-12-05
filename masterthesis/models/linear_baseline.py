import argparse

import sklearn.linear_model

from masterthesis.results import save_results
from masterthesis.utils import load_split
from masterthesis.features.build_features import bag_of_words, filename_iter
from masterthesis.models.report import report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-cefr', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    train_x, vectorizer = bag_of_words('train')
    print("Number of features is %d" % len(vectorizer.vocabulary_))
    dev_meta = load_split('dev', round_cefr=args.round_cefr)
    dev_x = vectorizer.transform(filename_iter(dev_meta))

    train_meta = load_split('train', round_cefr=args.round_cefr)
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
