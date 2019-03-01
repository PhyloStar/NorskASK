import argparse
import os
import tempfile
from typing import Iterable, Sequence

from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from masterthesis.features.build_features import (
    bag_of_words, filename_iter, iterate_mixed_pos_docs, iterate_pos_docs
)
from masterthesis.models.callbacks import F1Metrics
from masterthesis.models.report import multi_task_report, report
from masterthesis.results import save_results
from masterthesis.utils import (
    DATA_DIR, get_file_name, load_split, REPRESENTATION_LAYER, safe_plt as plt
)


conll_folder = DATA_DIR / 'conll'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('featuretype', choices={'pos', 'word', 'char', 'mixed'})
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max-features', type=int, default=10000)
    parser.add_argument('--multi', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--round-cefr', action='store_true')
    return parser.parse_args()


def build_model(vocab_size: int, num_classes: Sequence[int]):
    input_ = Input((vocab_size,))
    hidden_1 = Dense(256, activation='relu')(input_)
    dropout_1 = Dropout(0.5)(hidden_1)
    hidden_2 = Dense(256, activation='relu', name=REPRESENTATION_LAYER)(dropout_1)
    dropout_2 = Dropout(0.5)(hidden_2)
    outputs = [Dense(n_c, activation='softmax')(dropout_2) for n_c in num_classes]
    return Model(inputs=[input_], outputs=outputs)


def pos_line_iter(split) -> Iterable[str]:
    for doc in iterate_pos_docs(split):
        yield ' '.join(doc)


def mixed_pos_line_iter(split) -> Iterable[str]:
    for doc in iterate_mixed_pos_docs(split):
        yield ' '.join(doc)


def preprocess(kind: str, max_features: int, train_meta, dev_meta):
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
            'train', analyzer='char', ngram_range=(2, 4), max_features=max_features)
        dev_x = vectorizer.transform(filename_iter(dev_meta))
        num_features = len(vectorizer.vocabulary_)
    elif kind == 'word':
        train_x, vectorizer = bag_of_words('train', max_features=max_features)
        dev_x = vectorizer.transform(filename_iter(dev_meta))
        num_features = len(vectorizer.vocabulary_)
    else:
        raise ValueError('Feature type "%s" is not supported' % kind)
    return train_x, dev_x, num_features


def main():
    args = parse_args()
    train_meta = load_split('train', round_cefr=args.round_cefr)
    dev_meta = load_split('dev', round_cefr=args.round_cefr)

    kind = args.featuretype
    train_x, dev_x, num_features = preprocess(kind, args.max_features, train_meta, dev_meta)

    cefr_labels = sorted(train_meta.cefr.unique())
    train_y = [to_categorical([cefr_labels.index(c) for c in train_meta.cefr])]
    dev_y = [to_categorical([cefr_labels.index(c) for c in dev_meta.cefr])]
    num_classes = [len(cefr_labels)]

    if args.multi:
        lang_labels = sorted(train_meta.lang.unique())
        train_y.append(to_categorical([lang_labels.index(l) for l in train_meta.lang]))
        dev_y.append(to_categorical([lang_labels.index(l) for l in dev_meta.lang]))
        num_classes.append(len(lang_labels))

    print(num_classes)

    model = build_model(num_features, num_classes)
    model.summary()
    model.compile(
        optimizer=Adam(lr=args.lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # Context manager fails on Windows (can't open an open file again)
    temp_handle, weights_path = tempfile.mkstemp(suffix='.h5')
    callbacks = [F1Metrics(dev_x, dev_y, weights_path)]
    history = model.fit(
        train_x, train_y, epochs=args.epochs, callbacks=callbacks, validation_data=(dev_x, dev_y),
        verbose=2)
    model.load_weights(weights_path)
    os.close(temp_handle)
    os.remove(weights_path)

    if args.multi:
        predictions = model.predict(dev_x)[0]
        true = np.argmax(dev_y[0], axis=1)
    else:
        predictions = model.predict(dev_x)
        true = np.argmax(dev_y, axis=1)

    pred = np.argmax(predictions, axis=1)

    if args.multi:
        multi_task_report(history.history, true, pred, cefr_labels)
    else:
        report(true, pred, cefr_labels)
    plt.show()

    if not args.nosave:
        prefix = 'mlp_%s' % args.featuretype
        fname = get_file_name(prefix)
        save_results(fname, args.__dict__, history.history, true, pred)


if __name__ == '__main__':
    main()
