import argparse
import tempfile
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

from masterthesis.features.build_features import bag_of_words, filename_iter
from masterthesis.utils import load_split, project_root, conll_reader
from masterthesis.models.callbacks import F1Metrics
from masterthesis.models.report import report


conll_folder = project_root / 'ASK/conll'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('featuretype', choices=['pos', 'word', 'char'])
    parser.add_argument('--round-cefr', action='store_true')
    parser.add_argument('--max-features', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=2e-4)
    return parser.parse_args()


def build_model(vocab_size: int, num_classes: int):
    input_ = Input((vocab_size,))
    hidden_1 = Dense(256, activation='relu')(input_)
    dropout_1 = Dropout(0.5)(hidden_1)
    hidden_2 = Dense(256, activation='relu')(dropout_1)
    dropout_2 = Dropout(0.5)(hidden_2)
    output = Dense(num_classes, activation='softmax')(dropout_2)
    return Model(inputs=[input_], outputs=[output])


def pos_line_iter(meta) -> Iterable[str]:
    for stem in meta.filename:
        file = (conll_folder / stem).with_suffix('.conll')
        sents = []
        for sent in conll_reader(file, ['UPOS'], tags=False):
            sents.append(' '.join(tag for (tag,) in sent))
        yield '\n'.join(sents)


def preprocess(kind: str, max_features: int, train_meta, dev_meta):
    if kind == 'pos':
        vectorizer = CountVectorizer(
            lowercase=False, token_pattern="[^\s]+",
            ngram_range=(2, 4), max_features=max_features)
        train_x = vectorizer.fit_transform(pos_line_iter(train_meta))
        dev_x = vectorizer.transform(pos_line_iter(dev_meta))
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

    labels = sorted(train_meta.cefr.unique())

    train_y = to_categorical([labels.index(c) for c in train_meta.cefr])
    dev_y = to_categorical([labels.index(c) for c in dev_meta.cefr])

    model = build_model(num_features, len(labels))
    model.summary()
    model.compile(
        optimizer=Adam(lr=args.lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    with tempfile.NamedTemporaryFile(suffix='.h5') as weights_path:
        callbacks = [F1Metrics(dev_x, dev_y, weights_path.name)]
        model.fit(train_x, train_y, epochs=20, callbacks=callbacks, validation_data=(dev_x, dev_y))
        model.load_weights(weights_path.name)

    predictions = model.predict(dev_x)

    true = np.argmax(dev_y, axis=1)
    pred = np.argmax(predictions, axis=1)
    report(true, pred, labels)


if __name__ == '__main__':
    main()
