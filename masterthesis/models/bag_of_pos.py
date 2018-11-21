import argparse
import tempfile
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

from masterthesis.utils import load_split, project_root, conll_reader
from masterthesis.models.callbacks import F1Metrics
from masterthesis.models.report import report


conll_folder = project_root / 'conll'


def build_model(vocab_size: int):
    input_ = Input((vocab_size,))
    hidden = Dense(1000)(input_)
    output = Dense(7, activation='softmax')(hidden)
    return Model(inputs=[input_], outputs=[output])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round_cefr', action='store_true')
    parser.add_argument('--max_features', type=int, default=10000)
    return parser.parse_args()


def pos_line_iter(meta) -> Iterable[str]:
    for stem in meta.filename:
        file = (conll_folder / stem).with_suffix('.conll')
        for sent in conll_reader(file, 'UPOS', tags=False):
            yield ' '.join(tag for (tag,) in sent)


def main():
    args = parse_args()
    train_meta = load_split('train', round_cefr=args.round_cefr)
    dev_meta = load_split('dev', round_cefr=args.round_cefr)
    vectorizer = CountVectorizer(
        lowercase=False, token_pattern="[^\s]+",
        ngram_range=(2, 4), max_features=args.max_features)
    train_x = vectorizer.fit_transform(pos_line_iter(train_meta))
    dev_x = vectorizer.transform(pos_line_iter(dev_meta))

    labels = sorted(train_meta.cefr.unique())

    train_y = to_categorical([labels.index(c) for c in train_meta.cefr])
    dev_y = to_categorical([labels.index(c) for c in dev_meta.cefr])

    model = build_model(len(vectorizer.vocabulary_))
    model.summary()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

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
