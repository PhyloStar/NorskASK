import argparse
import os
import tempfile
from itertools import chain
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Embedding
from keras.layers import (
    Input, Conv1D, Dropout, Dense, GlobalMaxPooling1D, Concatenate, GlobalAveragePooling1D
)
from keras.utils import to_categorical

from masterthesis.features.build_features import words_to_sequences, make_w2i
from masterthesis.results import save_results
from masterthesis.models.report import report
from masterthesis.utils import load_train_and_dev, conll_reader
from masterthesis.models.callbacks import F1Metrics


def iter_all_tokens(train) -> Iterable[str]:
    """Yield all tokens"""
    for seq in iter_all_docs(train):
        for token in seq:
            yield token


def iter_all_docs(split: pd.DataFrame, column='UPOS') -> Iterable[List[str]]:
    """Iterate over all docs in the split.

    yields:
        Each document as a list of lists of tuples of the given column
    """
    for filename in split.filename:
        filepath = Path('ASK/conll') / (filename + '.conll')
        cr = conll_reader(filepath, [column], tags=True)
        # Only using a single column, extract the value
        tokens = (tup[0] for tup in chain.from_iterable(cr))
        yield list(tokens)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nli', action='store_true')
    parser.add_argument('--epochs', '-e', type=int)
    parser.add_argument('--batch-size', '-b', type=int)
    parser.add_argument('--doc-length', '-l', type=int)
    parser.add_argument('--vocab-size', '-s', type=int)
    parser.add_argument('--vectors', '-V', type=Path)
    parser.set_defaults(epochs=30, doc_length=700, vocab_size=4000, batch_size=32)
    return parser.parse_args()


def build_model(vocab_size: int, sequence_length: int, num_classes: int) -> Model:
    input_shape = (sequence_length,)
    input_layer = Input(shape=input_shape)
    embedding_layer = Embedding(vocab_size, 50)(input_layer)
    pooled_feature_maps = []
    for kernel_size in [4, 5, 6]:
        conv_layer = Conv1D(
            filters=100, kernel_size=kernel_size, activation='relu')(embedding_layer)
        pooled_feature_maps.extend([
            GlobalAveragePooling1D()(conv_layer),
            GlobalMaxPooling1D()(conv_layer)
        ])
    merged = Concatenate()(pooled_feature_maps)
    dropout_layer = Dropout(0.5)(merged)
    output_layer = Dense(num_classes, activation='softmax')(dropout_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    args = parse_args()
    seq_length = args.doc_length
    train, dev = load_train_and_dev()

    y_column = 'lang' if args.nli else 'cefr'
    labels = sorted(train[y_column].unique())

    w2i = make_w2i(args.vocab_size)
    train_x, dev_x = words_to_sequences(seq_length, ['train', 'dev'], w2i)

    train_y = to_categorical([labels.index(c) for c in train[y_column]])
    dev_y = to_categorical([labels.index(c) for c in dev[y_column]])

    model = build_model(args.vocab_size, seq_length, len(labels))
    model.summary()

    temp_handle, weights_path = tempfile.mkstemp(suffix='.h5')
    callbacks = [F1Metrics(dev_x, dev_y, weights_path)]
    history = model.fit(
        train_x, train_y, epochs=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, validation_data=(dev_x, dev_y),
        verbose=2)
    model.load_weights(weights_path)
    os.close(temp_handle)
    os.remove(weights_path)

    predictions = model.predict(dev_x)
    true = np.argmax(dev_y, axis=1)
    pred = np.argmax(predictions, axis=1)
    report(true, pred, labels)

    name = 'cnn'
    if args.nli:
        name = 'cnn-nli'
    save_results(name, args.__dict__, history.history, true, pred)


if __name__ == '__main__':
    main()
