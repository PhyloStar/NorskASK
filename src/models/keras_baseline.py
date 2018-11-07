import argparse
from itertools import chain
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Embedding
from keras.layers import Input, Conv1D, Dropout, Dense, GlobalMaxPooling1D, Concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import load_train_and_dev, conll_reader, heatmap


def iter_all_tokens(train):
    for seq in iter_all_docs(train):
        for token in seq:
            yield token


def iter_all_docs(split):
    for filename in split.filename:
        filepath = Path('ASK/conll') / (filename + '.conll')
        cr = conll_reader(filepath)
        yield list(chain.from_iterable(cr))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('target_column', nargs='?', choices=['cefr', 'lang'], default='cefr')
    return parser.parse_args()


def build_model(vocab_size: int, sequence_length: int) -> Model:
    input_shape = (sequence_length,)
    input_layer = Input(shape=input_shape)
    embedding_layer = Embedding(vocab_size + 1, 50)(input_layer)
    pooled_feature_maps = []
    for kernel_size in [4, 5, 6]:
        conv_layer = Conv1D(
            filters=100, kernel_size=kernel_size, activation='relu')(embedding_layer)
        pooling = GlobalMaxPooling1D()(conv_layer)
        pooled_feature_maps.append(pooling)
    merged = Concatenate()(pooled_feature_maps)
    dropout_layer = Dropout(0.5)(merged)
    output_layer = Dense(7, activation='softmax')(dropout_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    args = parse_args()
    seq_length = 100
    train, dev = load_train_and_dev()

    y_column = args.target_column
    labels = sorted(train[y_column].unique())
    print(labels)

    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(iter_all_tokens(train))
    vocab_size = len(tokenizer.index_word)
    print('vocab size = %d' % vocab_size)

    train_seqs = tokenizer.texts_to_sequences(iter_all_docs(train))
    dev_seqs = tokenizer.texts_to_sequences(iter_all_docs(dev))

    train_x = pad_sequences(train_seqs, maxlen=seq_length)
    dev_x = pad_sequences(dev_seqs, maxlen=seq_length)

    train_y = to_categorical([labels.index(c) for c in train[y_column]])
    dev_y = to_categorical([labels.index(c) for c in dev[y_column]])

    print(train_x.shape)
    print(dev_x.shape)

    model = build_model(vocab_size, seq_length)
    model.summary()
    model.fit(
        train_x, train_y, epochs=20, batch_size=8,
        validation_data=(dev_x, dev_y), verbose=2)

    predictions = np.argmax(model.predict(dev_x), axis=1)
    gold = np.argmax(dev_y, axis=1)
    print(classification_report(gold, predictions, target_names=labels))
    print("== Confusion matrix ==")
    conf_matrix = confusion_matrix(gold, predictions)
    print(conf_matrix)
    heatmap(conf_matrix, labels, labels)
    plt.show()


if __name__ == '__main__':
    main()
