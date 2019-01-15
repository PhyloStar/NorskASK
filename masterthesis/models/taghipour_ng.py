import argparse
import os
import tempfile
from collections import Counter

import numpy as np
from keras.layers import Input, Dense, Embedding, LSTM, Dropout
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import to_categorical

from masterthesis.features.build_features import iterate_tokens, iterate_docs
from masterthesis.utils import load_split, project_root
from masterthesis.models.callbacks import F1Metrics
from masterthesis.models.report import report
from masterthesis.models.layers import GlobalAveragePooling1D
from masterthesis.results import save_results


conll_folder = project_root / 'ASK/conll'

SEQ_LEN = 700  # 95th percentile of documents


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-cefr', action='store_true')
    parser.add_argument('--vocab-size', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--embed-dim', type=int, default=50)
    parser.add_argument('--rnn-dim', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--decay-rate', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout-rate', type=float, default=0.5)
    return parser.parse_args()


def build_model(vocab_size: int, sequence_len: int, num_classes: int,
                embed_dim: int, rnn_dim: int, dropout_rate: float):
    input_ = Input((sequence_len,))
    lookup = Embedding(vocab_size, embed_dim, mask_zero=True)(input_)
    lstm = LSTM(rnn_dim, return_sequences=True)(lookup)
    mean_over_time = GlobalAveragePooling1D()(lstm)
    dropout = Dropout(dropout_rate)(mean_over_time)
    output = Dense(num_classes, activation='softmax')(dropout)
    return Model(inputs=[input_], outputs=[output])


def preprocess(seq_len, train_meta, dev_meta, w2i):
    train_x = np.zeros((len(train_meta), seq_len), int)
    dev_x = np.zeros((len(dev_meta), seq_len), int)

    for row, doc in enumerate(iterate_docs('train')):
        for col, token in zip(range(seq_len), doc):
            if token not in w2i:
                token = '__UNK__'
            train_x[row, col] = w2i[token]

    for row, doc in enumerate(iterate_docs('dev')):
        for col, token in zip(range(seq_len), doc):
            if token not in w2i:
                token = '__UNK__'
            dev_x[row, col] = w2i[token]

    return train_x, dev_x


def make_w2i(vocab_size):
    tokens = Counter(iterate_tokens('train'))
    most_common = (token for (token, __) in tokens.most_common())

    w2i = {'__PAD__': 0, '__UNK__': 1}
    for rank, token in zip(range(2, vocab_size), most_common):
        w2i[token] = rank
    return w2i


def main():
    args = parse_args()
    train_meta = load_split('train', round_cefr=args.round_cefr)
    dev_meta = load_split('dev', round_cefr=args.round_cefr)

    vocab_size = args.vocab_size
    w2i = make_w2i(vocab_size)
    train_x, dev_x = preprocess(SEQ_LEN, train_meta, dev_meta, w2i)

    labels = sorted(train_meta.cefr.unique())

    train_y = to_categorical([labels.index(c) for c in train_meta.cefr])
    dev_y = to_categorical([labels.index(c) for c in dev_meta.cefr])

    model = build_model(
        vocab_size, SEQ_LEN, len(labels), args.embed_dim, args.rnn_dim, args.dropout_rate)
    model.summary()
    model.compile(
        optimizer=RMSprop(lr=args.lr, rho=args.decay_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # Context manager fails on Windows (can't open an open file again)
    temp_handle, weights_path = tempfile.mkstemp(suffix='.h5')
    callbacks = [F1Metrics(dev_x, dev_y, weights_path)]
    history = model.fit(
        train_x, train_y, epochs=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, validation_data=(dev_x, dev_y))
    model.load_weights(weights_path)
    os.close(temp_handle)
    os.remove(weights_path)

    predictions = model.predict(dev_x)

    true = np.argmax(dev_y, axis=1)
    pred = np.argmax(predictions, axis=1)
    report(true, pred, labels)
    save_results('taghipour_ng', args.__dict__, history.history, true, pred)


if __name__ == '__main__':
    main()
