import argparse
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from keras.layers import (
    Input, Dense, Embedding, LSTM, Dropout, Bidirectional, Activation,
    TimeDistributed, Lambda, RepeatVector, Multiply, Permute, Flatten
)
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras import backend as K
from tqdm import tqdm

from masterthesis.features.build_features import iterate_tokens, iterate_docs
from masterthesis.utils import load_split, DATA_DIR
from masterthesis.models.callbacks import F1Metrics
from masterthesis.models.report import report
from masterthesis.models.layers import GlobalAveragePooling1D
from masterthesis.results import save_results
from masterthesis.gensim_utils import load_fasttext_embeddings


conll_folder = DATA_DIR / 'conll'
vectors_path = Path('/projects/nlpl/data/vectors/11/128.zip')  # 50-D FastText embeddings

SEQ_LEN = 700  # 95th percentile of documents
INPUT_DROPOUT = 0.5
RECURRENT_DROPOUT = 0.1
EMB_LAYER_NAME = 'embedding_layer'


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
    parser.add_argument('--attention', action="store_true")
    parser.add_argument('--bidirectional', action="store_true")
    parser.add_argument('--fasttext', action="store_true", help='Initialize embeddings')
    parser.add_argument('--nli', action="store_true", help='Classify NLI')
    return parser.parse_args()


def build_model(vocab_size: int, sequence_len: int, num_classes: int,
                embed_dim: int, rnn_dim: int, dropout_rate: float,
                bidirectional: bool, attention: bool):
    mask_zero = not attention  # The attention mechanism does not support masked inputs
    input_ = Input((sequence_len,))
    lookup = Embedding(vocab_size, embed_dim, mask_zero=mask_zero, name=EMB_LAYER_NAME)(input_)

    lstm_factory = LSTM(rnn_dim, return_sequences=True, dropout=INPUT_DROPOUT,
                        recurrent_dropout=RECURRENT_DROPOUT)
    if bidirectional:
        lstm_factory = Bidirectional(lstm_factory)
    lstm = lstm_factory(lookup)
    dropout = Dropout(dropout_rate)(lstm)

    if attention:
        units = 2 * rnn_dim if bidirectional else rnn_dim
        # compute importance for each step
        attention = TimeDistributed(Dense(1, activation='tanh'))(dropout)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(units)(attention)
        attention = Permute([2, 1])(attention)

        # apply the attention
        sent_representation = Multiply()([dropout, attention])
        pooled = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    else:
        pooled = GlobalAveragePooling1D()(dropout)

    output = Dense(num_classes, activation='softmax')(pooled)
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

    if args.nli:
        labels = sorted(train_meta.lang.unique())
    else:
        labels = sorted(train_meta.cefr.unique())

    train_y = to_categorical([labels.index(c) for c in train_meta.cefr])
    dev_y = to_categorical([labels.index(c) for c in dev_meta.cefr])

    model = build_model(
        vocab_size=vocab_size, sequence_len=SEQ_LEN, num_classes=len(labels),
        embed_dim=args.embed_dim, rnn_dim=args.rnn_dim, dropout_rate=args.dropout_rate,
        bidirectional=args.bidirectional, attention=args.attention)
    model.summary()

    if args.fasttext:
        if not vectors_path.is_file():
            print('Embeddings path not available')
        else:
            kv = load_fasttext_embeddings(vectors_path)
            embeddings_matrix = np.zeros((vocab_size, 50))
            print('Making embeddings:')
            for word, idx in tqdm(w2i.items()):
                vec = kv.word_vec(word)
                embeddings_matrix[idx, :] = vec
            model.get_layer(EMB_LAYER_NAME).set_weights(embeddings_matrix)

    model.compile(
        optimizer=RMSprop(lr=args.lr, rho=args.decay_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # Context manager fails on Windows (can't open an open file again)
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
    result_prefix = 'taghipour_ng'
    if args.nli:
        result_prefix += '_nli'
    save_results(result_prefix, args.__dict__, history.history, true, pred)


if __name__ == '__main__':
    main()
