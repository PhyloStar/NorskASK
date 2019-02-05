import argparse
import os
from pathlib import Path
import pickle
import tempfile

from keras import backend as K
from keras.layers import (
    Activation, Bidirectional, Dense, Dropout, Embedding, Flatten, Input, Lambda, LSTM,
    Multiply, Permute, RepeatVector, TimeDistributed
)
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm

from masterthesis.features.build_features import make_w2i, words_to_sequences
from masterthesis.gensim_utils import load_embeddings
from masterthesis.models.callbacks import F1Metrics
from masterthesis.models.layers import GlobalAveragePooling1D
from masterthesis.models.report import report
from masterthesis.results import save_results
from masterthesis.utils import ATTENTION_LAYER, DATA_DIR, load_split, REPRESENTATION_LAYER


conll_folder = DATA_DIR / 'conll'

SEQ_LEN = 700  # 95th percentile of documents
INPUT_DROPOUT = 0.5
RECURRENT_DROPOUT = 0.1
EMB_LAYER_NAME = 'embedding_layer'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-cefr', action='store_true')
    parser.add_argument('--save-model', action='store_true')
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
    parser.add_argument('--vectors', type=Path, help='Embedding vectors')
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
        attention = Activation('softmax', name=ATTENTION_LAYER)(attention)
        attention = RepeatVector(units)(attention)
        attention = Permute([2, 1])(attention)

        # apply the attention
        sent_representation = Multiply()([dropout, attention])
        pooled = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    else:
        pooled = GlobalAveragePooling1D(name=REPRESENTATION_LAYER)(dropout)

    output = Dense(num_classes, activation='softmax')(pooled)
    return Model(inputs=[input_], outputs=[output])


def main():
    args = parse_args()
    train_meta = load_split('train', round_cefr=args.round_cefr)
    dev_meta = load_split('dev', round_cefr=args.round_cefr)

    vocab_size = args.vocab_size
    w2i = make_w2i(vocab_size)
    train_x, dev_x = words_to_sequences(SEQ_LEN, ['train', 'dev'], w2i)

    target_col = 'lang' if args.nli else 'cefr'
    labels = sorted(train_meta[target_col].unique())

    train_y = to_categorical([labels.index(c) for c in train_meta[target_col]])
    dev_y = to_categorical([labels.index(c) for c in dev_meta[target_col]])

    model = build_model(
        vocab_size=vocab_size, sequence_len=SEQ_LEN, num_classes=len(labels),
        embed_dim=args.embed_dim, rnn_dim=args.rnn_dim, dropout_rate=args.dropout_rate,
        bidirectional=args.bidirectional, attention=args.attention)
    model.summary()

    if args.vectors:
        if not args.vectors.is_file():
            print('Embeddings path not available')
        else:
            kv = load_embeddings(args.vectors, fasttext=args.fasttext)
            embeddings_matrix = np.zeros((vocab_size, 50))
            print('Making embeddings:')
            for word, idx in tqdm(w2i.items(), total=len(w2i)):
                vec = kv.word_vec(word)
                embeddings_matrix[idx, :] = vec
            model.get_layer(EMB_LAYER_NAME).set_weights([embeddings_matrix])

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

    if args.save_model:
        model.save('rnn_model.h5')
        pickle.dump(w2i, open('rnn_model_w2i.pkl', 'wb'))

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
