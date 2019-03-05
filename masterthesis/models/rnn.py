import argparse
import os
from pathlib import Path
import tempfile
from typing import Iterable

from keras import backend as K
from keras.layers import (
    Activation, Bidirectional, Dense, Dropout, Flatten, GRU, Lambda,
    Layer, LSTM, Multiply, Permute, RepeatVector, TimeDistributed
)
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import to_categorical
import numpy as np

from masterthesis.features.build_features import (
    make_pos2i, make_w2i, pos_to_sequences, words_to_sequences
)
from masterthesis.models.callbacks import F1Metrics
from masterthesis.models.layers import (
    build_inputs_and_embeddings, GlobalAveragePooling1D, InputLayerArgs
)
from masterthesis.models.report import multi_task_report, report
from masterthesis.models.utils import init_pretrained_embs
from masterthesis.results import save_results
from masterthesis.utils import (
    ATTENTION_LAYER, get_file_name, load_split, REPRESENTATION_LAYER, safe_plt as plt, save_model
)

SEQ_LEN = 700  # 95th percentile of documents
INPUT_DROPOUT = 0.5
RECURRENT_DROPOUT = 0.1
POS_EMB_DIM = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention', action="store_true")
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--bidirectional', action="store_true")
    parser.add_argument('--decay-rate', type=float)
    parser.add_argument('--dropout-rate', type=float)
    parser.add_argument('--embed-dim', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--fasttext', action="store_true", help='Initialize embeddings')
    parser.add_argument('--freeze-embeddings', action='store_true')
    parser.add_argument('--include-pos', action='store_true')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--nli', action="store_true", help='Classify NLI')
    parser.add_argument('--rnn-cell', choices={'gru', 'lstm'})
    parser.add_argument('--rnn-dim', type=int)
    parser.add_argument('--round-cefr', action='store_true')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--vectors', type=Path, help='Embedding vectors')
    parser.add_argument('--vocab-size', type=int)
    parser.set_defaults(batch_size=32, decay_rate=0.9, dropout_rate=0.5, embed_dim=50, epochs=50,
                        lr=1e-3, rnn_cell='lstm', rnn_dim=300, vocab_size=4000)
    return parser.parse_args()


def _build_rnn(rnn_cell: str, rnn_dim: int, bidirectional: bool) -> Layer:
    if rnn_cell == 'lstm':
        cell_factory = LSTM
    elif rnn_cell == 'gru':
        cell_factory = GRU
    rnn_factory = cell_factory(rnn_dim, return_sequences=True, dropout=INPUT_DROPOUT,
                               recurrent_dropout=RECURRENT_DROPOUT)
    if bidirectional:
        rnn_factory = Bidirectional(rnn_factory)
    return rnn_factory


def build_model(vocab_size: int, sequence_len: int, num_classes: Iterable[int],
                embed_dim: int, rnn_dim: int, dropout_rate: float,
                bidirectional: bool, attention: bool, freeze_embeddings: bool, rnn_cell: str,
                num_pos: int = 0):
    mask_zero = not attention  # The attention mechanism does not support masked inputs

    input_layer_args = InputLayerArgs(
        num_pos=num_pos, mask_zero=mask_zero, embed_dim=embed_dim, pos_embed_dim=POS_EMB_DIM,
        vocab_size=vocab_size, sequence_len=sequence_len, static_embeddings=freeze_embeddings
    )
    inputs, embedding_layer = build_inputs_and_embeddings(input_layer_args)

    rnn_factory = _build_rnn(rnn_cell, rnn_dim, bidirectional)
    rnn = rnn_factory(embedding_layer)

    dropout = Dropout(dropout_rate)(rnn)

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
        pooled = Lambda(lambda xin: K.sum(xin, axis=1),
                        name=REPRESENTATION_LAYER)(sent_representation)
    else:
        pooled = GlobalAveragePooling1D(name=REPRESENTATION_LAYER)(dropout)

    outputs = [Dense(n_c, activation='softmax')(pooled)
               for name, n_c in zip(['output', 'aux_output'], num_classes)]
    return Model(inputs=inputs, outputs=outputs)


def main():
    args = parse_args()
    train = load_split('train', round_cefr=args.round_cefr)
    dev = load_split('dev', round_cefr=args.round_cefr)

    vocab_size = args.vocab_size
    w2i = make_w2i(vocab_size)
    train_x, dev_x = words_to_sequences(SEQ_LEN, ['train', 'dev'], w2i)

    target_col = 'lang' if args.nli else 'cefr'
    labels = sorted(train[target_col].unique())

    if args.include_pos:
        pos2i = make_pos2i()
        num_pos = len(pos2i)
        train_pos, dev_pos = pos_to_sequences(SEQ_LEN, ['train', 'dev'], pos2i)
        train_x = [train_x, train_pos]
        dev_x = [dev_x, dev_pos]
    else:
        num_pos = 0

    train_y = [to_categorical([labels.index(c) for c in train[target_col]])]
    dev_y = [to_categorical([labels.index(c) for c in dev[target_col]])]
    num_classes = [len(labels)]

    if args.multi:
        assert not args.nli, "Both NLI and multi-task specified"
        lang_labels = sorted(train.lang.unique())
        train_y.append(to_categorical([lang_labels.index(l) for l in train.lang]))
        dev_y.append(to_categorical([lang_labels.index(l) for l in dev.lang]))
        num_classes.append(len(lang_labels))

    model = build_model(
        vocab_size=vocab_size, sequence_len=SEQ_LEN, num_classes=num_classes,
        embed_dim=args.embed_dim, rnn_dim=args.rnn_dim, dropout_rate=args.dropout_rate,
        bidirectional=args.bidirectional, attention=args.attention,
        freeze_embeddings=args.freeze_embeddings, rnn_cell=args.rnn_cell, num_pos=num_pos)
    model.summary()

    if args.vectors:
        init_pretrained_embs(model, args.vectors, w2i)

    optimizer = RMSprop(lr=args.lr, rho=args.decay_rate)
    model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])

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

    if args.multi:
        predictions = model.predict(dev_x)[0]
        pred = np.argmax(predictions, axis=1)
        true = np.argmax(dev_y[0], axis=1)
        multi_task_report(history.history, true, pred, labels)
    else:
        predictions = model.predict(dev_x)
        pred = np.argmax(predictions, axis=1)
        true = np.argmax(dev_y, axis=1)
        report(true, pred, labels)

    if args.nli:
        name = 'rnn-nli'
    elif args.multi:
        name = 'rnn-multi'
    else:
        name = 'rnn'
    name = get_file_name(name)

    if args.save_model:
        save_model(name, model, w2i)

    save_results(name, args.__dict__, history.history, true, pred)

    plt.show()


if __name__ == '__main__':
    main()
