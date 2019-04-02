import argparse
import os
import tempfile
from typing import Iterable, Sequence, Union  # noqa: F401

from keras import backend as K
from keras.layers import (
    Activation, Bidirectional, Dense, Dropout, Flatten, GlobalMaxPooling1D, GRU, Lambda,
    Layer, LSTM, Multiply, Permute, RepeatVector, TimeDistributed
)
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import to_categorical
import numpy as np

from masterthesis.models.callbacks import F1Metrics
from masterthesis.models.layers import (
    build_inputs_and_embeddings, GlobalAveragePooling1D, InputLayerArgs
)
from masterthesis.models.report import multi_task_report, report
from masterthesis.models.utils import (
    add_common_args, add_seq_common_args, get_sequence_input_reps, get_targets_and_output_units,
    init_pretrained_embs, ranked_accuracy, ranked_prediction
)
from masterthesis.results import save_results
from masterthesis.utils import (
    ATTENTION_LAYER, AUX_OUTPUT_NAME, get_file_name, load_split, OUTPUT_NAME,
    REPRESENTATION_LAYER, rescale_regression_results, safe_plt as plt, save_model, set_reproducible
)

INPUT_DROPOUT = 0.5
RECURRENT_DROPOUT = 0.1
POS_EMB_DIM = 10


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_seq_common_args(parser)
    parser.add_argument('--pool-method', choices={'mean', 'max', 'attention'})
    parser.add_argument('--bidirectional', action="store_true")
    parser.add_argument('--decay-rate', type=float)
    parser.add_argument('--dropout-rate', type=float)
    parser.add_argument('--fasttext', action="store_true", help='Initialize embeddings')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--rnn-cell', choices={'gru', 'lstm'})
    parser.add_argument('--rnn-dim', type=int)
    parser.set_defaults(batch_size=32, decay_rate=0.9, dropout_rate=0.5, embed_dim=100, epochs=50,
                        lr=1e-3, rnn_cell='lstm', rnn_dim=300, vocab_size=None, pool_method='mean')
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


def build_model(vocab_size: int, sequence_len: int, output_units: Sequence[int],
                embed_dim: int, rnn_dim: int, dropout_rate: float,
                bidirectional: bool, pool_method: str, static_embs: bool, rnn_cell: str,
                num_pos: int = 0, classification: bool = False):
    # Only the global average pooling supports masked input
    mask_zero = pool_method == 'mean'

    input_layer_args = InputLayerArgs(
        num_pos=num_pos, mask_zero=mask_zero, embed_dim=embed_dim, pos_embed_dim=POS_EMB_DIM,
        vocab_size=vocab_size, sequence_len=sequence_len, static_embeddings=static_embs
    )
    inputs, embedding_layer = build_inputs_and_embeddings(input_layer_args)

    rnn_factory = _build_rnn(rnn_cell, rnn_dim, bidirectional)
    rnn = rnn_factory(embedding_layer)

    dropout = Dropout(dropout_rate)(rnn)

    if pool_method == 'attention':
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
    elif pool_method == 'mean':
        pooled = GlobalAveragePooling1D(name=REPRESENTATION_LAYER)(dropout)
    elif pool_method == 'max':
        pooled = GlobalMaxPooling1D(name=REPRESENTATION_LAYER)(dropout)
    else:
        raise ValueError('Unrecognized pooling strategy: ' + pool_method)

    activation = 'softmax' if classification else 'sigmoid'
    outputs = [Dense(output_units[0], activation=activation, name=OUTPUT_NAME)(pooled)]
    if len(output_units) > 1:
        aux_out = Dense(output_units[1], activation='softmax', name=AUX_OUTPUT_NAME)(pooled)
        outputs.append(aux_out)
    return Model(inputs=inputs, outputs=outputs)


def get_compile_args(args):
    if args.method == 'classification':
        optimizer = RMSprop(lr=args.lr, rho=args.decay_rate)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']  # type: List[Union[str, Callable]]
    elif args.method == 'ranked':
        optimizer = RMSprop(lr=args.lr, rho=args.decay_rate)
        loss = 'mean_squared_error'
        metrics = [ranked_accuracy]
    elif args.method == 'regression':
        optimizer = 'rmsprop'
        loss = 'mean_squared_error'
        metrics = ['mae']
    else:
        raise ValueError('Unknown method')
    return optimizer, loss, metrics


def main():
    args = parse_args()

    set_reproducible()

    train_meta = load_split('train', round_cefr=args.round_cefr)
    dev_meta = load_split('dev', round_cefr=args.round_cefr)

    target_col = 'lang' if args.nli else 'cefr'
    labels = sorted(train_meta[target_col].unique())

    train_x, dev_x, num_pos, w2i = get_sequence_input_reps(args)

    train_target_scores = np.array([labels.index(c) for c in train_meta[target_col]], dtype=int)
    dev_target_scores = np.array([labels.index(c) for c in dev_meta[target_col]], dtype=int)
    del target_col

    train_y, dev_y, output_units = get_targets_and_output_units(
        train_target_scores, dev_target_scores, args.method)

    multi_task = args.aux_loss_weight > 0
    if multi_task:
        assert not args.nli, "Both NLI and multi-task specified"
        lang_labels = sorted(train_meta.lang.unique())
        train_y.append(to_categorical([lang_labels.index(l) for l in train_meta.lang]))
        dev_y.append(to_categorical([lang_labels.index(l) for l in dev_meta.lang]))
        output_units.append(len(lang_labels))
        loss_weights = {
            AUX_OUTPUT_NAME: args.aux_loss_weight,
            OUTPUT_NAME: 1.0 - args.aux_loss_weight
        }
    else:
        loss_weights = None

    model = build_model(
        vocab_size=args.vocab_size, sequence_len=args.doc_length, num_classes=output_units,
        embed_dim=args.embed_dim, rnn_dim=args.rnn_dim, dropout_rate=args.dropout_rate,
        bidirectional=args.bidirectional, pool_method=args.pool_method,
        static_embs=args.static_embs, rnn_cell=args.rnn_cell, num_pos=num_pos,
        classification=args.method == 'classification')
    model.summary()

    if args.vectors:
        init_pretrained_embs(model, args.vectors, w2i)

    optimizer = RMSprop(lr=args.lr, rho=args.decay_rate)
    model.compile(optimizer, 'categorical_crossentropy',
                  loss_weights=loss_weights, metrics=['accuracy'])

    optimizer, loss, metrics = get_compile_args(args.method, args.lr)
    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)

    # Context manager fails on Windows (can't open an open file again)
    temp_handle, weights_path = tempfile.mkstemp(suffix='.h5')
    val_y = dev_target_scores
    callbacks = [F1Metrics(dev_x, val_y, weights_path, ranked=args.method == 'ranked')]
    history = model.fit(
        train_x, train_y, epochs=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, validation_data=(dev_x, dev_y),
        verbose=2)
    model.load_weights(weights_path)
    os.close(temp_handle)
    os.remove(weights_path)

    true = dev_target_scores
    if multi_task:
        predictions = model.predict(dev_x)[0]
    else:
        predictions = model.predict(dev_x)
    if args.method == 'classification':
        pred = np.argmax(predictions, axis=1)
    elif args.method == 'regression':
        # Round to integers and clip to score range
        highest_class = train_target_scores.max()
        pred = rescale_regression_results(predictions, highest_class).ravel()
    elif args.method == 'ranked':
        pred = K.eval(ranked_prediction(predictions))
    if multi_task:
        multi_task_report(history.history, true, pred, labels)
    else:
        report(true, pred, labels)

    if args.nli:
        name = 'rnn-nli'
    elif multi_task:
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
