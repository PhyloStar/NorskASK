import argparse
from math import isfinite
import os
import tempfile
from typing import Iterable, List, Optional, Sequence

from keras.constraints import max_norm
from keras.layers import Concatenate, Conv1D, Dense, Dropout, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np

from masterthesis.features.build_features import (
    make_mixed_pos2i, make_pos2i, make_w2i, mixed_pos_to_sequences, pos_to_sequences,
    words_to_sequences
)
from masterthesis.models.callbacks import F1Metrics
from masterthesis.models.layers import build_inputs_and_embeddings, InputLayerArgs
from masterthesis.models.report import multi_task_report, report
from masterthesis.models.utils import add_common_args, add_seq_common_args, init_pretrained_embs
from masterthesis.results import save_results
from masterthesis.utils import (
    AUX_OUTPUT_NAME, get_file_name, load_split, OUTPUT_NAME, REPRESENTATION_LAYER,
    rescale_regression_results, safe_plt as plt, save_model, set_reproducible
)

POS_EMB_DIM = 10


def int_list(strlist: str) -> List[int]:
    """Turn a string of comma separated values into a list of integers.

    >>> int_list('2,3,6')
    [2, 3, 6]
    """
    ints = []
    for strint in strlist.split(','):
        intval = int(strint)
        ints.append(intval)
    return ints


def positive_float(s: str) -> Optional[float]:
    if s.lower() == 'none':
        return None
    f = float(s)
    if f > 0.0 and isfinite(f):
        return f
    raise ValueError('Invalid constraint value (must be positive, finite float)')


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_seq_common_args(parser)
    parser.add_argument('--constraint', type=positive_float)
    parser.add_argument('--windows', '-w', type=int_list)
    parser.set_defaults(batch_size=32, doc_length=700, embed_dim=50, epochs=50, vocab_size=20000,
                        windows=[3, 4, 5])
    return parser.parse_args()


def build_model(vocab_size: int, sequence_length: int, output_units: Sequence[int], embed_dim: int,
                windows: Iterable[int], num_pos: int = 0,
                constraint: Optional[float] = None, static_embs: bool = False,
                do_classification: bool = False) -> Model:
    """Build CNN model."""
    input_layer_args = InputLayerArgs(
        num_pos=num_pos, mask_zero=False, embed_dim=embed_dim, pos_embed_dim=POS_EMB_DIM,
        vocab_size=vocab_size, sequence_len=sequence_length, static_embeddings=static_embs
    )
    inputs, embedding_layer = build_inputs_and_embeddings(input_layer_args)

    pooled_feature_maps = []
    for kernel_size in windows:
        conv_layer = Conv1D(
            filters=100, kernel_size=kernel_size, activation='relu')(embedding_layer)
        pooled_feature_maps.extend([
            # GlobalAveragePooling1D()(conv_layer),
            GlobalMaxPooling1D()(conv_layer)
        ])
    merged = Concatenate(name=REPRESENTATION_LAYER)(pooled_feature_maps)
    dropout_layer = Dropout(0.5)(merged)

    kernel_constraint = constraint and max_norm(constraint)
    activation = 'softmax' if do_classification else 'sigmoid'
    outputs = [Dense(output_units[0], activation=activation,
                     kernel_constraint=kernel_constraint, name=OUTPUT_NAME)(dropout_layer)]
    if len(output_units) > 1:
        aux_out = Dense(output_units[1], activation='softmax', name=AUX_OUTPUT_NAME)(dropout_layer)
        outputs.append(aux_out)

    return Model(inputs=inputs, outputs=outputs)


def main():
    args = parse_args()

    set_reproducible()

    do_classification = args.classification
    seq_length = args.doc_length
    train = load_split('train', round_cefr=args.round_cefr)
    dev = load_split('dev', round_cefr=args.round_cefr)

    target_col = 'lang' if args.nli else 'cefr'
    labels = sorted(train[target_col].unique())

    if args.mixed_pos:
        t2i = make_mixed_pos2i()
        train_x, dev_x = mixed_pos_to_sequences(seq_length, ['train', 'dev'], t2i)
        args.vocab_size = len(t2i)
        num_pos = 0
    else:
        w2i = make_w2i(args.vocab_size)
        train_x, dev_x = words_to_sequences(seq_length, ['train', 'dev'], w2i)
        if args.include_pos:
            pos2i = make_pos2i()
            num_pos = len(pos2i)
            train_pos, dev_pos = pos_to_sequences(seq_length, ['train', 'dev'], pos2i)
            train_x = [train_x, train_pos]
            dev_x = [dev_x, dev_pos]
        else:
            num_pos = 0

    train_target_scores = np.array([labels.index(c) for c in train[target_col]], dtype=int)
    dev_target_scores = np.array([labels.index(c) for c in dev[target_col]], dtype=int)

    if do_classification:
        train_y = to_categorical(train_target_scores)
        dev_y = to_categorical(dev_target_scores)
        output_units = [len(labels)]
    else:  # Regression
        highest_class = max(train_target_scores)
        train_y = np.array(train_target_scores) / highest_class
        dev_y = np.array(dev_target_scores) / highest_class
        output_units = [1]

    train_y = [train_y]
    dev_y = [dev_y]

    multi_task = args.aux_loss_weight > 0
    if multi_task:
        assert not args.nli, "Both NLI and multi-task specified"
        lang_labels = sorted(train.lang.unique())
        train_y.append(to_categorical([lang_labels.index(l) for l in train.lang]))
        dev_y.append(to_categorical([lang_labels.index(l) for l in dev.lang]))
        output_units.append(len(lang_labels))
        loss_weights = {
            AUX_OUTPUT_NAME: args.aux_loss_weight,
            OUTPUT_NAME: 1.0 - args.aux_loss_weight
        }
    else:
        loss_weights = None

    model = build_model(args.vocab_size, seq_length, output_units, args.embed_dim,
                        windows=args.windows, num_pos=num_pos, constraint=args.constraint,
                        static_embs=args.static_embs, do_classification=do_classification)
    model.summary()

    if args.vectors:
        init_pretrained_embs(model, args.vectors, w2i)

    compile_model(model, do_classification, args.lr, loss_weights)

    temp_handle, weights_path = tempfile.mkstemp(suffix='.h5')
    val_y = dev_y if do_classification else [dev_target_scores]
    callbacks = [F1Metrics(dev_x, val_y, weights_path)]
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
    if do_classification:
        pred = np.argmax(predictions, axis=1)
    else:
        # Round to integers and clip to score range
        pred = rescale_regression_results(predictions, highest_class).ravel()
    if multi_task:
        multi_task_report(history.history, true, pred, labels)
    else:
        report(true, pred, labels)

    name = get_name(args.nli, multi_task)
    name = get_file_name(name)

    if args.save_model:
        save_model(name, model, w2i)

    save_results(name, args.__dict__, history.history, true, pred)

    plt.show()


def get_name(nli: bool, multi_task: bool) -> str:
    if nli:
        return 'cnn-nli'
    elif multi_task:
        return 'cnn-multi'
    return 'cnn'


def compile_model(model: Model, classification: bool, lr: float, loss_weights):
    if classification:
        optimizer = Adam(lr=lr)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:
        optimizer = 'rmsprop'
        loss = 'mean_squared_error'
        metrics = ['mae']
    model.compile(
        optimizer=optimizer,
        loss=loss,
        loss_weights=loss_weights,
        metrics=metrics)


if __name__ == '__main__':
    main()
