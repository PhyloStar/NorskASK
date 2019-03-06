import argparse
from math import isfinite
import os
import tempfile
from typing import Iterable, List, Optional

from keras.constraints import max_norm
from keras.layers import Concatenate, Conv1D, Dense, Dropout, GlobalMaxPooling1D
from keras.models import Model
from keras.utils import to_categorical
import numpy as np

from masterthesis.features.build_features import (
    make_mixed_pos2i, make_pos2i, make_w2i, mixed_pos_to_sequences, pos_to_sequences,
    words_to_sequences
)
from masterthesis.models.callbacks import F1Metrics
from masterthesis.models.layers import build_inputs_and_embeddings, InputLayerArgs
from masterthesis.models.report import multi_task_report, report
from masterthesis.models.utils import init_pretrained_embs, add_common_args, add_seq_common_args
from masterthesis.results import save_results
from masterthesis.utils import (
    AUX_OUTPUT_NAME, get_file_name, load_split, OUTPUT_NAME, REPRESENTATION_LAYER,
    save_model, safe_plt as plt
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
    parser.set_defaults(batch_size=32, doc_length=700, embed_dim=50, epochs=50, vocab_size=4000,
                        windows=[3, 4, 5])
    return parser.parse_args()


def build_model(vocab_size: int, sequence_length: int, num_classes: Iterable[int], embed_dim: int,
                windows: Iterable[int], num_pos: int = 0,
                constraint: Optional[float] = None, static_embs: bool = False) -> Model:
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
    if constraint is not None:
        kernel_constraint = max_norm(constraint)
    else:
        kernel_constraint = None
    outputs = [Dense(n_c, activation='softmax',
                     kernel_constraint=kernel_constraint, name=name)(dropout_layer)
               for name, n_c in zip([OUTPUT_NAME, AUX_OUTPUT_NAME], num_classes)]
    return Model(inputs=inputs, outputs=outputs)


def main():
    args = parse_args()
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

    train_y = [to_categorical([labels.index(c) for c in train[target_col]])]
    dev_y = [to_categorical([labels.index(c) for c in dev[target_col]])]
    num_classes = [len(labels)]

    if args.multi:
        assert not args.nli, "Both NLI and multi-task specified"
        lang_labels = sorted(train.lang.unique())
        train_y.append(to_categorical([lang_labels.index(l) for l in train.lang]))
        dev_y.append(to_categorical([lang_labels.index(l) for l in dev.lang]))
        num_classes.append(len(lang_labels))

    model = build_model(args.vocab_size, seq_length, num_classes, args.embed_dim,
                        windows=args.windows, num_pos=num_pos, constraint=args.constraint,
                        static_embs=args.static_embs)
    model.summary()

    if args.vectors:
        init_pretrained_embs(model, args.vectors, w2i)

    loss_weights = {
        AUX_OUTPUT_NAME: args.aux_loss_weight,
        OUTPUT_NAME: 1.0 - args.aux_loss_weight
    }
    optimizer = 'adam'
    model.compile(optimizer, 'categorical_crossentropy',
                  loss_weights=loss_weights, metrics=['accuracy'])

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
        name = 'cnn-nli'
    elif args.multi:
        name = 'cnn-multi'
    else:
        name = 'cnn'
    name = get_file_name(name)

    if args.save_model:
        save_model(name, model, w2i)

    save_results(name, args.__dict__, history.history, true, pred)

    plt.show()


if __name__ == '__main__':
    main()
