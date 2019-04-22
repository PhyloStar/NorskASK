import argparse
import os
import tempfile
from typing import Callable, Iterable, List, Sequence, Union  # noqa: F401

import keras.backend as K
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from masterthesis.features.build_features import (
    bag_of_words,
    filename_iter,
    iterate_mixed_pos_docs,
    iterate_pos_docs,
)
from masterthesis.models.callbacks import F1Metrics
from masterthesis.models.report import multi_task_report, report
from masterthesis.models.utils import (
    add_common_args,
    get_targets_and_output_units,
    ranked_accuracy,
    ranked_prediction,
)
from masterthesis.results import save_results
from masterthesis.utils import (
    AUX_OUTPUT_NAME,
    DATA_DIR,
    get_file_name,
    load_split,
    OUTPUT_NAME,
    REPRESENTATION_LAYER,
    rescale_regression_results,
    safe_plt as plt,
    save_model,
    set_reproducible,
)

conll_folder = DATA_DIR / 'conll'


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument('featuretype', choices={'pos', 'bow', 'char', 'mix'})
    parser.add_argument('--max-features', type=int, default=20000)
    return parser.parse_args()


def build_model(vocab_size: int, output_units: Sequence[int], classification: bool):
    input_ = Input((vocab_size, ))

    hidden_1 = Dense(100, activation='relu')(input_)
    dropout_1 = Dropout(0.5)(hidden_1)
    hidden_2 = Dense(300, activation='relu', name=REPRESENTATION_LAYER)(dropout_1)
    dropout_2 = Dropout(0.5)(hidden_2)

    activation = 'softmax' if classification else 'sigmoid'
    outputs = [Dense(output_units[0], activation=activation, name=OUTPUT_NAME)(dropout_2)]
    if len(output_units) > 1:
        aux_out = Dense(output_units[1], activation='softmax', name=AUX_OUTPUT_NAME)(dropout_2)
        outputs.append(aux_out)

    return Model(inputs=[input_], outputs=outputs)


def pos_line_iter(split) -> Iterable[str]:
    for doc in iterate_pos_docs(split):
        yield ' '.join(doc)


def mixed_pos_line_iter(split) -> Iterable[str]:
    for doc in iterate_mixed_pos_docs(split):
        yield ' '.join(doc)


def preprocess(kind: str, max_features: int, train_meta, dev_meta):
    if kind == 'pos':
        vectorizer = CountVectorizer(
            lowercase=False, token_pattern=r"[^\s]+", ngram_range=(2, 4), max_features=max_features
        )
        train_x = vectorizer.fit_transform(pos_line_iter('train'))
        dev_x = vectorizer.transform(pos_line_iter('dev'))
        num_features = len(vectorizer.vocabulary_)
    elif kind == 'mix':
        vectorizer = CountVectorizer(
            lowercase=False, token_pattern=r"[^\s]+", ngram_range=(1, 3), max_features=max_features
        )
        train_x = vectorizer.fit_transform(mixed_pos_line_iter('train'))
        dev_x = vectorizer.transform(mixed_pos_line_iter('dev'))
        num_features = len(vectorizer.vocabulary_)
    elif kind == 'char':
        train_x, vectorizer = bag_of_words(
            'train',
            analyzer='char',
            ngram_range=(2, 4),
            max_features=max_features,
            lowercase=False
        )
        dev_x = vectorizer.transform(filename_iter(dev_meta))
        num_features = len(vectorizer.vocabulary_)
    elif kind == 'bow':
        train_x, vectorizer = bag_of_words(
            'train', token_pattern=r"[^\s]+", max_features=max_features, lowercase=False
        )
        dev_x = vectorizer.transform(filename_iter(dev_meta))
        num_features = len(vectorizer.vocabulary_)
    else:
        raise ValueError('Feature type "%s" is not supported' % kind)
    return train_x, dev_x, num_features


def get_compile_args(method: str, lr: float):
    if method == 'classification':
        optimizer = Adam(lr=lr)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']  # type: List[Union[str, Callable]]
    elif method == 'ranked':
        optimizer = Adam(lr=lr)
        loss = 'mean_squared_error'
        metrics = [ranked_accuracy]
    elif method == 'regression':
        optimizer = 'rmsprop'
        loss = 'mean_squared_error'
        metrics = ['mae']
    else:
        raise ValueError('Unknown method')
    return optimizer, loss, metrics


def main():
    args = parse_args()

    set_reproducible(args.seed_delta)
    do_classification = args.method == 'classification'

    train_meta = load_split('train', round_cefr=args.round_cefr)
    dev_meta = load_split('dev', round_cefr=args.round_cefr)

    kind = args.featuretype
    train_x, dev_x, num_features = preprocess(kind, args.max_features, train_meta, dev_meta)

    target_col = 'lang' if args.nli else 'cefr'
    labels = sorted(train_meta[target_col].unique())

    train_target_scores = np.array([labels.index(c) for c in train_meta[target_col]], dtype=int)
    dev_target_scores = np.array([labels.index(c) for c in dev_meta[target_col]], dtype=int)

    train_y, dev_y, output_units = get_targets_and_output_units(
        train_target_scores, dev_target_scores, args.method
    )

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
    del train_meta, dev_meta

    model = build_model(num_features, output_units, do_classification)
    model.summary()

    optimizer, loss, metrics = get_compile_args(args.method, args.lr)
    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)

    # Context manager fails on Windows (can't open an open file again)
    temp_handle, weights_path = tempfile.mkstemp(suffix='.h5')
    val_y = dev_target_scores
    callbacks = [F1Metrics(dev_x, val_y, weights_path, ranked=args.method == 'ranked')]
    history = model.fit(
        train_x,
        train_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        validation_data=(dev_x, dev_y),
        verbose=2
    )
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

    plt.show()

    prefix = 'mlp_%s' % args.featuretype
    fname = get_file_name(prefix)
    save_results(fname, args.__dict__, history.history, true, pred)

    if args.save_model:
        save_model(fname, model, None)


if __name__ == '__main__':
    main()
