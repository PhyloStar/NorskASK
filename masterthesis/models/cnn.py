import argparse
import logging
from math import isfinite
import os
import tempfile
from typing import (  # noqa: F401
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)

import keras.backend as K
from keras.constraints import max_norm
from keras.layers import Concatenate, Conv1D, Dense, Dropout, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np

from masterthesis.models.callbacks import F1Metrics
from masterthesis.models.layers import build_inputs_and_embeddings, InputLayerArgs
from masterthesis.models.report import multi_task_report, report
from masterthesis.models.utils import (
    add_common_args,
    add_seq_common_args,
    get_sequence_input_reps,
    get_targets_and_output_units,
    init_pretrained_embs,
    ranked_accuracy,
    ranked_prediction,
)
from masterthesis.results import save_results
from masterthesis.utils import (
    AUX_OUTPUT_NAME,
    get_file_name,
    load_split,
    OUTPUT_NAME,
    REPRESENTATION_LAYER,
    rescale_regression_results,
    safe_plt as plt,
    save_model,
    set_reproducible,
)

POS_EMB_DIM = 10
logging.basicConfig()
logger = logging.getLogger(__name__)


def int_list(strlist: str) -> List[int]:
    """Turn a string of comma separated values into a list of integers.

    >>> int_list('2,3,6')
    [2, 3, 6]
    """
    ints = []
    for strint in strlist.split(","):
        intval = int(strint)
        ints.append(intval)
    return ints


def positive_float(s: str) -> Optional[float]:
    if s.lower() == "none":
        return None
    f = float(s)
    if f > 0.0 and isfinite(f):
        return f
    raise ValueError("Invalid constraint value (must be positive, finite float)")


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_seq_common_args(parser)
    parser.add_argument("--constraint", type=positive_float)
    parser.add_argument("--windows", "-w", type=int_list)
    parser.set_defaults(
        batch_size=32, embed_dim=100, epochs=50, vocab_size=None, windows=[3, 4, 5]
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger(None).setLevel(logging.DEBUG)
    return args


def build_model(
    vocab_size: int,
    sequence_length: int,
    output_units: Sequence[int],
    embed_dim: int,
    windows: Iterable[int],
    num_pos: int = 0,
    constraint: Optional[float] = None,
    static_embs: bool = False,
    classification: bool = False,
) -> Model:
    """Build CNN model."""
    input_layer_args = InputLayerArgs(
        num_pos=num_pos,
        mask_zero=False,
        embed_dim=embed_dim,
        pos_embed_dim=POS_EMB_DIM,
        vocab_size=vocab_size,
        sequence_len=sequence_length,
        static_embeddings=static_embs,
    )
    inputs, embedding_layer = build_inputs_and_embeddings(input_layer_args)

    pooled_feature_maps = []
    for kernel_size in windows:
        conv_layer = Conv1D(filters=100, kernel_size=kernel_size, activation="relu")(
            embedding_layer
        )
        pooled_feature_maps.extend(
            [
                # GlobalAveragePooling1D()(conv_layer),
                GlobalMaxPooling1D()(conv_layer)
            ]
        )
    merged = Concatenate(name=REPRESENTATION_LAYER)(pooled_feature_maps)
    dropout_layer = Dropout(0.5)(merged)

    kernel_constraint = constraint and max_norm(constraint)
    activation = "softmax" if classification else "sigmoid"
    outputs = [
        Dense(
            output_units[0],
            activation=activation,
            kernel_constraint=kernel_constraint,
            name=OUTPUT_NAME,
        )(dropout_layer)
    ]
    if len(output_units) > 1:
        aux_out = Dense(output_units[1], activation="softmax", name=AUX_OUTPUT_NAME)(
            dropout_layer
        )
        outputs.append(aux_out)

    return Model(inputs=inputs, outputs=outputs)


def get_name(nli: bool, multi_task: bool) -> str:
    if nli:
        return "cnn-nli"
    elif multi_task:
        return "cnn-multi"
    return "cnn"


def get_compile_args(method: str, lr: float):
    losses = {AUX_OUTPUT_NAME: "categorical_crossentropy"}
    metrics = {
        AUX_OUTPUT_NAME: ["accuracy"]
    }  # type: Dict[str, List[Union[str, Callable]]]
    if method == "classification":
        optimizer = Adam(lr=lr)
        losses[OUTPUT_NAME] = "categorical_crossentropy"
        metrics[OUTPUT_NAME] = ["accuracy"]
    elif method == "ranked":
        optimizer = Adam(lr=lr)
        losses[OUTPUT_NAME] = "mean_squared_error"
        metrics[OUTPUT_NAME] = [ranked_accuracy]
    elif method == "regression":
        optimizer = "rmsprop"
        losses[OUTPUT_NAME] = "mean_squared_error"
        metrics[OUTPUT_NAME] = ["mae"]
    else:
        raise ValueError("Unknown method")
    return optimizer, losses, metrics


def main():
    args = parse_args()

    set_reproducible(args.seed_delta)

    train_meta = load_split("train", round_cefr=args.round_cefr)
    dev_meta = load_split("dev", round_cefr=args.round_cefr)

    target_col = "lang" if args.nli else "cefr"
    labels = sorted(train_meta[target_col].unique())

    train_x, dev_x, num_pos, w2i = get_sequence_input_reps(args)
    args.vocab_size = len(w2i)
    print("Vocabulary size is {}".format(args.vocab_size))

    train_target_scores = np.array(
        [labels.index(c) for c in train_meta[target_col]], dtype=int
    )
    dev_target_scores = np.array(
        [labels.index(c) for c in dev_meta[target_col]], dtype=int
    )
    del target_col

    train_y, dev_y, output_units = get_targets_and_output_units(
        train_target_scores, dev_target_scores, args.method
    )

    optimizer, loss, metrics = get_compile_args(args.method, args.lr)
    multi_task = args.aux_loss_weight > 0
    if multi_task:
        assert not args.nli, "Both NLI and multi-task specified"
        lang_labels = sorted(train_meta.lang.unique())
        train_y.append(to_categorical([lang_labels.index(l) for l in train_meta.lang]))
        dev_y.append(to_categorical([lang_labels.index(l) for l in dev_meta.lang]))
        output_units.append(len(lang_labels))
        loss_weights = {
            AUX_OUTPUT_NAME: args.aux_loss_weight,
            OUTPUT_NAME: 1.0 - args.aux_loss_weight,
        }
    else:
        loss = loss[OUTPUT_NAME]
        metrics = metrics[OUTPUT_NAME]
        loss_weights = None
    del train_meta, dev_meta

    model = build_model(
        args.vocab_size,
        args.doc_length,
        output_units,
        args.embed_dim,
        windows=args.windows,
        num_pos=num_pos,
        constraint=args.constraint,
        static_embs=args.static_embs,
        classification=args.method == "classification",
    )
    model.summary()

    if args.vectors:
        init_pretrained_embs(model, args.vectors, w2i)

    model.compile(
        optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics
    )

    logger.debug("Train y\n%r", train_y[0][:5])
    logger.debug("Model config\n%r", model.get_config())

    temp_handle, weights_path = tempfile.mkstemp(suffix=".h5")
    val_y = dev_target_scores
    callbacks = [F1Metrics(dev_x, val_y, weights_path, ranked=args.method == "ranked")]
    history = model.fit(
        train_x,
        train_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        validation_data=(dev_x, dev_y),
        verbose=2,
    )
    model.load_weights(weights_path)
    os.close(temp_handle)
    os.remove(weights_path)

    true = dev_target_scores
    if multi_task:
        predictions = model.predict(dev_x)[0]
    else:
        predictions = model.predict(dev_x)
    if args.method == "classification":
        pred = np.argmax(predictions, axis=1)
    elif args.method == "regression":
        # Round to integers and clip to score range
        highest_class = train_target_scores.max()
        pred = rescale_regression_results(predictions, highest_class).ravel()
    elif args.method == "ranked":
        pred = K.eval(ranked_prediction(predictions))
    try:
        if multi_task:
            multi_task_report(history.history, true, pred, labels)
        else:
            report(true, pred, labels)
    except Exception:
        pass

    name = get_name(args.nli, multi_task)
    name = get_file_name(name)

    if args.save_model:
        save_model(name, model, w2i)

    save_results(name, args.__dict__, history.history, true, pred)

    plt.show()


if __name__ == "__main__":
    main()
