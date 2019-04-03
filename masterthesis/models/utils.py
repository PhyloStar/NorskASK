import argparse
import os
from pathlib import Path

import keras.backend as K
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm

from masterthesis.features.build_features import (
    make_mixed_pos2i, make_pos2i, make_w2i, mixed_pos_to_sequences, pos_to_sequences,
    words_to_sequences
)
from masterthesis.gensim_utils import load_embeddings
from masterthesis.utils import EMB_LAYER_NAME


def init_pretrained_embs(model: Model, vector_path: Path, w2i) -> None:
    if not vector_path.is_file():
        if 'SUBMITDIR' in os.environ:
            vector_path = Path(os.environ['SUBMITDIR']) / vector_path
        print('New path: %r' % vector_path)
    if not vector_path.is_file():
        print('Embeddings path not available, searching for submitdir')
    else:
        kv = load_embeddings(vector_path)
        embed_dim = kv.vector_size
        emb_layer = model.get_layer(EMB_LAYER_NAME)
        vocab_size = emb_layer.input_dim
        assert embed_dim == emb_layer.output_dim
        assert len(w2i) == vocab_size
        embeddings_matrix = np.zeros((vocab_size, embed_dim))
        print('Making embeddings:')
        for word, idx in tqdm(w2i.items(), total=vocab_size):
            vec = kv.word_vec(word)
            embeddings_matrix[idx, :] = vec
        emb_layer.set_weights([embeddings_matrix])


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--aux-loss-weight', type=float, default=0)
    parser.add_argument('--batch-size', '-b', type=int)
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--method', choices={'classification', 'regression', 'ranked'},
                        default='regression')
    parser.add_argument('--nli', action='store_true')
    parser.add_argument('--round-cefr', action='store_true')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--verbose', action='store_true')


def add_seq_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--doc-length', '-l', type=int)
    parser.add_argument('--embed-dim', type=int)
    parser.add_argument('--include-pos', action='store_true')
    parser.add_argument('--mixed-pos', action='store_true')
    parser.add_argument('--static-embs', action='store_true')
    parser.add_argument('--vectors', '-V', type=Path)
    parser.add_argument('--vocab-size', '-s', type=int)


def ranked_prediction(y_pred):
    extra_column = K.zeros((K.shape(y_pred)[0], 1), dtype='int32')
    threshold = K.constant(0.5, dtype=K.floatx())
    over_threshold = K.cast(K.greater(y_pred, threshold), 'int32')
    concatted = K.concatenate([over_threshold, extra_column], axis=1)
    return K.cast(K.argmin(concatted, axis=1), 'int32')


def ranked_accuracy(y_true, y_pred):
    true = ranked_prediction(y_true)
    pred = ranked_prediction(y_pred)
    return K.mean(K.equal(true, pred))


def get_targets_and_output_units(train_targets, dev_targets, method: str):
    if method == 'classification':
        train_y = to_categorical(train_targets)
        dev_y = to_categorical(dev_targets)
        output_units = [train_targets.max() + 1]
    elif method == 'regression':
        highest_class = max(train_targets)
        train_y = np.array(train_targets) / highest_class
        dev_y = np.array(dev_targets) / highest_class
        output_units = [1]
    elif method == 'ranked':
        train_y = to_ranked_rep(train_targets)
        dev_y = to_ranked_rep(dev_targets)
        output_units = [train_targets.max()]
    else:
        raise ValueError('Unknown method %r' % method)
    return [train_y], [dev_y], output_units


def to_ranked_rep(targets):
    rows = targets.shape[0]
    cols = int(targets.max())

    rep = np.zeros((rows, cols), dtype=int)
    for row, v in enumerate(targets):
        rep[row, :v] = 1
    return rep


def get_sequence_input_reps(args):
    if args.mixed_pos:
        w2i = make_mixed_pos2i()
        train_x, dev_x = mixed_pos_to_sequences(args.doc_length, ['train', 'dev'], w2i)
        args.vocab_size = len(w2i)
        num_pos = 0
    else:
        w2i = make_w2i(args.vocab_size)
        train_x, dev_x = words_to_sequences(args.doc_length, ['train', 'dev'], w2i)
        if args.include_pos:
            pos2i = make_pos2i()
            num_pos = len(pos2i)
            train_pos, dev_pos = pos_to_sequences(args.doc_length, ['train', 'dev'], pos2i)
            train_x = [train_x, train_pos]
            dev_x = [dev_x, dev_pos]
        else:
            num_pos = 0
    return train_x, dev_x, num_pos, w2i
