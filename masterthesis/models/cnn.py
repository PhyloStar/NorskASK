import argparse
import os
from pathlib import Path
import tempfile
from typing import Iterable, List

from keras.layers import Concatenate, Conv1D, Dense, Dropout, Embedding, GlobalMaxPooling1D, Input
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm

from masterthesis.features.build_features import (
    make_mixed_pos2i, make_pos2i, make_w2i, mixed_pos_to_sequences, pos_to_sequences,
    words_to_sequences
)
from masterthesis.gensim_utils import load_embeddings
from masterthesis.models.callbacks import F1Metrics
from masterthesis.models.report import report
from masterthesis.results import save_results
from masterthesis.utils import get_file_name, load_split, REPRESENTATION_LAYER, save_model

EMB_LAYER_NAME = 'embedding_layer'


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-b', type=int)
    parser.add_argument('--doc-length', '-l', type=int)
    parser.add_argument('--embed-dim', type=int)
    parser.add_argument('--epochs', '-e', type=int)
    parser.add_argument('--include-pos', action='store_true')
    parser.add_argument('--mixed-pos', action='store_true')
    parser.add_argument('--nli', action='store_true')
    parser.add_argument('--round-cefr', action='store_true')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--vectors', '-V', type=Path)
    parser.add_argument('--vocab-size', '-s', type=int)
    parser.add_argument('--windows', '-w', type=int_list)
    parser.set_defaults(batch_size=32, doc_length=700, embed_dim=50, epochs=50, vocab_size=4000,
                        windows=[4, 5, 6])
    return parser.parse_args()


def build_model(vocab_size: int, sequence_length: int, num_classes: int,
                windows: Iterable[int], num_pos: int = 0) -> Model:
    word_input_layer = Input((sequence_length,))
    word_embedding_layer = Embedding(vocab_size, 50, name=EMB_LAYER_NAME)(word_input_layer)
    if num_pos > 0:
        pos_input_layer = Input((sequence_length,))
        pos_embedding_layer = Embedding(num_pos, 10)(pos_input_layer)
        embedding_layer = Concatenate()([word_embedding_layer, pos_embedding_layer])
        inputs = [word_input_layer, pos_input_layer]
    else:
        embedding_layer = word_embedding_layer
        inputs = [word_input_layer]

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
    output_layer = Dense(num_classes, activation='softmax')(dropout_layer)
    return Model(inputs=inputs, outputs=output_layer)


def main():
    args = parse_args()
    seq_length = args.doc_length
    train = load_split('train', round_cefr=args.round_cefr)
    dev = load_split('dev', round_cefr=args.round_cefr)

    y_column = 'lang' if args.nli else 'cefr'
    labels = sorted(train[y_column].unique())

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

    train_y = to_categorical([labels.index(c) for c in train[y_column]])
    dev_y = to_categorical([labels.index(c) for c in dev[y_column]])

    model = build_model(args.vocab_size, seq_length, len(labels),
                        windows=args.windows, num_pos=num_pos)
    model.summary()

    if args.vectors:
        if not args.vectors.is_file():
            if 'SUBMITDIR' in os.environ:
                args.vectors = Path(os.environ['SUBMITDIR'] / args.vectors)
            print('New path: %r' % args.vectors)
        if not args.vectors.is_file():
            print('Embeddings path not available, searching for submitdir')
        else:
            kv = load_embeddings(args.vectors)
            embed_dim = kv.vector_size
            embeddings_matrix = np.zeros((args.vocab_size, embed_dim))
            print('Making embeddings:')
            for word, idx in tqdm(w2i.items(), total=len(w2i)):
                vec = kv.word_vec(word)
                embeddings_matrix[idx, :] = vec
            model.get_layer(EMB_LAYER_NAME).set_weights([embeddings_matrix])

    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    temp_handle, weights_path = tempfile.mkstemp(suffix='.h5')
    callbacks = [F1Metrics(dev_x, dev_y, weights_path)]
    history = model.fit(
        train_x, train_y, epochs=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, validation_data=(dev_x, dev_y),
        verbose=2)
    model.load_weights(weights_path)
    os.close(temp_handle)
    os.remove(weights_path)

    name = 'cnn'
    if args.nli:
        name = 'cnn-nli'
    name = get_file_name(name)

    if args.save_model:
        save_model(name, model, w2i)

    predictions = model.predict(dev_x)
    true = np.argmax(dev_y, axis=1)
    pred = np.argmax(predictions, axis=1)
    report(true, pred, labels)

    save_results(name, args.__dict__, history.history, true, pred)


if __name__ == '__main__':
    main()
