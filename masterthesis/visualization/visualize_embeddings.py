import argparse
import logging
from pathlib import Path
import pickle
from typing import Iterable

from keras.models import load_model, Model
import numpy as np
from sklearn.manifold import TSNE
import tqdm

from masterthesis.features.build_features import words_to_sequences
from masterthesis.gensim_utils import fingerprint, load_embeddings
from masterthesis.utils import (
    CEFR_LABELS, DATA_DIR, document_iterator, iso639_3, LANG_LABELS, load_split,
    REPRESENTATION_LAYER, safe_plt as plt
)

logging.basicConfig(level=logging.INFO)


class MockKeyedVectors:
    def __init__(self, vector_size: int) -> None:
        self.vector_size = vector_size

    def word_vec(self, w: str) -> np.ndarray:
        return np.random.randn(self.vector_size)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.word_vec(key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--split', choices=['train', 'dev', 'test'], default='dev')
    return parser.parse_args()


def get_fingerprints(embeddings: Path, filenames: Iterable[str]) -> np.ndarray:
    wv = load_embeddings(embeddings)
    txt_folder = DATA_DIR / 'txt'
    fingerprints = []
    print('Computing fingerprints of all documents ...')
    for filename in tqdm.tqdm(filenames):
        infile = txt_folder / Path(filename).with_suffix('.txt')
        with open(str(infile), encoding='utf-8') as f:
            fingerprints.append(fingerprint(wv, document_iterator(f)))
    return np.stack(fingerprints)


def get_model_representations(model_path: Path, split: str) -> np.ndarray:
    model = load_model(str(model_path))
    representation_model = Model(inputs=model.input,
                                 outputs=model.get_layer(REPRESENTATION_LAYER).output)
    w2i_path = model_path.parent / (model_path.stem + '_w2i.pkl')
    w2i = pickle.load(w2i_path.open('rb'))
    (x,) = words_to_sequences(700, [split], w2i)
    return representation_model.predict(x)


def main():
    args = parse_args()
    meta = load_split(args.split)
    meta.loc[:, 'lang'] = [iso639_3[l] for l in meta.lang]

    if args.embeddings:
        representations = get_fingerprints(args.embeddings, meta.filename)
    elif args.model:
        representations = get_model_representations(args.model, args.split)

    cefr_list = CEFR_LABELS
    testlevel_list = sorted(meta.testlevel.unique())
    lang_list = LANG_LABELS
    column_list = ['cefr', 'testlevel', 'lang']

    print('Computing t-SNE embeddings ...')
    embedded = TSNE(n_components=2, verbose=True).fit_transform(representations)

    fig, axes = plt.subplots(1, 3)
    for ax, label_list, col in zip(axes, [cefr_list, testlevel_list, lang_list], column_list):
        for label in label_list:
            mask = meta.loc[:, col] == label
            xs = embedded[mask, 0]
            ys = embedded[mask, 1]
            ax.plot(xs, ys, 'o', label=label)
        ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
