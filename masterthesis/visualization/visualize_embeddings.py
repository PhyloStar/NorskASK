import argparse
import logging
from pathlib import Path

from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import tqdm

from masterthesis.utils import document_iterator, load_train_and_dev
from masterthesis.utils import safe_plt as plt
from masterthesis.gensim_utils import load_fasttext_embeddings, fingerprint

logging.basicConfig(level=logging.INFO)


class MockKeyedVectors:
    def __init__(self, vector_size):
        self.vector_size = vector_size

    def word_vec(self, w):
        return np.random.randn(self.vector_size)

    def __getitem__(self, key):
        return self.word_vec(key)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('embeddings', nargs='?')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    txt_folder = Path('ASK/txt')
    meta = pd.concat(load_train_and_dev()).reset_index()

    if args.debug:
        wv = MockKeyedVectors(vector_size=10)
    else:
        try:
            wv = load_fasttext_embeddings(args.embeddings)
        except:
            wv = KeyedVectors.load(args.embeddings)

    fingerprints = []
    print('Computing fingerprints of all documents ...')
    for filename in tqdm.tqdm(meta.filename):
        infile = txt_folder / Path(filename).with_suffix('.txt')
        with open(str(infile), encoding='utf-8') as f:
            fingerprints.append(fingerprint(wv, document_iterator(f)))
    fingerprints_matrix = np.stack(fingerprints)

    cefr_list = sorted(meta.cefr.unique())
    testlevel_list = sorted(meta.testlevel.unique())
    lang_list = sorted(meta.lang.unique())
    column_list = ['cefr', 'testlevel', 'lang']

    print('Computing t-SNE embeddings ...')
    embedded = TSNE(n_components=2).fit_transform(fingerprints_matrix)

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
