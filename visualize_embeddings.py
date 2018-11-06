import logging
import sys
from collections import namedtuple
from pathlib import Path

from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import document_iterator
from gensim_utils import load_fasttext_embeddings, fingerprint

logging.basicConfig(level=logging.INFO)

class MockKeyedVectors:
    def __init__(self, vector_size):
        self.vector_size = vector_size

    def word_vec(self, w):
        return np.random.randn(self.vector_size)


def main():
    txt_folder = Path('ASK/txt')
    meta = pd.read_csv('metadata.csv').dropna(subset=['cefr'])

    wv = load_fasttext_embeddings(sys.argv[1])
    # wv = MockKeyedVectors(vector_size=10)

    fingerprints = []
    print('Computing fingerprints of all documents ...')
    for filename in meta.filename:
        infile = txt_folder / Path(filename).with_suffix('.txt')
        with open(str(infile)) as f:
            fingerprints.append(fingerprint(wv, document_iterator(f)))
    fingerprints_matrix = np.stack(fingerprints)
    cefr_list = ['A2', 'A2/B1', 'B1', 'B1/B2', 'B2', 'B2/C1', 'C1']
    testlevel_list = ['Språkprøven', 'Høyere nivå']
    lang_list = ['tysk', 'russisk', 'polsk', 'somali', 'vietnametisk', 'spansk', 'engelsk']
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
