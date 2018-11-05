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

MockKeyedVectors = namedtuple('MockKeyedVectors', ['vector_size'])


def main():
    txt_folder = Path('ASK/txt')
    meta = pd.read_csv('metadata.csv')

    wv = load_fasttext_embeddings(sys.argv[1])
    # wv = MockKeyedVectors(vector_size=100)

    labels = []
    fingerprints = []
    print('Computing fingerprints of all documents ...')
    for input_file in txt_folder.iterdir():
        metadata_row = meta[meta.filename == input_file.stem].iloc[0]
        if metadata_row.cefr is np.nan:
            continue
        label = metadata_row.cefr
        labels.append(label)
        with open(str(input_file)) as f:
            fingerprints.append(fingerprint(wv, document_iterator(f)))
    fingerprints_matrix = np.stack(fingerprints)
    label_list = ['A2', 'A2/B1', 'B1', 'B1/B2', 'B2', 'B2/C1', 'C1']
    print('Computing t-SNE embeddings ...')
    embedded = TSNE(n_components=2).fit_transform(fingerprints_matrix)
    for cefr in label_list:
        mask = np.array(labels) == cefr
        xs = embedded[mask, 0]
        ys = embedded[mask, 1]
        plt.plot(xs, ys, 'o', label=cefr)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
