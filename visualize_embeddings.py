import logging
import sys
from collections import namedtuple
from pathlib import Path
from typing import TextIO, Iterable

from gensim.models.fasttext import FastTextKeyedVectors
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_fasttext_embeddings

logging.basicConfig(level=logging.INFO)

MockKeyedVectors = namedtuple('MockKeyedVectors', ['vector_size'])


def fingerprint(wv: FastTextKeyedVectors, document: Iterable[str]) -> np.ndarray:
    """Calculate the ``semantic fingerprint'' of a document.

    This algorithm is also known as ``continuous bag of words'' (CBOW).
    The result is the average of the embedding vectors for all tokens
    in the document.

    Args:
        wv: A fastText keyed vectors object.
        document: The document represented as an iterable of tokens

    Returns:
        A float array with the same shape as the embeddings in wv
    """
    cbow = np.zeros(wv.vector_size, dtype=float)
    token_count = 0
    for token in document:
        try:
            cbow += wv.word_vec(token)
        except KeyError:
            continue
        token_count += 1
    cbow /= token_count
    return cbow


def document_iterator(doc: TextIO) -> Iterable[str]:
    """Iterate over tokens in a document.

    Args:
        doc: A file object where each line is a line of tokens
            separated by a space

    Yields:
        The tokens in the document.
    """
    for line in doc:
        tokens_iter = iter(line.split(' '))
        yield next(tokens_iter)


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
