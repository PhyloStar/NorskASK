import logging
import sys
from pathlib import Path
from typing import TextIO, Iterable

from gensim.models.wrappers.fasttext import FastTextKeyedVectors
import numpy as np
import pandas as pd

from utils import load_fasttext_embeddings

logging.basicConfig(level=logging.INFO)


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
        cbow += wv.word_vector(token)
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

    labels = []
    fingerprints = []
    for input_file in txt_folder.iterdir():
        label = meta[meta.filename == input_file.stem].cefr[0]
        labels.append(label)
        with open(input_file) as f:
            fingerprints.append(fingerprint(
                wv, document_iterator(f)))
    fingerprints_matrix = np.stack(fingerprints)
    print(labels)
    print(fingerprints_matrix.shape)


if __name__ == '__main__':
    main()
