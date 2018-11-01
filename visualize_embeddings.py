import logging
import zipfile
import sys
from pathlib import Path
from typing import Union, TextIO

from gensim.models.wrappers.fasttext import FastTextKeyedVectors
from gensim.models import FastText
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)


def load_fasttext_embeddings(file: Union[Path, str]) -> FastText:
    """Load embeddings from file and unit normalize vectors."""
    if isinstance(file, str):
        file = Path(file)
    # Detect the model format by its extension:
    if '.bin' in file.suffixes or '.vec' in file.suffixes:
        # Binary word2vec format
        emb_model = FastText.load_fasttext_format(str(file))
    elif file.suffix == '.zip':
        # ZIP archive from the NLPL vector repository
        with zipfile.ZipFile(str(file), "r") as archive:
            model_file = archive.extract('parameters.bin')
            emb_model = FastText.load_fasttext_format(model_file)
    else:
        # Native Gensim format?
        emb_model = FastText.load(str(file))

    emb_model.init_sims(replace=True)  # Unit-normalizing the vectors (if they aren't already)
    return emb_model.wv


def fingerprint(wv: FastTextKeyedVectors, document):
    cbow = np.zeros(wv.vector_size, dtype=float)
    token_count = 0
    for token in document:
        cbow += wv.word_vector(token)
        token_count += 1
    cbow /= token_count
    return cbow


def document_iterator(doc: TextIO):
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
