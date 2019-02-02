"""Utils using Gensim.

A separate module because it can take a long time to import gensim, and
we want to avoid that when not necessary.
"""

import logging
from pathlib import Path
from typing import Iterable, Union
import zipfile

from gensim.models import FastText
from gensim.models.keyedvectors import FastTextKeyedVectors, KeyedVectors
import numpy as np

logger = logging.getLogger(__name__)


def load_embeddings(file: Union[Path, str], fasttext: bool = False) -> KeyedVectors:
    """Load embeddings from file and unit normalize vectors."""
    if fasttext:
        return load_fasttext_embeddings(file)

    if isinstance(file, str):
        file = Path(file)

    # Native Gensim format?
    emb_model = KeyedVectors.load(str(file))

    emb_model.init_sims(replace=True)  # Unit-normalizing the vectors (if they aren't already)
    return emb_model.wv


def load_fasttext_embeddings(file: Union[Path, str]) -> FastTextKeyedVectors:
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


def fingerprint(wv: KeyedVectors, document: Iterable[str]) -> np.ndarray:
    """Calculate the ``semantic fingerprint'' of a document.

    This algorithm is also known as ``continuous bag of words'' (CBOW).
    The result is the average of the embedding vectors for all tokens
    in the document.

    Args:
        wv: A keyed vectors object.
        document: The document represented as an iterable of tokens

    Returns:
        A float array with the same shape as the embeddings in wv
    """
    cbow = np.zeros(wv.vector_size, dtype=float)
    token_count = 0
    for token in document:
        try:
            cbow += wv[token]
        except KeyError:
            logger.debug("Could not make embedding vector for %s", token)
            continue
        token_count += 1
    cbow /= token_count
    return cbow
