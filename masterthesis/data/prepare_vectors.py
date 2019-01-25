"""Store pre-trained vectors for the vocabulary in training set"""
import logging
import sys

import numpy as np
import tqdm
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors

from masterthesis.gensim_utils import load_fasttext_embeddings
from masterthesis.features.build_features import iterate_tokens

logging.basicConfig(level=logging.INFO)


def main():
    if len(sys.argv) != 2:
        raise ValueError("Needs exactly one argument (embedding path)")
    emb_path = sys.argv[1]
    kv = load_fasttext_embeddings(emb_path)
    vector_size = kv.vector_size
    words = [set(iterate_tokens('train'))]
    embeddings = np.zeros((len(words), vector_size))
    for row, word in tqdm.tqdm(enumerate(words)):
        embeddings[row, :] = kv.word_vec(word)

    new_kv = WordEmbeddingsKeyedVectors(vector_size)
    new_kv.add(words, embeddings)
    new_kv.save('new_vectors.vec')


if __name__ == '__main__':
    main()
