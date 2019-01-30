"""Store pre-trained vectors for the vocabulary in training set"""
import argparse
import logging
from pathlib import Path

import numpy as np
import tqdm
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors

from masterthesis.gensim_utils import load_fasttext_embeddings
from masterthesis.features.build_features import iterate_tokens

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('outputdir', type=Path)
    parser.add_argument('--outputname', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError('%r is not a file' % args.input)
    if not args.outputdir.is_dir():
        raise FileNotFoundError('%r is not a directory' % args.outputdir)
    if args.outputname:
        outfile = args.outputdir / args.outputname
    else:
        name = args.input.stem + '-small.pkl'
        outfile = args.outputdir / name

    kv = load_fasttext_embeddings(args.input)

    vector_size = kv.vector_size
    words = list(set(iterate_tokens('train'))) + ['__UNK__', '__PAD__']
    embeddings = np.zeros((len(words), vector_size))
    for row, word in tqdm.tqdm(enumerate(words)):
        embeddings[row, :] = kv.word_vec(word)

    new_kv = WordEmbeddingsKeyedVectors(vector_size)
    new_kv.add(words, embeddings)
    new_kv.save(str(outfile))


if __name__ == '__main__':
    main()
