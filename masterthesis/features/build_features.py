from itertools import chain
from typing import Callable, Iterable, List, Mapping
try:
    from typing import Counter
except ImportError:
    from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import tqdm

from masterthesis.utils import conll_reader, get_split_len, load_split, PROJECT_ROOT


data_folder = PROJECT_ROOT / 'ASK'


def iterate_tokens(split: str = 'train') -> Iterable[str]:
    return chain.from_iterable(iterate_docs(split))


def iterate_pos_tags(split: str = 'train') -> Iterable[str]:
    return chain.from_iterable(iterate_pos_docs(split))


def iterate_pos_docs(split: str = 'train') -> Iterable[Iterable[str]]:
    def _inner_iter(stream):
        for sent in sents:
            for (pos,) in sent:
                yield pos

    meta = load_split(split)
    filenames = filename_iter(meta, suffix='conll')
    for filename in filenames:
        sents = conll_reader(filename, cols=['UPOS'], tags=False)
        yield _inner_iter(sents)


def iterate_docs(split: str = 'train') -> Iterable[Iterable[str]]:
    def _inner_iter(stream):
        for line in stream:
            for token in line.split():
                yield token

    meta = load_split(split)
    for filename in meta.filename:
        path = (data_folder / 'txt' / filename).with_suffix('.txt')
        with path.open(encoding='utf-8') as stream:
            yield _inner_iter(stream)


def bag_of_words(split, **kwargs):
    """Fit a CountVectorizer on a split.

    Args:
        split: Name of the split {'train', 'test', 'dev}
        **kwargs: Are passed to CountVectorizer's constructor

    Returns:
        The transformed documents and the trained vectorizer.
    """
    vectorizer = CountVectorizer(input='filename', **kwargs)
    meta = load_split(split)
    x = vectorizer.fit_transform(filename_iter(meta))
    return x, vectorizer


def filename_iter(meta, suffix='txt') -> Iterable[str]:
    for filename in meta.filename:
        yield str((data_folder / suffix / filename).with_suffix('.' + suffix))


def make_w2i(vocab_size):
    print('Counting tokens ...')
    tokens = Counter(tqdm.tqdm(iterate_tokens('train')))
    most_common = (token for (token, __) in tokens.most_common())

    w2i = {'__PAD__': 0, '__UNK__': 1}
    for rank, token in zip(range(2, vocab_size), most_common):
        w2i[token] = rank
    return w2i


def make_pos2i():
    print('Counting POS tags ...')
    tokens = Counter(tqdm.tqdm(iterate_pos_tags('train')))
    most_common = (token for (token, __) in tokens.most_common())

    w2i = {'__PAD__': 0, '__UNK__': 1}
    for rank, token in zip(range(2, len(tokens) + 2), most_common):
        w2i[token] = rank
    return w2i


def _x_to_sequences(seq_len: int,
                    splits: Iterable[str],
                    mapping: Mapping[str, int],
                    doc_iterator: Callable[[str], Iterable[Iterable[str]]]) -> List[np.ndarray]:
    out = []
    for split in splits:
        split_len = get_split_len(split)
        print("Preprocessing split '%s' ..." % split)
        x = np.zeros((split_len, seq_len), int)
        for row, doc in tqdm.tqdm(enumerate(doc_iterator(split)), total=split_len):
            for col, token in zip(range(seq_len), doc):
                if token not in mapping:
                    token = '__UNK__'
                x[row, col] = mapping[token]
        out.append(x)
    return out


def pos_to_sequences(seq_len: int,
                     splits: Iterable[str],
                     pos2i: Mapping[str, int]) -> List[np.ndarray]:
    return _x_to_sequences(seq_len, splits, pos2i, iterate_pos_docs)


def words_to_sequences(seq_len: int,
                       splits: Iterable[str],
                       w2i: Mapping[str, int]) -> List[np.ndarray]:
    return _x_to_sequences(seq_len, splits, w2i, iterate_docs)
