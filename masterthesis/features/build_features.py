from itertools import chain
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Mapping
try:
    from typing import Counter
except ImportError:
    from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import tqdm

from masterthesis.utils import conll_reader, get_split_len, get_stopwords, load_split, PROJECT_ROOT


data_folder = PROJECT_ROOT / 'ASK'


def iterate_tokens(split: str = 'train') -> Iterable[str]:
    return chain.from_iterable(iterate_docs(split))


def iterate_mixed_pos_tags(split: str = 'train') -> Iterable[str]:
    return chain.from_iterable(iterate_mixed_pos_docs(split))


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


def iterate_mixed_pos_docs(split: str = 'train'):
    """Mixed POS-Function Word n-grams.

    E.g. NOUN kan også VERB NOUN til å VERB dem
    """
    sw = get_stopwords()

    def _inner_iter(stream):
        for sent in sents:
            for (form, pos) in sent:
                if form.lower() in sw:
                    yield form
                else:
                    yield pos

    meta = load_split(split)
    filenames = filename_iter(meta, suffix='conll')
    for filename in filenames:
        sents = conll_reader(filename, cols=['FORM', 'UPOS'], tags=False)
        yield _inner_iter(sents)


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


def _make_any2i(tokens_by_rank: Iterable[str]) -> Dict[str, int]:
    any2i = {'__PAD__': 0, '__UNK__': 1}
    for rank, token in enumerate(tokens_by_rank, start=2):
        any2i[token] = rank
    return any2i


def make_w2i(vocab_size: Optional[int]) -> Dict[str, int]:
    print('Counting tokens ...')
    tokens = Counter(tqdm.tqdm(iterate_tokens('train')))
    # If vocab_size is not None, make room for __PAD__ and __UNK__
    vocab_size = vocab_size and vocab_size - 2
    most_common = (token for (token, __) in tokens.most_common(vocab_size))
    return _make_any2i(most_common)


def make_pos2i() -> Dict[str, int]:
    print('Counting POS tags ...')
    tokens = Counter(tqdm.tqdm(iterate_pos_tags('train')))
    most_common = (token for (token, __) in tokens.most_common())
    return _make_any2i(most_common)


def make_mixed_pos2i() -> Dict[str, int]:
    print('Counting mixed POS tags ...')
    tokens = Counter(tqdm.tqdm(iterate_mixed_pos_tags('train')))
    most_common = (token for (token, __) in tokens.most_common())
    return _make_any2i(most_common)


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


def mixed_pos_to_sequences(seq_len: int,
                           splits: Iterable[str],
                           w2i: Mapping[str, int]) -> List[np.ndarray]:
    return _x_to_sequences(seq_len, splits, w2i, iterate_mixed_pos_docs)


def file_to_sequence(seq_len: int, filepath: Path, w2i: Mapping[str, int]) -> np.ndarray:
    x = np.zeros(seq_len, int)
    with filepath.open() as f:
        idx = 0
        for line in f:
            for token in line.strip().split():
                if token not in w2i:
                    token = '__UNK__'
                x[idx] = w2i[token]
                idx += 1
                if idx >= seq_len:
                    return x
    return x
