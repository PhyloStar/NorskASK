from itertools import chain
from typing import Iterable

from sklearn.feature_extraction.text import CountVectorizer

from masterthesis.utils import load_split, project_root


data_folder = project_root / 'ASK'


def iterate_tokens(split: str = 'train') -> Iterable[str]:
    return chain.from_iterable(iterate_docs(split))


def iterate_docs(split: str = 'train') -> Iterable[Iterable[str]]:
    def _inner_iter(stream):
        for line in stream:
            for token in line.split():
                yield token

    meta = load_split(split)
    for filename in meta.filename:
        path = (data_folder / 'txt' / filename).with_suffix('.txt')
        with path.open() as stream:
            yield _inner_iter(stream)


def bag_of_words(split, **kwargs):
    vectorizer = CountVectorizer(input='filename', **kwargs)
    meta = load_split(split)
    x = vectorizer.fit_transform(filename_iter(meta))
    return x, vectorizer


def filename_iter(meta, suffix='txt'):
    for filename in meta.filename:
        yield str((data_folder / suffix / filename).with_suffix('.' + suffix))
