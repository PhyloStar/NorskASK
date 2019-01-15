"""Assorted utility functions and values.

safe_plt: A Pyplot that won't attempt to open a display if not available
iso639_3: A mapping of Norwegian language names (as used in the data) to
    ISO639_3 codes.
"""
import itertools
import os
from pathlib import Path
from typing import TextIO, Iterable, Tuple, Union, Sequence, Optional, List

import pandas as pd
import numpy as np
import matplotlib
if 'SLURM_JOB_NODELIST' in os.environ or \
        (os.name == 'posix' and 'DISPLAY' not in os.environ):  # noqa: E402
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

safe_plt = plt

conll_cols = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']

iso639_3 = dict(
    engelsk='eng',
    polsk='pol',
    russisk='rus',
    somali='som',
    spansk='spa',
    tysk='deu',
    vietnamesisk='vie'
)

project_root = Path(__file__).resolve().parents[1]  # type: Path


def heatmap(values: np.ndarray,
            xticks: Sequence[str],
            yticks: Sequence[str],
            ax: Optional[plt.Axes] = None) -> None:
    """Plot a 2D array as a heatmap with overlayed values.

    Args:
        values: The 2D array to plot
        xticks: The labels to place on the X axis
        yticks: The labels to place on the Y axis
        ax: An optional Axes object to plot the heatmap on
    """
    if ax is None:
        ax = plt.gca()
    ax.imshow(values, cmap='viridis')
    ax.set(
        yticks=range(len(yticks)),
        xticks=range(len(xticks)),
        yticklabels=yticks,
        xticklabels=xticks
    )
    color_cutoff = values.max() / 2

    for row, col in itertools.product(range(values.shape[0]), range(values.shape[1])):
        val = values[row, col]
        color = 'white' if val < color_cutoff else 'black'
        if np.issubdtype(values.dtype, np.floating):
            label = '%.2f' % val
        else:
            label = str(int(val))
        ax.text(col, row, label,
                horizontalalignment='center',
                verticalalignment='center', color=color)


def load_train_and_dev() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and dev splits as dataframes.

    Args:
        project_root: Useful if running a script from somewhere else
            than the project root dir, such as a notebook

    Returns:
        Frames with the metadata for the documents in the train and
        dev splits.
    """
    return load_split('train'), load_split('dev')


def round_cefr_score(cefr: str) -> str:
    """Round intermediate CEFR levels up.

    >>> round_cefr('A2')
    'A2'
    >>> round_cefr('B1/B2')
    'B2'
    """
    return cefr[-2:] if '/' in cefr else cefr


def load_split(split: str, round_cefr: bool = False) -> pd.DataFrame:
    """Load the test split as a dataframe.

    Args:
        split: {train, dev, test}

    Returns:
        A frame with the metadata for documents in the requested split.
    """
    if split not in ['train', 'dev', 'test']:
        raise ValueError('Split must be train, dev or test')
    filepath = project_root / 'ASK/metadata.csv'
    df = pd.read_csv(filepath).dropna(subset=['cefr'])
    if round_cefr:
        df.loc[:, 'cefr'] = df.cefr.apply(round_cefr_score)
    return df[df.split == split]


def load_test() -> pd.DataFrame:
    """Load the test split as a dataframe.

    Args:
        project_root: Useful if running a script from somewhere else
            than the project root dir, such as a notebook

    Returns:
        A frame with the metadata for the documents in the test split.
    """
    return load_split('test')


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


def conll_reader(file: Union[str, Path],
                 cols: Sequence[str],
                 tags: bool = False) -> Iterable[List[Tuple[str, ...]]]:
    """Iterate over sentences in a CoNLL file.

    Args:
        file: The CoNLL file to read from
        cols: The columns to read
        tags: Whether to include start and end tags around sentences.
            The tags are '<s>' and '</s>' regardless of column.

    Yields:
        Each sentence in file as a list of tuples corresponding to the
        specified cols. If only a single column is specified, the list
        will contain simple values instead of tuples.
    """
    if isinstance(file, str):
        file = Path(file)
    try:
        col_idx = [conll_cols.index(c) for c in cols]
    except ValueError as e:
        raise ValueError('All column names must be one of %s' % set(conll_cols))
    if tags:
        start_tags = tuple('<s>' for __ in cols)
        end_tags = tuple('</s>' for __ in cols)
    with file.open(encoding='utf8') as stream:
        tuple_sequence = []  # type: List[Tuple[str, ...]]
        for line in stream:
            line = line.strip()
            if line.startswith('#'):
                continue
            if not line:  # Empty line = end of sentence
                if tags:
                    yield [start_tags] + tuple_sequence + [end_tags]
                else:
                    yield tuple_sequence
                tuple_sequence = []
            else:
                fields = line.split('\t')
                tup = tuple(fields[i] for i in col_idx)
                tuple_sequence.append(tup)
    if tuple_sequence:  # Flush if there is no empty line at end of file
        yield tuple_sequence
