import itertools
from pathlib import Path
from typing import TextIO, Iterable, Tuple, Union, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


iso639_3 = dict(
    engelsk='eng',
    polsk='pol',
    russisk='rus',
    somali='som',
    spansk='spa',
    tysk='deu',
    vietnamesisk='vie'
)


def heatmap(values, xticks, yticks, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.imshow(values, cmap='viridis')
    ax.set(
        yticks=range(len(yticks)),
        xticks=range(len(xticks)),
        yticklabels=yticks,
        xticklabels=xticks
    )
    col_cutoff = values.max() / 2

    for row, col in itertools.product(range(values.shape[0]), range(values.shape[1])):
        val = values[row, col]
        color = 'white' if val < col_cutoff else 'black'
        if np.issubdtype(values.dtype, np.floating):
            label = '%.2f' % val
        else:
            label = str(int(val))
        ax.text(col, row, label,
                horizontalalignment='center',
                verticalalignment='center', color=color)


def load_train_and_dev(
        project_root: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and dev splits as dataframes.

    Returns:
        Frames with the metadata for the documents in the train and
        dev splits.
    """
    filepath = Path('ASK/metadata.csv')
    if project_root is not None:
        filepath = Path(project_root) / filepath
    df = pd.read_csv(filepath).dropna(subset=['cefr'])
    train = df[df.split == 'train']
    dev = df[df.split == 'dev']
    return train, dev


def load_test(project_root: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load the test split as a dataframe.

    Returns:
        A frame with the metadata for the documents in the test split.
    """
    filepath = Path('ASK/metadata.csv')
    if project_root is not None:
        filepath = Path(project_root) / filepath
    df = pd.read_csv(filepath).dropna(subset=['cefr'])
    return df[df.split == 'test']


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


def conll_reader(file: Union[str, Path]) -> Iterable[List[str]]:
    if isinstance(file, str):
        file = Path(file)
    with file.open(encoding='utf8') as stream:
        pos_sequence = ['<s>']
        for line in stream:
            line = line.strip()
            if line.startswith('#'):
                continue
            if not line:
                pos_sequence.append('</s>')
                yield pos_sequence
                pos_sequence = ['<s>']
            else:
                pos = line.split('\t')[3]
                pos_sequence.append(pos)
