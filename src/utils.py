from pathlib import Path
from typing import TextIO, Iterable, Tuple, Union, List

import pandas as pd
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

    col_cutoff = values.max().max() / 2

    for i, row in enumerate(values):
        for j, count in enumerate(row):
            col = 'white' if count < col_cutoff else 'black'
            ax.text(j, i, int(count),
                    horizontalalignment='center',
                    verticalalignment='center', color=col)


def load_train_and_dev() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and dev splits as dataframes.

    Returns:
        Frames with the metadata for the documents in the train and
        dev splits.
    """
    df = pd.read_csv('ASK/metadata.csv').dropna(subset=['cefr'])
    train = df[df.split == 'train']
    dev = df[df.split == 'dev']
    return train, dev


def load_test() -> pd.DataFrame:
    """Load the test split as a dataframe.

    Returns:
        A frame with the metadata for the documents in the test split.
    """
    df = pd.read_csv('ASK/metadata.csv').dropna(subset=['cefr'])
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
