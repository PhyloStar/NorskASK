from typing import TextIO, Iterable, Tuple

import pandas as pd


def load_train_and_dev() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and dev splits as dataframes.

    Returns:
        Frames with the metadata for the documents in the train and
        dev splits.
    """
    df = pd.read_csv('metadata.csv').dropna(subset=['cefr'])
    train = df[df.split == 'train']
    dev = df[df.split == 'dev']
    return train, dev


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
