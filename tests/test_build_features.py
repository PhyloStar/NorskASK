from pathlib import Path
from unittest.mock import patch

from masterthesis.features.build_features import iterate_docs, iterate_tokens

test_data_dir = Path(__file__).parent / 'test_data'


class MockMeta:
    filename = ['sample_doc']


@patch('masterthesis.features.build_features.load_split')
@patch('masterthesis.features.build_features.data_folder', new=test_data_dir)
def test_iterate_docs(mock_load_split):
    mock_load_split.return_value = MockMeta()
    sents = [s for d in iterate_docs() for s in d]
    assert sents[0] == 'Dette'
    assert sents[-1] == '.'


@patch('masterthesis.features.build_features.load_split')
@patch('masterthesis.features.build_features.data_folder', new=test_data_dir)
def test_iterate_tokens(mock_load_split):
    mock_load_split.return_value = MockMeta()
    tokens = list(iterate_tokens())
    assert tokens[:4] == 'Dette er første linje'.split()
    assert tokens[-4:] == 'har to setninger .'.split()
