from pathlib import Path
from unittest.mock import patch

from numpy.testing import assert_equal

from masterthesis.features.build_features import iterate_docs, iterate_tokens
from masterthesis.models.taghipour_ng import preprocess

test_data_dir = Path(__file__).parent / 'test_data'
test_doc_len = 21


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


@patch('masterthesis.features.build_features.load_split')
@patch('masterthesis.features.build_features.data_folder', new=test_data_dir)
def test_vectorizer(mock_load_split):
    seq_len = test_doc_len + 1
    w2i = {
        "__PAD__": 0,
        "__UNK__": 1,
        "Dette": 2,
        "er": 3,
        "første": 4,
        "linje": 5,
    }
    mock_load_split.return_value = MockMeta()
    t, d = preprocess(seq_len, [0], [0], w2i)
    assert_equal(t[0, :4], [2, 3, 4, 5])  # Dette er første linje
    assert_equal(t[0, 4:7], [1, 1, 1])  # i dokumentet .
    assert t[0, -1] == 0  # __PAD__
