from masterthesis.utils import round_cefr_score


def test_round_cefr_score():
    assert round_cefr_score('A2') == 'A2'
    assert round_cefr_score('B1/B2') == 'B2'
    assert round_cefr_score('B2/C1') == 'C1'
