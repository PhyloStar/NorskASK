import argparse
import logging
from pathlib import Path
import pickle
from typing import Any, DefaultDict, Iterable, List  # noqa: F401

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
import seaborn as sns

from masterthesis.utils import RESULTS_DIR

sns.set()

logger = logging.getLogger(__name__)
logging.basicConfig()

extra = [
    RESULTS_DIR / 'cnn-26515464_6.pkl', RESULTS_DIR / 'cnn-26515464_13.pkl',
    RESULTS_DIR / 'cnn-26518498_6.pkl', RESULTS_DIR / 'cnn-26518498_13.pkl',
    RESULTS_DIR / 'rnn-26536430_15.pkl', RESULTS_DIR / 'rnn-26536431_15.pkl',
    RESULTS_DIR / 'rnn-26536430_27.pkl', RESULTS_DIR / 'rnn-26536431_27.pkl'
]

runs = ['26625017', '26628128']


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--debug', dest='loglevel', action='store_const', const=logging.DEBUG)
    parser.add_argument('--quiet', dest='loglevel', action='store_const', const=logging.WARN)
    parser.set_defaults(loglevel=logging.INFO)
    args = parser.parse_args()
    logging.getLogger(None).setLevel(args.loglevel)
    return args


def get_model_name(filename: str, cfg) -> str:
    if filename.startswith('cnn'):
        return 'cnn1' if cfg['mixed_pos'] else 'cnn2'
    elif filename.startswith('rnn'):
        return 'rnn1' if cfg['include_pos'] else 'rnn2'
    else:
        raise ValueError("Can't find rnn or cnn in filename: %s", filename)


def main():
    args = parse_args()
    model = []
    macrof1 = []
    collapsed = []
    lossweight = []
    for f in args.files + extra:
        res = pickle.load(f.open('rb'))
        cfg = res.config
        lossweight.append(cfg['aux_loss_weight'])
        mf1 = f1_score(res.true, res.predictions, average='macro')
        macrof1.append(mf1)
        collapsed.append(cfg['round_cefr'])
        model.append(get_model_name(f.name, cfg))
    df = pd.DataFrame.from_dict(
        {
            'Model': model,
            'Macro F1': macrof1,
            'Collapsed': collapsed,
            'Aux loss weight': lossweight
        }
    )
    sns.relplot(
        x='Aux loss weight',
        y='Macro F1',
        hue='Collapsed',
        col='Model',
        data=df,
        kind='line',
        col_wrap=2
    )
    print(df)
    plt.show()


if __name__ == '__main__':
    main()
