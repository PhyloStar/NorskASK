import argparse
from itertools import chain
import logging
from pathlib import Path
import pickle
from typing import Any, DefaultDict, Iterable, List  # noqa: F401

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
import seaborn as sns

from masterthesis.utils import RESULTS_DIR

sns.set(context='paper', style='whitegrid')

logger = logging.getLogger(__name__)
logging.basicConfig()

cnn1 = RESULTS_DIR / 'cnn-26515464_6.pkl'
cnn2 = RESULTS_DIR / 'cnn-26515464_13.pkl'
rnn1 = RESULTS_DIR / 'rnn-26536430_15.pkl'
rnn2 = RESULTS_DIR / 'rnn-26536430_27.pkl'

extra = [
    cnn1, RESULTS_DIR / 'cnn-26518498_6.pkl', cnn2, RESULTS_DIR / 'cnn-26518498_12.pkl', rnn1,
    RESULTS_DIR / 'rnn-26536431_15.pkl', rnn2, RESULTS_DIR / 'rnn-26536431_27.pkl'
]
runs = [
    '26646835', '26646836', '26646837', '26646838', '26646843', '26646844', '26646845', '26646846',
    '26646847', '26646848', '26646849', '26646850', '26646851', '26646852', '26647292', '26647293',
    '26647294', '26647295', '26648419', '26648420', '26648421', '26648422', '26648423', '26648424',
    '26648425', '26648427', '26649222', '26649223', '26649224', '26649225'
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*', type=Path)
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


def validate_model(model_name, cfg):
    if model_name == 'cnn1':
        true = pickle.load(cnn1.open('rb'))
    elif model_name == 'cnn2':
        true = pickle.load(cnn2.open('rb'))
    elif model_name == 'rnn1':
        true = pickle.load(rnn1.open('rb'))
    elif model_name == 'rnn2':
        true = pickle.load(rnn2.open('rb'))
    else:
        raise ValueError

    for k, v in true.config.items():
        if k in ['aux_loss_weight', 'round_cefr']:
            continue
        if v != cfg[k]:
            logger.warn('Wrong value for %s: %s != %s' % (k, cfg[k], v))
            logger.warn('Not a %s' % model_name)
            return False
    return True


def main():
    args = parse_args()
    model = []
    macrof1 = []
    collapsed = []
    lossweight = []
    globs = chain.from_iterable(Path('results').glob('*%s*' % run) for run in runs)
    for f in chain(args.files, globs, extra):
        res = pickle.load(f.open('rb'))
        cfg = res.config
        model_name = get_model_name(f.name, cfg)
        if not validate_model(model_name, cfg):
            logger.warn('Skipping %r', f)
            continue
        lossweight.append(cfg['aux_loss_weight'])
        mf1 = f1_score(res.true, res.predictions, average='macro')
        macrof1.append(mf1)
        collapsed.append(cfg['round_cefr'])
        model.append(model_name)
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
        style='Collapsed',
        col='Model',
        data=df,
        ci='sd',
        kind='line',
        col_wrap=2,
    )
    print(df.groupby(['Collapsed', 'Model']).count())
    plt.show()


if __name__ == '__main__':
    main()
