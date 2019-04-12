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
    '26648425', '26648427', '26649222', '26649223', '26649224', '26649225', '26679428', '26679429',
    '26679430', '26679431'
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
    file_lookup = {'cnn1': cnn1, 'cnn2': cnn2, 'rnn1': rnn1, 'rnn2': rnn2}
    true = pickle.load(file_lookup[model_name].open('rb'))

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
    seed_delta = []
    globs = chain.from_iterable(Path('results').glob('*%s*' % run) for run in runs)
    for f in chain(args.files, globs, extra):
        res = pickle.load(f.open('rb'))
        cfg = res.config
        model_name = get_model_name(f.name, cfg)
        if not validate_model(model_name, cfg):
            logger.warn('Skipping %r', f)
            continue
        lossweight.append(cfg['aux_loss_weight'])
        seed_delta.append(cfg.get('seed_delta', 0))
        mf1 = f1_score(res.true, res.predictions, average='macro')
        macrof1.append(mf1)
        collapsed.append(cfg['round_cefr'])
        model.append(model_name)
    df = pd.DataFrame.from_dict(
        {
            'Model': model,
            'Macro F1': macrof1,
            'Collapsed': collapsed,
            'Aux loss weight': lossweight,
            'Seed delta': seed_delta
        }
    )
    g = sns.FacetGrid(data=df, col='Model', col_wrap=2, col_order=['cnn1', 'cnn2', 'rnn1', 'rnn2'])
    g.map_dataframe(sns.lineplot, 'Aux loss weight', 'Macro F1', style='Collapsed')
    g.map_dataframe(sns.scatterplot, "Aux loss weight", 'Macro F1', style='Collapsed')
    g.add_legend(label_order=['Collapsed', 'True', 'False'])
    print(df.groupby(['Collapsed', 'Model']).size())
    print(df.query('~Collapsed').nlargest(5, 'Macro F1'))
    print(df.query('Collapsed').nlargest(5, 'Macro F1'))
    plt.show()


if __name__ == '__main__':
    main()
