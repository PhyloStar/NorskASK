from collections import Counter
from math import sqrt
import pickle
from typing import List

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from masterthesis.utils import RESULTS_DIR


def pi_k(a: List[int], b: List[int]) -> float:
    n_a = len(a)
    n_b = len(b)
    count_a = Counter(a)
    count_b = Counter(b)
    ks = list(range(min(a + b), max(a + b)))
    pis = []
    for k in ks:
        pis.append((count_a[k] / n_a + count_b[k] / n_b) / 2)
    num_classes = len(pis)
    ac = (1 / (num_classes - 1)) * sum(p * (1 - p) for p in pis)
    return ac


def main():
    data = {'filename': [], 'pearson': [], 'spearman': [], 'rmse': [], 'n_class': []}
    for results_file in RESULTS_DIR.iterdir():
        try:
            res = pickle.load(results_file.open('rb'))
        except Exception:
            print('Could not read file %s' % results_file)
            continue
        try:
            gold = res.true
            pred = res.predictions
        except AttributeError:
            print('Could not find gold and pred for file %s' % results_file)
            continue
        data['filename'].append(results_file.name)
        data['pearson'].append(pearsonr(gold, pred)[0])
        data['spearman'].append(spearmanr(gold, pred)[0])
        data['n_class'].append(max(max(gold), max(pred)) + 1)
        mse = sum((a - b) ** 2 for a, b in zip(gold, pred)) / len(gold)
        rmse = sqrt(mse)
        data['rmse'].append(rmse)
    df = pd.DataFrame.from_dict(data)
    print('== All labels ==')
    lim_df = df[df.n_class == 7]
    print('\nTop Pearson:')
    print(lim_df.sort_values('pearson', ascending=False)
                .loc[:, ['filename', 'pearson', 'n_class']]
                .head(5))
    print('\nTop Spearman:')
    print(lim_df.sort_values('spearman', ascending=False)
                .loc[:, ['filename', 'spearman', 'n_class']]
                .head(5))
    print('\nTop RMSE:')
    print(lim_df.sort_values('rmse')
                .loc[:, ['filename', 'rmse', 'n_class']]
                .head(5))

    print('== Collapsed labels ==')
    lim_df = df[df.n_class == 4]
    print('\nTop Pearson:')
    print(lim_df.sort_values('pearson', ascending=False)
                .loc[:, ['filename', 'pearson', 'n_class']]
                .head(5))
    print('\nTop Spearman:')
    print(lim_df.sort_values('spearman', ascending=False)
                .loc[:, ['filename', 'spearman', 'n_class']]
                .head(5))
    print('\nTop RMSE:')
    print(lim_df.sort_values('rmse')
                .loc[:, ['filename', 'rmse', 'n_class']]
                .head(5))


if __name__ == '__main__':
    main()
