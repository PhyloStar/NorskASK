import argparse
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from sklearn.metrics import f1_score
sns.set()


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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    return parser.parse_args()


def get_corr_mask(n: int):
    mask = np.ones((n, n), dtype=bool)
    return np.triu(mask, 1)


def main():
    args = parse_args()
    data = defaultdict(list)
    for filename in args.files:
        results_file = Path(filename)
        try:
            res = pickle.load(results_file.open('rb'))
        except Exception as e:
            print(e)
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
        data['macrof1'].append(f1_score(gold, pred, average='macro'))
        data['microf1'].append(f1_score(gold, pred, average='micro'))
        data['weightedf1'].append(f1_score(gold, pred, average='weighted'))
        data['n_class'].append(max(max(gold), max(pred)) + 1)
        mse = sum((a - b) ** 2 for a, b in zip(gold, pred)) / len(gold)
        rmse = sqrt(mse)
        data['rmse'].append(rmse)
        mae = sum(abs(a - b) for a, b in zip(gold, pred)) / len(gold)
        data['mae'].append(mae)
    df = pd.DataFrame.from_dict(data)
    print('== All labels ==')
    lim_df = df[df.n_class == 7].dropna()
    corr_matrix = lim_df.drop(columns=['filename', 'n_class']).corr()
    sns.heatmap(corr_matrix, center=0, mask=get_corr_mask(len(corr_matrix)), annot=True, fmt='.3f')
    plt.show()
    print('\nTop Pearson:')
    print(lim_df.sort_values('pearson', ascending=False)
                .loc[:, ['filename', 'pearson']]
                .head(5))
    print('\nTop Spearman:')
    print(lim_df.sort_values('spearman', ascending=False)
                .loc[:, ['filename', 'spearman']]
                .head(5))
    print('\nTop RMSE:')
    print(lim_df.sort_values('rmse')
                .loc[:, ['filename', 'rmse']]
                .head(5))
    print('\nTop MAE:')
    print(lim_df.sort_values('mae')
                .loc[:, ['filename', 'mae']]
                .head(5))
    # sns.pairplot(lim_df.drop(columns=['filename', 'n_class']))
    plt.show()

    print('== Collapsed labels ==')
    lim_df = df[df.n_class == 4].dropna()
    corr_matrix = lim_df.drop(columns=['filename', 'n_class']).corr()
    sns.heatmap(corr_matrix, center=0, mask=get_corr_mask(len(corr_matrix)), annot=True, fmt='.3f')
    plt.show()
    print('\nTop Pearson:')
    print(lim_df.sort_values('pearson', ascending=False)
                .loc[:, ['filename', 'pearson']]
                .head(5))
    print('\nTop Spearman:')
    print(lim_df.sort_values('spearman', ascending=False)
                .loc[:, ['filename', 'spearman']]
                .head(5))
    print('\nTop RMSE:')
    print(lim_df.sort_values('rmse')
                .loc[:, ['filename', 'rmse']]
                .head(5))
    print('\nTop MAE:')
    print(lim_df.sort_values('mae')
                .loc[:, ['filename', 'mae']]
                .head(5))
    # sns.pairplot(lim_df.drop(columns=['filename', 'n_class', 'pearson']))
    plt.show()


if __name__ == '__main__':
    main()
