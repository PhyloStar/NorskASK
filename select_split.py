from typing import Tuple

from scipy.stats import entropy
import numpy as np
import pandas as pd

CEFR_LABELS = ['A2', 'A2/B1', 'B1', 'B1/B2', 'B2', 'B2/C1', 'C1']
LANG_LABELS = ['russisk', 'polsk', 'tysk', 'vietnamesisk', 'engelsk', 'spansk', 'somali']


def cefr_distribution(df):
    return df.cefr.value_counts().reindex(CEFR_LABELS, fill_value=0).values


def lang_distribution(df):
    return df.lang.value_counts().reindex(LANG_LABELS, fill_value=0).values


def selection_distributions(df, indices) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    dev = df[indices == 1]
    test = df[indices == 2]
    return (
        (cefr_distribution(dev), cefr_distribution(test)),
        (lang_distribution(dev), lang_distribution(test))
    )


def topics_to_df(df, topics):
    return df[df.topic.isin(topics)]


def evaluate_candidate(df, sample_topics, topics, overall_cefr, overall_lang):
    dev_topics = set(topics[sample_topics == 1])
    dev_set = topics_to_df(df, dev_topics)
    test_topics = set(topics[sample_topics == 2])
    test_set = topics_to_df(df, test_topics)

    entropies = np.array([
        entropy(cefr_distribution(dev_set), overall_cefr),
        entropy(cefr_distribution(test_set), overall_cefr),
        entropy(lang_distribution(dev_set), overall_lang),
        entropy(lang_distribution(test_set), overall_lang)
    ])

    sample_sizes = (len(dev_set), len(test_set))
    sample_size_penalty = (abs(len(dev_set) - 121) + abs(len(test_set) - 121)) / 121

    return {
        'dev_topics': dev_topics,
        'test_topics': test_topics,
        'entropies': entropies,
        'sample_sizes': sample_sizes,
        'penalty': np.sum(entropies ** 2) + sample_size_penalty
    }


def main():
    df = pd.read_csv('metadata.csv').dropna(subset=['cefr'])
    overall_cefr = cefr_distribution(df)
    overall_lang = lang_distribution(df)

    topics = df.topic.unique()

    print(overall_cefr)
    print(overall_lang)

    best_split = None

    for _ in range(10000):
        sample_topics = np.random.choice([0, 1, 2], len(topics), p=[.8, .1, .1])
        res = evaluate_candidate(df, sample_topics, topics, overall_cefr, overall_lang)
        if best_split is None or best_split['penalty'] > res['penalty']:
            best_split = res
            print(best_split)

    dev_set = topics_to_df(df, best_split['dev_topics'])
    test_set = topics_to_df(df, best_split['test_topics'])

    print('== DEV SET ==')
    print(dev_set.cefr.value_counts().reindex(CEFR_LABELS, fill_value=0))
    print(dev_set.lang.value_counts().reindex(LANG_LABELS, fill_value=0))

    print('== TEST SET ==')
    print(test_set.cefr.value_counts().reindex(CEFR_LABELS, fill_value=0))
    print(test_set.lang.value_counts().reindex(LANG_LABELS, fill_value=0))


if __name__ == '__main__':
    main()
