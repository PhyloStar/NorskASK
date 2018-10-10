from functools import partial
from itertools import chain
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

    best_split = evolution(df, topics, overall_cefr, overall_lang)

    dev_set = topics_to_df(df, best_split['dev_topics'])
    test_set = topics_to_df(df, best_split['test_topics'])

    print('== DEV SET ==')
    print(dev_set.cefr.value_counts().reindex(CEFR_LABELS, fill_value=0))
    print(dev_set.lang.value_counts().reindex(LANG_LABELS, fill_value=0))

    print('== TEST SET ==')
    print(test_set.cefr.value_counts().reindex(CEFR_LABELS, fill_value=0))
    print(test_set.lang.value_counts().reindex(LANG_LABELS, fill_value=0))


def mutate(individual):
    n = len(individual)
    mask = np.random.random(n) > 0.2
    new_individual = np.random.choice([0, 1, 2], n)
    new_individual[mask] = individual[mask]
    return new_individual


def evolution(df, topics, overall_cefr, overall_lang):
    evaluate_f = partial(evaluate_candidate, df=df, topics=topics,
                         overall_cefr=overall_cefr, overall_lang=overall_lang)
    pop_size = 100
    elite_size = 20
    mutants_per_cand = pop_size // elite_size
    best_ever = None
    generations = 0
    population = [
        np.random.choice([0, 1, 2], len(topics), p=[.8, .1, .1])
        for _ in range(pop_size)
    ]
    print('Starting evolution algorithm ...')
    try:
        while True:
            results = [evaluate_f(sample_topics=cand) for cand in population]
            penalties = [r['penalty'] for r in results]
            best = results[np.argmin(penalties)]
            if best_ever is None or best['penalty'] < best_ever['penalty']:
                best_ever = best
                print(best_ever)
            elite_indices = np.argpartition(penalties, elite_size)[:elite_size]
            elite = [population[i] for i in elite_indices]
            population = list(chain.from_iterable(
                (mutate(cand) for _ in range(mutants_per_cand)) for cand in elite
            ))
            generations += 1
    except KeyboardInterrupt:
        print('Stopped after {} generations.'.format(generations))
        return best_ever


if __name__ == '__main__':
    main()
