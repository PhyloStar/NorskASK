import pandas as pd

from masterthesis.utils import iso639_3, safe_plt as plt

df = pd.read_csv('metadata.csv')
counts = df.groupby(['lang', 'cefr']).size().unstack().fillna(0)

normalized = counts.div(counts.sum(axis=1), axis=0)

plt.imshow(counts, cmap='viridis')
plt.yticks(range(len(counts)), [iso639_3[l] for l in counts.index])
plt.xticks(range(len(counts.columns)), counts.columns)

col_cutoff = counts.max().max() / 2

for i, row in enumerate(counts.itertuples()):
    for j, count in enumerate(row[1:]):
        col = 'white' if count < col_cutoff else 'black'
        plt.text(
            j, i, int(count), horizontalalignment='center', verticalalignment='center', color=col
        )

plt.ylabel('Native language')
plt.xlabel('CEFR proficiency')
plt.savefig('lang_vs_cefr.pdf')

plt.show()
