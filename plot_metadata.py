import pandas as pd
import matplotlib.pyplot as plt

iso639_3 = {
    'engelsk': 'eng',
    'polsk': 'pol',
    'russisk': 'rus',
    'somali': 'som',
    'spansk': 'spa',
    'tysk': 'deu',
    'vietnamesisk': 'vie'
}

df = pd.read_csv('metadata.csv')
counts = df.groupby(['lang', 'cefr']).size().unstack().fillna(0)

normalized = counts.div(counts.sum(axis=1), axis=0)

plt.imshow(counts)
plt.yticks(range(len(counts)), [iso639_3[l] for l in counts.index])
plt.xticks(range(len(counts.columns)), counts.columns)

plt.show()
