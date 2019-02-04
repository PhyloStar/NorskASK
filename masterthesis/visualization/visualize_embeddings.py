import argparse
import logging
from pathlib import Path

from keras.models import Model, load_model
import numpy as np
from sklearn.manifold import TSNE
import tqdm

from masterthesis.features.build_features import words_to_sequences, make_w2i
from masterthesis.gensim_utils import fingerprint, load_embeddings
from masterthesis.utils import document_iterator, iso639_3, load_split
from masterthesis.utils import CEFR_LABELS, DATA_DIR, LANG_LABELS, safe_plt as plt

logging.basicConfig(level=logging.INFO)


class MockKeyedVectors:
    def __init__(self, vector_size):
        self.vector_size = vector_size

    def word_vec(self, w):
        return np.random.randn(self.vector_size)

    def __getitem__(self, key):
        return self.word_vec(key)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--split', choices=['train', 'dev', 'test'], default='dev')
    return parser.parse_args()


def get_fingerprints(embeddings, filenames):
    wv = load_embeddings(embeddings)
    txt_folder = DATA_DIR / 'txt'
    fingerprints = []
    print('Computing fingerprints of all documents ...')
    for filename in tqdm.tqdm(filenames):
        infile = txt_folder / Path(filename).with_suffix('.txt')
        with open(str(infile), encoding='utf-8') as f:
            fingerprints.append(fingerprint(wv, document_iterator(f)))
    return np.stack(fingerprints)


def get_model_representations(model, split):
    representation_model = Model(inputs=model.input,
                                 outputs=model.get_layer('vector_representation').output)
    w2i = make_w2i(4000)
    (x,) = words_to_sequences(700, [split], w2i)
    return representation_model.predict(x)


def main():
    args = parse_args()
    meta = load_split(args.split)
    meta.loc[:, 'lang'] = [iso639_3[l] for l in meta.lang]

    if args.embeddings:
        representations = get_fingerprints(args.embeddings, meta.filename)
    elif args.model:
        model = load_model(str(args.model))
        representations = get_model_representations(model, args.split)

    cefr_list = CEFR_LABELS
    testlevel_list = sorted(meta.testlevel.unique())
    lang_list = LANG_LABELS
    column_list = ['cefr', 'testlevel', 'lang']

    print('Computing t-SNE embeddings ...')
    embedded = TSNE(n_components=2, verbose=True).fit_transform(representations)

    fig, axes = plt.subplots(1, 3)
    for ax, label_list, col in zip(axes, [cefr_list, testlevel_list, lang_list], column_list):
        for label in label_list:
            mask = meta.loc[:, col] == label
            xs = embedded[mask, 0]
            ys = embedded[mask, 1]
            ax.plot(xs, ys, 'o', label=label)
        ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
