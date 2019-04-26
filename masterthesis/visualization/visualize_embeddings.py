import argparse
import logging
from pathlib import Path
import pickle
from typing import Iterable

from keras.models import load_model, Model
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tqdm

from masterthesis.features.build_features import pos_to_sequences, words_to_sequences
from masterthesis.gensim_utils import fingerprint, load_embeddings
from masterthesis.utils import (
    CEFR_LABELS,
    DATA_DIR,
    document_iterator,
    iso639_3,
    load_split,
    REPRESENTATION_LAYER,
    safe_plt as plt,
)

sns.set(style='white', context='paper')
logger = logging.getLogger(__name__)
logging.basicConfig()


class MockKeyedVectors:
    def __init__(self, vector_size: int) -> None:
        self.vector_size = vector_size

    def word_vec(self, w: str) -> np.ndarray:
        return np.random.randn(self.vector_size)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.word_vec(key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument("--decomposition", choices={"tsne", "pca"}, default="pca")
    parser.add_argument("--hue", choices={"CEFR", "L1"}, default="CEFR")
    parser.add_argument(
        '--debug', dest='loglevel', action='store_const', const=logging.DEBUG
    )
    parser.add_argument(
        '--quiet', dest='loglevel', action='store_const', const=logging.WARN
    )
    parser.add_argument('--split', choices={'train', 'dev', 'test'}, default='dev')
    parser.set_defaults(loglevel=logging.INFO)
    args = parser.parse_args()
    logging.getLogger(None).setLevel(args.loglevel)
    return args


def get_fingerprints(embeddings: Path, filenames: Iterable[str]) -> np.ndarray:
    wv = load_embeddings(embeddings)
    txt_folder = DATA_DIR / 'txt'
    fingerprints = []
    logger.info("Computing fingerprints of all documents ...")
    for filename in tqdm.tqdm(filenames):
        infile = txt_folder / Path(filename).with_suffix('.txt')
        with open(str(infile), encoding='utf-8') as f:
            fingerprints.append(fingerprint(wv, document_iterator(f)))
    return np.stack(fingerprints)


def get_model_representations(model_path: Path, split: str) -> np.ndarray:
    model = load_model(str(model_path))
    representation_model = Model(
        inputs=model.input, outputs=model.get_layer(REPRESENTATION_LAYER).output
    )
    logger.debug(model.input)
    w2i_path = model_path.parent / (model_path.stem + '_w2i.pkl')
    w2i = pickle.load(w2i_path.open('rb'))
    (x,) = words_to_sequences(700, [split], w2i)
    if isinstance(model.input, list) and len(model.input) == 2:
        pos2i_path = model_path.parent / ('pos2i.pkl')
        pos2i = pickle.load(pos2i_path.open('rb'))
        (x_pos,) = pos_to_sequences(700, [split], pos2i)
        x = [x, x_pos]
    return representation_model.predict(x)


def main():
    args = parse_args()
    meta = load_split(args.split)
    meta["lang"].replace(iso639_3, inplace=True)

    if args.embeddings:
        representations = get_fingerprints(args.embeddings, meta.filename)
    elif args.model:
        representations = get_model_representations(args.model, args.split)

    logger.info("Computing embeddings ...")
    if args.decomposition == "tsne":
        decomposer = TSNE(n_components=2, verbose=True)
    elif args.decomposition == "pca":
        decomposer = PCA(n_components=2)
    embedded = decomposer.fit_transform(representations)
    meta["Component 1"] = embedded[:, 0]
    meta["Component 2"] = embedded[:, 1]
    meta.rename(
        {"cefr": "CEFR", "testlevel": "Test level", "num_tokens": "Length", "lang": "L1"},
        axis="columns",
        inplace=True
    )
    meta["Test level"].replace(
        {"Språkprøven": "IL test", "Høyere nivå": "AL test"},
        inplace=True
    )

    fig, ax = (plt.gcf(), plt.gca())
    if args.hue == "CEFR":
        palette = sns.mpl_palette('cool', 7)
        hue_order = CEFR_LABELS
    else:
        palette = None
        hue_order = None
    sns.scatterplot(
        x="Component 1",
        y="Component 2",
        hue=args.hue,
        style="Test level",
        data=meta,
        ax=ax,
        size="Length",
        palette=palette,
        hue_order=hue_order,
    )
    ax.tick_params(
        axis="both",
        which="both",
        bottom="off",
        top="off",
        labelbottom="off",
        right="off",
        left="off",
        labelleft="off",
    )
    handles, labels = ax.get_legend_handles_labels()
    cefr_legend = ax.legend(
        handles[:8], labels[:8], loc="center right", bbox_to_anchor=(-0.1, 0.5)
    )
    ax.legend(
        handles[8:], labels[8:], loc="center left", bbox_to_anchor=(1.05, 0.5)
    )
    ax.add_artist(cefr_legend)
    fig.set_size_inches(5, 3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
