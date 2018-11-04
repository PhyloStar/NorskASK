import zipfile
from pathlib import Path
from typing import Union

from gensim.models import FastText


def load_fasttext_embeddings(file: Union[Path, str]) -> FastText:
    """Load embeddings from file and unit normalize vectors."""
    if isinstance(file, str):
        file = Path(file)
    # Detect the model format by its extension:
    if '.bin' in file.suffixes or '.vec' in file.suffixes:
        # Binary word2vec format
        emb_model = FastText.load_fasttext_format(str(file))
    elif file.suffix == '.zip':
        # ZIP archive from the NLPL vector repository
        with zipfile.ZipFile(str(file), "r") as archive:
            model_file = archive.extract('parameters.bin')
            emb_model = FastText.load_fasttext_format(model_file)
    else:
        # Native Gensim format?
        emb_model = FastText.load(str(file))

    emb_model.init_sims(replace=True)  # Unit-normalizing the vectors (if they aren't already)
    return emb_model.wv
