import argparse
import tempfile

import numpy as np
from keras.optimizers import Adam
from keras.utils import to_categorical

from masterthesis.features.build_features import bag_of_words, filename_iter
from masterthesis.utils import load_split
from masterthesis.models.callbacks import F1Metrics
from masterthesis.models.report import report
from masterthesis.models.bag_of_chars import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-cefr', action='store_true')
    parser.add_argument('--vocab', type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    train_x, vectorizer = bag_of_words('train', max_features=args.vocab)
    dev_meta = load_split('dev', round_cefr=args.round_cefr)
    dev_x = vectorizer.transform(filename_iter(dev_meta))

    train_meta = load_split('train', round_cefr=args.round_cefr)
    labels = sorted(train_meta.cefr.unique())

    train_y = to_categorical([labels.index(c) for c in train_meta.cefr])
    dev_y = to_categorical([labels.index(c) for c in dev_meta.cefr])

    model = build_model(len(vectorizer.vocabulary_), num_classes=len(labels))
    model.summary()
    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    with tempfile.NamedTemporaryFile(suffix='.h5') as weights_path:
        callbacks = [F1Metrics(dev_x, dev_y, weights_path.name)]
        model.fit(train_x, train_y, epochs=20, callbacks=callbacks, validation_data=(dev_x, dev_y))
        model.load_weights(weights_path.name)

    predictions = model.predict(dev_x)

    true = np.argmax(dev_y, axis=1)
    pred = np.argmax(predictions, axis=1)
    report(true, pred, labels)


if __name__ == '__main__':
    main()
