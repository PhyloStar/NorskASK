import sklearn.linear_model

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

from masterthesis.features.build_features import bag_of_words, filename_iter
from masterthesis.utils import load_split
from masterthesis.models.callbacks import F1Metrics


def build_model(vocab_size: int):
    input_ = Input((vocab_size,))
    hidden = Dense(1000)(input_)
    output = Dense(7, activation='softmax')(hidden)
    return Model(inputs=[input_], outputs=[output])


def main():
    train_x, vectorizer = bag_of_words(
        'train', analyzer='char', ngram_range=(2, 4), max_features=5000)
    dev_meta = load_split('dev')
    dev_x = vectorizer.transform(filename_iter(dev_meta))

    print(vectorizer.get_feature_names()[:20])
    train_meta = load_split('train')
    labels = sorted(train_meta.cefr.unique())

    train_y = to_categorical([labels.index(c) for c in train_meta.cefr])
    dev_y = to_categorical([labels.index(c) for c in dev_meta.cefr])

    model = build_model(len(vectorizer.vocabulary_))
    model.summary()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    weights_path = 'jalla.h5'
    callbacks = [
        F1Metrics(dev_x, dev_y, weights_path)
    ]

    model.fit(train_x, train_y, epochs=5, callbacks=callbacks, validation_data=(dev_x, dev_y))

    model.load_weights(weights_path)

    predictions = model.predict(dev_x)

    true = np.argmax(dev_y, axis=1)
    pred = np.argmax(predictions, axis=1)
    print(sklearn.metrics.classification_report(true, pred, target_names=labels))
    print('Accuracy: %.3f' % sklearn.metrics.accuracy_score(true, pred))


if __name__ == '__main__':
    main()
