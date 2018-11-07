import sys

import sklearn
import sklearn.svm
import sklearn.linear_model
import sklearn.neural_network

from src.utils import load_train_and_dev


def main():
    print('Loading data ...')
    train, dev = load_train_and_dev()

    print('Preprocessing data ...')
    scaler = sklearn.preprocessing.StandardScaler(copy=True)
    train.loc[:, 'num_tokens'] = scaler.fit_transform(
        train.num_tokens.values[:, None].astype(float))
    dev.loc[:, 'num_tokens'] = scaler.transform(
        dev.num_tokens.values[:, None].astype(float))

    train.loc[:, 'testlevel'] = (train.testlevel == 'Språkprøven').astype(int)
    dev.loc[:, 'testlevel'] = (dev.testlevel == 'Språkprøven').astype(int)

    train_x = train.loc[:, ['testlevel', 'num_tokens']].values
    dev_x = dev.loc[:, ['testlevel', 'num_tokens']].values

    print('Training classifier ...')
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == 'svm':
            clf = sklearn.svm.LinearSVC()
        elif arg == 'mlp':
            clf = sklearn.neural_network.MLPClassifier((50))
        else:
            print('Unsupported classifier type %s, fallback to logistic regression')
    else:
        clf = sklearn.linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')

    clf.fit(train_x, train.cefr)

    predictions = clf.predict(dev_x)
    print(sklearn.metrics.classification_report(dev.cefr, predictions))
    print('Accuracy: %.3f' % sklearn.metrics.accuracy_score(dev.cefr, predictions))
    print('Confusion matrix:')
    conf_matrix = sklearn.metrics.confusion_matrix(dev.cefr, predictions)
    print(conf_matrix)


if __name__ == '__main__':
    main()
