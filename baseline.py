import sys

import pandas as pd
import sklearn
import sklearn.svm
import sklearn.linear_model
import matplotlib.pyplot as plt


def main():
    print('Loading data ...')
    df = pd.read_csv('metadata.csv').dropna(subset=['cefr'])

    train = df[df.split == 'train']
    dev = df[df.split == 'dev']

    if sys.argv[1] == 'svm':
        clf = sklearn.svm.LinearSVC()
    else:
        clf = sklearn.linear_model.LogisticRegression()
    
    print('Preprocessing data ...')
    scaler = sklearn.preprocessing.StandardScaler()
    train.loc[:, 'num_tokens'] = scaler.fit_transform(
        train.num_tokens.values[:, None].astype(float))
    dev.loc[:, 'num_tokens'] = scaler.transform(
        dev.num_tokens.values[:, None].astype(float))

    train.loc[:, 'testlevel'] = train.testlevel == 'Språkprøven'
    dev.loc[:, 'testlevel'] = dev.testlevel == 'Språkprøven'

    train_x = train.loc[:, ['testlevel', 'num_tokens']].values
    dev_x = dev.loc[:, ['testlevel', 'num_tokens']].values

    print('Training classifier ...')
    clf.fit(train_x, train.cefr)

    predictions = clf.predict(dev_x)
    print(sklearn.metrics.classification_report(dev.cefr, predictions))
    print('Accuracy: %.3f' % sklearn.metrics.accuracy_score(dev.cefr, predictions))
    print('Confusion matrix:')
    conf_matrix = sklearn.metrics.confusion_matrix(dev.cefr, predictions)
    print(conf_matrix)
    plt.imshow(conf_matrix)
    plt.show()


if __name__ == '__main__':
    main()
