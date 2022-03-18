# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas
# Press the green button in the gutter to run the script.
import pandas as pd
import sklearn
import CNN3
import utils.utils
from sklearn.model_selection import train_test_split


def fit_classifier():
    df = pd.read_csv("data\\arrays.csv", header=None)
    df = df.sample(frac=1).reset_index(drop=True)
    train_size = int(0.7 * len(df))
    train_set = df[:train_size]
    test_set = df[train_size:]
    y_train = np.array(train_set[train_set.columns[0]])
    y_test = np.array(test_set[test_set.columns[0]])
    x_train = np.array(train_set.iloc[:, 1:])
    x_test = np.array(test_set.iloc[:, 1:])

    # x_train = np.array(pd.read_csv('Earthquakes_TRAIN.csv', sep='  '))
    # y_train = np.array(pd.read_csv('Earthquakes_TRAIN.csv', usecols=[0], sep='  '))
    # x_test = np.array(pd.read_csv('Earthquakes_TEST.csv', sep='  '))
    # y_test = np.array(pd.read_csv('Earthquakes_TEST.csv', usecols=[0], sep='  '))

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = CNN3.Classifier_CNN(output_directory, input_shape, nb_classes, verbose=False)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)
    prediction = classifier.predict(x_test, y_true, x_train, y_train, y_test)
    print(prediction)


if __name__ == '__main__':
    output_directory = os.getcwd()
    fit_classifier()
