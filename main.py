import numpy as np
import matplotlib.pyplot as plt
import os
import pandas
# Press the green button in the gutter to run the script.
import pandas as pd
import sklearn
import CNN3
import tensorflow as tf
from utils.utils import pred_beautifier, pred_uglier, pred_another
from sklearn.model_selection import train_test_split
from LPPL_fitting import LPPL_fit


def load_data(path, columns):
    return pd.read_csv(path, header=None).sample(frac=1).reset_index(drop=True).iloc[:, :columns]


def prepare_data(df, length):
    train_size = int(0.7 * len(df))
    train_set = df[:train_size]
    test_set = df[train_size:]
    _y_train = np.array(train_set[train_set.columns[0]])
    _y_test = np.array(test_set[test_set.columns[0]])
    _x_train = np.array(train_set.iloc[:, 1:])
    _x_test = np.array(test_set.iloc[:, 1:])

    _nb_classes = len(np.unique(np.concatenate((_y_train, _y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((_y_train, _y_test), axis=0).reshape(-1, 1))
    _y_train = enc.transform(_y_train.reshape(-1, 1)).toarray()
    _y_test = enc.transform(_y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    _y_true = np.argmax(_y_test, axis=1)

    if len(_x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        _x_train = _x_train.reshape((_x_train.shape[0], _x_train.shape[1], 1))
        _x_test = _x_test.reshape((_x_test.shape[0], _x_test.shape[1], 1))

    _input_shape = _x_train.shape[1:]

    return _x_train, _y_train, _x_test, _y_test, _y_true, _input_shape, _nb_classes


def predict(clf, df):
    # PREDICTION
    pred_rows_used = 1500
    df_smaller = df.iloc[:pred_rows_used, :DATA_SIZE - 1]
    print(pred_beautifier(clf.predicting(df_smaller)))
    print(pred_another(clf.predicting(df_smaller), df_smaller))


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    output_directory = os.getcwd() + '\\classifier\\'
    DATA_SIZE = 500
    arr_filenames = ["array_500.csv", "df_noise.csv", "df_noise2.csv", "df_noise3.csv"]

    for filename in arr_filenames:
        df = load_data(f"data\\{filename}", DATA_SIZE)
        x_train, y_train, x_test, y_test, y_true, input_shape, nb_classes = prepare_data(df, DATA_SIZE)
        classifier = CNN3.Classifier_CNN(output_directory, input_shape, nb_classes, verbose=False)
        classifier.fit(x_train, y_train, x_test, y_test, y_true)
        prediction = classifier.predict(x_test, y_true, x_train, y_train, y_test)
        print(prediction)
        predict(classifier, pd.read_csv("data\\array_btc2.csv", header=None))
        predict(classifier, pd.read_csv("data\\array_eurpln.csv", header=None))
        predict(classifier, pd.read_csv("data\\array_usdpln.csv", header=None))
