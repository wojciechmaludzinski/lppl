import numpy as np
import matplotlib.pyplot as plt
import os
import pandas
# Press the green button in the gutter to run the script.
import pandas as pd
import sklearn
import CNN3
import tensorflow as tf
from utils.utils import pred_beautifier
from sklearn.model_selection import train_test_split


def fit_classifier(df, length):
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

    #VISUALISATION
    # classifier.visualize_filter(x_train, y_train)

    #PREDICTION
    df_btc = pd.read_csv("data\\array_btc2.csv", header=None)
    pred_rows_used = 10000
    df_btc_first = df_btc.iloc[:pred_rows_used, :length-1]
    print(pred_beautifier(classifier.predicting(df_btc_first)))



if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    output_directory = os.getcwd()

    # columns_used = 200  # out of length 999
    # print('200 columns used')
    columns_array = [500]

    # for i in range(len(columns_array)):
    #     print(f"clean with {columns_array[i]} columns")
    #     df_clean = pd.read_csv("data\\arrays.csv", header=None)
    #     df_clean = df_clean.sample(frac=1).reset_index(drop=True)
    #     df_clean = df_clean.iloc[:, :columns_array[i]]
    #     fit_classifier(df_clean, columns_array[i])

    # for i in range(len(columns_array)):
    #     print(f"noise1 with {columns_array[i]} columns")
    #     df = pd.read_csv("data\\df_noise.csv", header=None)
    #     df = df.sample(frac=1).reset_index(drop=True)
    #     df = df.iloc[: , :columns_array[i]]
    #     fit_classifier(df)
    #
    for i in range(len(columns_array)):
        print(f"noise2 with {columns_array[i]} columns")
        df2 = pd.read_csv("data\\df_noise2.csv", header=None)
        df2 = df2.sample(frac=1).reset_index(drop=True)
        df2 = df2.iloc[: , :columns_array[i]]
        fit_classifier(df2, columns_array[i])

    for i in range(len(columns_array)):
        print(f"noise3 with {columns_array[i]} columns")
        df3 = pd.read_csv("data\\df_noise3.csv", header=None)
        df3 = df3.sample(frac=1).reset_index(drop=True)
        df3 = df3.iloc[: , :columns_array[i]]
        fit_classifier(df3, columns_array[i])

    for i in range(len(columns_array)):
        print(f"noise4 with {columns_array[i]} columns")
        df4 = pd.read_csv("data\\df_noise4.csv", header=None)
        df4 = df4.sample(frac=1).reset_index(drop=True)
        df4 = df4.iloc[: , :columns_array[i]]
        fit_classifier(df4, columns_array[i])




