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


def load_data(path_or_df, columns, dataframe=False):
    if dataframe:
        return path_or_df.sample(frac=1).reset_index(drop=True).iloc[:, :columns]

    return pd.read_csv(path_or_df, header=None).sample(frac=1).reset_index(drop=True).iloc[:, :columns]


def prepare_data(df):
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


def predict(clf, df, pred_rows_used=500):
    df_smaller = df.iloc[:pred_rows_used, :DATA_SIZE - 1]
    print(pred_beautifier(clf.predicting(df_smaller)))
    print(pred_another(clf.predicting(df_smaller), df_smaller))


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    output_directory_path = os.getcwd() + '\\classifier\\'
    DATA_SIZE = 500
    # arr_filenames = ["df_noise2.csv",  "df_noise.csv", "df_noise3.csv"]


    # dataframes = [load_data(f"data\\df_noise.csv", DATA_SIZE),
    #               load_data(pd.read_csv("data\\df_noise.csv", header=None).iloc[0:2000, :DATA_SIZE], DATA_SIZE,
    #                         dataframe=True),  # 1,0
    #               load_data(pd.read_csv("data\\df_noise.csv", header=None,
    #                                     skiprows=[i for i in range(1000, 2000)]), DATA_SIZE, dataframe=True),  # 1,2
    #               load_data(pd.read_csv("data\\df_noise.csv", header=None).iloc[1000:3000], DATA_SIZE, dataframe=True)
    #               # 0,2
    #               ]
    dataframes = [
        load_data(f"data\\test_df_noise.csv", DATA_SIZE),
        load_data(f"data\\test_df_noise2.csv", DATA_SIZE),
    ]

    # directories = ['noise', '10', '12', '02']
    directories = ['test00', 'test22']


    # dataframes2 = [
    #     load_data(f"data\\df_noise.csv", DATA_SIZE),
    #     load_data(f"data\\df_noise2.csv", DATA_SIZE),
    #     load_data(f"data\\df_noise3.csv", DATA_SIZE),
    #     load_data(pd.read_csv("data\\df_noise.csv", header=None).iloc[0:2000, :DATA_SIZE], DATA_SIZE, dataframe=True),  # 1,0
    #     load_data(pd.read_csv("data\\df_noise.csv", header=None,
    #                           skiprows=[i for i in range(1000, 2000)]), DATA_SIZE, dataframe=True),  # 1,2
    #     load_data(pd.read_csv("data\\df_noise.csv", header=None).iloc[1000:3000], DATA_SIZE, dataframe=True) #0,2
    # ]

    dataframes2 = [
        load_data(f"data\\df_noise.csv", DATA_SIZE),
        load_data(f"data\\df_noise2.csv", DATA_SIZE),
        load_data(f"data\\df_noise3.csv", DATA_SIZE),
        load_data(f"data\\df_noise4.csv", DATA_SIZE)
    ]

    for df, directory in zip(dataframes, directories):
        print('data loaded')
        output_directory = output_directory_path+directory+'\\'

        x_train, y_train, x_test, y_test, y_true, input_shape, nb_classes = prepare_data(df)
        classifier = CNN3.Classifier_CNN(output_directory, input_shape, nb_classes, verbose=False)
        classifier.fit(x_train, y_train, x_test, y_test, y_true)
        prediction = classifier.predict(x_test, y_true, x_train, y_train, y_test)
        print('PREDICTION:', prediction)
        # predict(classifier, pd.read_csv("data\\array_eurpln.csv", header=None))

        # for _df in dataframes2:
        #     _x_train, _y_train, _x_test, _y_test, _y_true, _input_shape, _nb_classes = prepare_data(_df)
        #     print(_input_shape)
        #     _prediction = classifier.predict(_x_test, _y_true, _x_train, _y_train, _y_test)
        #     print('PREDICTION:', _prediction)


        print('prediction of 0,1,2:')
        predict(classifier, pd.read_csv("data\\df_noise.csv", header=None).iloc[:, 1:],
                pred_rows_used=3000)  # all

        print('prediction of 0,1:')
        predict(classifier, pd.read_csv("data\\df_noise.csv", header=None).iloc[0:2000, 1:],
                pred_rows_used=2000)  # 1 and 0
        #
        print('prediction of 1,2:')
        predict(classifier, pd.read_csv("data\\df_noise.csv", header=None,
                                        skiprows=[i for i in range(1000, 2001)]), pred_rows_used=2000)  # 1 and 2

        print('prediction of 0,2:')
        predict(classifier, pd.read_csv("data\\df_noise.csv", header=None).iloc[1000:3000, 1:],
                pred_rows_used=2000)  # 0 and 2



        print('prediction of 0,1,2:')
        predict(classifier, pd.read_csv("data\\df_noise2.csv", header=None).iloc[:, 1:],
                pred_rows_used=3000)  # all

        print('prediction of 0,1:')
        predict(classifier, pd.read_csv("data\\df_noise2.csv", header=None).iloc[0:2000, 1:],
                pred_rows_used=2000)  # 1 and 0
        #
        print('prediction of 1,2:')
        predict(classifier, pd.read_csv("data\\df_noise2.csv", header=None,
                                        skiprows=[i for i in range(1000, 2001)]), pred_rows_used=2000)  # 1 and 2

        print('prediction of 0,2:')
        predict(classifier, pd.read_csv("data\\df_noise2.csv", header=None).iloc[1000:3000, 1:],
                pred_rows_used=2000)  # 0 and 2

        print("-------------------------------finished")
