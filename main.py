import numpy as np
import matplotlib.pyplot as plt
import os
import pandas
# Press the green button in the gutter to run the script.
import pandas as pd
import sklearn
import CNN3
import tensorflow as tf
from utils.utils import pred_counter, pred_plot_all_lppls
from LPPL_fitting import LPPL_fit
from data_handler import load_data, prepare_data


def predict(clf, df, pred_rows_used=500):
    df_smaller = df.iloc[:pred_rows_used, :DATA_SIZE - 1]
    print(pred_counter(clf.predicting(df_smaller)))
    print(pred_plot_all_lppls(clf.predicting(df_smaller), df_smaller))


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    output_directory_path = os.getcwd() + '\\classifier\\'
    DATA_SIZE = 500
    # arr_filenames = ["df_noise2.csv",  "df_noise.csv", "df_noise3.csv"]
    data_sizes = [500, 450, 400, 350, 300, 100]


    # dataframes = [load_data(f"data\\df_noise.csv", DATA_SIZE),
    #               load_data(pd.read_csv("data\\df_noise.csv", header=None).iloc[0:2000, :DATA_SIZE], DATA_SIZE,
    #                         dataframe=True),  # 1,0
    #               load_data(pd.read_csv("data\\df_noise.csv", header=None,
    #                                     skiprows=[i for i in range(1000, 2000)]), DATA_SIZE, dataframe=True),  # 1,2
    #               load_data(pd.read_csv("data\\df_noise.csv", header=None).iloc[1000:3000], DATA_SIZE, dataframe=True)
    #               # 0,2
    #               ]
    for data_size in data_sizes:
        dataframes = [
            # load_data(f"data\\test_df_noise.csv", DATA_SIZE),
            load_data(f"data\\test_df_noise3.csv", data_size),
            # load_data(pd.read_csv("data\\updf_noise4.csv", header=None).iloc[0:2000, :DATA_SIZE], data_sizes[0],
            #                          dataframe=True),  # 1,0
            # load_data(pd.read_csv("data\\updf_noise4.csv", header=None).iloc[0:2000, :DATA_SIZE], data_sizes[1],
            #           dataframe=True),  # 1,0
            # load_data(pd.read_csv("data\\updf_noise4.csv", header=None).iloc[0:2000, :DATA_SIZE], data_sizes[2],
            #           dataframe=True),  # 1,0
            # load_data(pd.read_csv("data\\updf_noise4.csv", header=None).iloc[0:2000, :DATA_SIZE], data_sizes[3],
            #           dataframe=True),  # 1,0
            # load_data(pd.read_csv("data\\updf_noise4.csv", header=None).iloc[0:2000, :DATA_SIZE], data_sizes[4],
            #           dataframe=True),  # 1,0
        ]

        # directories = ['noise', '10', '12', '02']
        directories = ['git']


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
            load_data(f"data\\df_noise.csv", data_size),
            load_data(f"data\\df_noise2.csv", data_size),
            load_data(f"data\\df_noise3.csv", data_size),
            load_data(f"data\\df_noise4.csv", data_size)
        ]

        for df, directory in zip(dataframes, directories):
            print('data loaded')
            output_directory = output_directory_path+directory+'\\'

            x_train, y_train, x_test, y_test,  y_true, input_shape, nb_classes = prepare_data(df)
            classifier = CNN3.Classifier_CNN(output_directory, input_shape, nb_classes, verbose=False)
            classifier.fit(x_train, y_train, x_test, y_test, y_true)
            prediction = classifier.predict(x_test, y_true, x_train, y_train, y_test)
            print('PREDICTION:', prediction)

            for _df in dataframes2:
                _x_train, _y_train, _x_test, _y_test, _y_true, _input_shape, _nb_classes = prepare_data(_df)
                _prediction = classifier.predict(_x_test, _y_true, _x_train, _y_train, _y_test)
                print('PREDICTION inner:', _prediction)


            # print('prediction of 0,1,2:')
            # predict(classifier, pd.read_csv("data\\df_noise.csv", header=None).iloc[:, 1:],
            #         pred_rows_used=3000)  # all
            #
            # print('prediction of 0,1:')
            # predict(classifier, pd.read_csv("data\\df_noise.csv", header=None).iloc[0:2000, 1:],
            #         pred_rows_used=2000)  # 1 and 0
            #
            # print('prediction of 1,2:')
            # predict(classifier, pd.read_csv("data\\df_noise.csv", header=None,
            #                                 skiprows=[i for i in range(1000, 2001)]), pred_rows_used=2000)  # 1 and 2
            #
            # print('prediction of 0,2:')
            # predict(classifier, pd.read_csv("data\\df_noise.csv", header=None).iloc[1000:3000, 1:],
            #         pred_rows_used=2000)  # 0 and 2



            # print('prediction of 0,1,2:')
            # predict(classifier, pd.read_csv("data\\df_noise2.csv", header=None).iloc[:, 1:],
            #         pred_rows_used=3000)  # all
            #
            # print('prediction of 0,1:')
            # predict(classifier, pd.read_csv("data\\df_noise2.csv", header=None).iloc[0:2000, 1:],
            #         pred_rows_used=2000)  # 1 and 0
            # #
            # print('prediction of 1,2:')
            # predict(classifier, pd.read_csv("data\\df_noise2.csv", header=None,
            #                                 skiprows=[i for i in range(1000, 2001)]), pred_rows_used=2000)  # 1 and 2
            #
            # print('prediction of 0,2:')
            # predict(classifier, pd.read_csv("data\\df_noise2.csv", header=None).iloc[1000:3000, 1:],
            #         pred_rows_used=2000)  # 0 and 2
