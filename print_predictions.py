
import pandas as pd
from utils.utils import pred_counter, pred_plot_all_lppls
from data_handler import get_data_size

DATA_SIZE = get_data_size()

def predict(clf, df, pred_rows_used=500):
    df_smaller = df.iloc[:pred_rows_used, :DATA_SIZE - 1]
    print(pred_counter(clf.predicting(df_smaller)))
    print(pred_plot_all_lppls(clf.predicting(df_smaller), df_smaller))


def print_predictions(classifier):
    print('prediction of 0,1,2:')
    predict(classifier, pd.read_csv("data\\df_noise.csv", header=None).iloc[:, 1:],
            pred_rows_used=3000)  # all

    print('prediction of 0,1:')
    predict(classifier, pd.read_csv("data\\df_noise.csv", header=None).iloc[0:2000, 1:],
            pred_rows_used=2000)  # 1 and 0

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
