import numpy as np
import pandas as pd
import sklearn

'''Loads data and shuffles it'''
def load_data(path_or_df, columns, dataframe=False):
    if dataframe:
        return path_or_df.sample(frac=1).reset_index(drop=True).iloc[:, :columns]

    return pd.read_csv(path_or_df, header=None).sample(frac=1).reset_index(drop=True).iloc[:, :columns]

'''Retruns all necessery x and y'''
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

'''
Returns only list of length group_size, for skip_size=5 it's [[1,2,..500], [6,7,..505]..]
If list from file, then _list=open('numbers.txt').read().splitlines()
'''
def split_list(_list, group_size=500, skip_size=5):
    result = []
    for i in range(0, len(_list), skip_size):
        group = _list[i:i + group_size]
        if len(group) == group_size:
            result.append(group)
    return result

'''
example path: "C:\\Users\\Wojtek\\Repos\\download\\eurpln-h1-bid-2008-01-01-2009-01-01.csv"
get from script program
'''
def modify_currencies(path):
    df = pd.read_csv(path,
                 header=[0])
    nicelist = split_list(df.iloc[:, -1].values.tolist(), 500, 5)

    dataframe = pd.DataFrame(nicelist, columns=None)
    dataframe.to_csv('data\\array_eurpln.csv', index=False, header=False)



    
#Deprecated
def split_list_diff_sizes(_list, group_size=500, skip_size=5):
        return [_list[i:i + group_size] for i in range(0, len(_list), skip_size)]
