import pandas as pd
import os
from data_handler import load_data, get_data_size, get_output_dir
DATA_SIZE = get_data_size() 
data_sizes = [500, 450, 400, 350, 300, 100]

'''4 Dataframes, first have 3000 elements, 2-4 have 2000 elements, each 1000 elements are from different class'''
def load_dataframes_casual(path="data\\df_noise.csv"):
    dataframes = [
                load_data(path, DATA_SIZE), #0,1,2 classes
                load_data(pd.read_csv(path, header=None).iloc[0:2000, :DATA_SIZE], DATA_SIZE,
                        dataframe=True),  # 1,0
                load_data(pd.read_csv(path, header=None,
                                    skiprows=[i for i in range(1000, 2000)]), DATA_SIZE, dataframe=True),  # 1,2
                load_data(pd.read_csv(path, header=None).iloc[1000:3000], DATA_SIZE, dataframe=True)
                # 0,2
                ]
    return dataframes

def load_dataframes_diff_sizes(paths, data_sizes=data_sizes):
    dataframe = []
    for path in paths:
        for data_size in data_sizes:
            dataframe.append(load_data(path, data_size))
    return dataframe


def load_4_dataframes(data_size):
    dataframes2 = [
                load_data(f"data\\df_noise.csv", data_size),
                load_data(f"data\\df_noise2.csv", data_size),
                load_data(f"data\\df_noise3.csv", data_size),
                load_data(f"data\\df_noise4.csv", data_size)
            ]
    return dataframes2

def load_directories(dir_names):
    for dir_name in dir_names:
        if not os.path.exists(get_output_dir()+dir_name):
            os.makedirs(get_output_dir()+dir_name)
    
    return dir_names



def load_noise123_and_noise_10_12_02():
    dataframes2 = [
        load_data(f"data\\df_noise.csv", DATA_SIZE),
        load_data(f"data\\df_noise2.csv", DATA_SIZE),
        load_data(f"data\\df_noise3.csv", DATA_SIZE),
        load_data(pd.read_csv("data\\df_noise.csv", header=None).iloc[0:2000, :DATA_SIZE], DATA_SIZE, dataframe=True),  # 1,0
        load_data(pd.read_csv("data\\df_noise.csv", header=None,
                                skiprows=[i for i in range(1000, 2000)]), DATA_SIZE, dataframe=True),  # 1,2
        load_data(pd.read_csv("data\\df_noise.csv", header=None).iloc[1000:3000], DATA_SIZE, dataframe=True) #0,2
    ]
    return dataframes2

def load_updataframes():
    dataframes = [
    load_data(f"data\\test_df_noise.csv", DATA_SIZE),
    load_data(pd.read_csv("data\\updf_noise4.csv", header=None).iloc[0:2000, :DATA_SIZE], data_sizes[0],
                                dataframe=True),  # 1,0
    load_data(pd.read_csv("data\\updf_noise4.csv", header=None).iloc[0:2000, :DATA_SIZE], data_sizes[1],
                dataframe=True),  # 1,0
    load_data(pd.read_csv("data\\updf_noise4.csv", header=None).iloc[0:2000, :DATA_SIZE], data_sizes[2],
                dataframe=True),  # 1,0
    load_data(pd.read_csv("data\\updf_noise4.csv", header=None).iloc[0:2000, :DATA_SIZE], data_sizes[3],
                dataframe=True),  # 1,0
    load_data(pd.read_csv("data\\updf_noise4.csv", header=None).iloc[0:2000, :DATA_SIZE], data_sizes[4],
                dataframe=True),  # 1,0
    ]
    return dataframes