import pandas as pd
import csv
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

all_df = pd.read_csv("data\\array_500.csv", header=None)
first = all_df.iloc[:, 0].round()  # 0,1,2
df = all_df.iloc[:, 1:]  # rest


def add_noise(x, noise, y):
    noised = x + np.random.normal(np.mean(x), noise, x.shape) - np.mean(x)
    noised = noised.round(4)
    noised.insert(0, "0", y, True)
    return noised


def add_noise2(x, noise, y):
    noised = x + np.random.normal(0, noise)
    noised = noised.round(4)
    noised.insert(0, "0", y, True)
    return noised


def save_noise(dataframe, filename):
    _ = pd.DataFrame(dataframe, columns=None)
    _.to_csv(f'data\\{filename}.csv', index=False, header=False)


def plot_noise(dataframe, row, with_original=False, original=df):
    plt.plot(dataframe.iloc[row, 1:])
    if with_original:
        plt.plot(original.iloc[row])
        plt.show()


df_noise = add_noise(df, 0.5, first)
df_noise2 = add_noise(df, 1, first)
df_noise3 = add_noise(df, 3, first)
df_noise4 = add_noise(df, 5, first)

save_noise(df_noise, "df_noise")
save_noise(df_noise2, "df_noise2")
save_noise(df_noise3, "df_noise3")
save_noise(df_noise4, "df_noise4")