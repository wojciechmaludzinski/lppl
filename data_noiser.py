import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

all_df = pd.read_csv("data\\array_500.csv", header=None)
first_col = all_df.iloc[:, 0].round()  # 0,1,2
df = all_df.iloc[:, 1:]  # rest
ROUNDING_DECIMALS = 8


def add_noise(x, noise):
    noised = x + np.random.normal(np.mean(x), noise, x.shape) - np.mean(x)
    noised = noised.round(4)
    return noised
    # mozna rozwazyc dodanie wartosci minimalnej


def add_noise2(x, noise):
    noised = x + np.random.normal(0, noise, x.shape)
    noised = noised.round(ROUNDING_DECIMALS)
    return noised


# Too much noising
def add_noise3(x, noise):
    diff = x.max(axis=1)-x.min(axis=1)
    row_noises = np.random.normal(diff, noise)
    xd = []
    for index in range(0, x.shape[0]):
        row_noise = x.iloc[index] + np.random.normal(row_noises[index], noise, x.iloc[index].shape)
        xd.append(row_noise)
    return pd.DataFrame(xd)


# Too much noising
def add_noise4(x, noise):
    xmin = x.min(axis=1)
    xmax = x.max(axis=1)
    print('xmin:', xmin, xmax)
    xd = []

    for index in range(0, x.shape[0]):
        row_noise = xmin[index] + np.random.normal(0, noise, x.iloc[index].shape) * (xmax[index]-xmin[index]) #x.iloc[index]
        xd.append(x.iloc[index]+row_noise)
    return pd.DataFrame(xd)


def add_noise5(x, noise):
    p = x.shape[1]  # zostawic jaki jest
    xmin = x.min(axis=1)
    xmax = x.max(axis=1)
    n = noise * 1000
    xd = []

    for index in range(0, x.shape[0]):
        row_noise = xmin[index] + (xmax[index]-xmin[index]) * sum(np.random.random((int(n), p)), 2) / p #x.iloc[index]
        xd.append(x.iloc[index] + row_noise-xmax[index])
    return pd.DataFrame(xd)


def add_first_column(x, first):
    x.insert(0, "0", first, True)
    return x


def save_noise(dataframe, filename, first):
    dataframe = add_first_column(dataframe, first)
    _ = pd.DataFrame(dataframe, columns=None)
    _.to_csv(f'data\\{filename}.csv', index=False, header=False)


def plot_noise(dataframe, row, with_original=False, original=df, from_zero=False):
    plt.plot(dataframe.iloc[row, 1:])
    if with_original:
        plt.plot(original.iloc[row])
    if from_zero:
        plt.ylim(ymin=0)
    plt.show()


def plot_four_noises(df1, df2, df3, df4, row, with_original=False, original=df, from_zero=False,
                     title00='Axis [0, 0]', title01='Axis [0, 1]', title10='Axis [1, 0]', title11='Axis [1, 1]'):
    if from_zero:
        plt.ylim(ymin=0)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(df1.iloc[row, 1:])
    axs[0, 0].set_title(title00)
    axs[0, 1].plot(df2.iloc[row, 1:])
    axs[0, 1].set_title(title01)
    axs[1, 0].plot(df3.iloc[row, 1:])
    axs[1, 0].set_title(title10)
    axs[1, 1].plot(df4.iloc[row, 1:])
    axs[1, 1].set_title(title11)

    if with_original:
        axs[0, 0].plot(original.iloc[row])
        axs[0, 1].plot(original.iloc[row])
        axs[1, 0].plot(original.iloc[row])
        axs[1, 1].plot(original.iloc[row])

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()


df_noise = add_noise2(df, 0.5)
df_noise2 = add_noise2(df, 1)
df_noise3 = add_noise2(df, 3)
df_noise4 = add_noise2(df, 5)


# plot_four_noises(df, df_noise, df_noise2, df_noise3, 15, with_original=True)
indexes = [20, 1020, 21, 1021, 22, 1022, 23, 1023]
for num in indexes:
    plot_four_noises(df, df_noise, df_noise2, df_noise3, num, with_original=True)

save_noise(df_noise, "df_noise", first_col)
save_noise(df_noise2, "df_noise2", first_col)
save_noise(df_noise3, "df_noise3", first_col)
save_noise(df_noise4, "df_noise4", first_col)
