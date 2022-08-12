import pandas as pd
import csv
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

all_df = pd.read_csv("data\\array_500.csv", header=None)
first_col = all_df.iloc[:, 0].round()  # 0,1,2
df = all_df.iloc[:, 1:]  # rest





def add_noise(x, noise):
    noised = x + np.random.normal(np.mean(x), noise, x.shape) - np.mean(x)
    noised = noised.round(4)
    return noised
    # mozna rozwazyc dodanie wartosci minimalnej


def add_noise2(x, noise):
    noised = x + np.random.normal(0, noise)
    noised = noised.round(4)
    return noised


def add_noise3(x, noise):
    diff = np.max(x) - np.min(x)
    noised = x + np.random.normal(diff, noise)
    return noised


def add_noise4(x, noise):
    # przerobic żeby każd
    xmin = x.min()
    xmax = x.max()
    print('xmin:', xmin, xmax)
    # n = xx.shape
    noised = x
    # noised = xmin + np.random.normal(0, 1, n) * (xmax - xmin)
    noised.round(4)
    noised_df = pd.DataFrame(noised)
    return noised_df


def add_noise5(x, noise):
    p = 6  # zostawic jaki jest
    xmin = 0.38;
    xmax = 0.5;
    n = 10000;
    X = xmin + (xmax - xmin) * sum(rand(n, p), 2) / p


def add_noise6(x, noise):
    for row in x:
        print(x.iloc[row])
    return x

def add_noise7(df, sigma):
    mu = 0
    noise = np.random.normal(mu, sigma, df.shape)
    return (df+noise).round(4)



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


def plot_four_noises(df1, df2, df3, df4, row, with_original=False, original=df, from_zero=False):

    if from_zero:
        plt.ylim(ymin=0)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(df1.iloc[row, 1:])
    axs[0, 0].set_title('Axis [0, 0]')
    axs[0, 1].plot(df2.iloc[row, 1:])
    axs[0, 1].set_title('Axis [0, 1]')
    axs[1, 0].plot(df3.iloc[row, 1:])
    axs[1, 0].set_title('Axis [1, 0]')
    axs[1, 1].plot(df4.iloc[row, 1:])
    axs[1, 1].set_title('Axis [1, 1]')

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


df_noise = add_noise7(df, 0.5)
df_noise2 = add_noise7(df, 1)
df_noise3 = add_noise7(df, 3)
df_noise4 = add_noise7(df, 5)

# x_df_noise = add_noise3(df, 0.5, first)
# x_df_noise2 = add_noise3(df, 1, first)
# x_df_noise3 = add_noise3(df, 3, first)
# x_df_noise4 = add_noise3(df, 5, first)

# plot_noise(df, 1036)
# plot_noise(df_noise, 1035, with_original=True)
# plot_noise(df_noise2, 1035)
# plot_noise(df_noise3, 1035)
# plot_noise(df_noise4, 1035)

# plot_noise(df, 2035)
# plot_noise(df_noise, 2035, with_original=True)
# plot_noise(df_noise2, 2035)
# plot_noise(df_noise3, 2035)
# plot_noise(df_noise4, 2035)

# plot_noise(df, 35)
# plot_noise(df_noise, 35, with_original=True)
# plot_noise(df_noise2, 35)
# plot_noise(df_noise3, 35)
# plot_noise(df_noise4, 35)

# plot_four_noises(df, df_noise, df_noise2, df_noise3, 15, with_original=True)
indexes = [16,1016, 2016]
for num in indexes:
    plot_four_noises(df, df_noise, df_noise2, df_noise3, num, with_original=True)


# save_noise(df_noise, "test_df_noise", first_col)
# save_noise(df_noise2, "test_df_noise2", first_col)
# save_noise(df_noise3, "test_df_noise3", first_col)
# save_noise(df_noise4, "test_df_noise4", first_col)
