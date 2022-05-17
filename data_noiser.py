import pandas as pd
import csv
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data\\array_500.csv", header=None)
first = df.iloc[:, 0]
df = df.iloc[:, 1:]
df_noise = df + np.random.normal(np.mean(df), 0.5, df.shape) - np.mean(df)
df_noise2 = df + np.random.normal(np.mean(df), 1, df.shape) - np.mean(df)
df_noise3 = df + np.random.normal(np.mean(df), 3, df.shape) - np.mean(df)
df_noise4 = df + np.random.normal(np.mean(df), 5, df.shape) - np.mean(df)

# plt.plot(df_noise.iloc[1])
# plt.plot(df.iloc[1])
# plt.show()
#
# plt.plot(df_noise2.iloc[1])
# plt.plot(df.iloc[1])
# plt.show()
#
# plt.plot(df_noise3.iloc[1])
# plt.plot(df.iloc[1])
# plt.show()
#
# plt.plot(df_noise4.iloc[1])
# plt.plot(df.iloc[1])
# plt.show()
#
# plt.plot(df_noise.iloc[501])
# plt.plot(df.iloc[501])
# plt.show()
#
# plt.plot(df_noise2.iloc[501])
# plt.plot(df.iloc[501])
# plt.show()
#
# plt.plot(df_noise3.iloc[501])
# plt.plot(df.iloc[501])
# plt.show()
#
# plt.plot(df_noise4.iloc[501])
# plt.plot(df.iloc[501])
# plt.show()


df_noise.insert(0, "0", first, True)
df_noise2.insert(0, "0", first, True)
df_noise3.insert(0, "0", first, True)
df_noise4.insert(0, "0", first, True)

df_noise = pd.DataFrame(df_noise, columns=None)
df_noise.to_csv('data\\df_noise.csv', index=False, header=False)

df_noise2 = pd.DataFrame(df_noise2, columns=None)
df_noise2.to_csv('data\\df_noise2.csv', index=False, header=False)

df_noise3 = pd.DataFrame(df_noise3, columns=None)
df_noise3.to_csv('data\\df_noise3.csv', index=False, header=False)

df_noise4 = pd.DataFrame(df_noise4, columns=None)
df_noise4.to_csv('data\\df_noise4.csv', index=False, header=False)


