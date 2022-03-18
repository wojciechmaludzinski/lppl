import pandas as pd
import csv
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

input_path = "C:\\Users\\mmigala\\Desktop\\"
input_file = "output.csv"

# LPPL 4 factor model

# disable warnings
np.seterr(divide='ignore', invalid='ignore', over='ignore')
plt.axis('off')
DATA_SIZE = 1000
array_lp = []
array_cs = []
for i in range(400):
    tc = 6
    m = np.random.uniform(0.1, 0.9)  # 0.1 <= m <= 0.9
    lambda_val = 2
    omega = (2 * np.pi) / (np.log(lambda_val))
    # omega = np.random.uniform(9, 13)  # 6 <= omega <= 13

    C = abs(np.random.normal())  # |C| < 1
    B = np.random.uniform(-10, 0)  # B < 0
    A = 200
    phi = 10

    # DATA_SIZE = np.random.randint(400, 1000) przeniesione przed pętlę
    NOISE_FACTOR = 0.5
    CUTOFF = 0.8

    t = np.linspace(0, tc, num=DATA_SIZE)
    line_data = A + B * (tc - t) ** m + C * (tc - t) ** m * np.cos(omega * np.log(tc - t) - phi)
    line_data_index = np.linspace(0, tc, len(line_data))
    log_prices = [x + np.random.normal(0, NOISE_FACTOR) for x in line_data]
    log_prices = log_prices[:int(DATA_SIZE * CUTOFF)]
    t_cutoff = t[:int(DATA_SIZE * CUTOFF)]
    factor = 1 / max(t_cutoff)
    t_cutoff = t_cutoff * factor
    line_data_index = line_data_index * factor
    t = t * factor
    tc = max(t)

    simulated_data = pd.Series(data=log_prices, index=t_cutoff)

    # print("tc: %.2f" % tc)
    # plt.plot(simulated_data, '-')

    # plt.plot(line_data_index, line_data, 'b-')

    noise = np.random.normal(np.mean(line_data[:-1]), 0.5, line_data.shape) - np.mean(line_data[:-1])
    new_signal = line_data  # + noise
    # plt.plot(new_signal)
    # plt.show()
    new_signal = np.insert(new_signal, 0, 1)  # 0 - nonlogperiodic, 1 - logperiodic
    # print(new_signal)
    # print(new_signal)
    # plt.plot(line_data_index, new_signal, 'b-')

    number = random.randint(1, 1000)
    # plt.show()
    array_lp.append(new_signal[:-1])

    # plt.savefig(f'data\\lppl\\{number}.png',bbox_inches='tight', pad_inches = 0)
    # plt.clf()
    # dataframe = pd.DataFrame(np_array)
    # print(np_array)
    # np.savetxt("foo.csv", np_array, delimiter=",")

    # White noise cumsum
    mean = 0
    std = 1
    samples = np.random.normal(mean, std, size=DATA_SIZE)
    cumsum = 200 + np.cumsum(samples)  # + noise
    # plt.plot(cumsum)
    # plt.show()
    cumsum = np.insert(cumsum, 0, 0)  # 0 - nonlogperiodic, 1 - logperiodic
    array_cs.append(cumsum[:-1])

dataframe_lp = pd.DataFrame(array_lp, columns=None)
dataframe_lp.to_csv('data\\array_lp.csv', index=False, header=False)

dataframe_cs = pd.DataFrame(array_cs, columns=None)
dataframe_cs.to_csv('data\\array_cs.csv', index=False, header=False)


