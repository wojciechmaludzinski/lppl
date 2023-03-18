import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from math import pi, fabs
from utils.utils import normalize_array, df_index
from data_handler import get_data_size

def show_plot(data, title=" ", bottom=False):
    plt.plot(data)
    plt.title(title)
    if bottom:
        plt.ylim(bottom=0)
    plt.show()


# LPPL 4 factor model

# disable warnings
np.seterr(divide='ignore', invalid='ignore', over='ignore')
plt.axis('off')
DATA_SIZE = get_data_size()
array_lp, array_lp2 = [], []
array_cs = []
array_sin = []
array_up = []
ROUNDING_DECIMALS = 8
for i in range(1000):
    A = np.random.uniform(10, 300)  # >0
    B = np.random.uniform(-A / 2, -A / 10)
    C = np.random.uniform(3, 10)  # C = abs(np.random.normal())  # |C| < 1=
    tc = DATA_SIZE - 1
    m = np.random.uniform(0.1, 0.9)  # 0.1 <= m <= 0.9

    lambda_val = 2  # lambda_val = np.random.uniform(1.8, 2.2)
    omega = (2 * pi) / (np.log(lambda_val))
    phi = np.random.uniform(-2 * pi, 2 * pi)
    NOISE_FACTOR = 0.0
    CUTOFF = 0.8

    t = np.linspace(0, tc, num=DATA_SIZE)
    line_data = A + B * (tc - t) ** m + C * (tc - t) ** m * np.cos(omega * np.log(tc - t) + phi)
    line_data_index = np.linspace(0, tc, len(line_data))
    log_prices = [x + np.random.normal(0, NOISE_FACTOR) for x in line_data]
    log_prices = log_prices[:int(DATA_SIZE * CUTOFF)]
    t_cutoff = t[:int(DATA_SIZE * CUTOFF)]
    factor = 1 / max(t_cutoff)
    t_cutoff = t_cutoff * factor
    line_data_index = line_data_index * factor
    t = t * factor
    tc = max(t)

    line_data = normalize_array(line_data[:-1])
    line_data = np.insert(line_data, 0, 1)  # 0 - nonlogperiodic, 1 - logperiodic, 2 - sinusoid
    array_lp.append(line_data)

    # saving with index
    simulated_data = pd.Series(data=log_prices, index=np.round(t_cutoff, ROUNDING_DECIMALS))
    logperiodic_sign = pd.Series([1], index=[-0.1])
    simulated_data = logperiodic_sign.append(simulated_data)
    array_lp2.append(simulated_data)

    #
    # WHITE NOISE CUMSUM
    #
    samples = np.random.normal(0, 1, size=DATA_SIZE)
    cumsum = A + np.cumsum(samples)
    cumsum = normalize_array(cumsum[:-1])
    cumsum = np.insert(cumsum, 0, 0)  # 0 - nonlogperiodic, 1 - logperiodic, 2 - sinusoid
    array_cs.append(cumsum)

    #
    # WHITE NOISE CUMSUM WITH UPWARDS TREND
    #
    upwardness = np.arange(DATA_SIZE)
    upwards = cumsum * upwardness
    upwards = normalize_array(upwards)
    array_up.append(upwards)




    #
    # SINUSOID
    #
    time = np.arange(0, DATA_SIZE / 10, 0.1)
    amplitude = A + np.sin(time + phi) * random.randint(int(0.05 * A), int(0.7 * A))
    amplitude = normalize_array(amplitude[:-1])
    amplitude = np.insert(amplitude, 0, 2)  # 0 - nonlogperiodic, 1 - logperiodic, 2 - sinusoid
    array_sin.append(amplitude)

    # show_plot(line_data[1:], f"Przykład logperiodycznosci A={A} B={B} C={C}", bottom=True)
    # show_plot(cumsum[1:], "Przykład ruch Browna")
    # show_plot(amplitude[1:], "Przykład sinusoidy", bottom=True)
    # show_plot(upwards[1:])

dataframe_lp = pd.DataFrame(array_lp, columns=None).round(ROUNDING_DECIMALS)
dataframe_cs = pd.DataFrame(array_cs, columns=None).round(ROUNDING_DECIMALS)
dataframe_sin = pd.DataFrame(array_sin, columns=None).round(ROUNDING_DECIMALS)
dataframe_up = pd.DataFrame(array_up, columns=None).round(ROUNDING_DECIMALS)

# dataframe_lp.to_csv('data\\array_lp.csv', index=False, header=False)
# dataframe_cs.to_csv('data\\array_cs.csv', index=False, header=False)
# dataframe_sin.to_csv('data\\array_sin.csv', index=False, header=False)

df_all_rows = pd.concat([dataframe_lp, dataframe_cs, dataframe_sin])
df_all_rows.to_csv('data\\array_500up.csv', index=False, header=False)

# dataframe_simulated = pd.DataFrame(array_lp2).round(ROUNDING_DECIMALS)
# dataframe_simulated.to_csv('data\\array_simulated.csv', index=False, header=True)
# # dataframe_sim.T.to_csv('data\\array_sim.csv', index=False, header=True)
