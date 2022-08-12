import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from math import pi, fabs



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
DATA_SIZE = 500
array_lp, array_lp2 = [], []
array_cs = []
array_sin = []
for i in range(1000):
    tc = 6
    m = np.random.uniform(0.1, 0.9)  # 0.1 <= m <= 0.9
    lambda_val = np.random.uniform(1.8, 2.2)
    omega = (2 * np.pi) / (np.log(lambda_val))

    C = abs(np.random.normal())  # |C| < 1
    B = np.random.uniform(-15, 0)  # B < 0
    # A = np.random.uniform(10, 150)
    A = 20
    phi = np.random.uniform(-2 * pi, 2 * pi)

    # DATA_SIZE = np.random.randint(400, 1000) przeniesione przed pętlę
    NOISE_FACTOR = 0.0
    CUTOFF = 0.8
    ROUNDING_DECIMALS = 8

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

    simulated_data = pd.Series(data=log_prices, index=np.round(t_cutoff, ROUNDING_DECIMALS))
    logperiodic_sign = pd.Series([1], index=[-0.1])
    simulated_data = logperiodic_sign.append(simulated_data)
    # show_plot(line_data, "Logperiodic")
    # show_plot(simulated_data, f"Logperiodic with cutoff B={B}")
    print(simulated_data)

    line_data = line_data + fabs(2*min(line_data))
    line_data = line_data / max(line_data)
    line_data *= 10
    # show_plot(line_data, "Przykład")

    line_data = np.insert(line_data, 0, 1)  # 0 - nonlogperiodic, 1 - logperiodic
    array_lp.append(line_data[:-1])




    array_lp2.append(simulated_data)



    #
    # WHITE NOISE CUMSUM
    #

    mean = 0
    std = 1
    samples = np.random.normal(mean, std, size=DATA_SIZE)
    cumsum = A + np.cumsum(samples)

    cumsum = cumsum + fabs(2 * min(cumsum))
    cumsum = cumsum / max(cumsum)
    cumsum *= 10
    # show_plot(cumsum, "Przykład ruch Browna")

    # show_plot(cumsum, "Cumsum")

    cumsum = np.insert(cumsum, 0, 0)  # 0 - nonlogperiodic, 1 - logperiodic
    array_cs.append(cumsum[:-1])

    #
    # SINUSOID
    #
    time = np.arange(0, DATA_SIZE / 10, 0.1)
    amplitude = A + np.sin(time) * random.randint(int(0.05 * A), int(1 * A))

    amplitude = amplitude + fabs(2 * min(amplitude))
    amplitude = amplitude / max(amplitude)
    amplitude *= 10
    # show_plot(amplitude, "Przykład amplituda")


    # show_plot(amplitude, "Amplitude", True)

    amplitude = np.insert(amplitude, 0, 2)
    array_sin.append(amplitude[:-1])

dataframe_lp = pd.DataFrame(array_lp, columns=None)
# dataframe_lp.to_csv('data\\array_lp.csv', index=False, header=False)

dataframe_cs = pd.DataFrame(array_cs, columns=None)
# dataframe_cs.to_csv('data\\array_cs.csv', index=False, header=False)

dataframe_sin = pd.DataFrame(array_sin, columns=None)
# dataframe_sin.to_csv('data\\array_sin.csv', index=False, header=False)

df_all_rows = pd.concat([dataframe_lp, dataframe_cs, dataframe_sin])
df_all_rows.to_csv('data\\array_500.csv', index=False, header=False)

# dataframe_simulated = pd.DataFrame(array_lp2).round(ROUNDING_DECIMALS)
# dataframe_simulated.to_csv('data\\array_simulated.csv', index=False, header=True)
# # dataframe_sim.T.to_csv('data\\array_sim.csv', index=False, header=True)

