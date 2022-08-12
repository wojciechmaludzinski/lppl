import pandas as pd
import csv
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import LPPL_fitting
from LPPL_fitting import LPPL_fit

df = pd.read_csv('data\\array_simulated.csv')
df = df.iloc[:, 1:]
first_row = df.iloc[8]
data = pd.Series(first_row, dtype=float)
data.index = data.index.astype(float)
print(type(data.index))


tc = 6
m = 4
lambda_val = 2
omega = omega = (2 * np.pi) / (np.log(lambda_val))
A = 50
B = -3
phi = 10
C = abs(np.random.normal())
C1 = C * np.cos(phi)
C2 = C * np.sin(phi)

rv = LPPL_fitting.LPPL_fit(data)

if rv.success:
    line_points = len(data.values) * rv.tc
    t_ = np.linspace(0, rv.tc, num=int(line_points))
    est_line_data = rv.A + rv.B * (rv.tc - t_) ** rv.m + \
                    rv.C1 * (rv.tc - t_) ** rv.m * np.cos(rv.omega * np.log(rv.tc - t_)) + \
                    rv.C2 * (rv.tc - t_) ** rv.m * np.sin(rv.omega * np.log(rv.tc - t_))
    est_line_data_index = np.linspace(0, rv.tc, len(est_line_data))

    price_tc = rv.A + rv.B * (0.001) ** rv.m + \
               rv.C1 * (0.001) ** rv.m * np.cos(rv.omega * np.log(0.001)) + \
               rv.C2 * (0.001) ** rv.m * np.sin(rv.omega * np.log(0.001))

    # print()
    # print("== RESULTS ==")
    # print("  price tc: %.2f (%.2f)" % (price_tc, est_line_data[-2]))
    # print("   tc:   real value: % 8.2f    estimation: % 8.2f" % (tc, rv.tc))
    # print("    m:   real value: % 8.2f    estimation: % 8.2f" % (m, rv.m))
    # print("omega:   real value: % 8.2f    estimation: % 8.2f" % (omega, rv.omega))
    # print("    A:   real value: % 8.2f    estimation: % 8.2f" % (A, rv.A))
    # print("    B:   real value: % 8.2f    estimation: % 8.2f" % (B, rv.B))
    # print("   C1:   real value: % 8.2f    estimation: % 8.2f" % (C1, rv.C1))
    # print("   C2:   real value: % 8.2f    estimation: % 8.2f" % (C2, rv.C2))
    # print()
    # print("== ERROR STATISTICS ==")
    # print("Mean square error: % 20.2f" % (rv.mse,))
    # print(" MSE (normalized): % 20.2f" % (rv.norm_mse,))

    plt.plot(data.index, data.values, '.')
    plt.title('Real data')
    # plt.show()
    plt.plot(est_line_data_index, est_line_data, 'r-')
    plt.title("MSE: %d, NMSE: %.2f, tc: %.2f" % (rv.mse, rv.norm_mse, rv.tc))
    plt.show()


