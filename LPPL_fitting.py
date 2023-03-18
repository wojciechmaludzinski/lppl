import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy import optimize


def F1_get_linear_parameters(X, stock_data):
    tc, m, omega = X

    t = np.array(stock_data.index)
    y = np.array(stock_data.values)

    N = len(stock_data)
    f = (tc - t) ** m
    g = (tc - t) ** m * np.cos(omega * np.log(tc - t))
    h = (tc - t) ** m * np.sin(omega * np.log(tc - t))

    LHS = np.array([[N, sum(f), sum(g), sum(h)],
                    [sum(f), sum(f ** 2), sum(f * g), sum(f * h)],
                    [sum(g), sum(f * g), sum(g ** 2), sum(g * h)],
                    [sum(h), sum(f * h), sum(g * h), sum(h ** 2)]])

    RHS = np.array([[sum(y)],
                    [sum(y * f)],
                    [sum(y * g)],
                    [sum(y * h)]])

    A, B, C1, C2 = np.linalg.solve(LHS, RHS)
    return A, B, C1, C2


def F1(X, stock_data):
    tc, m, omega = X
    t = np.array(stock_data.index)
    y = np.array(stock_data.values)
    A, B, C1, C2 = F1_get_linear_parameters(X, stock_data)
    error = y - A - B * (tc - t) ** m - C1 * (tc - t) ** m * np.cos(omega * np.log(tc - t)) - \
            C2 * (tc - t) ** m * np.sin(omega * np.log(tc - t))
    cost = sum(error ** 2)
    return cost


def F1_normalized(result, stock_data):
    x1 = min(stock_data.values)
    x2 = max(stock_data.values)
    b = (x1 + x2) / (x1 - x2)
    a = (-1 - b) / x1
    data = np.array(stock_data.values) * a + b
    stock_data_norm = pd.Series(data=data, index=stock_data.index)
    return F1(result, stock_data_norm)


class Result:
    def __init__(self):
        self.success = None

        # model parameters
        self.tc = None
        self.m = None
        self.omega = None
        self.A = None
        self.B = None
        self.C1 = None
        self.C2 = None
        self.C = None
        self.pruned = None  # True if one of the parameters has been pruned to the
        # valid range after fitting.
        self.price_tc = None  # Estimated price at tc
        self.price_chg = None  # Price difference between est. price at tc and last price in percent

        self.mse = None  # mean square error
        self.mse_hist = []  # history of mean square errors
        self.norm_mse = None  # normalized mean square error
        self.opt_rv = None  # Return object from optimize function
        self.tc_start = []
        self.m_start = []
        self.omega_start = []


def LPPL_fit(data, tries=20, min_distance=0.2):
    rv = Result()
    fitted_parameters = None
    mse_min = None
    fitted_pruned = False

    tc_min, tc_max = 1, 1.6  # Critical time
    m_min, m_max = 0.1, 0.5  # Convexity: smaller is more convex
    omega_min, omega_max = (2 * np.pi) / (np.log(2.2)), (2 * np.pi) / (np.log(1.8))  # Number of oscillations


    # Scaling parameters to scale tc, m and omega to range 0 .. 1

    tc_scale_b = tc_min / (tc_min - tc_max)
    tc_scale_a = -tc_scale_b / tc_min

    m_scale_b = m_min / (m_min - m_max)
    m_scale_a = -m_scale_b / m_min

    omega_scale_b = omega_min / (omega_min - omega_max)
    omega_scale_a = -omega_scale_b / omega_min

    for n in range(tries):

        found = False

        while not found:
            tc_start = np.random.uniform(low=tc_min, high=tc_max)
            m_start = np.random.uniform(low=m_min, high=m_max)
            omega_start = np.random.uniform(low=omega_min, high=omega_max)
            found = True

            for i in range(len(rv.tc_start)):
                # Scale values to range 0 .. 1
                # Calculate distance and reject starting point if too close to
                # already used starting point
                a = np.array([tc_start * tc_scale_a + tc_scale_b,
                              m_start * m_scale_a + m_scale_b,
                              omega_start * omega_scale_a + omega_scale_b])
                b = np.array([rv.tc_start[i] * tc_scale_a + tc_scale_b,
                              rv.m_start[i] * m_scale_a + m_scale_b,
                              rv.omega_start[i] * omega_scale_a + omega_scale_b])
                distance = np.linalg.norm(a - b)
                if distance < min_distance:
                    found = False
                    # print("Points to close together: ", a, b)
                    break

        rv.tc_start.append(tc_start)
        rv.m_start.append(m_start)
        rv.omega_start.append(omega_start)

        x0 = [tc_start, m_start, omega_start]

        try:
            opt_rv = optimize.minimize(F1, x0, args=(data,), method='Nelder-Mead')
            if opt_rv.success:

                tc_est, m_est, omega_est = opt_rv.x
                pruned = False

                if tc_est < tc_min:
                    tc_est = tc_min
                    pruned = True
                elif tc_est > tc_max:
                    tc_est = tc_max
                    pruned = True

                if m_est < m_min:
                    m_est = m_min
                    pruned = True
                elif m_est > m_max:
                    m_est = m_max
                    pruned = True

                if omega_est < omega_min:
                    omega_est = omega_min
                    pruned = True
                elif omega_est > omega_max:
                    omega_est = omega_max
                    pruned = True

                mse = F1([tc_est, m_est, omega_est], data)

                if mse_min is None or mse < mse_min:
                    fitted_parameters = [tc_est, m_est, omega_est]
                    fitted_pruned = pruned
                    mse_min = mse
                    rv.mse_hist.append(mse)
                else:
                    rv.mse_hist.append(mse_min)
        except LinAlgError as e:
            # print("Exception occurred: ", e)
            pass

    if fitted_parameters is not None:
        rv.tc, rv.m, rv.omega = fitted_parameters
        rv.A, rv.B, rv.C1, rv.C2 = F1_get_linear_parameters(fitted_parameters, data)
        rv.C = abs(rv.C1) + abs(rv.C2)
        rv.price_tc = rv.A + rv.B * (0.001) ** rv.m + \
                      rv.C1 * (0.001) ** rv.m * np.cos(rv.omega * np.log(0.001)) + \
                      rv.C2 * (0.001) ** rv.m * np.sin(rv.omega * np.log(0.001))
        rv.price_chg = (rv.price_tc - data.iat[-1]) / data.iat[-1] * 100
        rv.pruned = fitted_pruned
        rv.mse = mse_min
        rv.norm_mse = F1_normalized(fitted_parameters, data) / len(data) * 1000
        rv.success = True
    return rv
