import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import odeint


def logistic_map(series_length=150, Phi=1):
    S = np.zeros((series_length, 1), "d")
    D = np.zeros((series_length, 1), "d")

    S[0] = 0.3
    for t in range(1, series_length):
        D[t] = np.sin(np.pi / 75.0 * (t)) - t / (150.0 * Phi)
        S[t] = (3.7 + 0.35 * D[t]) * S[t - 1] * (1 - S[t - 1])
    return pd.DataFrame(np.concatenate([S, D], axis=1), columns=["S", "true"])


def simple_2D(series_length=300):
    t = np.linspace(0, 2 * np.pi, series_length)
    x1 = np.sin(t) + np.cos(11 * t) ** 2 + np.random.randn(series_length) / 10
    x2 = np.cos(11 * t) + np.random.randn(series_length) / 10
    true = np.sin(t)
    return pd.DataFrame([x1, x2, true], index=["x1", "x2", "true"]).T


def simple_modulation(series_length=300):
    t = np.linspace(0, 3 * 2 * np.pi, series_length)
    true = np.cos(t) * t
    x1 = true * (np.cos(5 * t)) + np.random.randn(series_length) / 10

    return pd.DataFrame([x1, true], index=["x1", "true"]).T


def firstorder_steps(series_length=600):
    t = np.linspace(0, 20, series_length)
    Kp = 2.0
    taup = 0.10
    utot = ((t % 2.5 < 1)).reshape(series_length, 1)

    def model3(y, t):
        # u = 1
        u = t % 2.5 < 1
        # print(u)
        taup_ = taup
        if t > 10:
            taup_ *= 4
        return (-y + Kp * u) / taup_

    y = odeint(model3, 0, t)
    slow_varying = (t > 10).reshape(-1, 1)
    return pd.DataFrame(
        np.concatenate([y, utot, slow_varying], axis=1), columns=["x1", "u", "true"]
    )


def secondorder_steps(series_length=600):
    t = np.linspace(0, 80, series_length)
    Kp = 2.0
    Kp = 2.0  # gain
    tau = 0.3  # time constant
    zeta = 0.25  # damping factor
    theta = 0.0  # no time delay
    du = 1.0  # change in u
    utot = ((t % 10 < 5)).reshape(series_length, 1)
    # slow_varying = np.log(t) + 0.01
    limit = 25
    slow_varying = np.array([1 if _t < limit else (_t / limit) ** 3 for _t in t])

    def model3(x, t):
        y = x[0]
        u = t % 10 < 5
        tau_ = tau * (1 if t < limit else (t / limit) ** 3)
        # if t > 10:
        # tau_ *= 4
        dydt = x[1]
        dy2dt2 = (-2.0 * zeta * tau_ * dydt - y + Kp * u) / tau**3
        return [dydt, dy2dt2]

    x3 = odeint(model3, [0, 0], t)
    y = x3[:, 0].reshape((-1, 1)) + np.random.randn(series_length).reshape((-1, 1)) / 10
    slow_varying = slow_varying.reshape((-1, 1))
    return pd.DataFrame(
        np.concatenate([y, slow_varying], axis=1), columns=["x1", "true"]
    )


def rescale(x):
    return (x - x.mean()) / x.std()
