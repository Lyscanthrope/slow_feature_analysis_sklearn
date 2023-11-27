import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import odeint

from scipy import signal
from scipy.fft import fftshift


def logistic_map(series_length=150, Phi=1):
    """logisitic map examples from litterature

    Args:
        series_length (int, optional): _description_. Defaults to 150.
        Phi (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    S = np.zeros((series_length, 1), "d")
    D = np.zeros((series_length, 1), "d")

    S[0] = 0.3
    for t in range(1, series_length):
        D[t] = np.sin(np.pi / 75.0 * (t)) - t / (150.0 * Phi)
        S[t] = (3.7 + 0.35 * D[t]) * S[t - 1] * (1 - S[t - 1])
    return pd.DataFrame(np.concatenate([S, D], axis=1), columns=["S", "true"])


def simple_2D(series_length=300):
    """simple 2D example from the litterature from the scholarpedia of slow feature analysis

    Args:
        series_length (int, optional): _description_. Defaults to 300.

    Returns:
        _type_: _description_
    """
    t = np.linspace(0, 2 * np.pi, series_length)
    x1 = np.sin(t) + np.cos(11 * t) ** 2 + np.random.randn(series_length) / 20
    x2 = np.cos(11 * t) + np.random.randn(series_length) / 20
    true = np.sin(t)
    return pd.DataFrame([x1, x2, true], index=["x1", "x2", "true"]).T


def simple_modulation(series_length=300):
    """modulation example (difficulte one)

    Args:
        series_length (int, optional): _description_. Defaults to 300.

    Returns:
        _type_: _description_
    """
    t = np.linspace(0, 3 * 2 * np.pi, series_length)
    true = np.cos(t) * t
    x1 = true * (np.cos(15 * t)) + np.random.randn(series_length) / 10

    return pd.DataFrame([x1, true], index=["x1", "true"]).T


def firstorder_steps(series_length=600):
    t = np.linspace(0, 20, series_length)
    Kp = 2.0
    taup = 0.010
    utot = ((t % 1 < 0.5)).reshape(series_length, 1)

    def model3(y, t):
        # u = 1
        u = t % 1 < 0.5
        # print(u)
        taup_ = taup
        if t > 10:
            taup_ *= 4
        return (-y + Kp * u) / taup_

    y = odeint(model3, 0, t)
    slow_varying = (t > 10).reshape(-1, 1)
    return pd.DataFrame(
        # np.concatenate([y, utot, slow_varying], axis=1), columns=["x1", "u", "true"]
        np.concatenate([y, slow_varying], axis=1),
        columns=["x1", "true"],
    )


def secondorder_steps(series_length=600):
    t = np.linspace(0, 140, series_length)
    Kp = 2.0
    Kp = 2.0  # gain
    tau = 0.3  # time constant
    zeta = 0.25  # damping factor
    theta = 0.0  # no time delay
    du = 1.0  # change in u
    utot = ((t % 10 < 5)).reshape(series_length, 1)
    # slow_varying = np.log(t) + 0.01
    limit1 = 20
    limit2 = 80
    limit3 = 110

    def function(_t):
        if _t < limit1:
            out = 1
        elif _t > limit3:
            out = 0.3
        elif _t > limit2:
            out = limit2 / limit1
        else:
            out = _t / limit1
        return out

    slow_varying = np.array([function(_t) for _t in t])

    def model3(x, t):
        y = x[0]
        u = t % 10 < 5
        tau_ = tau / (function(t))
        # if t > 10:
        # tau_ *= 4
        dydt = x[1]
        dy2dt2 = (-2.0 * zeta * tau_ * dydt - y + Kp * u) / tau**3
        return [dydt, dy2dt2]

    x3 = odeint(model3, [0, 0], t)
    y = x3[:, 0].reshape((-1, 1)) + np.random.randn(series_length).reshape((-1, 1)) / 10
    y2 = (
        x3[:, 1].reshape((-1, 1)) + np.random.randn(series_length).reshape((-1, 1)) / 10
    )
    slow_varying = slow_varying.reshape((-1, 1))
    return pd.DataFrame(
        np.concatenate([y, slow_varying], axis=1), columns=["x1", "true"]
    )


def rescale(x):
    return (x - x.mean()) / x.std()


def sinus_signal_fft(series_length=1e5, nperseg=256):
    rng = np.random.default_rng()
    fs = 10e3
    N = series_length
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2
    time = np.arange(N) / float(fs)
    mod = 1500 * np.cos(2 * np.pi * 0.10 * time)
    carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
    noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    noise *= np.exp(-time / 10)
    x = carrier + noise
    f, t, Sxx = signal.spectrogram(x, fs, nperseg=nperseg)
    df = pd.DataFrame(Sxx.T, columns=[f"f_{c:02f}" for c in f])
    print(mod.shape)
    df["true"] = mod[:: int(N) // len(t)][1:]
    df["noise"] = np.exp(-time / 10)[:: int(N) // len(t)][1:]
    return df


def two_sinus_signal_fft(series_length=1e5):
    rng = np.random.default_rng()
    fs = 10e3
    N = series_length
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2
    time = np.arange(N) / float(fs)
    # mod1 = 200 * np.cos(2 * np.pi * 0.2 * time) + 2000 * time
    mod1 = time / 400
    mod2 = 1000 * np.cos(2 * np.pi * 0.33 * time)
    carrier1 = amp * np.sin(2 * np.pi * 1e4 * time * mod1)
    carrier2 = amp * np.sin(2 * np.pi * 1e3 * time + mod2)
    noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    noise *= np.exp(-time / 5)
    x = carrier1 + carrier2 + noise
    f, t, Sxx = signal.spectrogram(x, fs, nperseg=512)
    df = pd.DataFrame(Sxx.T, columns=[f"f_{c:02f}" for c in f])
    df["true"] = mod1[:: int(N) // len(t)][1:]
    df["true2"] = mod2[:: int(N) // len(t)][1:]
    df["noise"] = np.exp(-time / 5)[:: int(N) // len(t)][1:]
    plt.plot(x)
    plt.show()
    return df


def sweeping_signal_fft(series_length=1e5):
    rng = np.random.default_rng()
    fs = 10e3
    N = series_length
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2
    time = np.arange(N) / float(fs)
    mod1 = np.clip(time / 400, 0, 0.1)
    carrier1 = amp * np.sin(2 * np.pi * 1e4 * time * mod1)
    noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape) * 0.01
    x = carrier1 + noise
    f, t, Sxx = signal.spectrogram(x, fs, nperseg=512)
    df = pd.DataFrame(Sxx.T, columns=[f"f_{c:02f}" for c in f])
    df["true"] = mod1[:: int(N) // len(t)][1:]
    plt.plot(x)
    plt.show()
    return df
