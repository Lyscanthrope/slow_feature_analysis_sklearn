import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def logistic_map(series_length = 300):
    S = np.zeros((series_length, 1), 'd')
    D = np.zeros((series_length, 1), 'd')

    S[0] = 0.3
    for t in range(1, series_length):
        D[t] = np.sin(np.pi/75. * t) - t/150.
        S[t] = (3.7+0.35*D[t]) * S[t-1] * (1 - S[t-1])
    return pd.DataFrame(np.concatenate([S,D],axis=1),columns=["S","D"])

def simple_2D(series_length=300):
    t=np.linspace(0,2*np.pi,series_length)
    x1=np.sin(t)+np.cos(11*t)*2
    x2=np.cos(11*t)
    return pd.DataFrame([x1,x2],index=["x1","x2"]).T