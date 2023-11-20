import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, SparsePCA


# def make_lags(df, n_lags=3):
#     df_ = df.copy()
#     for lag in range(1, n_lags + 1):
#         df_[[f"{c}_lag_{lag:02}" for c in df.columns]] = df.shift(lag)
#     return df_.bfill()


def make_lags(x_df, n_lags=5):
    x_lag_past = []
    for i in range(n_lags):
        new_x = x_df.shift(i)
        new_x.columns = [f"{c}___{i}" for c in x_df.columns]
        x_lag_past.append(new_x)
    Xp = pd.concat(x_lag_past, axis=1).interpolate(limit_direction="backward")
    return Xp


def differentiate(x):
    out = x[:-1, :] - x[1:, :]
    out = np.concatenate([out[0:1, :], out], axis=0)
    return out


class PCA_whiten_kaiser(PCA):
    def __init__(self, singular_threshold=1, **kwargs):
        self.singular_threshold = singular_threshold
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        super().fit(X)
        print(self.singular_values_)
        self.n_kaiser = np.sum((self.singular_values_ > self.singular_threshold) * 1)
        print(self.n_kaiser)
        self.__init__(n_components=self.n_kaiser, whiten=True)
        super().fit(X)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
