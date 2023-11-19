
import pandas as pd

def make_lags(df,n_lags=3):
    df_=df.copy()
    for lag in range(1, n_lags + 1):
        df_[[f'{c}_lag_{lag:02}' for c in df.columns]] = df.shift(lag)
    return df_.bfill()

