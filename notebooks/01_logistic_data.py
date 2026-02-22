# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.10",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.20.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Illustration of Slow Feature Analysis: logistic map litterature example
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pathlib
    from importlib import reload
    from slow_feature_analysis import examples,sfa

    return examples, np, plt, reload, sfa


@app.cell
def _(examples, plt):
    df = examples.logistic_map(400)
    dfS = df.drop(["true"], axis=1)
    plt.plot(dfS)
    plt.show()
    return df, dfS


@app.cell
def _(dfS, examples, reload, sfa):
    from sklearn.preprocessing import FunctionTransformer

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.kernel_approximation import Nystroem, RBFSampler
    from sklearn.preprocessing import SplineTransformer

    reload(sfa)
    n_lags=3
    poly_order = 5
    expansion = Nystroem(n_components=50)
    # expansion=PolynomialFeatures(degree=poly_order)


    lags = FunctionTransformer(examples.make_lags, kw_args={"n_lags": n_lags})
    # laststep = sfa.SFA(n_components=3)

    laststep = sfa.SFASelector(sfa.SFA(), n_permutations=50, percentile=1,block_length=20)
    pipe = Pipeline(
        [
            ("lag", lags),
            ("expension", expansion),
            ("last", laststep),
        ]
    )
    pipe.fit(dfS)

    trS=pipe[:-1].transform(dfS)
    return (pipe,)


@app.cell
def _(pipe):
    pipe[-1].n_selected_
    return


@app.cell
def _(dfS, pipe):
    xt=pipe.transform(dfS)
    return (xt,)


@app.cell
def _(df, examples, np, plt, xt):
    i_min = 0
    true = df["true"]
    estimande = xt[:, i_min]
    # delta_value = pipe[-1].singular_values_[i_min]
    corr = np.corrcoef(true, estimande)[0, 1]
    plt.plot(examples.rescale(estimande) * np.sign(corr), label="estimande")
    # plt.title(f"corr:{corr:.4f}, singular value: {delta_value:.2f}")
    plt.plot(examples.rescale(true), label="true")
    plt.show()
    return


@app.cell
def _(plt, xt):
    plt.plot(xt)
    return


@app.cell
def _(dfS, pipe, plt):
    plt.plot(pipe[-1].get_residuals(pipe[:-1].transform(dfS)))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
