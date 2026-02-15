# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.10",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.19.11"
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
    from slow_graph_feature_analysis import examples

    return examples, np, plt, reload


@app.cell
def _(examples, plt):
    df = examples.logistic_map(350)
    dfS = df.drop(["true"], axis=1)
    plt.plot(dfS)
    plt.show()
    return df, dfS


@app.cell
def _(dfS, n_pca_selector, np, plt, reload, sfa):
    from sklearn.preprocessing import FunctionTransformer

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.kernel_approximation import Nystroem, RBFSampler
    from sklearn.preprocessing import SplineTransformer


    reload(n_pca_selector)
    reload(sfa)
    n_lags = 5
    poly_order = 3
    pre = StandardScaler()
    pre2 = StandardScaler()
    # expansion = SplineTransformer(degree=poly_order, n_knots=4)
    # expansion = FunctionTransformer(lambda x: x)
    expansion = Nystroem(n_components=200)
    whiten = sfa.PCA_whiten_enthropy(threshold=0.90)
    # whiten = n_pca_selector.PCAWithSignflipPA(
    #     whiten=True, thresholding="pairwise", plotting=True, alpha=95
    # )

    center = StandardScaler(with_std=False)
    diff = FunctionTransformer(sfa.differentiate)
    lags = FunctionTransformer(sfa.make_lags, kw_args={"n_lags": n_lags})
    laststep = sfa.PCA_whiten_kaiser(singular_threshold=1e-6)  # PCA()

    pipe = Pipeline(
        [
            ("lag", lags),
            ("expension", expansion),
            # ("pre2", pre2),
            ("whiten", whiten),
            ("diff", diff),
            ("centering", center),
            ("last", laststep),
        ]
    )
    pipe.fit(dfS)
    names = [l[0] for l in pipe.steps]
    index_whiten = [l[0] for l in pipe.steps].index("whiten")
    xt = pipe[0 : (index_whiten + 1)].transform(dfS)
    print(xt.shape)
    xt = pipe[-1:].transform(xt)

    ISSF = ((1 / pipe[-1].singular_values_) / (1 / pipe[-1].singular_values_).sum())[::-1]
    AISF = ISSF.cumsum() / ISSF.sum() * 100
    plt.plot(AISF)
    nb = np.sum(AISF < 90)
    plt.title(f"AISF for below 50 : {nb}")
    plt.show()
    return nb, pipe, xt


@app.cell
def _(df, examples, np, pipe, plt, xt):
    i_min = -1
    true = df["true"]
    estimande = xt[:, i_min]
    delta_value = pipe[-1].singular_values_[i_min]
    corr = np.corrcoef(true, estimande)[0, 1]
    plt.plot(examples.rescale(estimande) * np.sign(corr), label="estimande")
    plt.title(f"corr:{corr:.4f}, singular value: {delta_value:.2f}")
    plt.plot(examples.rescale(true), label="true")
    plt.show()
    return


@app.cell
def _(nb, plt, xt):
    plt.plot(xt[:, -nb:])
    plt.show()
    return


if __name__ == "__main__":
    app.run()
