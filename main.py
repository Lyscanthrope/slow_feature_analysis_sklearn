from typing import Any
import numpy as np
from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt
from src import examples
from src import sfa, n_pca_selector
from importlib import reload

reload(sfa)
reload(examples)

# df = examples.logistic_map(300, Phi=1)
# df2 = examples.logistic_map(1600, Phi=2)
# df = pd.concat([df, df2], axis=0, ignore_index=True)
# df = examples.simple_2D(4000)
# df = examples.simple_modulation(400)
# df = examples.firstorder_steps(800)
# df = examples.secondorder_steps(1500)
df = examples.sinus_signal_fft(512**2 * 4)
# df = examples.sweeping_signal_fft(512**2 * 2)
dfS = df.drop(["true", "noise"], axis=1)

# df = examples.sweeping_signal_raw(512**2 * 2)
# dfS = df.drop(["true"], axis=1)
plt.plot(dfS)
plt.show()
# plt.imshow(dfS, aspect="auto")
# plt.show()

# plt.plot(df["noise"])
# plt.show()


from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import StandardScaler, Iden
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.preprocessing import SplineTransformer
from sklearn.kernel_approximation import PolynomialCountSketch, AdditiveChi2Sampler

# faire un objet pour PCA singlar au dessus de 1
reload(n_pca_selector)
reload(sfa)
n_lags = 3
poly_order = 3
pre = StandardScaler()
pre2 = StandardScaler()
# expansion = PolynomialFeatures(poly_order, interaction_only=False)
# expansion = SplineTransformer(degree=poly_order, n_knots=4)
expansion = FunctionTransformer(lambda x: x)
# expansion= PolynomialCountSketch(degree=poly_order,n_components=300)
# expansion = RBFSampler()
# expansion = Nystroem(n_components=200)
# expansion = AdditiveChi2Sampler()
# whiten = PCA(n_components=1 - 1e-14, whiten=True)
# whiten = sfa.PCA_whiten_kaiser(singular_threshold=1e-0)  # 1 - (1e-14)
whiten = sfa.PCA_whiten_enthropy(threshold=0.95)
# whiten = n_pca_selector.PCAWithSignflipPA(
#     whiten=True, thresholding="pairwise", plotting=True, alpha=95
# )
# whiten.fit(dfS)
center = StandardScaler(with_std=False)
diff = FunctionTransformer(sfa.differentiate)
lags = FunctionTransformer(sfa.make_lags, kw_args={"n_lags": n_lags})
laststep = sfa.PCA_whiten_kaiser(singular_threshold=1e-14)  # PCA()
# laststep=n_pca_selector.PCAWithSignflipPA()


pipe = Pipeline(
    [
        ("lag", lags),
        # ("pre", pre),
        ("expension", expansion),
        ("pre2", pre2),
        ("whiten", whiten),
        # ("pre2", pre2),
        ("diff", diff),
        ("centering", center),
        ("last", laststep),
    ]
)
pipe.fit(dfS)
plt.plot(pipe["whiten"].singular_values_)
plt.show()


names = [l[0] for l in pipe.steps]
index_whiten = [l[0] for l in pipe.steps].index("whiten")
xt = pipe[0 : (index_whiten + 1)].transform(dfS)
print(xt.shape)
xt = pipe[-1:].transform(xt)

# pipe[0:4].fit(dfS)
# plt.show()
# len(pipe[-1].singular_values_)
# dir(pipe[-1])


ISSF = ((1 / pipe[-1].singular_values_) / (1 / pipe[-1].singular_values_).sum())[::-1]
AISF = ISSF.cumsum() / ISSF.sum() * 100
plt.plot(AISF)
nb = np.sum(AISF < 90)
plt.title(f"AISF for below 50 : {nb}")
plt.show()

# Idée : selection d'hyperparametre pour minimiser le nombre de variable qui ont une AISF>90%

# i_min = -1
# true = df["noise"]
# estimande = xt[:, i_min]
# delta_value = pipe[-1].singular_values_[i_min]
# corr = np.corrcoef(true, estimande)[0, 1]
# plt.plot(examples.rescale(estimande) * np.sign(corr), label="estimande")
# plt.title(f"corr:{corr:.4f}, singular value: {delta_value:.2f}")
# plt.plot(examples.rescale(true), label="true")
# plt.show()

i_min = -1
true = df["noise"]
estimande = xt[:, i_min]
delta_value = pipe[-1].singular_values_[i_min]
corr = np.corrcoef(true, estimande)[0, 1]
plt.plot(examples.rescale(estimande) * np.sign(corr), label="estimande")
plt.title(f"corr:{corr:.4f}, singular value: {delta_value:.2f}")
plt.plot(examples.rescale(true), label="true")
plt.show()

i_min = -2
true = df["true"]
estimande = xt[:, i_min]
delta_value = pipe[-1].singular_values_[i_min]
corr = np.corrcoef(true, estimande)[0, 1]
plt.plot(examples.rescale(estimande) * np.sign(corr), label="estimande")
plt.title(f"corr:{corr:.4f}, singular value: {delta_value:.2f}")
plt.plot(examples.rescale(true), label="true")
plt.show()


plt.plot(xt[:, -nb:])
plt.show()
