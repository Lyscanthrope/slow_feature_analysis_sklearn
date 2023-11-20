from typing import Any
import numpy as np
from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt
from sklearn._typing import MatrixLike
from src import examples
from src import sfa
from importlib import reload

reload(sfa)
reload(examples)

df = examples.logistic_map(300, Phi=1)
df2 = examples.logistic_map(600, Phi=2)
df = pd.concat([df, df2], axis=0, ignore_index=True)
# df = examples.simple_2D(400)
# df = examples.simple_modulation(400)
# df = examples.firstorder_steps(800)
# df = examples.secondorder_steps(2000)
plt.plot(df)
plt.show()


dfS = df.drop(["true"], axis=1)


from sklearn.preprocessing import FunctionTransformer


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.preprocessing import SplineTransformer

# faire un objet pour PCA singlar au dessus de 1

reload(sfa)
n_lags = 10
poly_order = 3
pre = StandardScaler()
pre2 = StandardScaler()
# expansion = PolynomialFeatures(poly_order, interaction_only=False)
expansion = SplineTransformer()
# expansion = RBFSampler()
# expansion = Nystroem(n_components=200)
# whiten = PCA(n_components=1 - 1e-14, whiten=True)
whiten = sfa.PCA_whiten_kaiser(singular_threshold=1e-15)  # 1 - (1e-14)
# whiten.fit(dfS)
center = StandardScaler(with_std=False)
diff = FunctionTransformer(sfa.differentiate)
lags = FunctionTransformer(sfa.make_lags, kw_args={"n_lags": n_lags})
laststep = PCA()


pipe = Pipeline(
    [
        ("lag", lags),
        # ("pre", pre),
        ("expension", expansion),
        ("pre2", pre2),
        ("whiten", whiten),
        ("diff", diff),
        # ("centering", center),
        ("last", laststep),
    ]
)
pipe.fit(dfS)

xt = pipe[0:4].transform(dfS)
xt = pipe[-1:].transform(xt)

# pipe[0:5].fit(dfS)
# len(pipe[-1].singular_values_)
# dir(pipe[-1])


print(pipe["whiten"].singular_values_)
print(xt.shape)
i_min = -1
true = df["true"]
estimande = xt[:, i_min]
delta_value = pipe[-1].singular_values_[i_min]
corr = np.corrcoef(true, estimande)[0, 1]
plt.plot(examples.rescale(estimande) * np.sign(corr), label="estimande")
plt.title(f"corr:{corr:.4f}, singular value: {delta_value:.2f}")
# plt.show()
plt.plot(examples.rescale(true), label="true")
# plt.plot(examples.rescale(df["x1"]), label="signal")
plt.show()
