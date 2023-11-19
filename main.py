import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import examples
from src import sfa
from importlib import reload

reload(sfa)
reload(examples)

df=examples.logistic_map(400)
# df.plot()
# plt.show()
dfS=pd.DataFrame(df["S"])

df=examples.simple_2D(400)
dfS=df

from sklearn.preprocessing import FunctionTransformer
def differentiate(x):
    out=x[:-1,:]-x[1:,:]
    out=np.concatenate([out[0:1,:],out],axis=0)
    return out

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem

#faire un objet pour PCA singlar au dessus de 1

n_lags=2
# poly_order=3
pre=StandardScaler()
# expansion=PolynomialFeatures(poly_order,interaction_only=False)
expansion=Nystroem()
whiten=PCA(n_components=1-(1e-16),whiten=True)#
center=StandardScaler(with_std=False)
diff=FunctionTransformer(differentiate)
lags=FunctionTransformer(sfa.make_lags,kw_args={"n_lags":n_lags})
laststep=PCA()
pipe=Pipeline([
    ("lag",lags),
    ("pre",pre),
    ("expension",expansion),
    ("whiten",whiten),
    ("diff",diff),
    # ("centering",center),
    ("last",laststep)
])
pipe.fit(dfS)

xt=pipe[0:4].transform(dfS)
xt=pipe[-1:].transform(xt)

# plt.plot(np.log(np.abs(differentiate(xt)**2).sum(axis=0)))
# plt.show()
print(np.abs(differentiate(xt)**2).sum(axis=0))
print(xt.shape)
i_min=np.argmin(np.abs(differentiate(xt)**2).sum(axis=0))
i_min=-1
plt.plot(xt[:,i_min])
plt.show()

# plt.plot(pipe["whiten"].singular_values_)
# plt.show()
# pipe["whiten"].explained_variance_ratio_.cumsum()[69]
# plt.show()