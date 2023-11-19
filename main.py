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

# df=examples.simple_2D(400)
# dfS=df

from sklearn.preprocessing import FunctionTransformer
def differentiate(x):
    out=x[:-1,:]-x[1:,:]
    out=np.concatenate([out[0:1,:],out],axis=0)
    return out

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

n_lags=10
poly_order=5
pre=StandardScaler()
expansion=PolynomialFeatures(poly_order,interaction_only=False)
whiten=PCA(n_components=1-1e7,whiten=True)
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

# plt.plot(np.abs(differentiate(xt)**2).sum(axis=0))
# plt.show()
print(np.abs(differentiate(xt)**2).sum(axis=0))
print(xt.shape)
i_min=np.argmin(np.abs(differentiate(xt)**2).sum(axis=0))
i_min=-1
plt.plot(xt[:,i_min])
plt.show()