from typing import Any
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import oas
import pandas as pd


def elbow_curve_automatic(elbow_curve):
    """take a curve with an elbow (decreasing concave curve)

    Args:
        elbow_curve (np.array): the curve

    Returns:
        int: number of elements at which the elbow happen
    """
    # inspired from : https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    import numpy.matlib as npm

    # recreating the x axis
    n_points = len(elbow_curve)
    all_coord = np.vstack((range(n_points), elbow_curve)).T
    first_point = all_coord[0]
    # compute the vector from first to last point
    line_vector = all_coord[-1] - all_coord[0]
    # normalize
    line_vector_norm = line_vector / np.sqrt(np.sum(line_vector**2))
    diff_from_first = all_coord - first_point
    scalar_product = np.sum(
        diff_from_first * npm.repmat(line_vector_norm, n_points, 1), axis=1
    )
    scalar_product_of_points = np.outer(scalar_product, line_vector_norm)
    vector_to_line = diff_from_first - scalar_product_of_points
    dist_to_line = np.sqrt(np.sum(vector_to_line**2, axis=1))
    idx_of_further = np.argmax(dist_to_line)
    n_components = idx_of_further
    return n_components + 1  # we add one as it counts from zero


class PCAWithElbow(PCA):
    def fit(self, X, y=None):
        super().fit(X)
        n_components = elbow_curve_automatic(self.explained_variance_ratio_)
        super().__init__(n_components=n_components)
        super().fit(X)


# This is the code that could be dropped into any project
# be carefull of imports
def compute_best_k(
    sigma: np.array,
    sigma_PA: np.array,
    thresholding: float,
    alpha: float,
    plotting: bool,
) -> int:
    """compute the best number of compenents of based for PCA

    Args:
        sigma (np.array): 1D array of singular values of the data
        sigma_PA (np.array): 2D singular values array for the T signfliped matrix
        thresholding (float): method of thresholding ("upper-edge" or "pariwise")
        alpha (float): percentile for cutoff (]0-100[)
        plotting (bool): Displaying a matplotlib figure

    Returns:
        int: number of components
    """
    percentile = np.percentile(sigma_PA, alpha, axis=0)
    tol = 1e-8
    k_optimal_upper_edge = np.sum((sigma > percentile[0]) * 1)
    k_optimal_pairwise = (
        np.where((sigma - percentile > tol) == True)[0][-1] + 1
    )  # np.where((sigma > percentile) == False)[0][0]# taking the lastest above the percentile

    if thresholding == "upper-edge":
        k_optimal = k_optimal_upper_edge
    elif thresholding == "pairwise":
        k_optimal = k_optimal_pairwise
    else:
        Exception("Thresholding method unknown")
    if plotting:
        fig = plt.figure()
        plt.plot(sigma, label="Singular value of PCA")
        plt.plot(percentile, label="Singular value of PA")
        plt.scatter(
            k_optimal_upper_edge - 1,
            sigma[k_optimal_upper_edge - 1],
            label=f"K optimal/Upper-edge: {k_optimal_upper_edge}",
        )
        plt.scatter(
            k_optimal_pairwise - 1,
            sigma[k_optimal_pairwise - 1],
            label=f"K_optimal/pairwise: {k_optimal_pairwise}",
        )
        plt.legend()
    return k_optimal


def Signflip_PA_singular(
    singular_values: np.array,
    X: np.array,
    T: int = 20,
    thresholding: str = "upper-edge",
    alpha: float = 95,
    plotting: bool = True,
) -> int:
    """perform signflip parrallel analysis

    Args:
        singular_values (np.array): singular values of the PCA
        X (np.array): data
        T (int, optional): number of repetitions. Defaults to 20.
        thresholding (str, optional): type of thresholding"upper-edge" is recommended. Defaults to "upper-edge".
        alpha (float, optional): Percentile. Defaults to 95.
        plotting (bool, optional): shall it makes a figure. Defaults to True.

    Returns:
        int: _description_
    """
    # based on signflip paper (Buja & Eyuboglu 1992 version)
    # using PCA to get SVD singular values for consistancy
    sigma = singular_values
    list_sigma_PA = []
    for t in range(T):
        R = np.random.choice([-1, 1], X.shape)
        X_ = np.multiply(R, X)
        pca = PCA()
        pca.fit(X_)
        list_sigma_PA.append(pca.singular_values_)
    sigma_PA = np.stack(list_sigma_PA)

    k_optimal = compute_best_k(sigma, sigma_PA, thresholding, alpha, plotting)
    return k_optimal


def Signflip_PA_singular_for_SVD(
    singular_values: np.array,
    X: np.array,
    T: int = 20,
    thresholding: str = "upper-edge",
    alpha: float = 95,
    plotting: bool = True,
) -> int:
    """perform signflip parrallel analysis

    Args:
        singular_values (np.array): singular values of the SVD
        X (np.array): data
        T (int, optional): number of repetitions. Defaults to 20.
        thresholding (str, optional): type of thresholding"upper-edge" is recommended. Defaults to "upper-edge".
        alpha (float, optional): Percentile. Defaults to 95.
        plotting (bool, optional): shall it makes a figure. Defaults to True.

    Returns:
        int: _description_
    """
    # based on signflip paper (Buja & Eyuboglu 1992 version)
    # using PCA to get SVD singular values for consistancy
    sigma = singular_values
    list_sigma_PA = []
    for t in range(T):
        R = np.random.choice([-1, 1], X.shape)
        X_ = np.multiply(R, X)
        U, s, Vh = np.linalg.svd(X_, full_matrices=False)
        list_sigma_PA.append(s)
    sigma_PA = np.stack(list_sigma_PA)

    k_optimal = compute_best_k(sigma, sigma_PA, thresholding, alpha, plotting)
    return k_optimal


class PCAWithSignflipPA(PCA):
    def __init__(
        self, T=20, thresholding="upper-edge", alpha=95, plotting=False, **kwargs
    ):
        self.T = T
        self.thresholding = thresholding
        self.alpha = alpha
        self.plotting = plotting
        self.kwargs = kwargs
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        super().fit(X)
        n_components = Signflip_PA_singular(
            self.singular_values_,
            X,
            T=self.T,
            thresholding=self.thresholding,
            alpha=self.alpha,
            plotting=self.plotting,
        )
        print(n_components)
        self.__init__(n_components=n_components + 1, **self.kwargs)
        super().fit(X)
        return self

    def fit_transform(self, X, y: Any = None):
        self.fit(X)
        return self.transform(X)
