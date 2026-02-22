import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils import shuffle


class SFA(BaseEstimator, TransformerMixin):
    """
    Linear Slow Feature Analysis (properly whitened)

    Guarantees on training data:
        W^T C_xx W = I
    """

    def __init__(self, n_components=None, eps=1e-12):
        self.n_components = n_components
        self.eps = eps

    # --------------------------------------------------
    # FIT
    # --------------------------------------------------
    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape

        if n_samples < 2:
            raise ValueError("SFA requires at least 2 samples.")

        self.n_features_in_ = n_features

        # Remove constant features
        variances = np.var(X, axis=0)
        self.non_constant_ = variances > self.eps
        self.constant_ = ~self.non_constant_

        X = X[:, self.non_constant_]

        # Center
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # Covariance
        C_xx = (Xc.T @ Xc) / (n_samples - 1)

        # Eigen-decomposition
        eigvals, U = np.linalg.eigh(C_xx)

        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        U = U[:, idx]

        valid = eigvals > self.eps
        eigvals = eigvals[valid]
        U = U[:, valid]

        # Whitening matrix
        Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
        self.whitening_ = U @ Lambda_inv_sqrt

        Z = Xc @ self.whitening_

        # Derivative covariance
        Z_dot = np.diff(Z, axis=0)
        C_zdot = (Z_dot.T @ Z_dot) / (n_samples - 2)

        # Solve eigenproblem
        eigenvalues, P = np.linalg.eigh(C_zdot)

        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        P = P[:, idx]

        if self.n_components is not None:
            P = P[:, :self.n_components]
            eigenvalues = eigenvalues[:self.n_components]

        # Final projection
        self.components_ = self.whitening_ @ P
        self.eigenvalues_ = eigenvalues

        # Pseudo-inverse for reconstruction
        self.pinv_ = np.linalg.pinv(self.components_)

        return self

    # --------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------
    def transform(self, X):
        check_is_fitted(self)

        X = check_array(X)
        Xf = X[:, self.non_constant_]
        Xc = Xf - self.mean_

        return Xc @ self.components_

    # --------------------------------------------------
    # INVERSE TRANSFORM (Reconstruction)
    # --------------------------------------------------
    def inverse_transform(self, Y):
        check_is_fitted(self)

        Xc_recon = Y @ self.pinv_

        Xf_recon = Xc_recon + self.mean_

        X_full = np.zeros((Y.shape[0], self.n_features_in_))
        X_full[:, self.non_constant_] = Xf_recon

        return X_full

    # --------------------------------------------------
    # RESIDUALS
    # --------------------------------------------------
    def get_residuals(self, X):
        X = check_array(X)

        Y = self.transform(X)
        X_recon = self.inverse_transform(Y)

        return X - X_recon
    
def block_shuffle_1d(x, block_length, rng):
    """
    Block shuffle a 1D array.
    Non-overlapping blocks of fixed length.
    """
    n = len(x)

    if block_length <= 1:
        return rng.permutation(x)

    n_blocks = n // block_length
    remainder = n % block_length

    # Trim to full blocks
    trimmed = x[:n_blocks * block_length]
    blocks = trimmed.reshape(n_blocks, block_length)

    # Shuffle blocks
    rng.shuffle(blocks, axis=0)

    shuffled = blocks.reshape(-1)

    # Append remainder unchanged (or shuffle separately if desired)
    if remainder > 0:
        shuffled = np.concatenate([shuffled, x[-remainder:]])

    return shuffled
class SFASelector(BaseEstimator):
    """
    Temporal Parallel Analysis using block shuffling.
    """

    def __init__(self,
                 estimator,
                 n_permutations=20,
                 percentile=5,
                 block_length=10,
                 random_state=None):

        self.estimator = estimator
        self.n_permutations = n_permutations
        self.percentile = percentile
        self.block_length = block_length
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)

        rng = np.random.default_rng(self.random_state)

        # --------------------------------------------------
        # 1️⃣ Fit full model first
        # --------------------------------------------------
        full_estimator = clone(self.estimator)
        full_estimator.fit(X)

        real_eigs = full_estimator.eigenvalues_
        n_comps = len(real_eigs)

        null_eigs = np.zeros((self.n_permutations, n_comps))

        print(f"Running Block Temporal PA "
            f"({self.n_permutations} perms, L={self.block_length})")

        # --------------------------------------------------
        # 2️⃣ Null distribution via block shuffle
        # --------------------------------------------------
        for i in range(self.n_permutations):

            X_perm = np.zeros_like(X)

            for j in range(X.shape[1]):
                X_perm[:, j] = block_shuffle_1d(
                    X[:, j],
                    self.block_length,
                    rng
                )

            temp = clone(self.estimator)
            temp.fit(X_perm)

            eigs = temp.eigenvalues_
            m = min(len(eigs), n_comps)
            null_eigs[i, :m] = eigs[:m]

        thresholds = np.percentile(
            null_eigs,
            self.percentile,
            axis=0
        )

        self.real_eigenvalues_ = real_eigs
        self.thresholds_ = thresholds

        self.significant_indices_ = np.where(
            real_eigs < thresholds
        )[0]

        self.n_selected_ = len(self.significant_indices_)

        # --------------------------------------------------
        # 3️⃣ Retrain estimator using only selected components
        # --------------------------------------------------
        if self.n_selected_ == 0:
            raise ValueError("No significant slow components found.")

        self.estimator_ = full_estimator

        # Keep only first k components
        if self.n_selected_ == 0:
            raise ValueError("No significant slow components found.")

        k = self.n_selected_

        self.estimator_.components_ = self.estimator_.components_[:, :k]
        self.estimator_.eigenvalues_ = self.estimator_.eigenvalues_[:k]
        self.estimator_.pinv_ = np.linalg.pinv(self.estimator_.components_)
        

        # Plot after retraining
        self._plot()

        return self

    def _plot(self):
        plt.figure()
        plt.plot(self.real_eigenvalues_, marker="o")
        plt.plot(self.thresholds_, marker="x")

        if self.n_selected_ > 0:
            idx = self.significant_indices_
            plt.scatter(idx,
                        self.real_eigenvalues_[idx],
                        s=100)

        plt.xlabel("Component Index")
        plt.ylabel("Eigenvalue (Slowness)")
        plt.title("Block Temporal Parallel Analysis")
        plt.legend(["Real", "Threshold", "Selected"])
        plt.show()

    def transform(self, X):
        check_is_fitted(self)

        Y_full = self.estimator_.transform(X)
        return Y_full

    def inverse_transform(self, X):
        check_is_fitted(self)

        Y = self.estimator_.inverse_transform(X)
        return Y
      
    def get_residuals(self, X):
        X = check_array(X)

        Y = self.transform(X)
        X_recon = self.inverse_transform(Y)

        return X - X_recon