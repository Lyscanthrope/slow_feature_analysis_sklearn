"""
Graph-Regularized Slow Feature Analysis (Graph-SFA)

A scikit-learn compatible transformer that extends Slow Feature Analysis
with graph-based regularization using the graph Laplacian.
"""

import numpy as np
from scipy import linalg
from scipy.sparse import issparse, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array


class GraphSlowFeatureAnalysis(BaseEstimator, TransformerMixin):
    """
    Graph-Regularized Slow Feature Analysis.
    
    Finds slowly-varying features in time series data while incorporating
    graph structure knowledge about input variables. The graph Laplacian
    regularizes the solution to encourage connected variables to have
    similar weights.
    
    Parameters
    ----------
    n_components : int, default=None
        Number of slow features to extract. If None, extracts all features.
        
    graph_laplacian : array-like of shape (n_features, n_features), default=None
        Graph Laplacian matrix encoding relationships between input variables.
        If None, standard SFA without graph regularization is performed.
        Can be dense or sparse matrix.
        
    lambda_graph : float, default=0.0
        Regularization strength for graph Laplacian term. Higher values
        enforce stronger adherence to graph structure.
        - lambda_graph = 0: Standard SFA (no graph constraint)
        - lambda_graph > 0: Increasingly graph-structured features
        
    laplacian_type : {'combinatorial', 'normalized', 'random_walk'}, default='combinatorial'
        Type of Laplacian normalization to use if raw adjacency/weight matrix
        is provided instead of Laplacian. This parameter only matters if you
        provide an adjacency/weight matrix rather than a pre-computed Laplacian.
        
        Different Laplacian types have different properties:
        
        - 'combinatorial': L = D - W
          * Most basic form, where D is the degree matrix
          * Sensitive to node degree; high-degree nodes have more influence
          * Eigenvalues in range [0, 2*max_degree]
          * Best for: Unweighted graphs or when all nodes should have equal importance
          
        - 'normalized': L_norm = I - D^(-1/2) W D^(-1/2)
          * Symmetric normalized Laplacian
          * Reduces influence of high-degree nodes
          * Eigenvalues in range [0, 2]
          * Best for: Graphs with varying node degrees, more robust to degree distribution
          * Most commonly used in spectral clustering and graph signal processing
          
        - 'random_walk': L_rw = I - D^(-1) W
          * Also called the random walk normalized Laplacian
          * Asymmetric (not self-adjoint)
          * Eigenvalues in range [0, 2]
          * Best for: When interpretation as random walk on graph is desired
          * Related to Markov chains and diffusion processes
          
        Note: The normalized Laplacian is generally recommended for most applications
        as it is less sensitive to the graph's degree distribution and has better
        numerical properties.
        
    standardize : bool, default=True
        Whether to standardize input data (zero mean, unit variance) before
        extracting slow features.
        
    whiten : bool, default=True
        Whether to whiten data (decorrelate) before computing slow features.
        This is part of the standard SFA preprocessing.
        
    time_derivative_method : {'finite_diff', 'central_diff'}, default='finite_diff'
        Method for computing temporal derivatives:
        - 'finite_diff': Forward difference (x_{t+1} - x_t)
        - 'central_diff': Central difference ((x_{t+1} - x_{t-1})/2)
        
    regularization : float, default=1e-8
        Small constant added to diagonal for numerical stability in
        matrix inversions.
        
    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The slow feature weight vectors (loading matrix). Each row is a
        slow feature, mapping from original space to slow feature space.
        
    eigenvalues_ : ndarray of shape (n_components,)
        The slowness values (eigenvalues) corresponding to each slow feature.
        Smaller values indicate slower features.
        
    mean_ : ndarray of shape (n_features,)
        Mean of training data, used for centering.
        
    std_ : ndarray of shape (n_features,)
        Standard deviation of training data, used for standardization.
        
    whitening_matrix_ : ndarray of shape (n_features, n_features)
        Whitening transformation matrix (if whiten=True).
        
    n_features_in_ : int
        Number of features seen during fit.
        
    explained_slowness_ratio_ : ndarray of shape (n_components,)
        Proportion of total slowness explained by each component.
        
    Examples
    --------
    >>> import numpy as np
    >>> from graph_sfa import GraphSlowFeatureAnalysis
    >>> 
    >>> # Generate time series data
    >>> t = np.linspace(0, 10, 1000)
    >>> X = np.column_stack([
    ...     np.sin(t),
    ...     np.cos(t),
    ...     np.sin(2*t),
    ...     np.random.randn(1000) * 0.1
    ... ])
    >>> 
    >>> # Define graph structure (variables 0,1 connected; 2,3 connected)
    >>> L = np.array([
    ...     [ 1, -1,  0,  0],
    ...     [-1,  1,  0,  0],
    ...     [ 0,  0,  1, -1],
    ...     [ 0,  0, -1,  1]
    ... ])
    >>> 
    >>> # Fit Graph-SFA
    >>> sfa = GraphSlowFeatureAnalysis(n_components=2, graph_laplacian=L, lambda_graph=0.1)
    >>> Y = sfa.fit_transform(X)
    >>> 
    >>> # Get derivatives of slow features
    >>> Y_dot = sfa.transform_derivatives(X)
    """
    
    def __init__(
        self,
        n_components=None,
        graph_laplacian=None,
        lambda_graph=0.0,
        laplacian_type='combinatorial',
        standardize=True,
        whiten=True,
        time_derivative_method='finite_diff',
        regularization=1e-8
    ):
        self.n_components = n_components
        self.graph_laplacian = graph_laplacian
        self.lambda_graph = lambda_graph
        self.laplacian_type = laplacian_type
        self.standardize = standardize
        self.whiten = whiten
        self.time_derivative_method = time_derivative_method
        self.regularization = regularization
        
    def _compute_temporal_derivative(self, X):
        """Compute temporal derivatives of the data."""
        if self.time_derivative_method == 'finite_diff':
            # Forward difference: x_{t+1} - x_t
            X_dot = np.diff(X, axis=0)
            X = X[:-1]  # Match dimensions
        elif self.time_derivative_method == 'central_diff':
            # Central difference: (x_{t+1} - x_{t-1}) / 2
            X_dot = (X[2:] - X[:-2]) / 2.0
            X = X[1:-1]  # Match dimensions
        else:
            raise ValueError(f"Unknown derivative method: {self.time_derivative_method}")
            
        return X, X_dot
    
    def _compute_laplacian(self, W):
        """
        Compute graph Laplacian from weight/adjacency matrix.
        
        Parameters
        ----------
        W : array-like
            Weight or adjacency matrix
            
        Returns
        -------
        L : ndarray
            Graph Laplacian matrix
        """
        W = np.asarray(W)
        D = np.diag(np.sum(np.abs(W), axis=1))
        
        if self.laplacian_type == 'combinatorial':
            L = D - W
        elif self.laplacian_type == 'normalized':
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + self.regularization))
            L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
        elif self.laplacian_type == 'random_walk':
            D_inv = np.diag(1.0 / (np.diag(D) + self.regularization))
            L = np.eye(W.shape[0]) - D_inv @ W
        else:
            raise ValueError(f"Unknown Laplacian type: {self.laplacian_type}")
            
        return L
    
    def fit(self, X, y=None):
        """
        Fit the Graph-SFA model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training time series data. Samples should be temporally ordered.
            
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        if n_samples < 2:
            raise ValueError("Need at least 2 samples to compute temporal derivatives")
        
        # Store mean and std for standardization
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ < self.regularization] = 1.0  # Avoid division by zero
        
        # Standardize data if requested
        if self.standardize:
            X = (X - self.mean_) / self.std_
        else:
            X = X - self.mean_  # Always center
            
        # Compute temporal derivatives
        X, X_dot = self._compute_temporal_derivative(X)
        n_samples_used = X.shape[0]
        
        # Compute covariance matrices
        C = (X.T @ X) / n_samples_used  # Covariance of data
        C_dot = (X_dot.T @ X_dot) / n_samples_used  # Covariance of derivatives
        
        # Add regularization for numerical stability
        C += self.regularization * np.eye(n_features)
        C_dot += self.regularization * np.eye(n_features)
        
        # Whitening (decorrelation) - part of standard SFA
        if self.whiten:
            # Compute whitening matrix: C^(-1/2)
            eigvals, eigvecs = linalg.eigh(C)
            eigvals = np.maximum(eigvals, self.regularization)
            self.whitening_matrix_ = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Apply whitening to derivative covariance
            C_dot_white = self.whitening_matrix_.T @ C_dot @ self.whitening_matrix_
        else:
            self.whitening_matrix_ = np.eye(n_features)
            C_dot_white = C_dot
            
        # Add graph regularization if provided
        if self.graph_laplacian is not None and self.lambda_graph > 0:
            L = np.asarray(self.graph_laplacian)
            
            if L.shape != (n_features, n_features):
                raise ValueError(
                    f"Graph Laplacian shape {L.shape} does not match "
                    f"n_features {n_features}"
                )
            
            # Check if L is actually a Laplacian or adjacency/weight matrix
            # Heuristic: Laplacian has non-negative diagonal and non-positive off-diagonal
            is_laplacian = (np.diag(L) >= -self.regularization).all()
            
            if not is_laplacian:
                # Assume it's a weight/adjacency matrix, compute Laplacian
                L = self._compute_laplacian(L)
            
            # Apply whitening to Laplacian if whitening is used
            if self.whiten:
                L_white = self.whitening_matrix_.T @ L @ self.whitening_matrix_
            else:
                L_white = L
                
            # Add graph regularization to derivative covariance
            C_dot_white = C_dot_white + self.lambda_graph * L_white
        
        # Solve generalized eigenvalue problem
        # We want to minimize w^T C_dot w subject to w^T C w = 1
        # In whitened space: minimize w^T C_dot_white w subject to w^T w = 1
        # This is just a standard eigenvalue problem in whitened space
        eigenvalues, eigenvectors = linalg.eigh(C_dot_white)
        
        # Sort by eigenvalues (slowest first)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Determine number of components
        if self.n_components is None:
            n_components = n_features
        else:
            n_components = min(self.n_components, n_features)
            
        # Store results
        self.eigenvalues_ = eigenvalues[:n_components]
        
        # Transform back from whitened space to original space
        # Components in original (standardized) space
        self.components_ = (self.whitening_matrix_ @ eigenvectors[:, :n_components]).T
        
        # Normalize components to have unit norm
        norms = np.sqrt(np.sum(self.components_**2, axis=1, keepdims=True))
        self.components_ = self.components_ / (norms + self.regularization)
        
        # Compute explained slowness ratio
        total_slowness = np.sum(eigenvalues)
        if total_slowness > 0:
            self.explained_slowness_ratio_ = eigenvalues[:n_components] / total_slowness
        else:
            self.explained_slowness_ratio_ = np.zeros(n_components)
        
        return self
    
    def transform(self, X):
        """
        Transform data to slow feature space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Slow features.
        """
        check_is_fitted(self, ['components_', 'mean_'])
        X = check_array(X, dtype=np.float64)
        
        # Standardize using training statistics
        if self.standardize:
            X = (X - self.mean_) / self.std_
        else:
            X = X - self.mean_
            
        # Project onto slow feature components
        return X @ self.components_.T
    
    def transform_derivatives(self, X):
        """
        Transform temporal derivatives of data to slow feature derivative space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data whose derivatives to transform. Samples should be temporally ordered.
            
        Returns
        -------
        X_dot_transformed : ndarray of shape (n_samples - k, n_components)
            Derivatives of slow features, where k depends on derivative method
            (k=1 for finite_diff, k=2 for central_diff).
        """
        check_is_fitted(self, ['components_', 'mean_'])
        X = check_array(X, dtype=np.float64)
        
        # Standardize
        if self.standardize:
            X = (X - self.mean_) / self.std_
        else:
            X = X - self.mean_
            
        # Compute derivatives
        X, X_dot = self._compute_temporal_derivative(X)
        
        # Project derivatives onto slow feature components
        return X_dot @ self.components_.T
    
    def inverse_transform(self, Y):
        """
        Transform slow features back to original feature space.
        
        Parameters
        ----------
        Y : array-like of shape (n_samples, n_components)
            Slow features.
            
        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
            Reconstructed data in original feature space.
        """
        check_is_fitted(self, ['components_', 'mean_'])
        Y = check_array(Y, dtype=np.float64)
        
        # Project back to original space
        X_reconstructed = Y @ self.components_
        
        # Reverse standardization
        if self.standardize:
            X_reconstructed = X_reconstructed * self.std_ + self.mean_
        else:
            X_reconstructed = X_reconstructed + self.mean_
            
        return X_reconstructed
    
    def get_loadings(self):
        """
        Get the loading matrix (component weight vectors).
        
        Returns
        -------
        components : ndarray of shape (n_components, n_features)
            The slow feature loading matrix. Each row is a component.
        """
        check_is_fitted(self, ['components_'])
        return self.components_.copy()
    
    def get_feature_importance(self):
        """
        Get absolute feature importance for each slow feature.
        
        Returns
        -------
        importance : ndarray of shape (n_components, n_features)
            Absolute values of loadings, indicating feature importance.
        """
        check_is_fitted(self, ['components_'])
        return np.abs(self.components_)
    
    def get_slowness_values(self):
        """
        Get the slowness values (eigenvalues) for each slow feature.
        
        Returns
        -------
        eigenvalues : ndarray of shape (n_components,)
            Slowness values. Smaller values indicate slower features.
        """
        check_is_fitted(self, ['eigenvalues_'])
        return self.eigenvalues_.copy()
    
    def score(self, X, y=None):
        """
        Return the negative mean slowness of the slow features.
        
        Lower slowness (more negative score) is better.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
            
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        score : float
            Negative mean slowness (higher is better).
        """
        check_is_fitted(self, ['components_'])
        
        # Transform to slow features
        Y = self.transform(X)
        
        # Compute derivatives of slow features
        Y_dot = self.transform_derivatives(X)
        
        # Compute slowness (variance of derivatives)
        slowness = np.mean(np.var(Y_dot, axis=0))
        
        return -slowness


class SlowFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Automatic selection of slow feature components via parallel analysis.
    
    This wrapper class uses temporal parallel analysis with block permutation
    to determine the optimal number of slow feature components. It generates
    null distributions by destroying slow temporal structure through block
    permutation, then selects components whose slowness is significantly
    below the null distribution.
    
    Parameters
    ----------
    estimator : GraphSlowFeatureAnalysis instance
        The SFA estimator to use. Its n_components parameter will be
        automatically determined and overridden.
        
    block_length : int
        Length of blocks for permutation. Should be chosen based on domain
        knowledge about the timescale of slow features you want to preserve.
        Temporal structures longer than block_length will be destroyed;
        structures shorter than block_length will be preserved within blocks.
        
    n_permutations : int, default=100
        Number of permuted null datasets to generate for building the
        null distribution.
        
    percentile_threshold : float, default=95.0
        Percentile of null distribution to use as threshold. Components
        with eigenvalues below this percentile are considered significantly
        slow. Should be between 0 and 100.
        
    permute_variables : {'independently', 'jointly'}, default='independently'
        How to permute multivariate time series:
        - 'independently': Each variable is block-permuted independently,
          destroying all cross-variable slow relationships. More conservative.
        - 'jointly': All variables experience the same block permutation,
          preserving instantaneous cross-variable correlations.
          
    min_components : int, default=1
        Minimum number of components to select, even if parallel analysis
        suggests fewer.
        
    max_components : int or None, default=None
        Maximum number of components to select. If None, uses n_features.
        
    random_state : int, RandomState instance or None, default=None
        Controls randomization for reproducibility.
        
    verbose : bool, default=False
        If True, prints progress during permutation testing.
        
    Attributes
    ----------
    estimator_ : GraphSlowFeatureAnalysis
        The fitted estimator with selected n_components.
        
    n_components_selected_ : int
        Number of components selected by parallel analysis.
        
    actual_eigenvalues_ : ndarray of shape (n_features,)
        Eigenvalues from the actual data (all components).
        
    null_eigenvalues_ : ndarray of shape (n_permutations, n_features)
        Eigenvalues from all permuted null datasets.
        
    threshold_eigenvalue_ : float
        The threshold value (percentile of null distribution) used for
        component selection.
        
    null_mean_ : ndarray of shape (n_features,)
        Mean eigenvalue across permutations for each component.
        
    null_std_ : ndarray of shape (n_features,)
        Standard deviation of eigenvalues across permutations.
        
    selection_mask_ : ndarray of shape (n_features,)
        Boolean mask indicating which components were selected.
        
    p_values_ : ndarray of shape (n_features,)
        Empirical p-values for each component (proportion of permutations
        with eigenvalue <= actual eigenvalue).
        
    Examples
    --------
    >>> import numpy as np
    >>> from graph_sfa import GraphSlowFeatureAnalysis, SlowFeatureSelector
    >>> 
    >>> # Generate time series
    >>> t = np.linspace(0, 10, 1000)
    >>> X = np.column_stack([np.sin(0.5*t), np.cos(0.5*t), 
    ...                       np.sin(5*t), np.random.randn(1000)])
    >>> 
    >>> # Automatic component selection
    >>> base_sfa = GraphSlowFeatureAnalysis(standardize=True)
    >>> selector = SlowFeatureSelector(
    ...     estimator=base_sfa,
    ...     block_length=20,
    ...     n_permutations=50
    ... )
    >>> Y = selector.fit_transform(X)
    >>> 
    >>> print(f"Selected {selector.n_components_selected_} components")
    >>> selector.plot_selection()
    """
    
    def __init__(
        self,
        estimator,
        block_length,
        n_permutations=100,
        percentile_threshold=95.0,
        permute_variables='independently',
        min_components=1,
        max_components=None,
        random_state=None,
        verbose=False
    ):
        self.estimator = estimator
        self.block_length = block_length
        self.n_permutations = n_permutations
        self.percentile_threshold = percentile_threshold
        self.permute_variables = permute_variables
        self.min_components = min_components
        self.max_components = max_components
        self.random_state = random_state
        self.verbose = verbose
        
    def _block_permute(self, X, rng):
        """
        Permute time series in blocks to destroy slow temporal structure.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Time series data to permute
            
        rng : RandomState
            Random number generator
            
        Returns
        -------
        X_permuted : ndarray of shape (n_samples, n_features)
            Block-permuted time series
        """
        n_samples, n_features = X.shape
        n_blocks = n_samples // self.block_length
        remainder = n_samples % self.block_length
        
        X_permuted = np.zeros_like(X)
        
        if self.permute_variables == 'independently':
            # Each variable gets its own random permutation
            for i in range(n_features):
                # Create blocks
                blocks = X[:n_blocks * self.block_length, i].reshape(n_blocks, self.block_length)
                
                # Permute blocks
                perm_idx = rng.permutation(n_blocks)
                permuted_blocks = blocks[perm_idx]
                
                # Flatten back
                X_permuted[:n_blocks * self.block_length, i] = permuted_blocks.flatten()
                
                # Handle remainder
                if remainder > 0:
                    X_permuted[n_blocks * self.block_length:, i] = X[n_blocks * self.block_length:, i]
                    
        elif self.permute_variables == 'jointly':
            # All variables experience the same permutation
            # Create blocks for all variables
            blocks = X[:n_blocks * self.block_length, :].reshape(n_blocks, self.block_length, n_features)
            
            # Permute blocks (same permutation for all variables)
            perm_idx = rng.permutation(n_blocks)
            permuted_blocks = blocks[perm_idx]
            
            # Flatten back
            X_permuted[:n_blocks * self.block_length, :] = permuted_blocks.reshape(-1, n_features)
            
            # Handle remainder
            if remainder > 0:
                X_permuted[n_blocks * self.block_length:, :] = X[n_blocks * self.block_length:, :]
        else:
            raise ValueError(f"Unknown permute_variables: {self.permute_variables}")
            
        return X_permuted
    
    def fit(self, X, y=None):
        """
        Fit the selector using parallel analysis to determine n_components.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training time series data.
            
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Fitted selector.
        """
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        n_samples, n_features = X.shape
        
        if self.block_length >= n_samples:
            raise ValueError(
                f"block_length ({self.block_length}) must be less than "
                f"n_samples ({n_samples})"
            )
        
        # Setup random state
        rng = np.random.RandomState(self.random_state)
        
        # First, fit on actual data with all components
        if self.verbose:
            print("Fitting SFA on actual data...")
            
        estimator_full = self.estimator.__class__(**self.estimator.get_params())
        estimator_full.set_params(n_components=None)  # Get all components
        estimator_full.fit(X)
        
        self.actual_eigenvalues_ = estimator_full.eigenvalues_.copy()
        
        # Generate null distributions
        if self.verbose:
            print(f"Generating {self.n_permutations} permuted datasets...")
            
        self.null_eigenvalues_ = np.zeros((self.n_permutations, n_features))
        
        for i in range(self.n_permutations):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Permutation {i + 1}/{self.n_permutations}")
                
            # Block permute the data
            X_perm = self._block_permute(X, rng)
            
            # Fit SFA on permuted data
            estimator_perm = self.estimator.__class__(**self.estimator.get_params())
            estimator_perm.set_params(n_components=None)
            estimator_perm.fit(X_perm)
            
            self.null_eigenvalues_[i] = estimator_perm.eigenvalues_
        
        # Compute null distribution statistics
        self.null_mean_ = np.mean(self.null_eigenvalues_, axis=0)
        self.null_std_ = np.std(self.null_eigenvalues_, axis=0)
        
        # Compute threshold (percentile across all null eigenvalues)
        # We want components significantly SLOWER (lower eigenvalues) than null
        self.threshold_eigenvalue_ = np.percentile(
            self.null_eigenvalues_.flatten(),
            self.percentile_threshold
        )
        
        # Alternative: Per-component thresholds (more conservative)
        # self.threshold_eigenvalues_per_component_ = np.percentile(
        #     self.null_eigenvalues_, self.percentile_threshold, axis=0
        # )
        
        # Select components below threshold
        self.selection_mask_ = self.actual_eigenvalues_ < self.threshold_eigenvalue_
        n_selected = np.sum(self.selection_mask_)
        
        # Apply min/max constraints
        if self.max_components is not None:
            n_selected = min(n_selected, self.max_components)
        n_selected = max(n_selected, self.min_components)
        
        # If constraints changed selection, just take the n_selected slowest
        if n_selected != np.sum(self.selection_mask_):
            self.selection_mask_ = np.zeros(n_features, dtype=bool)
            self.selection_mask_[:n_selected] = True
        
        self.n_components_selected_ = n_selected
        
        # Compute p-values for each component
        self.p_values_ = np.zeros(n_features)
        for i in range(n_features):
            # Proportion of null eigenvalues <= actual eigenvalue
            self.p_values_[i] = np.mean(
                self.null_eigenvalues_[:, i] <= self.actual_eigenvalues_[i]
            )
        
        if self.verbose:
            print(f"\nSelected {self.n_components_selected_} components")
            print(f"Threshold eigenvalue: {self.threshold_eigenvalue_:.6f}")
            print(f"Selected eigenvalues: {self.actual_eigenvalues_[:n_selected]}")
            print(f"P-values: {self.p_values_[:n_selected]}")
        
        # Fit final estimator with selected number of components
        self.estimator_ = self.estimator.__class__(**self.estimator.get_params())
        self.estimator_.set_params(n_components=self.n_components_selected_)
        self.estimator_.fit(X)
        
        return self
    
    def transform(self, X):
        """
        Transform data using the fitted estimator with selected components.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components_selected_)
            Transformed data.
        """
        check_is_fitted(self, ['estimator_'])
        return self.estimator_.transform(X)
    
    def inverse_transform(self, Y):
        """
        Transform data back to original space.
        
        Parameters
        ----------
        Y : array-like of shape (n_samples, n_components_selected_)
            Transformed data.
            
        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Data in original space.
        """
        check_is_fitted(self, ['estimator_'])
        return self.estimator_.inverse_transform(Y)
    
    def get_selection_summary(self):
        """
        Get a summary of the component selection results.
        
        Returns
        -------
        summary : dict
            Dictionary containing selection statistics and diagnostics.
        """
        check_is_fitted(self, ['actual_eigenvalues_'])
        
        n_features = len(self.actual_eigenvalues_)
        
        summary = {
            'n_components_selected': self.n_components_selected_,
            'threshold_eigenvalue': self.threshold_eigenvalue_,
            'block_length': self.block_length,
            'n_permutations': self.n_permutations,
            'percentile_threshold': self.percentile_threshold,
            'components': []
        }
        
        for i in range(n_features):
            comp_info = {
                'component': i,
                'eigenvalue': self.actual_eigenvalues_[i],
                'null_mean': self.null_mean_[i],
                'null_std': self.null_std_[i],
                'z_score': (self.actual_eigenvalues_[i] - self.null_mean_[i]) / (self.null_std_[i] + 1e-10),
                'p_value': self.p_values_[i],
                'selected': self.selection_mask_[i]
            }
            summary['components'].append(comp_info)
        
        return summary
    
    def plot_selection(self, figsize=(12, 8), save_path=None):
        """
        Visualize the component selection results.
        
        Creates a figure with:
        1. Eigenvalue comparison (actual vs null distribution)
        2. P-values for each component
        3. Z-scores showing significance
        
        Parameters
        ----------
        figsize : tuple, default=(12, 8)
            Figure size (width, height) in inches.
            
        save_path : str or None, default=None
            If provided, saves the figure to this path.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        """
        check_is_fitted(self, ['actual_eigenvalues_'])
        
        import matplotlib.pyplot as plt
        
        n_features = len(self.actual_eigenvalues_)
        component_idx = np.arange(n_features)
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Plot 1: Eigenvalues with null distribution
        ax = axes[0]
        
        # Plot null distribution as boxplot
        bp = ax.boxplot(self.null_eigenvalues_, positions=component_idx,
                        widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='lightgray', alpha=0.7),
                        medianprops=dict(color='darkblue', linewidth=2),
                        showfliers=False)
        
        # Plot actual eigenvalues
        colors = ['green' if sel else 'red' for sel in self.selection_mask_]
        ax.scatter(component_idx, self.actual_eigenvalues_, 
                  c=colors, s=100, zorder=5, edgecolors='black', linewidth=1.5,
                  label='Actual')
        
        # Add threshold line
        ax.axhline(self.threshold_eigenvalue_, color='orange', linestyle='--',
                  linewidth=2, label=f'{self.percentile_threshold}th Percentile Threshold')
        
        ax.set_xlabel('Component Index', fontsize=11)
        ax.set_ylabel('Eigenvalue (Slowness)', fontsize=11)
        ax.set_title('Parallel Analysis: Actual vs Null Eigenvalues', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add selection region
        ax.axvspan(-0.5, self.n_components_selected_ - 0.5, 
                  alpha=0.1, color='green', label='Selected')
        
        # Plot 2: P-values
        ax = axes[1]
        colors = ['green' if sel else 'red' for sel in self.selection_mask_]
        ax.bar(component_idx, self.p_values_, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(0.05, color='orange', linestyle='--', linewidth=2, 
                  label='p = 0.05')
        ax.set_xlabel('Component Index', fontsize=11)
        ax.set_ylabel('P-value', fontsize=11)
        ax.set_title('Component P-values (Empirical)', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Z-scores
        ax = axes[2]
        z_scores = (self.actual_eigenvalues_ - self.null_mean_) / (self.null_std_ + 1e-10)
        colors = ['green' if sel else 'red' for sel in self.selection_mask_]
        ax.bar(component_idx, z_scores, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(0, color='black', linewidth=1)
        ax.axhline(-1.96, color='orange', linestyle='--', linewidth=2, 
                  label='Z = -1.96 (p ≈ 0.025)')
        ax.set_xlabel('Component Index', fontsize=11)
        ax.set_ylabel('Z-score', fontsize=11)
        ax.set_title('Standardized Effect Sizes', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        return fig


if __name__ == "__main__":
    """
    Sample code demonstrating Graph-SFA usage with different scenarios.
    """
    import matplotlib.pyplot as plt
    
    print("=" * 80)
    print("Graph-Regularized Slow Feature Analysis - Sample Code")
    print("=" * 80)
    
    # ========================================================================
    # Example 1: Basic SFA without graph (standard SFA)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 1: Standard SFA (no graph regularization)")
    print("=" * 80)
    
    # Generate synthetic time series with slow and fast components
    np.random.seed(42)
    n_samples = 1000
    t = np.linspace(0, 10, n_samples)
    
    # Slow varying signals
    slow1 = np.sin(0.5 * t)  # Low frequency
    slow2 = np.cos(0.5 * t)  # Low frequency
    
    # Fast varying signals
    fast1 = np.sin(5 * t)  # High frequency
    fast2 = np.cos(5 * t)  # High frequency
    
    # Mix them together (each original signal influences multiple observed variables)
    X_basic = np.column_stack([
        slow1 + 0.3 * fast1 + 0.1 * np.random.randn(n_samples),
        slow2 + 0.3 * fast2 + 0.1 * np.random.randn(n_samples),
        0.5 * slow1 + fast1 + 0.1 * np.random.randn(n_samples),
        0.5 * slow2 + fast2 + 0.1 * np.random.randn(n_samples),
    ])
    
    # Fit standard SFA
    sfa_basic = GraphSlowFeatureAnalysis(
        n_components=4,
        standardize=True,
        whiten=True
    )
    Y_basic = sfa_basic.fit_transform(X_basic)
    
    print(f"\nInput shape: {X_basic.shape}")
    print(f"Output shape: {Y_basic.shape}")
    print(f"\nSlowness values (eigenvalues): {sfa_basic.get_slowness_values()}")
    print(f"Explained slowness ratio: {sfa_basic.explained_slowness_ratio_}")
    print("\nLoading matrix (components):")
    print(sfa_basic.get_loadings())
    
    # Compute derivatives
    Y_dot_basic = sfa_basic.transform_derivatives(X_basic)
    print(f"\nDerivative shape: {Y_dot_basic.shape}")
    print(f"Derivative variance per component: {np.var(Y_dot_basic, axis=0)}")
    
    # ========================================================================
    # Example 2: Graph-SFA with simple graph structure
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 2: Graph-SFA with known variable relationships")
    print("=" * 80)
    
    # Define graph structure: variables (0,1) are related, (2,3) are related
    # Using adjacency matrix (will be converted to Laplacian)
    W_adjacency = np.array([
        [0, 1, 0, 0],  # Variable 0 connected to variable 1
        [1, 0, 0, 0],  # Variable 1 connected to variable 0
        [0, 0, 0, 1],  # Variable 2 connected to variable 3
        [0, 0, 1, 0],  # Variable 3 connected to variable 2
    ], dtype=float)
    
    print("\nAdjacency matrix (graph structure):")
    print(W_adjacency)
    
    # Fit Graph-SFA with different regularization strengths
    for lambda_val in [0.0, 0.1, 1.0, 10.0]:
        sfa_graph = GraphSlowFeatureAnalysis(
            n_components=4,
            graph_laplacian=W_adjacency,
            lambda_graph=lambda_val,
            laplacian_type='normalized',
            standardize=True,
            whiten=True
        )
        Y_graph = sfa_graph.fit_transform(X_basic)
        
        print(f"\n--- Lambda = {lambda_val} ---")
        print(f"Slowness values: {sfa_graph.get_slowness_values()}")
        print("Loading matrix (first 2 components):")
        print(sfa_graph.get_loadings()[:2])
    
    # ========================================================================
    # Example 3: Pre-computed Laplacian matrix
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 3: Using pre-computed Laplacian matrix")
    print("=" * 80)
    
    # Manually compute combinatorial Laplacian
    D = np.diag(np.sum(W_adjacency, axis=1))
    L_combinatorial = D - W_adjacency
    
    print("\nCombinatorial Laplacian L = D - W:")
    print(L_combinatorial)
    
    sfa_laplacian = GraphSlowFeatureAnalysis(
        n_components=2,
        graph_laplacian=L_combinatorial,
        lambda_graph=0.5,
        standardize=True
    )
    Y_laplacian = sfa_laplacian.fit_transform(X_basic)
    
    print(f"\nSlowness values: {sfa_laplacian.get_slowness_values()}")
    print("Loadings:")
    print(sfa_laplacian.get_loadings())
    
    # ========================================================================
    # Example 4: Complex graph with multiple connected components
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 4: More complex graph structure")
    print("=" * 80)
    
    # Generate 6-dimensional data
    n_samples = 1000
    t = np.linspace(0, 20, n_samples)
    
    # Three groups of slow features
    group1_slow = np.sin(0.3 * t)
    group2_slow = np.cos(0.4 * t)
    group3_slow = np.sin(0.2 * t + 1.0)
    
    # Fast noise
    fast_noise = np.random.randn(n_samples, 6) * 0.2
    
    X_complex = np.column_stack([
        group1_slow + fast_noise[:, 0],      # Group 1
        group1_slow * 0.8 + fast_noise[:, 1],  # Group 1
        group2_slow + fast_noise[:, 2],      # Group 2
        group2_slow * 0.9 + fast_noise[:, 3],  # Group 2
        group3_slow + fast_noise[:, 4],      # Group 3
        group3_slow * 0.85 + fast_noise[:, 5], # Group 3
    ])
    
    # Graph: Three disconnected groups (0-1, 2-3, 4-5)
    W_complex = np.array([
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
    ], dtype=float)
    
    print("\nGraph structure (3 disconnected pairs):")
    print(W_complex)
    
    sfa_complex = GraphSlowFeatureAnalysis(
        n_components=3,
        graph_laplacian=W_complex,
        lambda_graph=1.0,
        laplacian_type='normalized',
        standardize=True,
        whiten=True
    )
    Y_complex = sfa_complex.fit_transform(X_complex)
    
    print(f"\nSlowness values: {sfa_complex.get_slowness_values()}")
    print("\nFeature importance (absolute loadings):")
    print(sfa_complex.get_feature_importance())
    print("\nNote: Each component should primarily use variables from one graph group")
    
    # ========================================================================
    # Example 5: Comparing different Laplacian types
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 5: Comparing different Laplacian normalizations")
    print("=" * 80)
    
    # Create a graph with varying degrees
    W_varied = np.array([
        [0, 1, 1, 1, 0, 0],  # Node 0: degree 3
        [1, 0, 1, 0, 0, 0],  # Node 1: degree 2
        [1, 1, 0, 1, 0, 0],  # Node 2: degree 3
        [1, 0, 1, 0, 1, 0],  # Node 3: degree 3
        [0, 0, 0, 1, 0, 1],  # Node 4: degree 2
        [0, 0, 0, 0, 1, 0],  # Node 5: degree 1
    ], dtype=float)
    
    print("\nAdjacency matrix (varying node degrees):")
    print(W_varied)
    print(f"\nNode degrees: {np.sum(W_varied, axis=1)}")
    
    for lap_type in ['combinatorial', 'normalized', 'random_walk']:
        sfa_type = GraphSlowFeatureAnalysis(
            n_components=2,
            graph_laplacian=W_varied,
            lambda_graph=0.5,
            laplacian_type=lap_type,
            standardize=True
        )
        Y_type = sfa_type.fit_transform(X_complex)
        
        print(f"\n--- Laplacian type: {lap_type} ---")
        print(f"Slowness values: {sfa_type.get_slowness_values()}")
        print(f"First component loadings: {sfa_type.get_loadings()[0]}")
    
    # ========================================================================
    # Example 6: Using transform_derivatives and inverse_transform
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 6: Working with derivatives and reconstruction")
    print("=" * 80)
    
    sfa_demo = GraphSlowFeatureAnalysis(
        n_components=2,
        standardize=True,
        whiten=True,
        time_derivative_method='central_diff'
    )
    sfa_demo.fit(X_basic)
    
    # Transform to slow feature space
    Y_demo = sfa_demo.transform(X_basic)
    print(f"\nSlow features shape: {Y_demo.shape}")
    print(f"Slow features mean: {np.mean(Y_demo, axis=0)}")
    print(f"Slow features std: {np.std(Y_demo, axis=0)}")
    
    # Get derivatives of slow features
    Y_dot_demo = sfa_demo.transform_derivatives(X_basic)
    print(f"\nSlow feature derivatives shape: {Y_dot_demo.shape}")
    print(f"Derivative variance (slowness): {np.var(Y_dot_demo, axis=0)}")
    
    # Reconstruct original data from slow features
    X_reconstructed = sfa_demo.inverse_transform(Y_demo)
    print(f"\nReconstructed data shape: {X_reconstructed.shape}")
    reconstruction_error = np.mean((X_basic - X_reconstructed)**2)
    print(f"Reconstruction error (MSE): {reconstruction_error:.6f}")
    
    # ========================================================================
    # Example 7: Model selection with score()
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 7: Model selection using score() method")
    print("=" * 80)
    
    # Test different numbers of components
    print("\nTesting different numbers of components:")
    for n_comp in [1, 2, 3, 4]:
        sfa_test = GraphSlowFeatureAnalysis(
            n_components=n_comp,
            standardize=True
        )
        sfa_test.fit(X_basic[:800])  # Train on first 800 samples
        score = sfa_test.score(X_basic[800:])  # Test on last 200 samples
        print(f"n_components={n_comp}: score={score:.6f} (higher is better)")
    
    # Test different regularization strengths
    print("\nTesting different graph regularization strengths:")
    for lambda_val in [0.0, 0.1, 0.5, 1.0, 5.0]:
        sfa_test = GraphSlowFeatureAnalysis(
            n_components=2,
            graph_laplacian=W_adjacency,
            lambda_graph=lambda_val,
            standardize=True
        )
        sfa_test.fit(X_basic[:800])
        score = sfa_test.score(X_basic[800:])
        print(f"lambda={lambda_val}: score={score:.6f}")
    
    # ========================================================================
    # Example 8: Visualization
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 8: Visualizing slow features")
    print("=" * 80)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Graph-Regularized Slow Feature Analysis', fontsize=14, fontweight='bold')
    
    # Plot original signals
    axes[0, 0].plot(t[:200], X_basic[:200, 0], label='Var 0', alpha=0.7)
    axes[0, 0].plot(t[:200], X_basic[:200, 1], label='Var 1', alpha=0.7)
    axes[0, 0].set_title('Original Signals (first 2 variables)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot slow features without graph
    Y_no_graph = sfa_basic.transform(X_basic)
    axes[0, 1].plot(t[:200], Y_no_graph[:200, 0], label='SF 1', alpha=0.7)
    axes[0, 1].plot(t[:200], Y_no_graph[:200, 1], label='SF 2', alpha=0.7)
    axes[0, 1].set_title('Slow Features (Standard SFA)')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot slow features with graph (low regularization)
    sfa_low = GraphSlowFeatureAnalysis(
        n_components=2, graph_laplacian=W_adjacency, 
        lambda_graph=0.1, laplacian_type='normalized'
    )
    Y_low = sfa_low.fit_transform(X_basic)
    axes[1, 0].plot(t[:200], Y_low[:200, 0], label='SF 1', alpha=0.7)
    axes[1, 0].plot(t[:200], Y_low[:200, 1], label='SF 2', alpha=0.7)
    axes[1, 0].set_title('Slow Features (Graph-SFA, λ=0.1)')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot slow features with graph (high regularization)
    sfa_high = GraphSlowFeatureAnalysis(
        n_components=2, graph_laplacian=W_adjacency,
        lambda_graph=1.0, laplacian_type='normalized'
    )
    Y_high = sfa_high.fit_transform(X_basic)
    axes[1, 1].plot(t[:200], Y_high[:200, 0], label='SF 1', alpha=0.7)
    axes[1, 1].plot(t[:200], Y_high[:200, 1], label='SF 2', alpha=0.7)
    axes[1, 1].set_title('Slow Features (Graph-SFA, λ=1.0)')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot loading patterns
    loadings_no_graph = sfa_basic.get_loadings()[:2]
    x_pos = np.arange(4)
    width = 0.35
    axes[2, 0].bar(x_pos - width/2, loadings_no_graph[0], width, label='Component 1', alpha=0.7)
    axes[2, 0].bar(x_pos + width/2, loadings_no_graph[1], width, label='Component 2', alpha=0.7)
    axes[2, 0].set_title('Loadings (Standard SFA)')
    axes[2, 0].set_xlabel('Variable Index')
    axes[2, 0].set_ylabel('Loading Value')
    axes[2, 0].set_xticks(x_pos)
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3, axis='y')
    axes[2, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    loadings_graph = sfa_high.get_loadings()[:2]
    axes[2, 1].bar(x_pos - width/2, loadings_graph[0], width, label='Component 1', alpha=0.7)
    axes[2, 1].bar(x_pos + width/2, loadings_graph[1], width, label='Component 2', alpha=0.7)
    axes[2, 1].set_title('Loadings (Graph-SFA, λ=1.0)')
    axes[2, 1].set_xlabel('Variable Index')
    axes[2, 1].set_ylabel('Loading Value')
    axes[2, 1].set_xticks(x_pos)
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3, axis='y')
    axes[2, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    print("\nVisualization created. Close the plot window to continue...")
    plt.savefig('/home/claude/graph_sfa_demo.png', dpi=150, bbox_inches='tight')
    print("Plot saved to: /home/claude/graph_sfa_demo.png")
    
    # ========================================================================
    # Example 9: Automatic Component Selection with SlowFeatureSelector
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 9: Automatic Component Selection via Parallel Analysis")
    print("=" * 80)
    
    # Generate data with clear separation between slow and fast features
    np.random.seed(42)
    n_samples = 1000
    t = np.linspace(0, 20, n_samples)
    
    # Two genuinely slow features
    slow_1 = np.sin(0.3 * t)
    slow_2 = np.cos(0.25 * t)
    
    # Two fast features
    fast_1 = np.sin(8 * t)
    fast_2 = np.cos(10 * t)
    
    # Mix into observed variables
    X_auto = np.column_stack([
        slow_1 + 0.2 * fast_1 + 0.1 * np.random.randn(n_samples),
        slow_2 + 0.2 * fast_2 + 0.1 * np.random.randn(n_samples),
        0.7 * slow_1 + fast_1 + 0.1 * np.random.randn(n_samples),
        0.7 * slow_2 + fast_2 + 0.1 * np.random.randn(n_samples),
        fast_1 + 0.1 * np.random.randn(n_samples),
        fast_2 + 0.1 * np.random.randn(n_samples),
    ])
    
    print("\nData has 6 variables: 2 slow features mixed in different ways + 2 pure fast features")
    print("Expected result: Should select ~2 components")
    
    # Create base SFA estimator
    base_sfa = GraphSlowFeatureAnalysis(
        standardize=True,
        whiten=True,
        time_derivative_method='finite_diff'
    )
    
    # Use SlowFeatureSelector for automatic component selection
    selector = SlowFeatureSelector(
        estimator=base_sfa,
        block_length=30,  # Based on slow feature period (~20 samples)
        n_permutations=50,  # Using 50 for speed in demo
        percentile_threshold=95.0,
        permute_variables='independently',
        min_components=1,
        random_state=42,
        verbose=True
    )
    
    # Fit and transform
    Y_auto = selector.fit_transform(X_auto)
    
    print("\n" + "-" * 80)
    print("SELECTION RESULTS")
    print("-" * 80)
    summary = selector.get_selection_summary()
    print(f"\nComponents selected: {summary['n_components_selected']}")
    print(f"Threshold eigenvalue: {summary['threshold_eigenvalue']:.6f}")
    print(f"Block length used: {summary['block_length']}")
    print(f"Number of permutations: {summary['n_permutations']}")
    
    print("\nPer-component details:")
    print(f"{'Comp':<6} {'Eigenval':<12} {'Null Mean':<12} {'Null Std':<12} {'Z-score':<10} {'P-value':<10} {'Selected':<10}")
    print("-" * 80)
    for comp in summary['components']:
        print(f"{comp['component']:<6} "
              f"{comp['eigenvalue']:<12.6f} "
              f"{comp['null_mean']:<12.6f} "
              f"{comp['null_std']:<12.6f} "
              f"{comp['z_score']:<10.3f} "
              f"{comp['p_value']:<10.4f} "
              f"{'YES' if comp['selected'] else 'NO':<10}")
    
    # Visualize selection
    print("\nGenerating selection visualization...")
    fig_selection = selector.plot_selection(save_path='/home/claude/component_selection.png')
    plt.close(fig_selection)
    
    # ========================================================================
    # Example 10: Comparing Manual vs Automatic Selection
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 10: Manual vs Automatic Component Selection")
    print("=" * 80)
    
    # Manual selection with different n_components
    results_manual = {}
    for n_comp in [2, 4, 6]:
        sfa_manual = GraphSlowFeatureAnalysis(
            n_components=n_comp,
            standardize=True,
            whiten=True
        )
        sfa_manual.fit(X_auto[:800])
        score = sfa_manual.score(X_auto[800:])
        results_manual[n_comp] = {
            'score': score,
            'eigenvalues': sfa_manual.get_slowness_values()
        }
    
    print("\nManual selection (validation scores):")
    for n_comp, result in results_manual.items():
        print(f"  n_components={n_comp}: score={result['score']:.6f}")
        print(f"    Eigenvalues: {result['eigenvalues']}")
    
    # Automatic selection
    selector_auto = SlowFeatureSelector(
        estimator=GraphSlowFeatureAnalysis(standardize=True, whiten=True),
        block_length=30,
        n_permutations=50,
        random_state=42,
        verbose=False
    )
    selector_auto.fit(X_auto[:800])
    score_auto = selector_auto.estimator_.score(X_auto[800:])
    
    print(f"\nAutomatic selection:")
    print(f"  n_components={selector_auto.n_components_selected_}: score={score_auto:.6f}")
    print(f"    Eigenvalues: {selector_auto.estimator_.get_slowness_values()}")
    
    # ========================================================================
    # Example 11: Effect of Block Length on Selection
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 11: Sensitivity to Block Length Parameter")
    print("=" * 80)
    
    block_lengths = [10, 20, 30, 50, 100]
    
    print("\nTesting different block lengths:")
    print(f"{'Block Length':<15} {'N Components':<15} {'Threshold':<15}")
    print("-" * 45)
    
    for bl in block_lengths:
        selector_bl = SlowFeatureSelector(
            estimator=GraphSlowFeatureAnalysis(standardize=True),
            block_length=bl,
            n_permutations=30,
            random_state=42,
            verbose=False
        )
        selector_bl.fit(X_auto)
        print(f"{bl:<15} {selector_bl.n_components_selected_:<15} {selector_bl.threshold_eigenvalue_:<15.6f}")
    
    print("\nNote: Larger block lengths preserve more slow structure in null data,")
    print("      leading to higher thresholds and potentially fewer selected components.")
    
    # ========================================================================
    # Example 12: Graph-SFA with Automatic Selection
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 12: Combining Graph Regularization with Automatic Selection")
    print("=" * 80)
    
    # Define graph structure for our 6 variables
    # Variables (0,1) related, (2,3) related, (4,5) related
    W_graph_auto = np.array([
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
    ], dtype=float)
    
    print("\nGraph structure (3 pairs of connected variables):")
    print(W_graph_auto)
    
    # Without graph regularization
    selector_no_graph = SlowFeatureSelector(
        estimator=GraphSlowFeatureAnalysis(
            standardize=True,
            graph_laplacian=None,
            lambda_graph=0.0
        ),
        block_length=30,
        n_permutations=30,
        random_state=42,
        verbose=False
    )
    selector_no_graph.fit(X_auto)
    
    # With graph regularization
    selector_with_graph = SlowFeatureSelector(
        estimator=GraphSlowFeatureAnalysis(
            standardize=True,
            graph_laplacian=W_graph_auto,
            lambda_graph=0.5,
            laplacian_type='normalized'
        ),
        block_length=30,
        n_permutations=30,
        random_state=42,
        verbose=False
    )
    selector_with_graph.fit(X_auto)
    
    print(f"\nWithout graph regularization:")
    print(f"  Selected components: {selector_no_graph.n_components_selected_}")
    print(f"  Eigenvalues: {selector_no_graph.actual_eigenvalues_[:selector_no_graph.n_components_selected_]}")
    
    print(f"\nWith graph regularization (λ=0.5):")
    print(f"  Selected components: {selector_with_graph.n_components_selected_}")
    print(f"  Eigenvalues: {selector_with_graph.actual_eigenvalues_[:selector_with_graph.n_components_selected_]}")
    
    print("\nLoadings without graph:")
    print(selector_no_graph.estimator_.get_loadings())
    
    print("\nLoadings with graph:")
    print(selector_with_graph.estimator_.get_loadings())
    
    print("\nNote: Graph regularization encourages connected variables to have similar weights.")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
Key features demonstrated:
1. Standard SFA without graph regularization (λ=0)
2. Graph-SFA with adjacency matrix (automatically converted to Laplacian)
3. Graph-SFA with pre-computed Laplacian matrix
4. Complex graphs with multiple connected components
5. Comparison of different Laplacian types (combinatorial, normalized, random_walk)
6. Using transform_derivatives() and inverse_transform()
7. Model selection with score() method
8. Visualization of slow features and loading patterns
9. Automatic component selection via SlowFeatureSelector with parallel analysis
10. Comparison of manual vs automatic component selection
11. Sensitivity analysis of block length parameter
12. Combining graph regularization with automatic selection

The graph regularization encourages variables that are connected in the graph
to have similar weights in the slow feature components, effectively incorporating
prior knowledge about variable relationships into the SFA decomposition.

The SlowFeatureSelector uses temporal parallel analysis with block permutation
to automatically determine the optimal number of components by comparing actual
eigenvalues against a null distribution from randomized data.

For more information, see the class docstrings and method documentation.
    """)
    print("=" * 80)

