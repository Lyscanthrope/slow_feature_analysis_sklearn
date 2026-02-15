# Graph-Regularized Slow Feature Analysis with Automatic Component Selection

## Scientific Documentation

---

## 1. Introduction

### 1.1 Background

**Slow Feature Analysis (SFA)** is an unsupervised learning algorithm for extracting slowly-varying features from high-dimensional time series data. Originally introduced by Wiskott and Sejnowski (2002), SFA is based on the principle that meaningful information in temporal signals often changes more slowly than the raw sensory data.

This implementation extends classical SFA in two important directions:

1. **Graph-Regularized SFA**: Incorporates prior knowledge about relationships between input variables through graph Laplacian regularization
2. **Automatic Component Selection**: Uses temporal parallel analysis with block permutation to objectively determine the number of slow features

---

## 2. Theoretical Framework

### 2.1 Classical Slow Feature Analysis

#### 2.1.1 Problem Formulation

Given a multivariate time series $x(t) ∈ ℝ^d$, SFA seeks to find a set of functions ${g_i}$ that extract slowly-varying features y(t) = [y₁(t), ..., y_k(t)]^T where:

$$y_i(t) = g_i(x(t)) = w_i^T xt)$$

for linear SFA with weight vectors **w**_i ∈ ℝ^d.

#### 2.1.2 Optimization Objective

The slowness of a feature y_i(t) is quantified by the variance of its temporal derivative:

$$Δ(y_i) = ⟨ẏ_i²⟩_t$$

where $⟨·⟩_t$ denotes temporal averaging and $ẏ_i = dy_i/dt$.

The SFA optimization problem is:

**Minimize**: $Δ(y_i) = ⟨ẏ_i²⟩_t$

**Subject to**:

- $⟨y_i⟩_t = 0$  (zero mean)
- $⟨y_i²⟩_t = 1$  (unit variance)
- $⟨y_i,y_j⟩_t = 0 $ for i ≠ j  (decorrelation)

#### 2.1.3 Solution via Generalized Eigenvalue Problem

For linear SFA with whitened input data, the optimization reduces to solving:

**Ċ w** = λ **w**

where:

- **Ċ** = ⟨**ẋ ẋ**^T⟩_t is the temporal derivative covariance matrix
- λ is the eigenvalue (slowness measure)
- **w** is the eigenvector (weight vector for slow feature)

The eigenvectors corresponding to the smallest eigenvalues yield the slowest features.

### 2.2 Graph-Regularized SFA

#### 2.2.1 Motivation

In many applications, we have prior knowledge about relationships between input variables:

- Spatial proximity in sensor networks
- Functional connectivity in neural recordings
- Causal or correlation structure in multivariate systems

Graph-regularized SFA incorporates this knowledge to encourage connected variables in the graph to have similar weights in the extracted features.

#### 2.2.2 Graph Laplacian

Let **G** = (V, E) be an undirected weighted graph where:

- V = {1, ..., d} represents the input variables
- **W** ∈ ℝ^(d×d) is the symmetric weight/adjacency matrix where W_ij > 0 if nodes i and j are connected

The **graph Laplacian** is defined as:

**L** = **D** - **W**

where **D** is the diagonal degree matrix: D_ii = Σ_j W_ij

#### 2.2.3 Types of Graph Laplacian

**Combinatorial Laplacian**:
L_comb = **D** - **W**

**Normalized Symmetric Laplacian**:
L_norm = **I** - **D**^(-1/2) **W D**^(-1/2)

**Random Walk Laplacian**:
L_rw = **I** - **D**^(-1) **W**

The normalized Laplacian is generally preferred as it is less sensitive to the degree distribution and has eigenvalues in [0, 2].

#### 2.2.4 Graph Regularization Term

The graph smoothness penalty for a weight vector **w** is:

R_graph(**w**) = **w**^T **L w** = (1/2) Σ_{(i,j)∈E} W_ij (w_i - w_j)²

This term is minimized when connected variables have similar weights.

#### 2.2.5 Modified Optimization Problem

Graph-SFA solves:

**Minimize**: **w**^T **Ċ w** + λ **w**^T **L w**

**Subject to**: **w**^T **C w** = 1

where:

- **C** is the data covariance matrix
- λ ≥ 0 is the regularization parameter controlling the strength of graph constraint
- λ = 0 recovers standard SFA
- λ → ∞ enforces pure graph-structured solutions

#### 2.2.6 Solution

After whitening transformation, the graph-regularized SFA becomes:

(**Ċ** + λ**L**) **w** = β **w**

This is a standard eigenvalue problem. The eigenvectors corresponding to smallest eigenvalues β yield the slowest graph-regularized features.

### 2.3 Automatic Component Selection via Parallel Analysis

#### 2.3.1 The Component Selection Problem

A fundamental challenge in SFA (and other dimensionality reduction methods) is determining how many components to retain. Unlike PCA where "explained variance" provides a natural criterion, in SFA we minimize derivative variance, making traditional selection methods less applicable.

#### 2.3.2 Temporal Parallel Analysis Framework

**Parallel analysis** (Horn, 1965) compares eigenvalues from actual data against those from appropriately randomized data. For SFA, we adapt this by:

1. Generating null datasets that destroy slow temporal structure while preserving fast dynamics
2. Computing eigenvalue distributions from these null datasets
3. Selecting components whose slowness is significantly greater (eigenvalues significantly lower) than the null distribution

#### 2.3.3 Block Permutation for Temporal Null Model

**Rationale**: We need a randomization that:

- Destroys temporal dependencies at slow timescales (breaks slow features)
- Preserves temporal dependencies at fast timescales (maintains realistic fast dynamics)
- Maintains marginal statistical properties (variance, distribution)

**Method**: Block permutation with block length L

For a time series **x**(t) where t = 1, ..., T:

1. Divide the series into blocks of length L:
   **B**_k = [**x**(kL+1), **x**(kL+2), ..., **x**((k+1)L)]  for k = 0, ..., ⌊T/L⌋-1

2. Randomly permute the order of blocks:
   **x**_perm(t) = [**B**_π(0), **B**_π(1), ..., **B**_π(K)]
   where π is a random permutation

**Properties**:

- Temporal structures with period ≫ L are destroyed (broken across block boundaries)
- Temporal structures with period ≪ L are preserved (within-block dynamics intact)
- Structures with period ≈ L are partially destroyed

**Choice of L**: Should be informed by domain knowledge about the slowest timescale of interest. Features slower than L will be destroyed in the null data; features faster than L will be preserved.

#### 2.3.4 Independent vs Joint Permutation

**Independent permutation**: Each variable x_i(t) is permuted with its own random permutation π_i

- Destroys all cross-variable slow relationships
- More conservative (aggressive null model)
- Tests whether slow features survive complete destruction of inter-variable temporal coherence

**Joint permutation**: All variables are permuted with the same permutation π

- Preserves instantaneous cross-variable correlations
- Tests whether slow features arise from temporal evolution rather than static correlations

For graph-regularized SFA, **independent permutation** is recommended as it provides a stronger test of slow feature significance.

#### 2.3.5 Null Distribution and Threshold

For K permutations, we obtain a distribution of eigenvalues:

{λ^(k)_i : k = 1, ..., K, i = 1, ..., d}

where λ^(k)_i is the i-th eigenvalue from the k-th permutation.

The **threshold** for component selection is defined as:

λ_threshold = percentile({λ^(k)_i}, α)

Typically α = 95, meaning we select components whose actual eigenvalue falls below the 95th percentile of the null distribution.

#### 2.3.6 Selection Criterion

Component i is selected if:

λ_i < λ_threshold

where λ_i is the eigenvalue from actual data.

Equivalently, we can compute empirical p-values:

p_i = (1/Kd) Σ_{k,j} 𝟙[λ^(k)_j ≤ λ_i]

and select components where p_i < (1 - α/100).

#### 2.3.7 Statistical Interpretation

Under the null hypothesis that a component is not genuinely slow (i.e., its slowness arises from chance), its eigenvalue should be comparable to those from the permuted data. Rejection of this null hypothesis (eigenvalue significantly below threshold) provides evidence that the component captures genuine slow temporal structure.

The z-score for component i is:

z_i = (λ_i - μ_null,i) / σ_null,i

where:

- μ_null,i = mean eigenvalue for i-th component across permutations
- σ_null,i = standard deviation across permutations

Highly negative z-scores (z_i < -2) indicate strong evidence for genuine slow features.

---

## 3. Mathematical Derivations

### 3.1 Whitening Transformation

Given data covariance **C**, the whitening transformation is:

**Z** = **C**^(-1/2) = **V Λ**^(-1/2) **V**^T

where **C** = **V Λ V**^T is the eigendecomposition of **C**.

After whitening, the data **x**_white = **Z x** satisfies:

⟨**x**_white **x**_white^T⟩ = **I**

### 3.2 Derivative Covariance in Whitened Space

For whitened data, the derivative covariance becomes:

**Ċ**_white = **Z**^T **Ċ Z**

### 3.3 Graph Laplacian in Whitened Space

The graph regularization term in whitened space:

**w**_white^T (**Z**^T **L Z**) **w**_white

### 3.4 Combined Objective

The graph-regularized SFA in whitened space becomes:

**Minimize**: **w**_white^T (**Ċ**_white + λ**L**_white) **w**_white

**Subject to**: **w**_white^T **w**_white = 1

where **L**_white = **Z**^T **L Z**

This is solved by eigendecomposition:

(**Ċ**_white + λ**L**_white) **w**_white = β **w**_white

### 3.5 Transformation Back to Original Space

The weight vectors in original space are:

**w** = **Z w**_white

These are then normalized: **w** ← **w** / ||**w**||

---

## 4. Algorithm Summary

### 4.1 Graph-Regularized SFA Algorithm

**Input**:

- Time series data **X** ∈ ℝ^(T×d)
- Graph Laplacian **L** ∈ ℝ^(d×d) (optional)
- Regularization parameter λ ≥ 0
- Number of components k

**Algorithm**:

1. **Preprocessing**:
   - Center: **X** ← **X** - mean(**X**)
   - Standardize (optional): **X** ← **X** / std(**X**)

2. **Compute Temporal Derivatives**:
   - **Ẋ** ← diff(**X***, axis=time)

3. **Compute Covariance Matrices**:
   - **C** = (1/T) **X**^T **X**
   - **Ċ** = (1/T) **Ẋ**^T **Ẋ**

4. **Whitening**:
   - Compute **Z** = **C**^(-1/2)
   - **Ċ**_white = **Z**^T **Ċ Z**
   - **L**_white = **Z**^T **L Z** (if graph provided)

5. **Add Graph Regularization**:
   - **M** = **Ċ**_white + λ**L**_white

6. **Eigendecomposition**:
   - Solve **M w** = β **w**
   - Sort eigenvectors by eigenvalue (ascending)
   - Select k eigenvectors with smallest eigenvalues

7. **Transform to Original Space**:
   - **W** = **Z** × [**w**₁, ..., **w**_k]
   - Normalize columns of **W**

8. **Project Data**:
   - **Y** = **X W**

**Output**: Slow features **Y** ∈ ℝ^(T×k)

### 4.2 Automatic Component Selection Algorithm

**Input**:

- Time series data **X** ∈ ℝ^(T×d)
- Block length L
- Number of permutations K
- Percentile threshold α (default: 95)

**Algorithm**:

1. **Fit SFA on Actual Data**:
   - Run Graph-SFA with all components (k = d)
   - Store eigenvalues: **λ**_actual = [λ₁, ..., λ_d]

2. **Generate Null Distribution**:
   - For k = 1 to K:
     - **X**_perm ← BlockPermute(**X**, L)
     - Run Graph-SFA on **X**_perm
     - Store eigenvalues: **λ**^(k) = [λ₁^(k), ..., λ_d^(k)]

3. **Compute Null Statistics**:
   - **μ**_null = mean(**λ**^(1), ..., **λ**^(K))
   - **σ**_null = std(**λ**^(1), ..., **λ**^(K))

4. **Determine Threshold**:
   - λ_threshold = percentile(flatten(**λ**^(1), ..., **λ**^(K)), α)

5. **Select Components**:
   - n_select = |{i : λ_i < λ_threshold}|
   - Apply min/max constraints if specified

6. **Refit with Selected Components**:
   - Run Graph-SFA with k = n_select

**Output**:

- Fitted SFA with n_select components
- Diagnostic information (null distribution, p-values, z-scores)

---

## 5. Practical Considerations

### 5.1 Choice of Block Length L

The block length should be chosen based on domain knowledge:

- **Lower bound**: Should be larger than the period of the fastest dynamics you want to preserve in the null model
- **Upper bound**: Should be smaller than or comparable to the period of the slowest features you want to detect
- **Rule of thumb**: L ≈ (τ_slow / 2) where τ_slow is the characteristic timescale of slow features

**Example**: For hourly financial data where you want to detect daily patterns, use L ≈ 12 hours.

### 5.2 Number of Permutations K

- Minimum: K ≥ 50 for stable estimates
- Recommended: K = 100-200 for reliable p-values
- High-stakes: K = 1000+ for publication-quality results

Trade-off: More permutations → better statistical estimates but longer computation time.

### 5.3 Percentile Threshold α

- Conservative (few false positives): α = 99 (select only very significant components)
- Moderate (balanced): α = 95 (standard choice)
- Liberal (more components): α = 90

### 5.4 Graph Regularization Parameter λ

- λ = 0: No graph constraint (standard SFA)
- λ ∈ [0.01, 0.1]: Weak graph influence
- λ ∈ [0.1, 1.0]: Moderate graph constraint
- λ > 1.0: Strong graph structure enforcement

Selection via cross-validation or information criteria recommended.

### 5.5 Independent vs Joint Permutation

- **Use independent** when:
  - Testing whether slow features depend on cross-variable temporal coherence
  - Graph captures important cross-variable relationships
  - Conservative testing desired

- **Use joint** when:
  - Instantaneous correlations are important
  - Variables have synchronized measurements
  - Want to test temporal evolution specifically

---

## 6. Interpretation of Results

### 6.1 Eigenvalues

- **Small eigenvalues** (λ_i < 0.01): Very slow features, change minimally over time
- **Moderate eigenvalues** (0.01 < λ_i < 0.1): Moderately slow features
- **Large eigenvalues** (λ_i > 0.1): Fast-changing features (usually discarded)

The scale depends on sampling rate and data characteristics.

### 6.2 Loading Vectors (Components)

The loading vector **w**_i indicates which input variables contribute to slow feature i:

- Large positive values: Strong positive contribution
- Large negative values: Strong negative contribution  
- Near-zero values: Minimal contribution

With graph regularization, connected variables tend to have similar loadings.

### 6.3 P-values and Z-scores

- **p < 0.05**: Strong evidence for genuine slow feature
- **p < 0.01**: Very strong evidence
- **z < -2**: Significantly slower than null expectation (roughly p < 0.025)
- **z < -3**: Highly significant (roughly p < 0.001)

### 6.4 Null Distribution Comparison

Visualizing actual eigenvalues against the null distribution boxplots shows:

- **Below box**: Component is significantly slower than random
- **Within box**: Component slowness consistent with noise
- **Above box**: Component is actually faster than expected (unusual)

---

## 7. Validation and Diagnostics

### 7.1 Cross-Validation

Split time series into training and validation sets (respecting temporal order):

- Fit on training data with selected components
- Evaluate slowness on held-out validation data
- Components should remain slow in validation set

### 7.2 Stability Analysis

Run selection with different block lengths or random seeds:

- Robust features should be selected consistently
- Marginal features may appear/disappear

### 7.3 Reconstruction Error

For k selected components:

- Reconstruct: **X̂** = **Y W**^T + mean
- Error: MSE = ||\*\*X\*\* - **X̂**||²
- High error: May need more components
- Low error: Components capture most variance

---

## 8. References

### Foundational Papers

1. **Wiskott, L., & Sejnowski, T. J. (2002)**. Slow feature analysis: Unsupervised learning of invariances. *Neural Computation*, 14(4), 715-770.
   - Original SFA paper

2. **Horn, J. L. (1965)**. A rationale and test for the number of factors in factor analysis. *Psychometrika*, 30(2), 179-185.
   - Original parallel analysis method

3. **Chung, F. R. (1997)**. Spectral graph theory (Vol. 92). American Mathematical Society.
   - Graph Laplacian theory

### Graph-Based Extensions

1. **Sprekeler, H. (2011)**. On the relation of slow feature analysis and Laplacian eigenmaps. *Neural Computation*, 23(12), 3287-3302.
   - Connections between SFA and graph methods

2. **Blaschke, T., Berkes, P., & Wiskott, L. (2006)**. What is the relation between slow feature analysis and independent component analysis? *Neural Computation*, 18(10), 2495-2508.
   - Theoretical foundations

### Parallel Analysis Applications

1. **Peres-Neto, P. R., Jackson, D. A., & Somers, K. M. (2005)**. How many principal components? Stopping rules for determining the number of non-trivial axes revisited. *Computational Statistics & Data Analysis*, 49(4), 974-997.
   - Comparison of component selection methods

---

## 9. Implementation Notes

### 9.1 Computational Complexity

- **SFA fitting**: O(d³ + Td²) where T = time points, d = dimensions
  - Dominated by eigendecomposition: O(d³)
  - Practical limit: d ~ 1000 without specialized methods

- **Block permutation**: O(T)
  - Very efficient, just array indexing

- **Parallel analysis**: O(K × SFA_cost)
  - Scales linearly with number of permutations
  - Embarrassingly parallel - can distribute across cores

### 9.2 Numerical Stability

- Add small regularization (ε = 10⁻⁸) to covariance matrices
- Use symmetric eigenvalue solvers (scipy.linalg.eigh)
- Normalize components to unit norm
- Check condition numbers of covariance matrices

### 9.3 Scikit-learn Compatibility

Both `GraphSlowFeatureAnalysis` and `SlowFeatureSelector` follow scikit-learn conventions:

- `fit(X, y)` method (y ignored)
- `transform(X)` method
- `fit_transform(X, y)` available
- Compatible with pipelines
- Implements `get_params()` and `set_params()`

---

## 10. Example Applications

### 10.1 Neuroscience

- Extract slow behavioral features from neural recordings
- Graph represents anatomical or functional connectivity
- Block length based on behavioral timescales (seconds to minutes)

### 10.2 Robotics

- Learn invariant representations from sensorimotor data
- Graph represents kinematic constraints
- Block length based on movement timescales

### 10.3 Finance

- Detect slow-moving market factors from high-frequency data
- Graph represents sector relationships or correlation structure
- Block length based on relevant trading timescales (minutes to days)

### 10.4 Climate Science

- Extract slow climate modes from spatiotemporal data
- Graph represents spatial proximity or teleconnections
- Block length based on atmospheric dynamics (weeks to seasons)

---

## 11. Future Extensions

### Potential Improvements

1. **Nonlinear SFA**: Kernel methods or neural network implementations
2. **Online/Incremental SFA**: Update components as new data arrives
3. **Hierarchical SFA**: Multiple timescale decomposition
4. **Sparse SFA**: Add L1 regularization for sparse loadings
5. **Robust SFA**: Handle outliers and non-Gaussian noise
6. **Multi-graph SFA**: Multiple relationship structures simultaneously

---

## Appendix: Mathematical Notation

| Symbol | Description |
|--------|-------------|
| **x**(t) | Input time series, **x** ∈ ℝ^d |
| **y**(t) | Slow features, **y** ∈ ℝ^k |
| **w**_i | Weight vector for i-th component |
| **W** | Weight matrix, columns are weight vectors |
| **C** | Data covariance matrix |
| **Ċ** | Temporal derivative covariance matrix |
| **L** | Graph Laplacian matrix |
| **D** | Degree matrix of graph |
| **W** | Weight/adjacency matrix of graph |
| λ_i | Eigenvalue (slowness) of i-th component |
| λ | Graph regularization parameter |
| T | Number of time points |
| d | Number of input dimensions |
| k | Number of slow feature components |
| L | Block length for permutation |
| K | Number of permutations |
| α | Percentile threshold (e.g., 95) |
| ⟨·⟩_t | Temporal average |
| ẋ | Temporal derivative dx/dt |

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Software Version**: graph_sfa v1.0  
**License**: MIT (or your preferred license)
