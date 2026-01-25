# Week 3: The Matrix Factorization Framework

## Learning Objectives

- Understand the low-rank matrix approximation formulation
- Grasp the concept of latent factors
- Connect MF to SVD and PCA
- Recognize why MF works for recommendation

---

## Core Idea: Latent Factor Models

### The Insight

**User preferences and item characteristics can be represented in a low-dimensional latent space.**

**Example**: Movies
- Latent factors might capture:
  - **Factor 1**: Seriousness (drama ↔ comedy)
  - **Factor 2**: Action level (peaceful ↔ explosive)
  - **Factor 3**: Age appropriateness (kids ↔ adults)
  - **Factor 4**: Fantasy level (realistic ↔ sci-fi)

**User**: Likes serious, action-packed, adult, sci-fi movies
- User vector: $\mathbf{u} = [0.9, 0.8, 0.9, 0.9]$ (high on factors 1,2,3,4)

**Movie (The Matrix)**:
- Movie vector: $\mathbf{v} = [0.7, 0.9, 0.8, 0.95]$ (serious, very action, adult, very sci-fi)

**Prediction**: $\mathbf{u}^T \mathbf{v} = 0.9×0.7 + 0.8×0.9 + 0.9×0.8 + 0.9×0.95 = 2.96$ (high → user will like it!)

---

## Mathematical Formulation

### The Rating Matrix

**Observed data**: User-item rating matrix $R \in \mathbb{R}^{|U| \times |I|}$

$$R = \begin{bmatrix}
r_{11} & r_{12} & ? & r_{14} & ? \\
? & r_{22} & r_{23} & ? & r_{25} \\
r_{31} & ? & ? & r_{34} & r_{35} \\
? & r_{42} & r_{43} & r_{44} & ?
\end{bmatrix}$$

**Properties**:
- Most entries are missing (?)
- Very sparse (99%+ unknown)
- High-dimensional ($|U| \times |I|$)

---

### Low-Rank Approximation

**Goal**: Approximate $R$ with product of two low-rank matrices.

$$R \approx U^T V$$

where:
- $U \in \mathbb{R}^{k \times |U|}$: User latent factor matrix
- $V \in \mathbb{R}^{k \times |I|}$: Item latent factor matrix
- $k \ll \min(|U|, |I|)$: Number of latent factors (typically 20-200)

**Dimensionality Reduction**:
- Original: $|U| \times |I|$ parameters (millions to billions)
- MF: $k \times (|U| + |I|)$ parameters (thousands to millions)
- **Massive compression!**

---

### Element-Wise View

For each user-item pair:

$$\hat{r}_{ui} = \mathbf{u}_u^T \mathbf{v}_i = \sum_{f=1}^k u_{uf} \cdot v_{if}$$

where:
- $\mathbf{u}_u \in \mathbb{R}^k$: User $u$'s latent factor vector
- $\mathbf{v}_i \in \mathbb{R}^k$: Item $i$'s latent factor vector
- $f$: Factor index

**Interpretation**:
- Each factor represents a hidden dimension (genre, mood, complexity, etc.)
- Rating is the dot product of user and item in latent space

---

## Visual Illustration

### 2D Example (k=2 factors)

**Latent Space**:
```
Factor 2 (Action Level)
    ↑
    |    • The Matrix (high action, high sci-fi)
    |    • Inception
    |
    |    • Interstellar
    |
    |        • Cast Away (low action, low sci-fi)
    |    • Forrest Gump
    └──────────────────────────→ Factor 1 (Sci-Fi Level)
```

**Users**:
```
Factor 2
    ↑
    |    • Alice (loves action sci-fi)
    |
    |    • Bob (moderate preferences)
    |
    |    • Carol (prefers drama)
    |
    └──────────────────────────→ Factor 1
```

**Prediction**:
- Alice is close to *The Matrix* in latent space → High predicted rating
- Carol is far from *The Matrix* → Low predicted rating

---

## Connection to SVD (Singular Value Decomposition)

### Standard SVD

For a complete matrix $R$:

$$R = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{|U| \times |U|}$: Left singular vectors (orthogonal)
- $\Sigma \in \mathbb{R}^{|U| \times |I|}$: Diagonal matrix of singular values
- $V \in \mathbb{R}^{|I| \times |I|}$: Right singular vectors (orthogonal)

### Truncated SVD

Keep only top-$k$ singular values:

$$R \approx U_k \Sigma_k V_k^T$$

where:
- $U_k \in \mathbb{R}^{|U| \times k}$: Top-$k$ left singular vectors
- $\Sigma_k \in \mathbb{R}^{k \times k}$: Top-$k$ singular values
- $V_k \in \mathbb{R}^{I| \times k}$: Top-$k$ right singular vectors

**This is a low-rank approximation!**

---

### Why Not Just Use SVD for RecSys?

**Problem 1: Missing Entries**
- SVD requires a complete matrix
- Recommendation matrices are 99%+ sparse
- **Solutions attempted**:
  - Fill missing values with 0 (bad: treats "unknown" as "dislike")
  - Fill with item/user averages (better, but still not ideal)

**Problem 2: Overfitting on Missing Data**
- If we fill missing values, SVD fits to those guesses
- May not generalize to true user preferences

**Problem 3: Not Optimized for Recommendation**
- SVD minimizes reconstruction error on ALL entries
- Recommendation only cares about observed entries
- Want to predict unknown entries well, not reconstruct known entries perfectly

**Matrix Factorization Solution**:
- Only fit to observed entries
- Add regularization to prevent overfitting
- Optimize directly for recommendation quality

---

## Connection to PCA (Principal Component Analysis)

### PCA Perspective

**User matrix as data matrix**: Each column is a user, each row is an item rating.

**PCA**: Find principal components (directions of maximum variance).

**MF**: Learn latent factors that explain variance in user preferences.

**Relationship**:
- PCA finds orthogonal components
- MF finds non-orthogonal factors (more flexible)
- MF optimized for prediction, PCA for variance explanation

---

## Why Does Matrix Factorization Work?

### 1. **Dimensionality Reduction**

**Curse of Dimensionality**:
- Original: $|U| \times |I|$ dimensions
- Sparse data → unreliable estimates

**MF Solution**:
- Project to $k$-dimensional latent space
- $k \ll |U|, |I|$
- Denser representation → more reliable

---

### 2. **Collaborative Signal Capture**

**Example**:
- Alice loves *The Matrix* and *Inception*
- Bob loves *The Matrix* and *Blade Runner*
- Carol loves *Inception* and *Blade Runner*

**Latent Factors**:
- Factor 1 (Sci-Fi): All three movies high
- Factor 2 (Mind-bending): *The Matrix* and *Inception* high, *Blade Runner* medium

**Collaborative Filtering Emerges**:
- Alice, Bob, Carol all have high Sci-Fi factor
- Similar users automatically clustered in latent space

---

### 3. **Generalization to Unseen Pairs**

**Direct CF** (user-based, item-based):
- Need overlap between users/items to compute similarity
- Doesn't generalize beyond observed patterns

**MF**:
- Learns latent factors from all observed data
- Can predict for ANY user-item pair
- Even if no direct overlap, latent factors bridge the gap

**Example**:
- Alice and Bob never rated the same item
- But both have high Sci-Fi factor (learned from other overlaps)
- MF can still infer they have similar tastes

---

### 4. **Implicit Feature Learning**

**No need to manually define features** (genre, director, actors).

**MF automatically discovers**:
- Latent factors that best explain observed ratings
- May capture genre, mood, complexity, popularity, etc.
- Learns from data, not human intuition

---

## Example: 3-User, 4-Movie System

### Observed Ratings

$$R = \begin{bmatrix}
5 & 3 & ? & 1 \\
? & 4 & 5 & ? \\
2 & ? & 1 & 5
\end{bmatrix}$$

### MF with k=2

**User factors** ($U \in \mathbb{R}^{2 \times 3}$):
$$U = \begin{bmatrix}
0.9 & 0.2 & -0.8 \\
0.8 & 0.9 & 0.3
\end{bmatrix}$$

- User 1: [0.9, 0.8] (high on both factors)
- User 2: [0.2, 0.9] (low factor 1, high factor 2)
- User 3: [-0.8, 0.3] (negative factor 1, low factor 2)

**Item factors** ($V \in \mathbb{R}^{2 \times 4}$):
$$V = \begin{bmatrix}
0.8 & 0.5 & 0.9 & -0.7 \\
0.7 & 0.6 & 0.8 & 0.2
\end{bmatrix}$$

- Movie 1: [0.8, 0.7]
- Movie 2: [0.5, 0.6]
- Movie 3: [0.9, 0.8]
- Movie 4: [-0.7, 0.2]

### Predictions

**User 1, Movie 3** (unknown):
$$\hat{r}_{13} = [0.9, 0.8] \cdot [0.9, 0.8] = 0.9×0.9 + 0.8×0.8 = 0.81 + 0.64 = 1.45$$

**After rescaling to 1-5**: $\hat{r}_{13} \approx 4.5$ (predicted rating: 4-5 stars)

**Interpretation**: User 1 will probably like Movie 3 (both high on factors 1 and 2).

---

## Latent Factor Interpretation

### Discovered Factors (Hypothetical)

Looking at learned factors, we might interpret:

**Factor 1**: Sci-Fi vs. Drama
- Movies 1, 3 have high values (sci-fi)
- Movie 4 has negative value (drama)

**Factor 2**: Action vs. Calm
- Movies 2, 3 have high values (action-packed)
- Movie 4 has low value (calm)

**User 1**: Loves sci-fi and action
**User 2**: Moderate on sci-fi, loves action
**User 3**: Dislikes sci-fi, prefers calm dramas

**Note**: Factors are learned, not pre-defined. Interpretation is post-hoc.

---

## Number of Factors (k)

### How to Choose k?

**Too small (k=1-5)**:
- Underfitting: Can't capture complexity
- Low accuracy

**Too large (k=500+)**:
- Overfitting: Memorizes noise
- High variance
- Computational cost

**Typical values**: k = 20-200

**Methods to choose**:
1. **Cross-validation**: Try k=10, 20, 50, 100, 200, pick best on validation set
2. **Elbow method**: Plot RMSE vs. k, look for elbow
3. **Domain knowledge**: More complex domains (movies) need more factors than simple domains (binary likes)

### k vs. Data Size

| Dataset | Users | Items | Interactions | Typical k |
|---------|-------|-------|--------------|-----------|
| MovieLens 100K | 943 | 1,682 | 100K | 10-50 |
| MovieLens 1M | 6,040 | 3,706 | 1M | 50-100 |
| MovieLens 10M | 71,567 | 10,681 | 10M | 100-200 |
| Netflix Prize | 480K | 17K | 100M | 200-500 |

**General rule**: More data → can support more factors

---

## Bias Terms

### Basic MF

$$\hat{r}_{ui} = \mathbf{u}_u^T \mathbf{v}_i$$

**Problem**: Doesn't capture global biases.

**Example**:
- *The Godfather*: average rating 4.5 (popular, highly rated)
- *Gigli*: average rating 1.5 (unpopular, poorly rated)
- User Alice: tends to rate 0.5 stars higher than average
- User Bob: tends to rate 0.5 stars lower than average

**Basic MF forces latent factors to capture these biases** → wastes capacity.

---

### MF with Bias Terms

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i$$

where:
- $\mu$: Global average rating (e.g., 3.5 stars)
- $b_u$: User bias (how much user $u$ deviates from average)
- $b_i$: Item bias (how much item $i$ deviates from average)
- $\mathbf{u}_u^T \mathbf{v}_i$: User-item interaction (latent factors)

**Benefits**:
- **Separates bias from interaction**: Latent factors focus on preferences, not rating scales
- **Better predictions**: Explicitly models known biases
- **Faster convergence**: Easier optimization

**Example**:
- $\mu = 3.5$
- *The Godfather*: $b_i = +1.0$ (rated 1 star above average)
- Alice: $b_u = +0.5$ (rates 0.5 higher than average)
- Latent interaction: $\mathbf{u}_{Alice}^T \mathbf{v}_{Godfather} = 0.2$

**Prediction**: $\hat{r} = 3.5 + 0.5 + 1.0 + 0.2 = 5.2$ (clip to 5.0)

---

## Matrix Factorization Variants Preview

### Basic MF
$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i$$

### SVD++ (Implicit Feedback)
$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{v}_i^T \left(\mathbf{u}_u + |N(u)|^{-0.5} \sum_{j \in N(u)} \mathbf{y}_j \right)$$

Adds implicit feedback from items user interacted with.

### TimeSVD++ (Temporal Dynamics)
$$\hat{r}_{ui}(t) = \mu + b_u(t) + b_i(t) + \mathbf{v}_i^T \mathbf{u}_u(t)$$

Models how preferences change over time.

### Factorization Machines (Feature Interactions)
$$\hat{r}_{ui} = w_0 + \sum_j w_j x_j + \sum_{j=1}^n \sum_{k=j+1}^n \langle \mathbf{v}_j, \mathbf{v}_k \rangle x_j x_k$$

Generalizes MF to arbitrary features.

**Covered in detail**: See **advanced-variants.md**

---

## Summary

**Matrix Factorization**:
- Approximate sparse rating matrix $R$ with low-rank product $U^T V$
- Learn latent factors for users and items
- $k$ factors capture hidden dimensions (genre, mood, complexity)
- Prediction: $\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i$

**Why it works**:
1. Dimensionality reduction (dense representation)
2. Collaborative signal capture (similar users/items cluster)
3. Generalization (predict beyond observed overlaps)
4. Implicit feature learning (no manual feature engineering)

**Key parameters**:
- $k$: Number of latent factors (20-200 typical)
- $\lambda$: Regularization strength (cross-validation)
- Bias terms: Capture global, user, item biases

**Next**:
- **optimization.md**: How to learn U and V from data
- **algorithms.md**: SGD, ALS, coordinate descent
- **advanced-variants.md**: SVD++, TimeSVD++, Factorization Machines

---

## References

1. **Koren, Y., Bell, R., & Volinsky, C. (2009)**. "Matrix factorization techniques for recommender systems". *Computer, IEEE*, 42(8), 30-37.
   - **Essential reading**: Comprehensive overview

2. **Funk, S. (2006)**. "Netflix Update: Try This at Home". Blog post.
   - Original Funk-SVD (gradient descent for MF)

3. **Salakhutdinov, R., & Mnih, A. (2008)**. "Probabilistic matrix factorization". *NIPS*.
   - Bayesian perspective on MF
