# Week 3: Optimization Algorithms for Matrix Factorization

## Learning Objectives

- Master Stochastic Gradient Descent (SGD) for MF
- Understand Alternating Least Squares (ALS)
- Learn when to use each algorithm
- Implement optimization techniques in practice

---

## The Optimization Problem (Recap)

**Goal**: Learn user and item latent factors.

**Objective function**:
$$\min_{U, V, b} \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \hat{r}_{ui})^2 + \lambda \left( ||U||^2_F + ||V||^2_F + ||b||^2 \right)$$

where:
- $\mathcal{K}$ = set of observed ratings
- $\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i$
- $\lambda$ = regularization parameter
- $||\cdot||^2_F$ = Frobenius norm (sum of squared elements)

**Challenges**:
- Non-convex (not globally optimal solution guaranteed)
- High-dimensional (millions of parameters)
- Sparse data (99%+ missing values)

---

## Algorithm 1: Stochastic Gradient Descent (SGD)

### Intuition

**Gradient descent**: Move in direction of steepest descent.

**Stochastic**: Update parameters after each training example (not full batch).

**Why stochastic?**
- **Fast**: Don't need to compute gradients for all data
- **Memory efficient**: Process one rating at a time
- **Effective for sparse data**: Most useful for RecSys

---

### Algorithm

```
Initialize U, V, b randomly (small values, e.g., ~ N(0, 0.01))

For epoch = 1 to max_epochs:
    Shuffle training data (observed ratings)

    For each rating (u, i, r_ui) in training data:
        # Predict rating
        r̂_ui = μ + b_u + b_i + u_u^T v_i

        # Compute error
        e_ui = r_ui - r̂_ui

        # Update parameters (gradient ascent to minimize squared error)
        b_u ← b_u + α · (e_ui - λ · b_u)
        b_i ← b_i + α · (e_ui - λ · b_i)
        u_u ← u_u + α · (e_ui · v_i - λ · u_u)
        v_i ← v_i + α · (e_ui · u_u - λ · v_i)

    # Optional: Evaluate on validation set
    if val_RMSE stopped improving:
        break  # Early stopping
```

**Parameters**:
- $\alpha$ = learning rate (e.g., 0.005-0.01)
- $\lambda$ = regularization (e.g., 0.01-0.1)
- max_epochs = typically 10-100

---

### Gradient Derivation

**Loss for single rating**:
$$\mathcal{L}_{ui} = (r_{ui} - \hat{r}_{ui})^2 + \lambda(||u_u||^2 + ||v_i||^2 + b_u^2 + b_i^2)$$

**Gradients**:

$$\frac{\partial \mathcal{L}_{ui}}{\partial b_u} = -2(r_{ui} - \hat{r}_{ui}) + 2\lambda b_u$$

$$\frac{\partial \mathcal{L}_{ui}}{\partial b_i} = -2(r_{ui} - \hat{r}_{ui}) + 2\lambda b_i$$

$$\frac{\partial \mathcal{L}_{ui}}{\partial \mathbf{u}_u} = -2(r_{ui} - \hat{r}_{ui}) \mathbf{v}_i + 2\lambda \mathbf{u}_u$$

$$\frac{\partial \mathcal{L}_{ui}}{\partial \mathbf{v}_i} = -2(r_{ui} - \hat{r}_{ui}) \mathbf{u}_u + 2\lambda \mathbf{v}_i$$

**Update rule** (ignoring constant 2):
$$\theta \leftarrow \theta - \alpha \cdot \frac{\partial \mathcal{L}}{\partial \theta}$$

Substituting:
$$b_u \leftarrow b_u + \alpha \cdot (e_{ui} - \lambda \cdot b_u)$$

(where $e_{ui} = r_{ui} - \hat{r}_{ui}$)

---

### Implementation (Python)

```python
import numpy as np

class SGD_MF:
    def __init__(self, n_factors=20, n_epochs=20, lr=0.005, reg=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr  # Learning rate (alpha)
        self.reg = reg  # Regularization (lambda)

    def fit(self, ratings):
        """
        Train matrix factorization with SGD.

        Args:
            ratings: List of (user_id, item_id, rating) tuples
        """
        n_users = max(r[0] for r in ratings) + 1
        n_items = max(r[1] for r in ratings) + 1

        # Initialize parameters
        self.global_mean = np.mean([r[2] for r in ratings])
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # Training loop
        for epoch in range(self.n_epochs):
            np.random.shuffle(ratings)  # Shuffle for stochasticity

            for user, item, rating in ratings:
                # Predict
                pred = self.predict(user, item)

                # Compute error
                error = rating - pred

                # Update biases
                self.user_bias[user] += self.lr * (error - self.reg * self.user_bias[user])
                self.item_bias[item] += self.lr * (error - self.reg * self.item_bias[item])

                # Update factors
                user_factors_old = self.user_factors[user].copy()
                self.user_factors[user] += self.lr * (error * self.item_factors[item] - self.reg * self.user_factors[user])
                self.item_factors[item] += self.lr * (error * user_factors_old - self.reg * self.item_factors[item])

            # Optional: Compute training RMSE
            if (epoch + 1) % 5 == 0:
                rmse = self.compute_rmse(ratings)
                print(f"Epoch {epoch+1}: RMSE = {rmse:.4f}")

    def predict(self, user, item):
        """Predict rating for user-item pair."""
        return (self.global_mean +
                self.user_bias[user] +
                self.item_bias[item] +
                np.dot(self.user_factors[user], self.item_factors[item]))

    def compute_rmse(self, ratings):
        """Compute RMSE on given ratings."""
        errors = [(r - self.predict(u, i))**2 for u, i, r in ratings]
        return np.sqrt(np.mean(errors))
```

---

### Learning Rate Scheduling

**Problem**: Fixed learning rate can be too large (divergence) or too small (slow convergence).

**Solution**: Decay learning rate over time.

**Common schedules**:

**1. Step decay**:
$$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / k \rfloor}$$

Example: $\alpha_0 = 0.01$, $\gamma = 0.5$, $k = 10$ epochs
- Epochs 0-9: $\alpha = 0.01$
- Epochs 10-19: $\alpha = 0.005$
- Epochs 20-29: $\alpha = 0.0025$

**2. Exponential decay**:
$$\alpha_t = \alpha_0 \cdot e^{-\beta t}$$

**3. Inverse time decay**:
$$\alpha_t = \frac{\alpha_0}{1 + \beta t}$$

**4. Cosine annealing** (modern deep learning):
$$\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})(1 + \cos(\frac{t}{T} \pi))$$

**Recommendation**: Start with step decay (simplest, effective).

---

### SGD Variants

**1. Mini-Batch SGD**:
- Update after $B$ examples (batch size)
- More stable than single-sample
- Parallelizable
- Typical batch size: 32-256

**2. SGD with Momentum**:
$$v_t = \beta v_{t-1} + \alpha \nabla \mathcal{L}$$
$$\theta_t = \theta_{t-1} - v_t$$

- Accumulates velocity in gradient direction
- Faster convergence, less oscillation
- Typical $\beta = 0.9$

**3. Adam (Adaptive Moment Estimation)**:
- Adaptive learning rate per parameter
- Combines momentum + RMSProp
- Popular in deep learning

**For MF**: Vanilla SGD usually sufficient. Adam can help for faster convergence.

---

## Algorithm 2: Alternating Least Squares (ALS)

### Intuition

**Key insight**: If we fix $U$, the problem is quadratic in $V$ (and vice versa).

**Quadratic problems** have closed-form solution (no iterative gradient descent needed).

**Alternating**:
1. Fix $V$, solve for $U$ (closed-form)
2. Fix $U$, solve for $V$ (closed-form)
3. Repeat until convergence

---

### Why ALS?

**Advantages over SGD**:
1. **No learning rate tuning**: Closed-form solution
2. **Parallelizable**: Each user/item update independent
3. **Implicit feedback**: Naturally handles all missing entries
4. **Stable**: No risk of divergence

**Disadvantages**:
1. **Memory**: Requires solving linear systems
2. **Slower per iteration**: Closed-form more expensive than single gradient step
3. **Explicit feedback**: Less common than SGD for ratings

**Best for**: Implicit feedback, distributed systems (Spark), large-scale

---

### Algorithm

```
Initialize V randomly

For iteration = 1 to max_iterations:
    # Step 1: Fix V, solve for all users
    For each user u:
        # Solve: u_u = argmin Σ_i (r_ui - u_u^T v_i)^2 + λ||u_u||^2
        # Closed-form solution (see derivation below)
        u_u = (V^T V + λI)^(-1) V^T r_u

    # Step 2: Fix U, solve for all items
    For each item i:
        # Solve: v_i = argmin Σ_u (r_ui - u_u^T v_i)^2 + λ||v_i||^2
        v_i = (U^T U + λI)^(-1) U^T r_i

    # Check convergence (e.g., RMSE change < threshold)
    if converged:
        break
```

**Note**: Steps 1 and 2 can be parallelized (all users updated independently, all items independently).

---

### Closed-Form Solution Derivation

**Objective** (for fixed $V$, solving for $\mathbf{u}_u$):
$$\min_{\mathbf{u}_u} \sum_{i: r_{ui} \text{ observed}} (r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda ||\mathbf{u}_u||^2$$

**Matrix form**:
Let:
- $\mathbf{r}_u$ = vector of user $u$'s ratings (only observed items)
- $V_u$ = matrix of item factors for items rated by user $u$

$$\min_{\mathbf{u}_u} ||\mathbf{r}_u - V_u^T \mathbf{u}_u||^2 + \lambda ||\mathbf{u}_u||^2$$

**Expand**:
$$\mathcal{L} = (\mathbf{r}_u - V_u^T \mathbf{u}_u)^T (\mathbf{r}_u - V_u^T \mathbf{u}_u) + \lambda \mathbf{u}_u^T \mathbf{u}_u$$

**Take gradient** w.r.t. $\mathbf{u}_u$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_u} = -2 V_u (\mathbf{r}_u - V_u^T \mathbf{u}_u) + 2\lambda \mathbf{u}_u$$

**Set to zero**:
$$V_u \mathbf{r}_u - V_u V_u^T \mathbf{u}_u + \lambda \mathbf{u}_u = 0$$

$$(V_u V_u^T + \lambda I) \mathbf{u}_u = V_u \mathbf{r}_u$$

**Solve**:
$$\mathbf{u}_u = (V_u V_u^T + \lambda I)^{-1} V_u \mathbf{r}_u$$

**Same for items** (symmetrically).

---

### Implementation (Python)

```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

class ALS_MF:
    def __init__(self, n_factors=20, n_iterations=10, reg=0.1):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg = reg

    def fit(self, ratings_matrix):
        """
        Train with ALS.

        Args:
            ratings_matrix: Sparse matrix (users × items), scipy.sparse format
        """
        n_users, n_items = ratings_matrix.shape

        # Initialize item factors randomly
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))

        # ALS iterations
        for iteration in range(self.n_iterations):
            # Step 1: Fix items, solve for users
            self.user_factors = self._als_step(ratings_matrix, self.item_factors, self.reg)

            # Step 2: Fix users, solve for items
            self.item_factors = self._als_step(ratings_matrix.T, self.user_factors, self.reg)

            # Optional: Compute RMSE
            if (iteration + 1) % 2 == 0:
                rmse = self.compute_rmse(ratings_matrix)
                print(f"Iteration {iteration+1}: RMSE = {rmse:.4f}")

    def _als_step(self, ratings, fixed_factors, reg):
        """
        One ALS step: solve for all users (or items).

        Args:
            ratings: Sparse matrix (n × m)
            fixed_factors: Fixed factor matrix (m × k)
            reg: Regularization

        Returns:
            Updated factor matrix (n × k)
        """
        n, m = ratings.shape
        k = fixed_factors.shape[1]
        new_factors = np.zeros((n, k))

        # Precompute V^T V + λI (same for all users)
        YtY = fixed_factors.T @ fixed_factors
        lambdaI = np.eye(k) * reg

        for idx in range(n):
            # Get ratings for this user (sparse row)
            r = ratings[idx].toarray().flatten()
            observed = r > 0  # Which items rated

            if observed.sum() == 0:
                continue  # User has no ratings

            # Get item factors for observed ratings
            V_u = fixed_factors[observed]

            # Solve: (V_u^T V_u + λI) u = V_u^T r_u
            # Optimized: Use precomputed YtY for dense case
            # For sparse, compute V_u^T V_u directly
            A = V_u.T @ V_u + lambdaI
            b = V_u.T @ r[observed]

            new_factors[idx] = np.linalg.solve(A, b)

        return new_factors

    def predict(self, user, item):
        """Predict rating."""
        return np.dot(self.user_factors[user], self.item_factors[item])

    def compute_rmse(self, ratings_matrix):
        """Compute RMSE on observed ratings."""
        predictions = self.user_factors @ self.item_factors.T
        observed = ratings_matrix.toarray() > 0
        errors = (ratings_matrix.toarray()[observed] - predictions[observed])**2
        return np.sqrt(np.mean(errors))
```

---

### Implicit Feedback with ALS (WRMF)

**For implicit feedback** (clicks, views, no ratings):
- Treat all missing entries as negative (not unknown)
- Weight by confidence

**Objective**:
$$\min_{U,V} \sum_{u,i} c_{ui}(p_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda(||U||^2 + ||V||^2)$$

where:
- $p_{ui} \in \{0,1\}$: 1 if interaction, 0 otherwise
- $c_{ui}$: Confidence (e.g., $c_{ui} = 1 + \alpha \cdot r_{ui}$ where $r_{ui}$ is interaction count)

**ALS still applies** (closed-form solution with weighted least squares).

**Popular for**: Spotify, Netflix, YouTube (implicit feedback dominant).

---

## Algorithm 3: Coordinate Descent

### Intuition

**Update one parameter at a time**, holding all others fixed.

**For MF**:
- Update $u_{u,1}$, holding all other parameters fixed
- Update $u_{u,2}$, holding all others fixed
- ...
- Update $v_{i,k}$, holding all others fixed
- Repeat until convergence

**Rarely used in practice** (SGD and ALS more popular), but simple to understand.

---

## Comparison of Algorithms

| Aspect | SGD | ALS | Coordinate Descent |
|--------|-----|-----|-------------------|
| **Convergence Speed** | Fast (per epoch) | Slow (per iteration) | Slow |
| **Per-Iteration Cost** | Low (one gradient step) | High (solve linear system) | Medium |
| **Parallelization** | Hard (sequential updates) | Easy (users/items independent) | Medium |
| **Learning Rate** | Must tune | No learning rate | No learning rate |
| **Implicit Feedback** | Tricky (need sampling) | Natural (WRMF) | Natural |
| **Memory** | Low | Medium (store matrices) | Low |
| **Best For** | Explicit ratings, medium-scale | Implicit feedback, distributed | Rare use |

---

## Convergence and Early Stopping

### Monitoring Convergence

**Metrics to track**:
1. **Training RMSE**: Should decrease
2. **Validation RMSE**: Should decrease, then plateau/increase (overfitting)
3. **Parameter change**: $||\theta_t - \theta_{t-1}||$ (should decrease)

**Convergence criterion**:
- Validation RMSE stops improving for $n$ epochs (e.g., $n=5$)
- Or: Parameter change < threshold (e.g., $10^{-4}$)

---

### Early Stopping

**Prevents overfitting**:
- Monitor validation RMSE
- If increases for $k$ consecutive epochs → stop

**Implementation**:
```python
best_rmse = float('inf')
patience_counter = 0
patience = 5  # Stop after 5 epochs without improvement

for epoch in range(max_epochs):
    # Train
    train_one_epoch()

    # Evaluate
    val_rmse = evaluate_validation()

    if val_rmse < best_rmse:
        best_rmse = val_rmse
        save_model()  # Save best model
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

load_best_model()  # Restore best model
```

---

## Hyperparameter Tuning

### Key Hyperparameters

**1. Number of Factors ($k$)**:
- Range: 20-200 (typically)
- Too small: Underfitting
- Too large: Overfitting, slow
- **Tune**: Cross-validation

**2. Regularization ($\lambda$)**:
- Range: 0.001-0.1
- Too small: Overfitting (especially with many factors)
- Too large: Underfitting
- **Tune**: Grid search

**3. Learning Rate ($\alpha$, for SGD)**:
- Range: 0.001-0.01
- Too small: Slow convergence
- Too large: Divergence
- **Tune**: Start with 0.005, adjust if diverging/slow

**4. Number of Epochs**:
- Range: 10-100
- **Tune**: Early stopping (don't set manually)

---

### Grid Search Example

```python
from sklearn.model_selection import KFold

# Define parameter grid
param_grid = {
    'n_factors': [20, 50, 100],
    'lr': [0.001, 0.005, 0.01],
    'reg': [0.01, 0.02, 0.05],
}

# Cross-validation
kf = KFold(n_splits=5)
best_params = None
best_rmse = float('inf')

for n_factors in param_grid['n_factors']:
    for lr in param_grid['lr']:
        for reg in param_grid['reg']:
            rmses = []
            for train_idx, val_idx in kf.split(ratings):
                train_data = ratings[train_idx]
                val_data = ratings[val_idx]

                model = SGD_MF(n_factors=n_factors, lr=lr, reg=reg)
                model.fit(train_data)
                rmse = model.compute_rmse(val_data)
                rmses.append(rmse)

            avg_rmse = np.mean(rmses)
            print(f"Factors={n_factors}, LR={lr}, Reg={reg}: RMSE={avg_rmse:.4f}")

            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_params = (n_factors, lr, reg)

print(f"Best params: {best_params}, Best RMSE: {best_rmse:.4f}")
```

---

## Practical Tips

### 1. Initialization

**Random initialization**:
```python
user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
```

**Why small variance (0.1)?**
- Large initial values → large gradients → instability
- Small values → gradual learning

**Alternative**: Xavier/He initialization (deep learning)
$$\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}$$

---

### 2. Bias Term Importance

**Always include bias terms** ($\mu$, $b_u$, $b_i$).

**Why?**
- Captures rating scale differences
- Reduces factor burden
- Faster convergence
- Better accuracy (typically 5-10% RMSE improvement)

---

### 3. Regularization is Critical

**Without regularization**:
- Overfitting on sparse data
- Poor generalization

**With regularization**:
- Prevents overfitting
- Smooths factors
- Improves validation performance

**Tip**: Start with $\lambda = 0.02$, tune from there.

---

### 4. Shuffling Training Data

**For SGD, shuffle each epoch**:
- Breaks correlation between consecutive samples
- Improves convergence
- Standard practice

```python
for epoch in range(n_epochs):
    np.random.shuffle(ratings)  # Important!
    for user, item, rating in ratings:
        # Update
        ...
```

---

## Summary

**Three main algorithms** for matrix factorization:

**1. SGD (Stochastic Gradient Descent)**:
- Most common for explicit ratings
- Fast per iteration
- Requires learning rate tuning
- Sequential updates (hard to parallelize)

**2. ALS (Alternating Least Squares)**:
- Best for implicit feedback
- Parallelizable (Spark)
- No learning rate
- Slower per iteration

**3. Coordinate Descent**:
- Rarely used
- Simple to understand

**Recommendations**:
- **Explicit ratings (MovieLens, Netflix)**: Start with SGD
- **Implicit feedback (Spotify, YouTube)**: Use ALS with WRMF
- **Distributed systems (Spark)**: ALS is natural fit

**Key hyperparameters**:
- Factors $k$: 20-200
- Regularization $\lambda$: 0.01-0.1
- Learning rate $\alpha$ (SGD): 0.001-0.01
- Early stopping: Monitor validation RMSE

**Next**:
- **advanced-variants.md**: SVD++, TimeSVD++, Factorization Machines
- **code-examples.md**: Full implementations and comparisons

---

## References

1. **Koren, Y. (2008)**. "Factorization meets the neighborhood". *KDD*.
2. **Hu, Y., Koren, Y., & Volinsky, C. (2008)**. "Collaborative filtering for implicit feedback datasets". *ICDM*. (ALS for implicit feedback)
3. **Zhou, Y., et al. (2008)**. "Large-scale parallel collaborative filtering for the Netflix Prize". *AAIM*. (Parallelizing SGD)
