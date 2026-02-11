# Week 3: Matrix Factorization - Optimization

## Overview

This document covers the **optimization problem** underlying matrix factorization for collaborative filtering. We'll explore the objective function, gradient derivation, regularization strategies, and practical considerations for training effective recommendation models.

**Prerequisites**: framework.md, algorithms.md (for SGD and ALS)

---

## Learning Objectives

By the end of this section, you will:
- Understand the MF optimization formulation
- Derive gradients for SGD updates
- Recognize the role of regularization
- Identify common pitfalls (overfitting, underfitting)
- Implement MF optimization from scratch

---

## The Problem: Why Is MF Optimization Tricky?

*Before we dive into equations, let's understand what makes this problem challenging.*

**Imagine you're trying to solve a puzzle** where you need to find two matrices $U$ and $V$ such that their product approximates the observed ratings:

$$R \approx U^T V$$

**Challenge 1: Missing Data**
- 99% of the matrix is empty (users haven't rated most items)
- We can ONLY learn from the 1% we observe
- But we want to predict the 99% we haven't seen!

**Challenge 2: Non-Convexity**
- The problem has many local minima
- Different random initializations can lead to different solutions
- We need to be careful about where we start

**Challenge 3: The Chicken-and-Egg Problem**
- To find good user vectors $\mathbf{u}_u$, we need to know item vectors $\mathbf{v}_i$
- To find good item vectors $\mathbf{v}_i$, we need to know user vectors $\mathbf{u}_u$
- Which do we solve first?

*Keep these challenges in mind* as we develop the solution.

---

## The Basic Optimization Problem

### Problem Setup

**Given**:
- User-item rating matrix $R \in \mathbb{R}^{|U| \times |I|}$ (sparse, mostly unobserved)
- Observed entries: $(u, i, r_{ui})$ where $r_{ui}$ is the rating

**Goal**: Find low-rank factorization
$$R \approx U^T V$$

where:
- $U \in \mathbb{R}^{k \times |U|}$: User latent factor matrix
- $V \in \mathbb{R}^{k \times |I|}$: Item latent factor matrix
- $k$: Number of latent factors (embedding dimension)

**Prediction**:
$$\hat{r}_{ui} = \mathbf{u}_u^T \mathbf{v}_i = \sum_{f=1}^k u_{uf} \cdot v_{fi}$$

---

## Objective Function

### Squared Error Loss

**Basic formulation** (without regularization):

$$\min_{U,V} \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2$$

where $\mathcal{K}$ = set of observed ratings (the key indices).

**Critical insight**: We ONLY sum over observed entries, not the entire matrix.

**Why?**
- Full matrix is mostly empty (99%+ sparsity)
- Missing values $\neq$ zero ratings
- Missing = "user hasn't seen item" (not "user dislikes item")

---

### With Regularization

**Complete objective function**:

$$J(U, V) = \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda \left( \sum_{u} \|\mathbf{u}_u\|^2 + \sum_{i} \|\mathbf{v}_i\|^2 \right)$$

**Components**:
1. **Data term**: $\sum_{(u,i) \in \mathcal{K}} (r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2$
   - Measures how well we fit observed ratings
   - Lower = better fit

2. **Regularization term**: $\lambda \left( \sum_{u} \|\mathbf{u}_u\|^2 + \sum_{i} \|\mathbf{v}_i\|^2 \right)$
   - Penalizes large latent factor values
   - Prevents overfitting
   - $\lambda$ controls strength (higher = more regularization)

**Notation**:
- $\|\mathbf{u}_u\|^2 = \sum_{f=1}^k u_{uf}^2$: L2 norm squared
- This is also called **Frobenius norm** for matrices

---

## With Bias Terms

**Real-world systems** add bias terms to capture baseline effects:

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i$$

where:
- $\mu$: Global average rating
- $b_u$: User bias (some users rate higher/lower on average)
- $b_i$: Item bias (some items are universally liked/disliked)

**Objective with biases**:

$$J = \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \mu - b_u - b_i - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda \left( \sum_u (\|\mathbf{u}_u\|^2 + b_u^2) + \sum_i (\|\mathbf{v}_i\|^2 + b_i^2) \right)$$

**Why biases matter**:
- User 1 always rates 1 star higher → $b_u = +1$
- Item "The Godfather" is universally loved → $b_i = +0.8$
- Separates systematic effects from true preferences

**Example**:
- User's average: 4.2 stars → $b_u = 0.2$ (if global $\mu = 4.0$)
- Movie's average: 4.5 stars → $b_i = 0.5$
- Predicted baseline: $\mu + b_u + b_i = 4.0 + 0.2 + 0.5 = 4.7$
- MF part learns deviations from this baseline

---

## Gradient Derivation

### For User Factors

Take derivative of $J$ with respect to $\mathbf{u}_u$:

$$\frac{\partial J}{\partial \mathbf{u}_u} = \sum_{i: (u,i) \in \mathcal{K}} \frac{\partial}{\partial \mathbf{u}_u} \left[ (r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 \right] + \lambda \frac{\partial}{\partial \mathbf{u}_u} \|\mathbf{u}_u\|^2$$

**Chain rule**:

$$\frac{\partial}{\partial \mathbf{u}_u} (r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 = 2(r_{ui} - \mathbf{u}_u^T \mathbf{v}_i) \cdot (-\mathbf{v}_i) = -2 e_{ui} \mathbf{v}_i$$

where $e_{ui} = r_{ui} - \hat{r}_{ui}$ is the prediction error.

**Regularization**:

$$\frac{\partial}{\partial \mathbf{u}_u} \|\mathbf{u}_u\|^2 = 2\mathbf{u}_u$$

**Combined**:

$$\frac{\partial J}{\partial \mathbf{u}_u} = \sum_{i: (u,i) \in \mathcal{K}} (-2 e_{ui} \mathbf{v}_i) + 2\lambda \mathbf{u}_u$$

$$= -2 \sum_{i: (u,i) \in \mathcal{K}} e_{ui} \mathbf{v}_i + 2\lambda \mathbf{u}_u$$

---

### For Item Factors

By symmetry:

$$\frac{\partial J}{\partial \mathbf{v}_i} = -2 \sum_{u: (u,i) \in \mathcal{K}} e_{ui} \mathbf{u}_u + 2\lambda \mathbf{v}_i$$

---

### For Biases

**User bias** $b_u$:

$$\frac{\partial J}{\partial b_u} = \sum_{i: (u,i) \in \mathcal{K}} (-2 e_{ui}) + 2\lambda b_u = -2 \sum_{i: (u,i) \in \mathcal{K}} e_{ui} + 2\lambda b_u$$

**Item bias** $b_i$:

$$\frac{\partial J}{\partial b_i} = -2 \sum_{u: (u,i) \in \mathcal{K}} e_{ui} + 2\lambda b_i$$

---

## SGD Update Rules

### Stochastic Gradient Descent

For each rating $(u, i, r_{ui})$:

1. **Compute prediction**:
   $$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i$$

2. **Compute error**:
   $$e_{ui} = r_{ui} - \hat{r}_{ui}$$

3. **Update user factors**:
   $$\mathbf{u}_u \leftarrow \mathbf{u}_u + \gamma (e_{ui} \cdot \mathbf{v}_i - \lambda \mathbf{u}_u)$$

4. **Update item factors**:
   $$\mathbf{v}_i \leftarrow \mathbf{v}_i + \gamma (e_{ui} \cdot \mathbf{u}_u - \lambda \mathbf{v}_i)$$

5. **Update biases**:
   $$b_u \leftarrow b_u + \gamma (e_{ui} - \lambda b_u)$$
   $$b_i \leftarrow b_i + \gamma (e_{ui} - \lambda b_i)$$

where:
- $\gamma$: Learning rate (step size)
- $\lambda$: Regularization parameter

---

### The Intuition: What Does the Gradient Mean?

*Before I show you numbers, let's develop intuition.*

**The user update**: $\mathbf{u}_u \leftarrow \mathbf{u}_u + \gamma \cdot e_{ui} \cdot \mathbf{v}_i$

*What happens when we underpredict?* ($e_{ui} > 0$, actual rating higher than prediction)

The update adds $\gamma \cdot e_{ui} \cdot \mathbf{v}_i$ to $\mathbf{u}_u$. Since $e_{ui} > 0$:
- We're adding a positive multiple of $\mathbf{v}_i$ to $\mathbf{u}_u$
- This moves $\mathbf{u}_u$ **toward** $\mathbf{v}_i$ in latent space
- Next time, $\mathbf{u}_u^T \mathbf{v}_i$ will be larger → higher prediction

*What happens when we overpredict?* ($e_{ui} < 0$)

- We're adding a negative multiple of $\mathbf{v}_i$
- This moves $\mathbf{u}_u$ **away from** $\mathbf{v}_i$
- Next time, prediction will be lower

**Think of it like this**: We're adjusting the user's position in "preference space" to better match their actual behavior. Users who liked an item should be close to that item; users who disliked it should be far away.

---

### Numerical SGD Walkthrough: One Complete Update

*Let's trace through a single SGD update with actual numbers.*

**Before the update:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| $\mu$ | 3.5 | Global mean rating |
| $b_u$ | +0.2 | User 7 rates 0.2 stars above average |
| $b_i$ | -0.3 | Movie 42 is 0.3 stars below average |
| $\mathbf{u}_u$ | [0.5, -0.3, 0.8] | User 7's latent vector (k=3) |
| $\mathbf{v}_i$ | [0.6, 0.4, 0.2] | Movie 42's latent vector |
| $\gamma$ | 0.01 | Learning rate |
| $\lambda$ | 0.02 | Regularization strength |

**The rating**: User 7 rated Movie 42 as **4.5 stars** (actual $r_{ui} = 4.5$)

---

**Step 1: Compute prediction**

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i$$

Dot product:
$$\mathbf{u}_u^T \mathbf{v}_i = 0.5 \times 0.6 + (-0.3) \times 0.4 + 0.8 \times 0.2 = 0.30 - 0.12 + 0.16 = 0.34$$

Prediction:
$$\hat{r}_{ui} = 3.5 + 0.2 + (-0.3) + 0.34 = 3.74$$

---

**Step 2: Compute error**

$$e_{ui} = r_{ui} - \hat{r}_{ui} = 4.5 - 3.74 = +0.76$$

*We underpredicted by 0.76 stars!* The user liked this movie more than we expected.

---

**Step 3: Update user vector** $\mathbf{u}_u$

$$\mathbf{u}_u^{new} = \mathbf{u}_u + \gamma (e_{ui} \cdot \mathbf{v}_i - \lambda \mathbf{u}_u)$$

Compute the gradient direction:
$$e_{ui} \cdot \mathbf{v}_i = 0.76 \times [0.6, 0.4, 0.2] = [0.456, 0.304, 0.152]$$

Regularization term:
$$\lambda \mathbf{u}_u = 0.02 \times [0.5, -0.3, 0.8] = [0.01, -0.006, 0.016]$$

Combined:
$$e_{ui} \cdot \mathbf{v}_i - \lambda \mathbf{u}_u = [0.456-0.01, 0.304-(-0.006), 0.152-0.016] = [0.446, 0.310, 0.136]$$

Apply learning rate:
$$\mathbf{u}_u^{new} = [0.5, -0.3, 0.8] + 0.01 \times [0.446, 0.310, 0.136]$$
$$= [0.5 + 0.00446, -0.3 + 0.0031, 0.8 + 0.00136]$$
$$= [0.504, -0.297, 0.801]$$

*Notice*: The user vector moved toward the item vector (all components increased in the direction of $\mathbf{v}_i$).

---

**Step 4: Update item vector** $\mathbf{v}_i$

$$\mathbf{v}_i^{new} = \mathbf{v}_i + \gamma (e_{ui} \cdot \mathbf{u}_u - \lambda \mathbf{v}_i)$$

Following the same logic:
$$e_{ui} \cdot \mathbf{u}_u = 0.76 \times [0.5, -0.3, 0.8] = [0.38, -0.228, 0.608]$$
$$\lambda \mathbf{v}_i = 0.02 \times [0.6, 0.4, 0.2] = [0.012, 0.008, 0.004]$$

$$\mathbf{v}_i^{new} = [0.6, 0.4, 0.2] + 0.01 \times ([0.38, -0.228, 0.608] - [0.012, 0.008, 0.004])$$
$$= [0.6, 0.4, 0.2] + 0.01 \times [0.368, -0.236, 0.604]$$
$$= [0.604, 0.398, 0.206]$$

---

**Step 5: Update biases**

$$b_u^{new} = b_u + \gamma(e_{ui} - \lambda b_u) = 0.2 + 0.01 \times (0.76 - 0.02 \times 0.2)$$
$$= 0.2 + 0.01 \times 0.756 = 0.2076$$

$$b_i^{new} = b_i + \gamma(e_{ui} - \lambda b_i) = -0.3 + 0.01 \times (0.76 - 0.02 \times (-0.3))$$
$$= -0.3 + 0.01 \times 0.766 = -0.292$$

---

**After the update - Let's verify!**

New prediction:
$$\hat{r}_{ui}^{new} = 3.5 + 0.2076 + (-0.292) + [0.504, -0.297, 0.801]^T [0.604, 0.398, 0.206]$$

New dot product:
$$= 0.504 \times 0.604 + (-0.297) \times 0.398 + 0.801 \times 0.206$$
$$= 0.304 - 0.118 + 0.165 = 0.351$$

$$\hat{r}_{ui}^{new} = 3.5 + 0.2076 - 0.292 + 0.351 = 3.767$$

*Wait, that's only slightly higher than before (3.74 → 3.77)?*

That's because we used a small learning rate ($\gamma = 0.01$). Over many iterations, these small steps accumulate. After 20 epochs with thousands of ratings, the prediction will converge much closer to 4.5.

---

### Summary of Update Intuition

| Condition | Error $e_{ui}$ | What Happens |
|-----------|---------------|--------------|
| Underpredicted | $e_{ui} > 0$ | Move $\mathbf{u}_u$ toward $\mathbf{v}_i$, increase biases |
| Overpredicted | $e_{ui} < 0$ | Move $\mathbf{u}_u$ away from $\mathbf{v}_i$, decrease biases |
| Perfect prediction | $e_{ui} = 0$ | Only regularization applies (shrink toward zero) |

---

## Regularization Strategies

### L2 Regularization (Ridge)

**Form**: $\lambda \|\mathbf{u}_u\|^2$

**Effect**:
- Penalizes large factor values
- Smooth, differentiable
- **Most common** in MF

**Gradient**: $2\lambda \mathbf{u}_u$ (easy to compute)

---

### L1 Regularization (Lasso)

**Form**: $\lambda \|\mathbf{u}_u\|_1 = \lambda \sum_{f} |u_{uf}|$

**Effect**:
- Encourages sparsity (many factors → 0)
- Compact models

**Gradient**: $\lambda \cdot \text{sign}(\mathbf{u}_u)$ (non-smooth at zero)

**Rarely used** in practice for MF (L2 works better empirically).

---

### Elastic Net

**Form**: $\lambda_1 \|\mathbf{u}_u\|_1 + \lambda_2 \|\mathbf{u}_u\|^2$

**Effect**: Combines L1 and L2 benefits

**Gradient**: $\lambda_1 \cdot \text{sign}(\mathbf{u}_u) + 2\lambda_2 \mathbf{u}_u$

---

### Per-Item and Per-User Regularization

Different items have different amounts of data:
- Popular items: Many ratings → less regularization needed
- Niche items: Few ratings → more regularization needed

**Adaptive regularization**:

$$J = \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \sum_u \lambda_u \|\mathbf{u}_u\|^2 + \sum_i \lambda_i \|\mathbf{v}_i\|^2$$

where:
- $\lambda_u = \frac{\lambda_0}{|\mathcal{K}_u|}$: Inversely proportional to number of ratings by user $u$
- $\lambda_i = \frac{\lambda_0}{|\mathcal{K}_i|}$: Inversely proportional to number of ratings for item $i$

**Benefit**: Better generalization for sparse users/items.

---

## Overfitting vs. Underfitting

### Overfitting

**Symptoms**:
- Low training error, high test error
- Model memorizes training data
- Poor generalization

**Causes**:
- Too many latent factors ($k$ too large)
- Too little regularization ($\lambda$ too small)
- Too many training epochs

**Solutions**:
- Increase $\lambda$
- Decrease $k$
- Early stopping (monitor validation error)

---

### Underfitting

**Symptoms**:
- High training error, high test error
- Model too simple to capture patterns

**Causes**:
- Too few latent factors ($k$ too small)
- Too much regularization ($\lambda$ too large)

**Solutions**:
- Decrease $\lambda$
- Increase $k$
- Train longer

---

### The Bias-Variance Tradeoff

**Low complexity** ($k$ small, $\lambda$ large):
- High bias (underfit)
- Low variance (stable predictions)

**High complexity** ($k$ large, $\lambda$ small):
- Low bias (fits training data well)
- High variance (unstable, overfits)

**Goal**: Find the sweet spot using validation data.

---

## Loss Landscape Visualization: The Mountainous Terrain

*Imagine you're blindfolded on a mountain range, trying to find the lowest valley.*

The MF optimization surface is **non-convex**:

```
High Loss
    ↑
    |     ╱╲    ╱╲
    |    ╱  ╲  ╱  ╲
    |___╱____╲╱____╲___ → Parameter space
         ↑       ↑
       local   global
       min     minimum

Multiple valleys - some deeper than others!
```

### The Ball-Rolling Analogy

*Think of SGD as rolling a ball down this surface.*

**Starting position** (initialization):
- Random initialization = dropping the ball from a random point
- Where you start determines which valley you might end up in

**Each SGD step**:
- Compute gradient = "feel which direction is downhill"
- Take a small step = "roll a bit in that direction"
- Learning rate = "how far to roll before checking again"

**What can go wrong?**

1. **Too high learning rate** (giant steps):
   - Ball overshoots valleys, bounces around chaotically
   - Loss oscillates wildly, never converges

2. **Too low learning rate** (tiny steps):
   - Ball rolls extremely slowly
   - May take forever to reach a valley
   - Might stop on a gentle slope (not a true minimum)

3. **Local minima**:
   - Ball settles in a shallow valley when a deeper one exists nearby
   - Different initializations help explore different valleys

---

### Why MF Loss Is Non-Convex

*Let me show you concretely why this happens.*

**Consider a simple case**: 1 user, 1 item, 1 latent factor (k=1)

Loss: $L = (r - u \cdot v)^2$ where $r$ is the true rating.

If $r = 4$, then $L = (4 - uv)^2$

**The solutions** (where $L = 0$):
- $u = 2, v = 2$
- $u = 4, v = 1$
- $u = 1, v = 4$
- $u = -2, v = -2$
- ... infinitely many!

**Visualizing**:
```
    v
    ↑
    |   *  (u=1, v=4)
    |    ╲
    |     ╲  ← uv = 4 contour
    |      ╲
    |       * (u=2, v=2)
    |        ╲
    |─────────*──────→ u
              (u=4, v=1)
```

The set of optimal solutions forms a hyperbola! SGD can converge to any point on this curve, depending on initialization.

*This is why* we add regularization: It breaks the symmetry and prefers solutions with smaller $|u|$ and $|v|$.

---

### Practical Implications

| Challenge | Solution |
|-----------|----------|
| Local minima | Multiple random initializations, keep best |
| Saddle points | Momentum (SGD with momentum escapes saddles) |
| Scaling differences | Normalize ratings, use adaptive learning rate (Adam) |
| Symmetry in solutions | Regularization breaks symmetry |

**In practice**: Multiple random initializations help, but surprisingly, one random initialization usually works well enough for recommendation systems.

---

## Convergence Criteria

### When to Stop Training?

**Option 1**: Fixed number of epochs
- Simple, predictable
- May stop too early or too late

**Option 2**: Loss threshold
- Stop when $J < \epsilon$ for some threshold $\epsilon$
- Risk of overfitting

**Option 3**: Early stopping (BEST)
- Monitor validation RMSE every epoch
- Stop when validation error stops improving
- Prevents overfitting

**Example**:
```python
best_val_rmse = float('inf')
patience = 5  # Stop after 5 epochs without improvement
epochs_without_improvement = 0

for epoch in range(max_epochs):
    train(...)
    val_rmse = evaluate_on_validation()

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        save_model()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break  # Early stop
```

---

## Practical Considerations

### Initialization

**Random initialization** (most common):
```python
U = np.random.normal(0, 0.1, (k, n_users))
V = np.random.normal(0, 0.1, (k, n_items))
b_u = np.zeros(n_users)
b_i = np.zeros(n_items)
```

**Why normal(0, 0.1)?**
- Mean 0: No bias
- Small variance: Start near origin (regularization-friendly)

**SVD initialization** (advanced):
- Use truncated SVD as starting point
- Can speed up convergence
- More expensive initially

---

### Learning Rate Scheduling

**Fixed learning rate**:
- $\gamma = 0.01$ throughout training
- Simple but may not converge well

**Step decay**:
- $\gamma_t = \gamma_0 \cdot \text{decay}^{\lfloor \frac{t}{T} \rfloor}$
- Example: Halve learning rate every 10 epochs

**Exponential decay**:
- $\gamma_t = \gamma_0 \cdot e^{-\lambda t}$

**Adaptive methods** (Adam, RMSprop):
- Automatically adjust per-parameter learning rates
- Work well in practice

---

### Shuffling

**CRITICAL**: Shuffle training data each epoch!

```python
for epoch in range(n_epochs):
    np.random.shuffle(ratings)  # Important!
    for user, item, rating in ratings:
        # SGD update
```

**Why?**
- Breaks correlation between consecutive ratings
- Better gradient estimates
- Faster convergence

---

## Complete Implementation

```python
import numpy as np

class MatrixFactorization:
    def __init__(self, n_factors=20, learning_rate=0.01,
                 reg=0.02, n_epochs=20):
        self.k = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs

    def fit(self, ratings, val_ratings=None):
        """
        ratings: list of (user, item, rating) tuples
        """
        # Initialize
        users = set([u for u, i, r in ratings])
        items = set([i for u, i, r in ratings])
        self.n_users = max(users) + 1
        self.n_items = max(items) + 1

        # Global mean
        self.mu = np.mean([r for u, i, r in ratings])

        # Latent factors
        self.U = np.random.normal(0, 0.1, (self.k, self.n_users))
        self.V = np.random.normal(0, 0.1, (self.k, self.n_items))

        # Biases
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)

        # Training loop
        best_val_rmse = float('inf')

        for epoch in range(self.n_epochs):
            # Shuffle for better convergence
            np.random.shuffle(ratings)

            # SGD updates
            for u, i, r in ratings:
                # Prediction
                pred = self.predict(u, i)

                # Error
                err = r - pred

                # Gradients and updates
                u_vec = self.U[:, u].copy()
                i_vec = self.V[:, i].copy()

                # Update factors
                self.U[:, u] += self.lr * (err * i_vec - self.reg * u_vec)
                self.V[:, i] += self.lr * (err * u_vec - self.reg * i_vec)

                # Update biases
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])

            # Evaluate
            train_rmse = self.evaluate(ratings)

            if val_ratings:
                val_rmse = self.evaluate(val_ratings)
                print(f"Epoch {epoch+1}: Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")

                # Early stopping
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                else:
                    print("Early stopping")
                    break
            else:
                print(f"Epoch {epoch+1}: Train RMSE = {train_rmse:.4f}")

    def predict(self, u, i):
        """Predict rating for user u and item i"""
        if u >= self.n_users or i >= self.n_items:
            return self.mu
        return self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.U[:, u], self.V[:, i])

    def evaluate(self, ratings):
        """Compute RMSE on ratings"""
        errors = [(r - self.predict(u, i))**2 for u, i, r in ratings]
        return np.sqrt(np.mean(errors))

# Example usage
if __name__ == "__main__":
    # Toy dataset
    train_ratings = [
        (0, 0, 5.0), (0, 1, 3.0), (0, 2, 4.0),
        (1, 0, 4.0), (1, 2, 5.0),
        (2, 1, 2.0), (2, 2, 1.0), (2, 3, 3.0),
        (3, 0, 1.0), (3, 3, 5.0)
    ]

    val_ratings = [
        (0, 3, 3.0), (1, 1, 4.0), (2, 0, 2.0), (3, 2, 4.0)
    ]

    # Train
    mf = MatrixFactorization(n_factors=5, learning_rate=0.01, reg=0.1, n_epochs=50)
    mf.fit(train_ratings, val_ratings)

    # Predict
    print(f"\nPrediction for user 0, item 3: {mf.predict(0, 3):.2f}")
```

---

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Symbol | Typical Range | Impact |
|-----------|--------|---------------|--------|
| Latent factors | $k$ | 10-200 | Model capacity |
| Learning rate | $\gamma$ | 0.001-0.1 | Convergence speed |
| Regularization | $\lambda$ | 0.001-1.0 | Overfitting control |
| Epochs | $T$ | 10-100 | Training time |

### Grid Search Example

```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'n_factors': [10, 20, 50],
    'learning_rate': [0.001, 0.01, 0.1],
    'reg': [0.01, 0.1, 1.0]
}

best_rmse = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    mf = MatrixFactorization(**params)
    mf.fit(train_ratings)
    val_rmse = mf.evaluate(val_ratings)

    if val_rmse < best_rmse:
        best_rmse = val_rmse
        best_params = params

print(f"Best params: {best_params}")
print(f"Best RMSE: {best_rmse:.4f}")
```

---

## Summary

**Key Takeaways**:
1. MF optimizes squared error over **observed ratings only**
2. Regularization prevents overfitting (essential for sparse data)
3. Bias terms capture systematic effects
4. SGD is simple and effective
5. Early stopping on validation data prevents overfitting
6. Hyperparameter tuning is critical for good performance

**Objective Function**:
$$J = \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \mu - b_u - b_i - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda \left( \sum_u (\|\mathbf{u}_u\|^2 + b_u^2) + \sum_i (\|\mathbf{v}_i\|^2 + b_i^2) \right)$$

**Update Rules**:
- $\mathbf{u}_u \leftarrow \mathbf{u}_u + \gamma (e_{ui} \cdot \mathbf{v}_i - \lambda \mathbf{u}_u)$
- $\mathbf{v}_i \leftarrow \mathbf{v}_i + \gamma (e_{ui} \cdot \mathbf{u}_u - \lambda \mathbf{v}_i)$
- $b_u \leftarrow b_u + \gamma (e_{ui} - \lambda b_u)$
- $b_i \leftarrow b_i + \gamma (e_{ui} - \lambda b_i)$

---

## References

1. **Koren, Y., Bell, R., & Volinsky, C. (2009)**. "Matrix Factorization Techniques for Recommender Systems". *IEEE Computer*.
   - Classic introduction to MF optimization

2. **Salakhutdinov, R., & Mnih, A. (2008)**. "Probabilistic Matrix Factorization". *NIPS*.
   - Bayesian perspective on MF

3. **Zhou, Y., et al. (2008)**. "Large-scale Parallel Collaborative Filtering for the Netflix Prize". *AAIM*.
   - Practical optimization strategies

4. **Bottou, L. (2010)**. "Large-Scale Machine Learning with Stochastic Gradient Descent". *COMPSTAT*.
   - SGD theory and practice
