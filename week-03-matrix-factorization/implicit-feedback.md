# Week 3: Matrix Factorization - Implicit Feedback

## Overview

Most real-world recommendation systems don't have explicit ratings. Instead, they observe **implicit feedback**: clicks, views, purchases, plays, time spent. This document covers how to adapt matrix factorization for implicit feedback data, focusing on two dominant approaches:

1. **Weighted Regularized Matrix Factorization (WRMF)** - Hu et al., 2008
2. **Bayesian Personalized Ranking (BPR)** - Rendle et al., 2009

These methods power recommendation systems at Spotify, Netflix, YouTube, Amazon, and most major platforms.

---

## Learning Objectives

By the end of this section, you will:
- Distinguish between explicit and implicit feedback
- Understand the one-class collaborative filtering problem
- Implement WRMF for confidence-weighted recommendations
- Master BPR for pairwise ranking
- Apply these techniques to real-world datasets

---

## Explicit vs. Implicit Feedback

### Comparison

| Aspect | Explicit Feedback | Implicit Feedback |
|--------|------------------|-------------------|
| **Examples** | Star ratings, thumbs up/down | Clicks, views, purchases, plays |
| **Signal** | Direct preference | Inferred preference |
| **Availability** | Sparse (1-5% of interactions) | Abundant (all user actions) |
| **Interpretation** | Clear (5 stars = love) | Ambiguous (click = interest?) |
| **Negative feedback** | Explicit (1 star = dislike) | Unclear (no click = ?) |
| **Scale** | Thousands to millions | Billions |

---

### The Challenge with Implicit Feedback

**Problem**: Implicit feedback is **one-class**.

**Observed interactions** ($y_{ui} = 1$):
- User clicked item → Positive signal (interest)
- User purchased item → Strong positive signal

**Missing interactions** ($y_{ui} = 0$):
- User hasn't seen item yet? (no information)
- User saw but ignored? (weak negative)
- User dislikes item? (strong negative)

**Key insight**: **Absence of evidence ≠ Evidence of absence**

We can't treat missing values as negatives!

---

## Mathematical Formulation

### Explicit Feedback (Recap)

**Data**: $r_{ui} \in \{1, 2, 3, 4, 5\}$ (rating matrix, sparse)

**Model**: $\hat{r}_{ui} = \mathbf{u}_u^T \mathbf{v}_i$

**Loss**: Squared error on observed ratings
$$\min_{U,V} \sum_{(u,i) \in \text{observed}} (r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda(\|U\|^2 + \|V\|^2)$$

---

### Implicit Feedback Formulation

**Data**: $y_{ui} \in \{0, 1\}$ (interaction matrix, dense)
- $y_{ui} = 1$: User $u$ interacted with item $i$
- $y_{ui} = 0$: No interaction (majority of entries)

**Challenge**: Can't ignore $y_{ui} = 0$ (too many), can't treat as strong negatives.

**Solution**: Introduce **confidence weights**.

---

## Approach 1: Weighted Regularized Matrix Factorization (WRMF)

### The Hu-Koren-Volinsky Model (2008)

**Paper**: "Collaborative Filtering for Implicit Feedback Datasets" (IEEE ICDM 2008)

**Used by**: Spotify, Netflix, YouTube, Pandora

---

### Key Idea: Confidence Weights

For each user-item pair $(u, i)$, define:

1. **Preference** $p_{ui}$:
   $$p_{ui} = \begin{cases} 1 & \text{if } y_{ui} > 0 \text{ (interaction exists)} \\ 0 & \text{otherwise} \end{cases}$$

2. **Confidence** $c_{ui}$:
   $$c_{ui} = 1 + \alpha \cdot y_{ui}$$

where:
- $y_{ui}$: Number of interactions (plays, clicks, views)
- $\alpha$: Confidence scaling factor (hyperparameter, typically 40)

**Interpretation**:
- $p_{ui} = 1, c_{ui} = 41$: User played song 1 time → confident positive
- $p_{ui} = 1, c_{ui} = 401$: User played song 10 times → very confident positive
- $p_{ui} = 0, c_{ui} = 1$: No interaction → low confidence negative

---

### Objective Function

**WRMF minimizes**:

$$\min_{U,V} \sum_{u,i} c_{ui} (p_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda (\|U\|^2 + \|V\|^2)$$

**Key differences from standard MF**:
1. **Sum over ALL entries** (not just observed)
2. **Weighted** by confidence $c_{ui}$
3. **Binary preferences** $p_{ui} \in \{0, 1\}$

**Why this works**:
- High confidence on observed interactions (force $\hat{y}_{ui} \to 1$)
- Low confidence on missing (gentle push towards 0, but not strong)
- More interactions → higher confidence

---

### ALS for WRMF

**Problem**: Summing over all $(u, i)$ pairs is expensive!
- Matrix size: $|U| \times |I|$ (millions × millions)

**Solution**: Alternating Least Squares (ALS) with sparse computations.

**Update for** $\mathbf{u}_u$:

$$\mathbf{u}_u = \left( V C^u V^T + \lambda I \right)^{-1} V C^u \mathbf{p}_u$$

where:
- $C^u = \text{diag}(c_{u1}, c_{u2}, \ldots, c_{u|I|})$: Confidence weights for user $u$
- $\mathbf{p}_u = (p_{u1}, p_{u2}, \ldots, p_{u|I|})^T$: Preference vector for user $u$

**Sparse trick**:
- Most $c_{ui} = 1$ (no interaction)
- Only compute explicitly for $c_{ui} > 1$ (interactions)

---

### Implementation

```python
import numpy as np
from scipy.sparse import csr_matrix

class WRMF:
    def __init__(self, n_factors=50, reg=0.01, alpha=40, n_iters=15):
        self.k = n_factors
        self.reg = reg
        self.alpha = alpha
        self.n_iters = n_iters

    def fit(self, interactions):
        """
        interactions: sparse matrix (users x items) with interaction counts
        """
        self.n_users, self.n_items = interactions.shape

        # Initialize factors
        self.U = np.random.normal(0, 0.01, (self.n_users, self.k))
        self.V = np.random.normal(0, 0.01, (self.n_items, self.k))

        # Precompute confidence matrix
        # C = 1 + alpha * interactions
        self.C = 1.0 + self.alpha * interactions

        # Preference matrix (binary)
        self.P = (interactions > 0).astype(np.float32)

        # ALS iterations
        for iteration in range(self.n_iters):
            # Fix V, solve for U
            self.U = self._als_step(self.V, self.P, self.C)

            # Fix U, solve for V
            self.V = self._als_step(self.U, self.P.T, self.C.T)

            # Optionally: compute loss (expensive)
            if iteration % 5 == 0:
                loss = self.compute_loss()
                print(f"Iteration {iteration}: Loss = {loss:.4f}")

    def _als_step(self, fixed_vecs, prefs, confs):
        """
        One ALS step: solve for one set of factors given the other.

        fixed_vecs: (n_items, k) - the fixed factor matrix (V)
        prefs: (n_users, n_items) - preference matrix (P)
        confs: (n_users, n_items) - confidence matrix (C)

        Returns: (n_users, k) updated factor matrix (U)
        """
        n_solve = prefs.shape[0]
        k = fixed_vecs.shape[1]
        updated_vecs = np.zeros((n_solve, k))

        # Regularization term
        reg_eye = self.reg * np.eye(k)

        for idx in range(n_solve):
            # Get preferences and confidences for this user
            p_u = prefs[idx].toarray().flatten()  # Dense vector
            c_u = confs[idx].toarray().flatten()

            # Sparse computation: only consider non-zero confidences
            # Standard: V^T C_u V + lambda I
            # Efficient: V^T V + V^T (C_u - I) V

            # V^T V (precompute once)
            VtV = fixed_vecs.T @ fixed_vecs

            # V^T (C_u - I) V (only for c_u > 1)
            nonzero = c_u > 1
            if np.any(nonzero):
                V_nonzero = fixed_vecs[nonzero]
                c_diff = c_u[nonzero] - 1  # C_u - I
                VtCuV = V_nonzero.T @ (c_diff[:, None] * V_nonzero)
            else:
                VtCuV = 0

            # Left side: V^T C_u V + lambda I
            A = VtV + VtCuV + reg_eye

            # Right side: V^T C_u p_u
            b = fixed_vecs.T @ (c_u * p_u)

            # Solve: A u_u = b
            updated_vecs[idx] = np.linalg.solve(A, b)

        return updated_vecs

    def compute_loss(self):
        """Compute WRMF objective (expensive, for monitoring)"""
        predictions = self.U @ self.V.T
        diff = self.P.toarray() - predictions
        weighted_error = np.sum(self.C.toarray() * (diff ** 2))
        reg_term = self.reg * (np.sum(self.U ** 2) + np.sum(self.V ** 2))
        return weighted_error + reg_term

    def recommend(self, user_id, top_n=10, exclude_interacted=True):
        """Generate top-N recommendations for a user"""
        scores = self.U[user_id] @ self.V.T

        if exclude_interacted:
            # Mask already interacted items
            interacted = self.P[user_id].toarray().flatten().astype(bool)
            scores[interacted] = -np.inf

        # Top-N items
        top_items = np.argsort(scores)[::-1][:top_n]
        return top_items, scores[top_items]

# Example usage
if __name__ == "__main__":
    # Toy data: 5 users, 10 items
    # Rows: users, Columns: items, Values: interaction counts
    data = np.array([
        [0, 5, 0, 0, 0, 3, 0, 0, 1, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 3, 4, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 3, 2, 0]
    ])

    interactions = csr_matrix(data)

    # Train WRMF
    model = WRMF(n_factors=10, alpha=40, n_iters=20)
    model.fit(interactions)

    # Recommend for user 0
    items, scores = model.recommend(user_id=0, top_n=5)
    print(f"Top 5 recommendations for user 0: {items}")
    print(f"Scores: {scores}")
```

---

### Hyperparameters

| Parameter | Symbol | Typical Range | Impact |
|-----------|--------|---------------|--------|
| Latent factors | $k$ | 20-200 | Model capacity |
| Regularization | $\lambda$ | 0.001-0.1 | Overfitting control |
| Confidence scaling | $\alpha$ | 10-100 | How much to trust interactions |
| ALS iterations | $T$ | 10-30 | Convergence |

**Tuning tips**:
- **$\alpha$ too small**: Model ignores interactions
- **$\alpha$ too large**: Overfits to observed, poor generalization
- Typical: $\alpha = 40$ (from original paper)

---

## Approach 2: Bayesian Personalized Ranking (BPR)

### The Pairwise Ranking Perspective

**Paper**: Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)

**Key insight**: Instead of predicting scores, **learn to rank**.

**Assumption**: User $u$ prefers observed item $i$ over unobserved item $j$:
$$i >_u j$$

where:
- $i \in I_u^+$: Items user $u$ interacted with
- $j \in I \setminus I_u^+$: Items user $u$ didn't interact with

---

### Pairwise Loss

**Goal**: For each user $u$, ensure:
$$\hat{y}_{ui} > \hat{y}_{uj} \quad \forall i \in I_u^+, j \notin I_u^+$$

**BPR Optimization Criterion**:

$$\max_{U,V} \sum_{u} \sum_{i \in I_u^+} \sum_{j \notin I_u^+} \ln \sigma(\hat{y}_{uij}) - \lambda \|U\|^2 - \lambda \|V\|^2$$

where:
- $\hat{y}_{uij} = \hat{y}_{ui} - \hat{y}_{uj} = \mathbf{u}_u^T (\mathbf{v}_i - \mathbf{v}_j)$
- $\sigma(x) = \frac{1}{1 + e^{-x}}$: Sigmoid function
- $\lambda$: Regularization

**Interpretation**:
- $\sigma(\hat{y}_{uij})$: Probability that user $u$ prefers $i$ over $j$
- Maximize log-likelihood of correct pairwise preferences

---

### Why Pairwise?

**Pointwise** (WRMF):
- Predict absolute scores
- "User likes item $i$ with score 0.8"

**Pairwise** (BPR):
- Predict relative preferences
- "User prefers item $i$ over item $j$"

**Advantage of pairwise**:
- Directly optimizes ranking (Top-N recommendation)
- Doesn't require assigning scores to negatives
- Better aligns with evaluation metrics (Precision@K, NDCG)

---

### Stochastic Gradient Descent for BPR

**Sampling strategy**:
1. Sample user $u$ uniformly
2. Sample positive item $i \in I_u^+$
3. Sample negative item $j \notin I_u^+$ uniformly

**Gradient**:

$$\frac{\partial}{\partial \mathbf{u}_u} \ln \sigma(\hat{y}_{uij}) = \sigma(-\hat{y}_{uij}) \cdot (\mathbf{v}_i - \mathbf{v}_j)$$

$$\frac{\partial}{\partial \mathbf{v}_i} \ln \sigma(\hat{y}_{uij}) = \sigma(-\hat{y}_{uij}) \cdot \mathbf{u}_u$$

$$\frac{\partial}{\partial \mathbf{v}_j} \ln \sigma(\hat{y}_{uij}) = -\sigma(-\hat{y}_{uij}) \cdot \mathbf{u}_u$$

**Update rules**:
$$\mathbf{u}_u \leftarrow \mathbf{u}_u + \gamma \left[ \sigma(-\hat{y}_{uij}) (\mathbf{v}_i - \mathbf{v}_j) - \lambda \mathbf{u}_u \right]$$

$$\mathbf{v}_i \leftarrow \mathbf{v}_i + \gamma \left[ \sigma(-\hat{y}_{uij}) \mathbf{u}_u - \lambda \mathbf{v}_i \right]$$

$$\mathbf{v}_j \leftarrow \mathbf{v}_j + \gamma \left[ -\sigma(-\hat{y}_{uij}) \mathbf{u}_u - \lambda \mathbf{v}_j \right]$$

---

### Implementation

```python
import numpy as np
from collections import defaultdict

class BPR:
    def __init__(self, n_factors=20, learning_rate=0.01, reg=0.01, n_iters=100000):
        self.k = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.n_iters = n_iters

    def fit(self, interactions):
        """
        interactions: list of (user, item) tuples representing positive feedback
        """
        # Build user-item sets
        self.user_items = defaultdict(set)
        users_set = set()
        items_set = set()

        for user, item in interactions:
            self.user_items[user].add(item)
            users_set.add(user)
            items_set.add(item)

        self.users = sorted(users_set)
        self.items = sorted(items_set)
        self.n_users = max(self.users) + 1
        self.n_items = max(self.items) + 1

        # Initialize factors
        self.U = np.random.normal(0, 0.01, (self.n_users, self.k))
        self.V = np.random.normal(0, 0.01, (self.n_items, self.k))

        # SGD training
        for iteration in range(self.n_iters):
            # Sample (user, positive item, negative item) triple
            user = np.random.choice(self.users)
            pos_items = list(self.user_items[user])

            if len(pos_items) == 0:
                continue

            pos_item = np.random.choice(pos_items)

            # Sample negative item (not interacted)
            neg_item = np.random.choice(self.items)
            while neg_item in self.user_items[user]:
                neg_item = np.random.choice(self.items)

            # Compute prediction difference
            y_uij = np.dot(self.U[user], self.V[pos_item] - self.V[neg_item])

            # Sigmoid gradient
            sigmoid = 1.0 / (1 + np.exp(-y_uij))
            grad = 1 - sigmoid

            # Update user factors
            self.U[user] += self.lr * (grad * (self.V[pos_item] - self.V[neg_item]) - self.reg * self.U[user])

            # Update positive item factors
            self.V[pos_item] += self.lr * (grad * self.U[user] - self.reg * self.V[pos_item])

            # Update negative item factors
            self.V[neg_item] += self.lr * (-grad * self.U[user] - self.reg * self.V[neg_item])

            # Periodic logging
            if iteration % 10000 == 0:
                auc = self.evaluate_auc()
                print(f"Iteration {iteration}: AUC = {auc:.4f}")

    def predict(self, user, item):
        """Predict score for (user, item)"""
        return np.dot(self.U[user], self.V[item])

    def recommend(self, user, top_n=10, exclude_interacted=True):
        """Generate top-N recommendations"""
        scores = self.U[user] @ self.V.T

        if exclude_interacted:
            for item in self.user_items[user]:
                scores[item] = -np.inf

        top_items = np.argsort(scores)[::-1][:top_n]
        return top_items, scores[top_items]

    def evaluate_auc(self, n_samples=1000):
        """Evaluate AUC (Area Under ROC Curve)"""
        correct = 0
        total = 0

        for _ in range(n_samples):
            user = np.random.choice(self.users)
            pos_items = list(self.user_items[user])

            if len(pos_items) == 0:
                continue

            pos_item = np.random.choice(pos_items)
            neg_item = np.random.choice(self.items)

            while neg_item in self.user_items[user]:
                neg_item = np.random.choice(self.items)

            # Check if positive item ranked higher
            if self.predict(user, pos_item) > self.predict(user, neg_item):
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.5

# Example usage
if __name__ == "__main__":
    # Toy data: (user, item) interaction pairs
    interactions = [
        (0, 1), (0, 5), (0, 8),
        (1, 0), (1, 9),
        (2, 3), (2, 4), (2, 7),
        (3, 6),
        (4, 1), (4, 7), (4, 8)
    ]

    # Train BPR
    model = BPR(n_factors=10, learning_rate=0.05, reg=0.01, n_iters=50000)
    model.fit(interactions)

    # Recommend for user 0
    items, scores = model.recommend(user=0, top_n=5)
    print(f"Top 5 recommendations for user 0: {items}")
    print(f"Scores: {scores}")
```

---

## WRMF vs. BPR: Comparison

| Aspect | WRMF | BPR |
|--------|------|-----|
| **Optimization** | Pointwise (predict scores) | Pairwise (rank items) |
| **Loss function** | Weighted squared error | Log-likelihood of rankings |
| **Training** | ALS (closed-form) | SGD (sampling) |
| **Speed** | Fast per iteration | Many iterations needed |
| **Scalability** | Good (sparse ALS) | Excellent (minibatch SGD) |
| **Cold start** | Better (uses all data) | Worse (needs positives) |
| **Typical use** | Audio (Spotify), video | E-commerce, news |

**In practice**:
- **WRMF**: When you have interaction counts (plays, views)
- **BPR**: When you have binary interactions (clicks, purchases)

---

## Negative Sampling Strategies

### Uniform Sampling

**Method**: Sample negative items uniformly at random.

**Pros**: Simple, unbiased
**Cons**: Wastes compute on easy negatives (unpopular items)

---

### Popularity-Based Sampling

**Method**: Sample negatives proportional to item popularity.

$$P(\text{sample item } j) \propto (\text{popularity of } j)^\alpha$$

**Typical**: $\alpha = 0.75$ (from word2vec)

**Pros**: More challenging negatives, faster learning
**Cons**: Biased towards popular items

---

### Hard Negative Mining

**Method**: Sample items with high predicted scores but not interacted.

**Pros**: Focus on hard examples, better discrimination
**Cons**: Computationally expensive, can be unstable

---

## Evaluation Metrics for Implicit Feedback

### Ranking Metrics

**Precision@K**:
$$\text{Precision@K} = \frac{|\text{relevant items in top K}|}{K}$$

**Recall@K**:
$$\text{Recall@K} = \frac{|\text{relevant items in top K}|}{|\text{all relevant items}|}$$

**NDCG@K** (Normalized Discounted Cumulative Gain):
$$\text{NDCG@K} = \frac{DCG@K}{IDCG@K}$$

where:
$$DCG@K = \sum_{i=1}^K \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

**MAP@K** (Mean Average Precision):
$$\text{MAP@K} = \frac{1}{|U|} \sum_u \frac{1}{K} \sum_{k=1}^K \text{Precision}@k \cdot \text{rel}(k)$$

---

### AUC (Area Under ROC Curve)

**Interpretation**: Probability that a random positive item ranks higher than a random negative item.

**Computation**: Sample many (user, pos_item, neg_item) triples, check ranking.

$$\text{AUC} = \frac{\sum_{(u,i,j)} \mathbb{1}[\hat{y}_{ui} > \hat{y}_{uj}]}{|{(u,i,j)}|}$$

---

## Practical Tips

### 1. Data Preprocessing

- **Threshold**: Remove users with < 5 interactions (cold start)
- **Cap**: Clip interaction counts (e.g., max 100 plays)
- **Normalize**: Consider temporal decay (recent interactions weighted higher)

### 2. Hyperparameter Tuning

- **WRMF**: Start with $\alpha = 40$, $\lambda = 0.01$
- **BPR**: Start with $lr = 0.05$, $\lambda = 0.01$
- Use validation set (20% of users)

### 3. Scalability

- **WRMF**: Parallelize ALS (user/item updates independent)
- **BPR**: Minibatch SGD, GPU acceleration
- Libraries: Implicit (Python), LensKit, Surprise

---

## Libraries and Tools

### Implicit Library (Python)

```python
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import coo_matrix

# Prepare data
interactions = coo_matrix((data, (users, items)))

# Train WRMF
wrmf = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=15)
wrmf.fit(interactions)

# Train BPR
bpr = BayesianPersonalizedRanking(factors=50, learning_rate=0.01, regularization=0.01)
bpr.fit(interactions)

# Recommend
user_id = 0
recommended_items = wrmf.recommend(user_id, interactions[user_id], N=10)
print(recommended_items)
```

---

## Summary

**Key Takeaways**:
1. **Implicit feedback is ubiquitous** in modern RecSys (clicks, views, plays)
2. **WRMF**: Confidence-weighted pointwise approach (good for counts)
3. **BPR**: Pairwise ranking approach (good for binary interactions)
4. **Negative sampling** is critical for efficiency
5. **Evaluation**: Use ranking metrics (Precision@K, NDCG), not RMSE

**When to use what**:
- **WRMF**: Audio/video streaming (Spotify, YouTube), interaction counts available
- **BPR**: E-commerce (Amazon), news, binary clicks/purchases

**Next steps**:
- Advanced variants: SVD++, TimeSVD++ (advanced-variants.md)
- Neural approaches: NCF, deep learning (week-05-neural-cf/)

---

## References

1. **Hu, Y., Koren, Y., & Volinsky, C. (2008)**. "Collaborative Filtering for Implicit Feedback Datasets". *IEEE ICDM*.
   - Original WRMF paper, foundational work

2. **Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009)**. "BPR: Bayesian Personalized Ranking from Implicit Feedback". *UAI*.
   - BPR algorithm, pairwise ranking

3. **Pan, R., et al. (2008)**. "One-Class Collaborative Filtering". *IEEE ICDM*.
   - Theoretical foundations of implicit feedback

4. **He, X., & McAuley, J. (2016)**. "VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback". *AAAI*.
   - Extending BPR with visual features

5. **Johnson, C. C. (2014)**. "Logistic Matrix Factorization for Implicit Feedback Data". *NIPS Workshop*.
   - Alternative probabilistic approach
