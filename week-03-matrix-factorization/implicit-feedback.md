# Week 3: Matrix Factorization - Implicit Feedback

## The Problem: Why Can't We Treat Missing as Negative?

*Before we dive into the algorithms, let me show you why implicit feedback is fundamentally different from explicit ratings.*

**The Setup**: You're building a music recommendation system. You have data on which songs users have played.

| User | Song A | Song B | Song C | Song D | Song E |
|------|--------|--------|--------|--------|--------|
| Alice | 5 plays | 0 plays | 12 plays | 0 plays | 0 plays |
| Bob | 0 plays | 8 plays | 0 plays | 0 plays | 3 plays |
| Carol | 0 plays | 0 plays | 0 plays | 7 plays | 0 plays |

**The naive approach**: "Let's treat 0 plays as 'dislikes' and run standard matrix factorization!"

*But wait... think about this carefully.*

---

### The Concrete Example: What Goes Wrong

**Let's calculate what happens if we naively treat unobserved = 0.**

Suppose we have 100,000 songs in our catalog. Alice has played 50 songs total.

**If we treat missing as negative:**
- Positive examples: 50 songs Alice played
- Negative examples: 99,950 songs Alice "didn't play"

**Training signal:**
- For every 1 positive, we have ~2000 negatives
- The model learns: "To minimize error, predict 0 for everything!"
- Why? Because 99.95% of the training signal says "predict 0"

**The math**:
$$\text{Loss} = \sum_{\text{positives}} (1 - \hat{y})^2 + \sum_{\text{negatives}} (0 - \hat{y})^2$$

If we predict $\hat{y} = 0$ for everything:
$$\text{Loss} = 50 \times (1-0)^2 + 99950 \times (0-0)^2 = 50$$

If we predict $\hat{y} = 0.1$ for everything:
$$\text{Loss} = 50 \times (1-0.1)^2 + 99950 \times (0-0.1)^2 = 50 \times 0.81 + 99950 \times 0.01 = 40.5 + 999.5 = 1040$$

*See the problem?* Predicting zeros minimizes loss even though it's useless!

---

### The Key Insight: Absence of Evidence ≠ Evidence of Absence

*Why didn't Alice play Song D?*

**Possibility 1**: Alice doesn't know Song D exists (no exposure)
**Possibility 2**: Alice saw Song D but wasn't in the mood (context)
**Possibility 3**: Alice doesn't like that genre (true negative)
**Possibility 4**: Song D wasn't available in her region (availability)

*We simply don't know which one!* Treating all zeros as negatives conflates these very different scenarios.

**Explicit feedback**: If Alice rated Song D as 1 star, we KNOW she dislikes it.
**Implicit feedback**: If Alice has 0 plays for Song D, we know... nothing definitive.

*This is the fundamental challenge we must solve.*

---

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
- **Derive why the confidence weighting makes sense**
- **Understand why pairwise ranking is natural for implicit data**
- **Know the computational tricks that make these methods scale**
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

*The brilliant insight: instead of treating all zeros equally, assign different confidence levels.*

For each user-item pair $(u, i)$, define:

1. **Preference** $p_{ui}$:
   $$p_{ui} = \begin{cases} 1 & \text{if } y_{ui} > 0 \text{ (interaction exists)} \\ 0 & \text{otherwise} \end{cases}$$

2. **Confidence** $c_{ui}$:
   $$c_{ui} = 1 + \alpha \cdot y_{ui}$$

where:
- $y_{ui}$: Number of interactions (plays, clicks, views)
- $\alpha$: Confidence scaling factor (hyperparameter, typically 40)

---

### The Intuition: "More Plays = More Sure"

*Let me walk you through the confidence function step by step.*

**The function**: $c_{ui} = 1 + \alpha \cdot y_{ui}$

*What does each part mean?*

**The "1"** (baseline confidence):
- Even for items with $y_{ui} = 0$, we have $c_{ui} = 1$
- *Why not zero?* Because we still want to push unobserved items toward 0, just gently
- The baseline says: "I have *some* belief that unobserved items might not be preferred"

**The "$\alpha \cdot y_{ui}$"** (interaction boost):
- More interactions → higher confidence
- If Alice played a song 10 times, we're MORE SURE she likes it than if she played it once
- $\alpha$ controls how much more sure we are

---

### Numerical Example: Different Confidence Levels

*Let's make this concrete with $\alpha = 40$ (the typical value).*

| Scenario | $y_{ui}$ | $c_{ui} = 1 + 40 \cdot y_{ui}$ | Interpretation |
|----------|----------|--------------------------------|----------------|
| Never played | 0 | 1 | "Weak belief: maybe doesn't like" |
| Played once | 1 | 41 | "Pretty confident: likes this" |
| Played 5 times | 5 | 201 | "Very confident: really likes this" |
| Played 10 times | 10 | 401 | "Extremely confident: loves this" |
| Played 100 times | 100 | 4001 | "Near-certain: this is a favorite" |

*Now think about what this means for training.*

**When we make a prediction error on a 10-play song**:
- The error contributes 401× more to the loss than an error on an unobserved song
- The model REALLY wants to get this prediction right
- It will adjust the user and item vectors significantly

**When we make a prediction error on an unobserved song**:
- The error contributes only 1× (baseline) to the loss
- The model will gently push the prediction toward 0
- But it won't overfit to this weak signal

---

### Visualizing High vs Low Confidence

*Think of confidence as the "volume" of a training example's voice.*

```
Confidence Level:     LOW (c=1)          HIGH (c=401)
                     ─────────           ─────────────

Training signal:     (whisper)           (SHOUTING)
                     "maybe zero..."     "DEFINITELY ONE!"

Effect on gradient:  tiny nudge          big push

What happens if      "oh well, I         "MUST FIX THIS!
model is wrong:      might be wrong"     ADJUST VECTORS!"
```

*This is why WRMF works:* The model pays attention to what it's confident about (actual interactions) while maintaining a gentle prior that unobserved items are probably not preferred.

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

### The ALS Sparse Trick: Why $V^T(C-I)V$ Works

*This is the computational magic that makes WRMF practical. Let's derive it step by step.*

**The Problem**:

For each user $u$, we need to compute:
$$V C^u V^T$$

where $C^u$ is a diagonal $|I| \times |I|$ matrix with $c_{ui}$ on the diagonal.

**Naive complexity**: $O(k^2 |I|)$ per user, where $|I|$ might be millions of items!

*But here's the key observation...*

---

**Step 1: Decompose $C^u$**

Since $c_{ui} = 1 + \alpha \cdot y_{ui}$, and most $y_{ui} = 0$:

$$C^u = I + (C^u - I)$$

where:
- $I$: Identity matrix (accounts for the baseline confidence of 1)
- $(C^u - I)$: A **sparse** matrix with $\alpha \cdot y_{ui}$ only for observed items

*Most entries of $(C^u - I)$ are zero!* Only $|I_u|$ entries are non-zero, where $I_u$ = items user $u$ interacted with.

---

**Step 2: Expand $V C^u V^T$**

$$V C^u V^T = V (I + (C^u - I)) V^T = V I V^T + V (C^u - I) V^T$$

$$= V V^T + V (C^u - I) V^T$$

**Key insight**:
- $V V^T$ is a $k \times k$ matrix that's the **same for all users**! Precompute once.
- $V (C^u - I) V^T$ is **sparse** - only involves items user $u$ interacted with.

---

**Step 3: Compute the Sparse Part Efficiently**

Let $I_u = \{i : y_{ui} > 0\}$ be the set of items user $u$ interacted with.

$$V (C^u - I) V^T = \sum_{i \in I_u} (c_{ui} - 1) \mathbf{v}_i \mathbf{v}_i^T$$

*This is a sum of $|I_u|$ rank-1 matrices!*

**Computation**:
- For each item $i$ that user $u$ interacted with:
  - Look up $\mathbf{v}_i$ (the item's $k$-dimensional vector)
  - Compute the outer product $\mathbf{v}_i \mathbf{v}_i^T$ (a $k \times k$ matrix)
  - Multiply by $(c_{ui} - 1)$
  - Add to the running sum

---

**Step 4: Final Complexity Analysis**

**For the user update:**

1. **Precompute $VV^T$**: $O(k^2 |I|)$ - done once, shared across all users
2. **Per-user sparse part**: $O(k^2 |I_u|)$ - only for items user $u$ interacted with
3. **Matrix inversion**: $O(k^3)$ - for the $k \times k$ matrix
4. **Matrix-vector multiply**: $O(k^2)$

**Total per user**: $O(k^2 |I_u| + k^3)$

**Compare to naive**: $O(k^2 |I|)$

**Savings**: $\frac{|I|}{|I_u|}$ = typical user interaction ratio

*For a catalog of 10 million items and a user who interacted with 1000:*
- Naive: 10,000,000 operations
- Sparse trick: 1,000 operations
- **10,000× speedup!**

---

**The Complete Algorithm (Pseudocode)**:

```python
# Precompute VV^T (done once)
VtV = V @ V.T  # k x k matrix

for each user u:
    # Start with precomputed VV^T
    A = VtV.copy()

    # Add sparse part for items user interacted with
    for i in items_user_u_interacted_with:
        c_diff = c[u, i] - 1  # This is alpha * y_ui
        A += c_diff * np.outer(V[:, i], V[:, i])

    # Add regularization
    A += lambda * I

    # Compute right-hand side (also sparse)
    b = V @ (C[u, :] * p[u, :])  # Also sparse!

    # Solve the linear system
    u[u] = solve(A, b)
```

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

### The Bayesian Perspective: Where BPR Comes From

*Let me derive BPR from first principles so you understand why pairwise ranking makes sense.*

**Step 1: The Bayesian Setup**

We want to find the best parameters $\Theta$ (user and item factors) given the observed data $D_S$:

$$p(\Theta | D_S) \propto p(D_S | \Theta) \cdot p(\Theta)$$

where:
- $p(D_S | \Theta)$: Likelihood of observed data given parameters
- $p(\Theta)$: Prior on parameters (regularization!)

---

**Step 2: What Is Our "Observed Data"?**

*This is the key insight.*

For implicit feedback, what do we actually observe? Not ratings, but **relative preferences**.

If Alice clicked on item $i$ but not item $j$, we can reasonably assume:
$$\text{Alice prefers } i \text{ over } j$$

We don't know how MUCH Alice likes $i$, but we know she prefers it to $j$.

**The observed data $D_S$:**
$$D_S = \{(u, i, j) : i \in I_u^+, j \notin I_u^+\}$$

All triples where user $u$ interacted with item $i$ but not with item $j$.

---

**Step 3: The Likelihood Function**

For each triple $(u, i, j)$, we model the probability that user $u$ prefers $i$ over $j$:

$$p(i >_u j | \Theta) = \sigma(\hat{x}_{uij})$$

where:
- $\hat{x}_{uij} = \hat{r}_{ui} - \hat{r}_{uj}$: Difference in predicted scores
- $\sigma(x) = \frac{1}{1 + e^{-x}}$: Sigmoid function

*Why sigmoid?* It maps any real number to a probability in (0, 1).

**Assuming independence across triples:**

$$p(D_S | \Theta) = \prod_{(u,i,j) \in D_S} \sigma(\hat{x}_{uij})$$

---

**Step 4: The Log-Likelihood**

Taking the log (easier to optimize):

$$\ln p(D_S | \Theta) = \sum_{(u,i,j) \in D_S} \ln \sigma(\hat{x}_{uij})$$

---

**Step 5: Adding the Prior (Regularization)**

We use a Gaussian prior on the parameters:

$$p(\Theta) \propto \exp\left(-\frac{\lambda}{2} \|\Theta\|^2\right)$$

Log-prior:

$$\ln p(\Theta) = -\frac{\lambda}{2} \|\Theta\|^2 + \text{const}$$

---

**Step 6: The BPR Optimization Criterion**

Putting it together (maximizing log-posterior):

$$\text{BPR-Opt} = \sum_{(u,i,j) \in D_S} \ln \sigma(\hat{x}_{uij}) - \lambda \|\Theta\|^2$$

**Or equivalently, for matrix factorization**:

$$\max_{U,V} \sum_{u} \sum_{i \in I_u^+} \sum_{j \notin I_u^+} \ln \sigma(\hat{y}_{uij}) - \lambda \|U\|^2 - \lambda \|V\|^2$$

where:
- $\hat{y}_{uij} = \hat{y}_{ui} - \hat{y}_{uj} = \mathbf{u}_u^T (\mathbf{v}_i - \mathbf{v}_j)$

---

### Why Pairwise Makes Sense: The Connection to Maximum Likelihood

*Let's dig deeper into why this is the right objective.*

**Pointwise approaches** (like treating implicit feedback as 0/1 labels):
- Model: $p(y_{ui} = 1 | \Theta) = \sigma(\hat{r}_{ui})$
- Problem: What's $p(y_{ui} = 0 | \Theta)$?
- We're trying to model the probability of "NOT clicking", but we don't know if the user even saw the item!

**Pairwise approaches** (BPR):
- Model: $p(i >_u j | \Theta) = \sigma(\hat{r}_{ui} - \hat{r}_{uj})$
- We only model *relative* preferences
- No need to model absolute probabilities of clicking

*The key advantage:* We're modeling something we can actually infer from the data (relative preferences), not something we can't (absolute click probabilities).

---

### Pairwise vs Pointwise: A Concrete Comparison

*Let's see why this matters with an example.*

**Scenario**: Alice clicked items {A, B}. Items {C, D, E} were not clicked.

**Pointwise approach (WRMF-style)**:
- Training targets: A=1, B=1, C=0, D=0, E=0
- Learns: $\hat{r}_A \approx 1$, $\hat{r}_B \approx 1$, $\hat{r}_C \approx 0$, ...
- Problem: Forces absolute scores. What if Alice would rate all of them highly if she saw them?

**Pairwise approach (BPR)**:
- Training signal: A > C, A > D, A > E, B > C, B > D, B > E
- Learns: $\hat{r}_A > \hat{r}_C$, $\hat{r}_A > \hat{r}_D$, etc.
- Key difference: Doesn't force any absolute scale! Just learns relative ordering.

*Why is this better for recommendation?*

We don't care if the predicted score is 0.8 or 0.2. We care about **ranking**: which items should appear at the top of the list.

BPR directly optimizes for ranking. WRMF optimizes for score prediction and hopes ranking follows.

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

## What Can Go Wrong?

*Let me warn you about the common failure modes specific to implicit feedback methods.*

### Failure Mode 1: Confidence Miscalibration (WRMF)

**Problem**: Your confidence function doesn't match reality.

**Symptoms**:
- Model overfits to users with many interactions
- Users with few interactions get poor recommendations
- Popular items dominate even for niche users

**Example**: You use $\alpha = 40$, but your platform has power users with 10,000+ interactions.
- Their confidence values: $c = 1 + 40 \times 10000 = 400,001$
- They completely dominate the loss function!
- Casual users (10 interactions, $c = 401$) are essentially ignored.

**Solutions**:
- Log-transform counts: $c_{ui} = 1 + \alpha \log(1 + y_{ui})$
- Cap interaction counts: $y_{ui} = \min(y_{ui}, 100)$
- Per-user normalization: Scale by user's total interactions

---

### Failure Mode 2: Negative Sampling Bias (BPR)

**Problem**: Your negative sampling distribution doesn't represent true negatives.

**Symptoms**:
- AUC looks great, but recommendations are bad
- Model learns to distinguish popular vs unpopular, not relevant vs irrelevant
- Niche items never get recommended

**Example**: With uniform sampling on a catalog where 90% of items have <10 interactions:
- Most negatives are obscure items
- Model easily learns: "obscure = negative"
- But it never learns to distinguish between popular items!

**Solutions**:
- Popularity-based negative sampling
- Hard negative mining (periodically)
- Mix of sampling strategies

---

### Failure Mode 3: Feedback Loop Amplification

**Problem**: Your recommendations create the data that trains future models.

**Symptoms**:
- Over time, recommendations become less diverse
- New items never get discovered
- User preferences appear to "narrow" (but it's a data artifact)

**The loop**:
1. Model recommends items A, B, C
2. User clicks on A, B, C (because that's all they see!)
3. Model learns "user likes A, B, C"
4. Model recommends A, B, C even more
5. Repeat...

**Solutions**:
- Exploration/exploitation (bandits)
- Diversification in recommendation lists
- Random exposure experiments
- Propensity scoring to debias training data

---

### Failure Mode 4: Position Bias

**Problem**: Users click on items because of WHERE they appear, not WHAT they are.

**Symptoms**:
- Items shown in position 1 have 10× more clicks
- Model thinks position-1 items are universally loved
- Retraining amplifies this bias

**Example**:
- Song A shown in position 1: 1000 clicks
- Song B shown in position 10: 100 clicks
- Reality: Song B might be better! Users just don't scroll down.

**Solutions**:
- Position-aware models: $y_{ui} = f(\text{rank}, \text{user}, \text{item})$
- Inverse propensity weighting: Weight by 1/P(position)
- Randomized experiments to measure true preferences

---

### Failure Mode 5: Treating Missing as Uniform Negative

**Problem**: Some missing entries are "strong negatives" (user saw and rejected), others are "unknown" (user never saw).

**Symptoms**:
- Model confused about items in "maybe" category
- Recommendations include items user explicitly skipped
- Click-through rate doesn't improve despite good offline metrics

**Example**:
- User scrolled past Song X: Strong negative (saw it, didn't click)
- User never saw Song Y: Unknown (might love it!)
- Both have $y = 0$, but they're fundamentally different.

**Solutions**:
- Incorporate exposure data if available
- Multi-level confidence: Seen-not-clicked < Never-seen
- Inverse propensity scoring

---

### Failure Mode 6: Scale Mismatch Between Users

**Problem**: Power users and casual users contribute very differently to loss.

**Symptoms (WRMF)**:
- Model optimizes for power users
- Casual users get generic/popular recommendations

**Symptoms (BPR)**:
- Power users generate many training triples
- Model overfits to their preferences

**Solutions**:
- Per-user loss normalization
- Sample users uniformly (not interactions)
- Weight by inverse user activity

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
2. **Can't treat missing as negative**: Absence of evidence ≠ evidence of absence
3. **WRMF**: Confidence-weighted pointwise approach (good for counts)
4. **BPR**: Pairwise ranking approach (good for binary interactions)
5. **Negative sampling** is critical for efficiency
6. **Evaluation**: Use ranking metrics (Precision@K, NDCG), not RMSE
7. **Watch out for** confidence miscalibration, feedback loops, position bias

**When to use what**:
- **WRMF**: Audio/video streaming (Spotify, YouTube), interaction counts available
- **BPR**: E-commerce (Amazon), news, binary clicks/purchases

**The Computational Tricks**:
- WRMF: Sparse $V^T(C-I)V$ computation → $O(k^2|I_u|)$ instead of $O(k^2|I|)$
- BPR: Stochastic sampling → Each update is $O(k)$

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
