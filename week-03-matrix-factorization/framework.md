# Week 3: The Matrix Factorization Framework

## The Problem: Why Can't We Just Store All Ratings?

*Before we dive into matrix factorization, let me show you why we need it in the first place.*

**Imagine you're Netflix** with 200 million users and 15,000 movies. You want to predict how every user would rate every movie.

**The naive approach**: Store a giant table with all ratings.

$$\text{Table size} = 200,000,000 \times 15,000 = 3 \times 10^{12} \text{ entries}$$

**Problem 1: Space**
- 3 trillion entries
- At 4 bytes each = 12 terabytes just for ratings!
- And this table is 99%+ empty (users rate ~100 movies, not 15,000)

**Problem 2: Sparsity**
- Each user has rated maybe 0.01% of items
- We can't compute similarity with so little overlap
- We can't generalize to new user-item pairs

**Problem 3: No Generalization**
- What if a new movie arrives? Zero ratings = zero predictions
- What if a new user arrives? No history = no recommendations

*Here's the key insight*: Users and items live in a much **lower-dimensional space** than the raw rating matrix suggests.

---

## The Insight That Changes Everything

*Let me ask you a question before showing the solution.*

**Why do you think Alice (who loves "The Matrix" and "Inception") would probably like "Blade Runner"?**

Think about it...

The answer isn't "because these are the same movie." It's because they share **underlying characteristics**:
- They're all sci-fi
- They have mind-bending plots
- They feature philosophical themes about reality

**The key insight**: We don't need to store 3 trillion ratings. We just need to learn these **hidden characteristics** (latent factors) for each user and item!

$$\text{Parameters needed} = k \times (|U| + |I|)$$

With $k = 100$ factors:
$$100 \times (200,000,000 + 15,000) = 20 \text{ billion}$$

That's still a lot, but it's **150x smaller** than the full matrix! And more importantly, it **generalizes** to unseen pairs.

---

## Learning Objectives

- Understand the low-rank matrix approximation formulation
- Grasp the concept of latent factors
- Connect MF to SVD and PCA
- **Derive step-by-step why $U^T V$ gives us predictions**
- **Work through a complete numerical example**
- Recognize why MF works for recommendation

---

## Core Idea: Latent Factor Models

### What Are Latent Factors?

**User preferences and item characteristics can be represented in a low-dimensional latent space.**

*Think of it this way*: Instead of asking "Does Alice like The Matrix?", we ask:

1. "How much does Alice like sci-fi?" (Factor 1)
2. "How much does Alice like action?" (Factor 2)
3. "How much does Alice like cerebral plots?" (Factor 3)

And for the movie:

1. "How sci-fi is The Matrix?" (Factor 1)
2. "How action-packed is The Matrix?" (Factor 2)
3. "How cerebral is The Matrix?" (Factor 3)

**The magic**: Alice's rating = how well her preferences **align** with the movie's characteristics.

---

### Visual Example: Movies in Latent Space

Let's visualize with $k=2$ factors:

**Latent Space**:
```
Factor 2 (Action Level)
    ↑
  5 |    • The Matrix (high action, high sci-fi)
    |    • Mad Max
  4 |              • Inception
    |
  3 |
    |              • Interstellar
  2 |
    |    • Cast Away (low action, low sci-fi)
  1 |    • Forrest Gump    • Her
    |
    └──────────────────────────────────→ Factor 1 (Sci-Fi Level)
         1    2    3    4    5
```

**Users**:
```
Factor 2
    ↑
  5 |    ★ Alice (loves action sci-fi)
    |
  4 |
    |
  3 |              ★ Bob (balanced)
    |
  2 |
    |                           ★ Carol (prefers drama)
  1 |
    |
    └──────────────────────────────────→ Factor 1
         1    2    3    4    5
```

**Prediction**:
- Alice (at [4.5, 4.8]) is close to The Matrix (at [4.2, 4.9]) → High rating predicted!
- Carol (at [1.5, 1.2]) is far from The Matrix → Low rating predicted

*Can you see why* the dot product captures this? When vectors point in the same direction, their dot product is large. When they're orthogonal or opposite, it's small or negative.

---

## Mathematical Formulation

### Step-by-Step: Why $U^T V$ Works

*Let's build up the math piece by piece, so you understand every symbol.*

**Step 1: Define the problem**

We have a rating matrix $R \in \mathbb{R}^{|U| \times |I|}$:

$$R = \begin{bmatrix}
r_{11} & r_{12} & ? & r_{14} & ? \\
? & r_{22} & r_{23} & ? & r_{25} \\
r_{31} & ? & ? & r_{34} & r_{35} \\
? & r_{42} & r_{43} & r_{44} & ?
\end{bmatrix}$$

Most entries are missing (?). We want to fill them in.

---

**Step 2: The low-rank assumption**

*Here's the key assumption*: Both users and items can be described by just $k$ numbers.

For each user $u$, we have a vector $\mathbf{u}_u \in \mathbb{R}^k$:
$$\mathbf{u}_u = \begin{bmatrix} u_{u1} \\ u_{u2} \\ \vdots \\ u_{uk} \end{bmatrix}$$

For each item $i$, we have a vector $\mathbf{v}_i \in \mathbb{R}^k$:
$$\mathbf{v}_i = \begin{bmatrix} v_{i1} \\ v_{i2} \\ \vdots \\ v_{ik} \end{bmatrix}$$

---

**Step 3: The prediction formula**

*How do we combine user and item vectors to predict a rating?*

The simplest way: **dot product**.

$$\hat{r}_{ui} = \mathbf{u}_u^T \mathbf{v}_i = \sum_{f=1}^k u_{uf} \cdot v_{if}$$

*Why dot product?* Think about what each term means:
- $u_{u1}$: How much user $u$ likes Factor 1 (e.g., "likes sci-fi")
- $v_{i1}$: How much item $i$ has Factor 1 (e.g., "is sci-fi")
- $u_{u1} \cdot v_{i1}$: Contribution of Factor 1 to the rating

The total rating is the sum of contributions from all factors.

---

**Step 4: Matrix form**

Stack all user vectors as columns of $U \in \mathbb{R}^{k \times |U|}$:
$$U = \begin{bmatrix} \mathbf{u}_1 & \mathbf{u}_2 & \cdots & \mathbf{u}_{|U|} \end{bmatrix}$$

Stack all item vectors as columns of $V \in \mathbb{R}^{k \times |I|}$:
$$V = \begin{bmatrix} \mathbf{v}_1 & \mathbf{v}_2 & \cdots & \mathbf{v}_{|I|} \end{bmatrix}$$

Then the entire prediction matrix is:
$$\hat{R} = U^T V$$

*Let's verify*: The $(u, i)$ entry of $U^T V$ is the dot product of row $u$ of $U^T$ (which is $\mathbf{u}_u^T$) and column $i$ of $V$ (which is $\mathbf{v}_i$).

$$(U^T V)_{ui} = \mathbf{u}_u^T \mathbf{v}_i = \hat{r}_{ui}$$

That's exactly our prediction formula.

---

### The Compression Magic

**Original representation**:
- $|U| \times |I|$ parameters (200M × 15K = 3 trillion)

**MF representation**:
- $k \times |U|$ for user factors
- $k \times |I|$ for item factors
- Total: $k \times (|U| + |I|)$

**With k=100**:
- 100 × (200,000,000 + 15,000) ≈ 20 billion parameters
- **150× compression!**

And we get **generalization for free**: Any user-item pair has a prediction, even if never observed!

---

## Complete Numerical Walkthrough: 3-User, 4-Movie System

*Now let's work through a complete example with actual numbers. Follow along carefully.*

### Setup

**Observed ratings** (5-star scale):

|           | Movie 1 (Action) | Movie 2 (Drama) | Movie 3 (Sci-Fi) | Movie 4 (Romance) |
|-----------|-----------------|-----------------|------------------|-------------------|
| User 1 (Action Fan) | 5 | 2 | ? | 1 |
| User 2 (Eclectic)   | ? | 4 | 5 | ? |
| User 3 (Romance Fan)| 2 | ? | 1 | 5 |

We want to predict the missing entries (?).

---

### Step 1: Choose k and Initialize

Let's use $k = 2$ latent factors. *What might these factors represent?*

After training (we'll cover how later), suppose we learned:

**User factors** $U \in \mathbb{R}^{2 \times 3}$:
$$U = \begin{bmatrix}
0.9 & 0.3 & -0.7 \\
0.7 & 0.8 & -0.8
\end{bmatrix}$$

- Column 1: User 1 = $[0.9, 0.7]^T$ (high on both factors → likes action/adventure)
- Column 2: User 2 = $[0.3, 0.8]^T$ (low factor 1, high factor 2 → eclectic taste)
- Column 3: User 3 = $[-0.7, -0.8]^T$ (negative on both → opposite preferences)

**Item factors** $V \in \mathbb{R}^{2 \times 4}$:
$$V = \begin{bmatrix}
0.8 & -0.2 & 0.9 & -0.8 \\
0.6 & 0.7 & 0.5 & -0.6
\end{bmatrix}$$

- Column 1: Movie 1 (Action) = $[0.8, 0.6]^T$ (high on Factor 1)
- Column 2: Movie 2 (Drama) = $[-0.2, 0.7]^T$ (high on Factor 2)
- Column 3: Movie 3 (Sci-Fi) = $[0.9, 0.5]^T$ (very high on Factor 1)
- Column 4: Movie 4 (Romance) = $[-0.8, -0.6]^T$ (negative on both)

---

### Step 2: Compute All Predictions

**Prediction matrix**: $\hat{R} = U^T V$

$$U^T = \begin{bmatrix}
0.9 & 0.7 \\
0.3 & 0.8 \\
-0.7 & -0.8
\end{bmatrix}$$

$$\hat{R} = U^T V = \begin{bmatrix}
0.9 & 0.7 \\
0.3 & 0.8 \\
-0.7 & -0.8
\end{bmatrix}
\begin{bmatrix}
0.8 & -0.2 & 0.9 & -0.8 \\
0.6 & 0.7 & 0.5 & -0.6
\end{bmatrix}$$

Let me compute each entry step by step:

---

**Row 1 (User 1):**

$\hat{r}_{11} = 0.9 \times 0.8 + 0.7 \times 0.6 = 0.72 + 0.42 = 1.14$

$\hat{r}_{12} = 0.9 \times (-0.2) + 0.7 \times 0.7 = -0.18 + 0.49 = 0.31$

$\hat{r}_{13} = 0.9 \times 0.9 + 0.7 \times 0.5 = 0.81 + 0.35 = 1.16$

$\hat{r}_{14} = 0.9 \times (-0.8) + 0.7 \times (-0.6) = -0.72 - 0.42 = -1.14$

---

**Row 2 (User 2):**

$\hat{r}_{21} = 0.3 \times 0.8 + 0.8 \times 0.6 = 0.24 + 0.48 = 0.72$

$\hat{r}_{22} = 0.3 \times (-0.2) + 0.8 \times 0.7 = -0.06 + 0.56 = 0.50$

$\hat{r}_{23} = 0.3 \times 0.9 + 0.8 \times 0.5 = 0.27 + 0.40 = 0.67$

$\hat{r}_{24} = 0.3 \times (-0.8) + 0.8 \times (-0.6) = -0.24 - 0.48 = -0.72$

---

**Row 3 (User 3):**

$\hat{r}_{31} = (-0.7) \times 0.8 + (-0.8) \times 0.6 = -0.56 - 0.48 = -1.04$

$\hat{r}_{32} = (-0.7) \times (-0.2) + (-0.8) \times 0.7 = 0.14 - 0.56 = -0.42$

$\hat{r}_{33} = (-0.7) \times 0.9 + (-0.8) \times 0.5 = -0.63 - 0.40 = -1.03$

$\hat{r}_{34} = (-0.7) \times (-0.8) + (-0.8) \times (-0.6) = 0.56 + 0.48 = 1.04$

---

### Step 3: Scale to Rating Range

The raw predictions are in range [-1.14, 1.16]. We need to rescale to [1, 5] stars.

**Rescaling formula**: $r_{\text{scaled}} = \frac{(\hat{r} - \min)}{(\max - \min)} \times 4 + 1$

With $\min = -1.14$ and $\max = 1.16$, the range is $2.30$.

Let me rescale the key predictions:

| Raw $\hat{r}$ | Calculation | 5-Star Rating |
|---------------|-------------|---------------|
| $\hat{r}_{13} = 1.16$ | $(1.16 - (-1.14)) / 2.30 \times 4 + 1$ | **5.0** |
| $\hat{r}_{11} = 1.14$ | $(1.14 - (-1.14)) / 2.30 \times 4 + 1$ | **4.97** |
| $\hat{r}_{24} = -0.72$ | $(-0.72 - (-1.14)) / 2.30 \times 4 + 1$ | **1.73** |
| $\hat{r}_{34} = 1.04$ | $(1.04 - (-1.14)) / 2.30 \times 4 + 1$ | **4.79** |

---

### Step 4: Interpret the Results

**Final prediction matrix (5-star scale)**:

|           | Movie 1 | Movie 2 | Movie 3 | Movie 4 |
|-----------|---------|---------|---------|---------|
| User 1    | **4.97** (actual: 5) | **2.26** (actual: 2) | **5.00** (predict: high!) | **1.00** (actual: 1) |
| User 2    | **3.24** | **2.86** (actual: 4) | **3.15** (actual: 5) | **1.73** |
| User 3    | **1.00** (actual: 2) | **2.26** | **1.04** (actual: 1) | **4.79** (actual: 5) |

*Look at what the model learned!*

1. **User 1 will love Movie 3 (Sci-Fi)** - predicted 5.0 stars. Makes sense: User 1 likes action (gave Movie 1 a 5), and Movie 3 is also action-oriented.

2. **User 2 won't like Movie 4 (Romance)** - predicted 1.73 stars. User 2's tastes don't align with romance.

3. **User 3 will love Movie 4 (Romance)** - predicted 4.79 stars. This matches actual rating of 5!

---

### What the Latent Factors Might Mean

Looking at the learned factors:

**Factor 1** (first row of V):
- Movie 1 (Action): +0.8
- Movie 2 (Drama): -0.2
- Movie 3 (Sci-Fi): +0.9
- Movie 4 (Romance): -0.8

*Interpretation*: Factor 1 captures "action/intensity level"

**Factor 2** (second row of V):
- Movie 1: +0.6
- Movie 2: +0.7
- Movie 3: +0.5
- Movie 4: -0.6

*Interpretation*: Factor 2 might capture "seriousness" or "plot complexity"

**Remember**: These factors are **learned from data**, not predefined. The model discovers them automatically!

---

## Connection to SVD (Singular Value Decomposition)

*You might be wondering: isn't this just SVD?*

### Standard SVD

For a **complete** matrix $R$:

$$R = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{|U| \times |U|}$: Left singular vectors (orthogonal)
- $\Sigma \in \mathbb{R}^{|U| \times |I|}$: Diagonal matrix of singular values
- $V \in \mathbb{R}^{|I| \times |I|}$: Right singular vectors (orthogonal)

### Truncated SVD (Low-Rank Approximation)

Keep only top-$k$ singular values:

$$R \approx U_k \Sigma_k V_k^T$$

This is mathematically optimal (Eckart-Young theorem): It minimizes $\|R - U_k \Sigma_k V_k^T\|_F$ among all rank-$k$ matrices.

---

### Why Can't We Just Use SVD for Recommendations?

*This is a critical question. Let me explain the three problems.*

**Problem 1: Missing Entries**

SVD requires a **complete** matrix. Our matrix is 99%+ empty!

*Attempted solutions and their failures:*
- Fill missing with 0 → Treats "unknown" as "dislike" (wrong!)
- Fill with row/column means → Still guessing, and SVD will overfit to these guesses
- Fill with global mean → Same problem

**Problem 2: We Care About the Wrong Thing**

SVD minimizes error on **all entries**:
$$\min \|R - U_k \Sigma_k V_k^T\|_F^2 = \min \sum_{u,i} (r_{ui} - \hat{r}_{ui})^2$$

But we only care about **observed entries**. We want:
$$\min \sum_{(u,i) \in \text{observed}} (r_{ui} - \hat{r}_{ui})^2$$

**Problem 3: No Regularization**

SVD finds the exact rank-$k$ approximation. For sparse data, this overfits!

We need regularization to prevent the model from memorizing the training data:
$$\min \sum_{(u,i) \in \text{observed}} (r_{ui} - \hat{r}_{ui})^2 + \lambda(\|U\|^2 + \|V\|^2)$$

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

*Let me give you the intuition behind each reason.*

### 1. Dimensionality Reduction

**Curse of Dimensionality**:
- Original: $|U| \times |I|$ dimensions
- Sparse data → unreliable estimates

**MF Solution**:
- Project to $k$-dimensional latent space
- $k \ll |U|, |I|$
- Denser representation → more reliable

*Analogy*: Instead of asking 15,000 yes/no questions about each movie, we ask 100 questions about underlying themes.

---

### 2. Collaborative Signal Capture

*This is where the "collaborative" part comes in.*

**Example**:
- Alice loves "The Matrix" and "Inception"
- Bob loves "The Matrix" and "Blade Runner"
- Carol loves "Inception" and "Blade Runner"

**What happens during training**:
- Alice and Bob both rate "The Matrix" highly → Their user vectors move closer together
- All three movies share sci-fi elements → Their item vectors cluster
- Result: "Blade Runner" gets recommended to Alice even though she never rated it!

**The magic**: Similar users and similar items automatically cluster in latent space.

---

### 3. Generalization to Unseen Pairs

**Direct CF** (user-based, item-based):
- Need overlap between users/items to compute similarity
- If Alice and Bob share no common items → Can't compute similarity

**MF**:
- Learns latent factors from **all** observed data
- Can predict for **any** user-item pair
- Even without direct overlap, latent factors bridge the gap

*Example*:
- Alice and Bob never rated the same item
- But Alice rated sci-fi movies A, B highly
- Bob rated sci-fi movies C, D highly
- Both get high "likes sci-fi" factor → Similar users!

---

### 4. Implicit Feature Learning

**No need to manually define features** (genre, director, actors).

**MF automatically discovers**:
- Latent factors that best explain observed ratings
- May capture genre, mood, complexity, popularity, etc.
- Learns from data, not human intuition

**Advantage**: Works even when good features are hard to define (e.g., "vibe" of a restaurant).

---

## Number of Factors (k)

### The Intuition: What If k=1?

*Before looking at tuning, let's understand extremes.*

**If k=1**: Every user and item is described by a single number.

$$\hat{r}_{ui} = u_u \cdot v_i$$

*What could one number capture?* Maybe "average quality" or "mainstream appeal".

**Problem**: Can't distinguish between action lovers and romance lovers. The single factor conflates everything.

**Example with k=1**:
- "The Matrix": $v = 0.9$ (highly rated)
- "The Notebook": $v = 0.8$ (also highly rated)
- These would be similar! But they appeal to different audiences.

---

### What If k=1000?

**With 1000 factors**: Maximum expressiveness, but...

**Problems**:
1. **Overfitting**: With enough factors, model memorizes training data
2. **No generalization**: Each user-item pair essentially gets its own parameter
3. **Computational cost**: 1000× more parameters to learn

**The sweet spot**: $k$ large enough to capture important factors, small enough to generalize.

---

### How to Choose k?

**Typical values**: k = 20-200

**Methods to choose**:

1. **Cross-validation** (recommended):
   - Try k = 10, 20, 50, 100, 200
   - Pick the one with best validation RMSE

2. **Elbow method**:
   - Plot RMSE vs. k
   - Look for the "elbow" where improvements slow down

3. **Domain knowledge**:
   - Complex domains (movies) → more factors
   - Simple domains (binary likes) → fewer factors

---

### Practical Guidelines

| Dataset | Users | Items | Interactions | Typical k |
|---------|-------|-------|--------------|-----------|
| MovieLens 100K | 943 | 1,682 | 100K | 10-50 |
| MovieLens 1M | 6,040 | 3,706 | 1M | 50-100 |
| MovieLens 10M | 71,567 | 10,681 | 10M | 100-200 |
| Netflix Prize | 480K | 17K | 100M | 200-500 |

**General rule**: More data → can support more factors without overfitting.

---

## Bias Terms

### The Problem with Basic MF

$$\hat{r}_{ui} = \mathbf{u}_u^T \mathbf{v}_i$$

*What's missing?* Consider:

- "The Godfather": Average rating 4.5 (universally loved)
- "Gigli": Average rating 1.5 (universally panned)
- User Alice: Tends to rate 0.5 stars higher than average
- User Bob: Tends to rate 0.5 stars lower than average

**Basic MF forces latent factors to capture these biases** → wastes model capacity on obvious patterns.

---

### MF with Bias Terms

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i$$

where:
- $\mu$: Global average rating (e.g., 3.5 stars)
- $b_u$: User bias (how much user $u$ deviates from average)
- $b_i$: Item bias (how much item $i$ deviates from average)
- $\mathbf{u}_u^T \mathbf{v}_i$: User-item interaction (what's left after accounting for biases)

---

### Numerical Example with Biases

**Setup**:
- $\mu = 3.5$ (global average)
- "The Godfather": $b_i = +1.0$ (rated 1 star above average)
- Alice: $b_u = +0.3$ (rates 0.3 higher than average)
- Latent interaction: $\mathbf{u}_{Alice}^T \mathbf{v}_{Godfather} = 0.4$

**Prediction**:
$$\hat{r} = 3.5 + 0.3 + 1.0 + 0.4 = 5.2$$

We'd clip to 5.0 (max rating).

**Interpretation**:
- Baseline: 3.5 (average movie, average user)
- +1.0: Great movie (everyone likes it)
- +0.3: Generous user (rates higher than most)
- +0.4: Good match (Alice's preferences align with Godfather's themes)

---

### Why Biases Help

1. **Separates bias from interaction**: Latent factors focus on *preferences*, not rating scales
2. **Better predictions**: Explicitly models known patterns
3. **Faster convergence**: Easier optimization (less to learn)
4. **Interpretability**: Can explain "Alice rates high" vs "Alice likes this genre"

---

## What Can Go Wrong?

*Let me warn you about common failure modes.*

### Failure Mode 1: Cold Start

**Problem**: New user or item with no ratings.

**Symptoms**:
- Predictions default to global mean
- New items never get recommended
- New users get generic recommendations

**Solutions**:
- Content features for cold items
- Demographic features for cold users
- Active learning (ask for ratings)
- Hybrid models (MF + content-based)

---

### Failure Mode 2: Popularity Bias

**Problem**: Popular items dominate recommendations.

**Symptoms**:
- Everyone gets recommended the same 50 movies
- Niche items never surface
- Long-tail items ignored

**Cause**: Popular items have more training data → better factor estimates → higher confidence → more recommendations → more data (cycle!)

**Solutions**:
- Regularize popular items more heavily
- Popularity-aware loss functions
- Diversity in top-k selection
- Inverse propensity weighting

---

### Failure Mode 3: Overfitting

**Problem**: Model memorizes training data, fails on new data.

**Symptoms**:
- Low training RMSE, high test RMSE
- Predictions too extreme (very high or very low)
- Factors have very large values

**Solutions**:
- Increase regularization $\lambda$
- Decrease number of factors $k$
- Early stopping (monitor validation error)
- Dropout during training (for neural MF)

---

### Failure Mode 4: Underfitting

**Problem**: Model too simple to capture patterns.

**Symptoms**:
- High training RMSE, high test RMSE
- Predictions cluster around mean
- Can't distinguish user preferences

**Solutions**:
- Increase $k$
- Decrease regularization $\lambda$
- Add bias terms
- Train longer

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

**What can go wrong**:
- Cold start (new users/items)
- Popularity bias (recommending only popular items)
- Overfitting (memorizing training data)
- Underfitting (model too simple)

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
