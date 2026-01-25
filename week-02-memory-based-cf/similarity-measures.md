# Week 2: Similarity Measures Deep Dive

## Learning Objectives

- Master all major similarity metrics for collaborative filtering
- Understand when to use each metric
- Learn significance weighting and normalization techniques
- Implement optimizations for sparse data

---

## Why Similarity Matters

**Core of memory-based CF**: Finding similar users or items.

**Quality of similarity → Quality of recommendations**

**Challenge**: Sparse data makes similarity unreliable.

**Goal**: Choose and optimize similarity measures for your domain.

---

## Similarity Metrics Comparison

| Metric | Best For | Handles Rating Scale? | Sparsity Sensitive? | Complexity |
|--------|----------|---------------------|-------------------|------------|
| **Cosine** | Binary data, implicit feedback | No | Medium | O(\|overlap\|) |
| **Pearson** | Explicit ratings, different scales | Yes (mean-centered) | High | O(\|overlap\|) |
| **Adjusted Cosine** | Items with different avg ratings | Yes | High | O(\|overlap\|) |
| **Jaccard** | Binary sets (clicked/not clicked) | N/A | Low | O(\|union\|) |
| **Euclidean** | Dense data | No | Very High | O(\|overlap\|) |

---

## 1. Cosine Similarity

### Formula

$$\text{cosine}(u, v) = \frac{\mathbf{r}_u \cdot \mathbf{r}_v}{||\mathbf{r}_u|| \cdot ||\mathbf{r}_v||} = \frac{\sum_i r_{ui} \cdot r_{vi}}{\sqrt{\sum_i r_{ui}^2} \cdot \sqrt{\sum_i r_{vi}^2}}$$

### Geometric Interpretation

**Angle between rating vectors**:
- 0° (cosine = 1): Identical preferences
- 90° (cosine = 0): Orthogonal, no similarity
- 180° (cosine = -1): Opposite preferences (rare in practice)

### When to Use

✅ **Good for**:
- Binary data (clicked/not clicked)
- Implicit feedback (views, plays)
- When magnitude doesn't matter (only direction)

❌ **Not ideal for**:
- Explicit ratings with different user scales
- Alice rates 4-5, Bob rates 1-2 (same preferences, different scales)

### Example

```python
import numpy as np

def cosine_similarity(user1, user2):
    """
    Compute cosine similarity between two users.

    Args:
        user1, user2: Rating vectors (numpy arrays)

    Returns:
        Similarity score [0, 1]
    """
    # Only consider co-rated items
    mask = (user1 > 0) & (user2 > 0)

    if mask.sum() == 0:
        return 0  # No overlap

    u1 = user1[mask]
    u2 = user2[mask]

    dot_product = np.dot(u1, u2)
    norm1 = np.linalg.norm(u1)
    norm2 = np.linalg.norm(u2)

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot_product / (norm1 * norm2)

# Example
user_a = np.array([5, 4, 0, 3, 0, 5])  # 0 = not rated
user_b = np.array([4, 0, 3, 2, 5, 0])

sim = cosine_similarity(user_a, user_b)
print(f"Cosine similarity: {sim:.3f}")
# Output: Cosine similarity: 0.998 (very similar)
```

### Properties

**Range**: [0, 1] for non-negative ratings, [-1, 1] for any ratings

**Symmetric**: cosine(u, v) = cosine(v, u)

**Not a metric**: Doesn't satisfy triangle inequality

---

## 2. Pearson Correlation Coefficient

### Formula

$$\text{Pearson}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \cdot \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}$$

where:
- $I_{uv}$ = items rated by both users
- $\bar{r}_u$ = mean rating of user $u$

### Mean-Centering

**Key difference from cosine**: Subtracts user's average rating.

**Why important?**
- Alice: rates 4-5 (generous)
- Bob: rates 1-2 (harsh)
- Both love the same movies, but cosine sees them as different
- Pearson: Accounts for rating scale differences

### When to Use

✅ **Best for**:
- Explicit ratings (1-5 stars)
- Users with different rating scales
- When you care about agreement on **relative** preferences

❌ **Not ideal for**:
- Binary data (can't mean-center)
- Very sparse data (mean unreliable with few ratings)

### Example

```python
def pearson_correlation(user1, user2):
    """
    Compute Pearson correlation.

    Returns:
        Correlation [-1, 1]
    """
    # Co-rated items
    mask = (user1 > 0) & (user2 > 0)

    if mask.sum() < 2:  # Need at least 2 overlaps
        return 0

    u1 = user1[mask]
    u2 = user2[mask]

    # Mean-center
    u1_centered = u1 - u1.mean()
    u2_centered = u2 - u2.mean()

    # Pearson correlation
    numerator = np.dot(u1_centered, u2_centered)
    denominator = np.linalg.norm(u1_centered) * np.linalg.norm(u2_centered)

    if denominator == 0:
        return 0

    return numerator / denominator

# Example: Users with different scales but similar tastes
alice = np.array([5, 5, 4, 5, 0])  # Generous rater (avg 4.75)
bob = np.array([2, 2, 1, 2, 0])    # Harsh rater (avg 1.75)

print(f"Cosine: {cosine_similarity(alice, bob):.3f}")
print(f"Pearson: {pearson_correlation(alice, bob):.3f}")
# Output: Cosine: 0.998, Pearson: 1.000 (perfect correlation after centering)
```

### Properties

**Range**: [-1, 1]
- +1: Perfect positive correlation
- 0: No correlation
- -1: Perfect negative correlation

**Handles different rating scales**: Yes (mean-centered)

**Sensitive to outliers**: Yes (squared deviations)

---

## 3. Adjusted Cosine Similarity

### Formula

For **items** (transpose of Pearson for users):

$$\text{adj\_cosine}(i, j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)(r_{uj} - \bar{r}_u)}{\sqrt{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)^2} \cdot \sqrt{\sum_{u \in U_{ij}} (r_{uj} - \bar{r}_u)^2}}$$

where:
- $U_{ij}$ = users who rated both items
- $\bar{r}_u$ = mean rating of **user** $u$ (not item)

### Key Difference

**Pearson** (for users): Subtracts user's mean
**Adjusted Cosine** (for items): Subtracts each user's mean

**Why?** Items rated by different users need to account for user rating scales.

### When to Use

✅ **Best for**:
- **Item-based collaborative filtering**
- Items rated by users with different scales

❌ **Not for**:
- User-based CF (use Pearson instead)

### Example

```python
def adjusted_cosine_items(item1, item2, user_means):
    """
    Adjusted cosine for items.

    Args:
        item1, item2: Rating vectors (users × 1)
        user_means: Mean rating for each user

    Returns:
        Similarity [-1, 1]
    """
    # Users who rated both items
    mask = (item1 > 0) & (item2 > 0)

    if mask.sum() < 2:
        return 0

    i1 = item1[mask]
    i2 = item2[mask]
    means = user_means[mask]

    # Adjust by user means
    i1_adj = i1 - means
    i2_adj = i2 - means

    numerator = np.dot(i1_adj, i2_adj)
    denominator = np.linalg.norm(i1_adj) * np.linalg.norm(i2_adj)

    if denominator == 0:
        return 0

    return numerator / denominator

# Example: Items rated by users with different scales
# Movie A ratings by 3 users: [5, 3, 4]
# Movie B ratings by same users: [5, 3, 5]
# User means: [4.5, 2.5, 3.0] (Alice generous, Bob harsh, Carol medium)

movie_a = np.array([5, 3, 4])
movie_b = np.array([5, 3, 5])
user_means = np.array([4.5, 2.5, 3.0])

sim = adjusted_cosine_items(movie_a, movie_b, user_means)
print(f"Adjusted cosine: {sim:.3f}")
# Adjusts for user rating scales
```

---

## 4. Jaccard Similarity

### Formula

$$\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

where:
- $A$ = set of items user A interacted with
- $B$ = set of items user B interacted with

### When to Use

✅ **Perfect for**:
- **Binary implicit feedback** (clicked/not clicked)
- Sets (no rating magnitude)
- Simple and interpretable

❌ **Not for**:
- Explicit ratings (ignores magnitudes)
- When rating values matter

### Example

```python
def jaccard_similarity(set1, set2):
    """
    Jaccard similarity for sets.

    Args:
        set1, set2: Sets of item IDs

    Returns:
        Similarity [0, 1]
    """
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0

    return intersection / union

# Example: Users as sets of watched movies
alice_movies = {1, 2, 3, 5, 8, 10}
bob_movies = {2, 3, 4, 5, 9}

sim = jaccard_similarity(alice_movies, bob_movies)
print(f"Jaccard similarity: {sim:.3f}")
# Intersection: {2, 3, 5} → 3 movies
# Union: {1, 2, 3, 4, 5, 8, 9, 10} → 8 movies
# Jaccard: 3/8 = 0.375
```

### Properties

**Range**: [0, 1]

**Symmetric**: Jaccard(A, B) = Jaccard(B, A)

**Simple**: Easy to compute and interpret

**Popularity bias**: Popular items dominate intersection

---

## 5. Euclidean Distance (and Similarity)

### Distance Formula

$$\text{distance}(u, v) = \sqrt{\sum_{i \in I_{uv}} (r_{ui} - r_{vi})^2}$$

### Similarity (Inverse)

$$\text{similarity}(u, v) = \frac{1}{1 + \text{distance}(u, v)}$$

### When to Use

✅ **Good for**:
- Dense data (most items rated)
- When absolute differences matter

❌ **Not ideal for**:
- Sparse data (sensitive to missing values)
- Different rating scales
- Recommendation systems (rarely used)

### Example

```python
def euclidean_similarity(user1, user2):
    """
    Euclidean distance-based similarity.

    Returns:
        Similarity (0, 1], higher is more similar
    """
    mask = (user1 > 0) & (user2 > 0)

    if mask.sum() == 0:
        return 0

    distance = np.linalg.norm(user1[mask] - user2[mask])
    return 1 / (1 + distance)

# Example
user_a = np.array([5, 4, 3, 0, 5])
user_b = np.array([5, 4, 3, 0, 4])

sim = euclidean_similarity(user_a, user_b)
print(f"Euclidean similarity: {sim:.3f}")
# Distance = sqrt((5-5)^2 + (4-4)^2 + (3-3)^2 + (5-4)^2) = 1.0
# Similarity = 1/(1+1) = 0.5
```

---

## Advanced Techniques

### 1. Significance Weighting

**Problem**: Similarity based on 1-2 co-rated items is unreliable.

**Solution**: Downweight similarities with few overlaps.

$$\text{sim}_{weighted}(u, v) = \text{sim}(u, v) \cdot \min\left(1, \frac{|I_{uv}|}{\tau}\right)$$

where:
- $\tau$ = significance threshold (e.g., 50)
- $|I_{uv}|$ = number of co-rated items

**Example**:
- Similarity = 0.9, but only 5 co-rated items
- $\tau = 50$
- Weighted: $0.9 \times \min(1, 5/50) = 0.9 \times 0.1 = 0.09$

**Effect**: Heavily discounts unreliable similarities.

```python
def significance_weighting(similarity, n_overlap, threshold=50):
    """Apply significance weighting."""
    weight = min(1.0, n_overlap / threshold)
    return similarity * weight

# Example
sim = 0.9  # High similarity
n_overlap = 5  # But only 5 co-rated items

weighted_sim = significance_weighting(sim, n_overlap)
print(f"Original: {sim:.3f}, Weighted: {weighted_sim:.3f}")
# Output: Original: 0.900, Weighted: 0.090 (unreliable, downweighted)
```

---

### 2. Variance Weighting

**Problem**: Users who rate everything the same (e.g., all 5s) aren't informative.

**Solution**: Weight by user variance.

$$\text{sim}_{weighted}(u, v) = \text{sim}(u, v) \cdot \sigma_u \cdot \sigma_v$$

where $\sigma_u$ = standard deviation of user $u$'s ratings.

**Example**:
- Alice: [5, 5, 5, 5] → $\sigma = 0$ (no variance, uninformative)
- Bob: [1, 3, 5, 2, 4] → $\sigma = 1.58$ (high variance, informative)

```python
def variance_weighting(similarity, user1_ratings, user2_ratings):
    """Apply variance weighting."""
    # Only non-zero ratings
    u1 = user1_ratings[user1_ratings > 0]
    u2 = user2_ratings[user2_ratings > 0]

    if len(u1) < 2 or len(u2) < 2:
        return 0  # Can't compute variance

    std1 = np.std(u1)
    std2 = np.std(u2)

    return similarity * std1 * std2

# Example
alice = np.array([5, 5, 5, 5, 0])  # No variance
bob = np.array([1, 3, 5, 2, 4])    # High variance

sim = 0.8
weighted = variance_weighting(sim, alice, bob)
print(f"Weighted by variance: {weighted:.3f}")
# Alice's std ≈ 0 → weighted similarity ≈ 0 (uninformative)
```

---

### 3. Default Voting

**Problem**: Users with few ratings have unreliable means.

**Solution**: Assume default rating for missing items when computing mean.

$$\bar{r}_u = \frac{\sum_i r_{ui} + k \cdot r_{default}}{|I_u| + k}$$

where:
- $k$ = strength of prior (e.g., 25)
- $r_{default}$ = default rating (e.g., 3.0 for 1-5 scale)

**Effect**: Shrinks mean toward default for users with few ratings.

---

### 4. Case Amplification

**Problem**: Similarities near 0.5 are ambiguous.

**Solution**: Amplify similarities to make distinctions clearer.

$$\text{sim}_{amplified} = \text{sim}^\rho$$

where $\rho > 1$ (typically 2.5).

**Effect**:
- High similarities (0.9) → even higher (0.90^2.5 = 0.87)
- Low similarities (0.3) → much lower (0.30^2.5 = 0.05)
- Sharpens distinctions

```python
def case_amplification(similarity, rho=2.5):
    """Amplify similarity to sharpen distinctions."""
    return similarity ** rho

# Example
sims = [0.9, 0.7, 0.5, 0.3]
amplified = [case_amplification(s) for s in sims]

print("Original:", sims)
print("Amplified:", [f"{s:.3f}" for s in amplified])
# Original: [0.9, 0.7, 0.5, 0.3]
# Amplified: ['0.871', '0.483', '0.177', '0.049']
# Sharpens high vs low similarities
```

---

## Practical Recommendations

### For Explicit Ratings (MovieLens, Netflix)

**Best**: Pearson correlation
- Handles rating scale differences
- Mean-centering essential

**Enhancements**:
- Significance weighting (threshold = 50)
- Variance weighting (optional)

**Code**:
```python
def robust_pearson(user1, user2, min_overlap=5, sig_threshold=50):
    """Pearson with significance weighting."""
    # Compute Pearson
    sim = pearson_correlation(user1, user2)

    # Count overlaps
    overlap = ((user1 > 0) & (user2 > 0)).sum()

    if overlap < min_overlap:
        return 0  # Too few overlaps

    # Significance weighting
    weight = min(1.0, overlap / sig_threshold)

    return sim * weight
```

---

### For Implicit Feedback (Clicks, Views)

**Best**: Cosine or Jaccard
- No rating magnitudes to worry about
- Cosine for weighted (view counts)
- Jaccard for pure binary

**Enhancement**: Significance weighting

---

### For Item-Based CF

**Best**: Adjusted cosine
- Accounts for user rating scale differences
- More reliable than Pearson for items

**Alternatives**: Cosine (if users have similar scales)

---

### For Very Sparse Data

**Best**: Jaccard or simple overlap count
- Less sensitive to sparsity
- More robust with few co-ratings

**Avoid**: Pearson (needs sufficient co-ratings for reliable mean)

---

## Computational Optimizations

### 1. Exploit Sparsity

**Don't iterate over all items**, only co-rated ones.

```python
# Inefficient
for i in range(n_items):
    if user1[i] > 0 and user2[i] > 0:
        # Compute

# Efficient (sparse matrix)
import scipy.sparse as sp

def cosine_sparse(user1_sparse, user2_sparse):
    """Cosine for sparse vectors."""
    numerator = user1_sparse.dot(user2_sparse.T).toarray()[0, 0]
    norm1 = sp.linalg.norm(user1_sparse)
    norm2 = sp.linalg.norm(user2_sparse)

    if norm1 == 0 or norm2 == 0:
        return 0

    return numerator / (norm1 * norm2)
```

---

### 2. Precompute Norms

For cosine, precompute $||\mathbf{r}_u||$ for all users.

```python
# Precompute
user_norms = np.linalg.norm(ratings_matrix, axis=1)

# Then for each pair
def cosine_fast(user1, user2, norm1, norm2):
    dot = np.dot(user1, user2)
    return dot / (norm1 * norm2)
```

---

### 3. Top-K Computation

**Don't compute all pairs**, only top-K neighbors.

**Approach**:
1. Compute similarities for random sample
2. Find approximate top-K
3. Refine with exact computation

**Libraries**: scikit-learn's NearestNeighbors, Annoy, FAISS

---

## Summary

### Similarity Metrics

| Metric | Formula | Range | Best For |
|--------|---------|-------|----------|
| **Cosine** | $\frac{\mathbf{u} \cdot \mathbf{v}}{\\|\mathbf{u}\\| \\|\mathbf{v}\\|}$ | [0,1] | Binary, implicit |
| **Pearson** | $\frac{\sum (r_i - \bar{r}_u)(r_i - \bar{r}_v)}{\sigma_u \sigma_v}$ | [-1,1] | Explicit, user-based |
| **Adjusted Cosine** | Cosine with user mean subtraction | [-1,1] | Item-based |
| **Jaccard** | $\frac{\\|A \cap B\\|}{\\|A \cup B\\|}$ | [0,1] | Binary sets |
| **Euclidean** | $\frac{1}{1 + \\|\mathbf{u} - \mathbf{v}\\|}$ | (0,1] | Dense data (rare) |

### Enhancements

- **Significance weighting**: Downweight few co-ratings
- **Variance weighting**: Downweight low-variance users
- **Case amplification**: Sharpen high vs low similarities

### Recommendations

**Explicit ratings**: Pearson + significance weighting
**Implicit feedback**: Cosine or Jaccard
**Item-based**: Adjusted cosine
**Sparse data**: Jaccard

**Next**: See **code-examples.md** for full implementations.

---

## References

1. **Herlocker, J. L., et al. (2002)**. "Evaluating collaborative filtering recommender systems". *ACM TOIS*.
2. **Sarwar, B., et al. (2001)**. "Item-based collaborative filtering recommendation algorithms". *WWW*.
3. **Breese, J. S., et al. (1998)**. "Empirical analysis of predictive algorithms for collaborative filtering". *UAI*.
