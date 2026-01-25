# Week 2: User-Based Collaborative Filtering

## Learning Objectives

- Understand the intuition behind user-based collaborative filtering
- Compute user-user similarity using various metrics
- Implement k-nearest neighbors for recommendation
- Analyze computational complexity and limitations

---

## Intuition: "Users Similar to You Liked..."

### Core Idea

**If users agreed in the past, they will likely agree in the future.**

**Example**:
- Alice and Bob both loved *The Matrix*, *Inception*, and *Interstellar*
- Alice also loved *Blade Runner 2049*
- Bob hasn't seen *Blade Runner 2049*
- **Recommendation**: Suggest *Blade Runner 2049* to Bob

**Mathematical Formulation**:
1. Find users similar to target user
2. Aggregate their ratings/preferences
3. Recommend items they liked but target user hasn't seen

---

## The Algorithm

### Step 1: Compute User-User Similarity

Given user-item rating matrix $R \in \mathbb{R}^{|U| \times |I|}$:

$$\text{sim}(u, v) = f(r_u, r_v)$$

where $r_u$ and $r_v$ are rating vectors for users $u$ and $v$.

### Step 2: Find k-Nearest Neighbors

$$N_k(u) = \{v_1, v_2, \ldots, v_k\}$$

where $v_1, \ldots, v_k$ are the $k$ users most similar to $u$.

### Step 3: Predict Rating

**Weighted Average**:

$$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} \text{sim}(u,v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N_k(u)} |\text{sim}(u,v)|}$$

where:
- $\bar{r}_u$ = average rating by user $u$
- $r_{vi}$ = rating by user $v$ for item $i$
- $\text{sim}(u,v)$ = similarity between users $u$ and $v$

**Interpretation**:
- Start with user $u$'s average rating
- Adjust based on how neighbors rated item $i$
- Weight by similarity (more similar users have more influence)
- Normalize by sum of similarities

---

## Similarity Metrics

### 1. Pearson Correlation Coefficient

**Measures linear correlation** between two users' rating patterns.

$$\text{Pearson}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}$$

where $I_{uv}$ = set of items rated by both $u$ and $v$.

**Range**: $[-1, +1]$
- $+1$: Perfect positive correlation (users agree perfectly)
- $0$: No correlation
- $-1$: Perfect negative correlation (users always disagree)

**Advantages**:
- Accounts for different rating scales (one user rates 3-5, another rates 1-5)
- Mean-centered: Focuses on deviations from average

**Disadvantages**:
- Sensitive to outliers
- Requires many co-rated items for reliability
- Undefined if variance is zero

**Example**:
```
User A ratings: [5, 4, 3, 5, 4]  (mean = 4.2)
User B ratings: [4, 3, 2, 4, 3]  (mean = 3.2)

Deviations A: [0.8, -0.2, -1.2, 0.8, -0.2]
Deviations B: [0.8, -0.2, -1.2, 0.8, -0.2]

Numerator: 0.8×0.8 + (-0.2)×(-0.2) + ... = 2.4
Denominator: √2.4 × √2.4 = 2.4

Pearson = 2.4 / 2.4 = 1.0 (perfect correlation)
```

---

### 2. Cosine Similarity

**Measures angle** between rating vectors.

$$\text{cosine}(u, v) = \frac{r_u \cdot r_v}{||r_u|| \cdot ||r_v||} = \frac{\sum_{i \in I_{uv}} r_{ui} \cdot r_{vi}}{\sqrt{\sum_{i \in I_{uv}} r_{ui}^2} \sqrt{\sum_{i \in I_{uv}} r_{vi}^2}}$$

**Range**: $[0, +1]$ for ratings (always positive)

**Geometric Interpretation**:
- Angle of 0°: $\cos(0) = 1$ (identical preferences)
- Angle of 90°: $\cos(90) = 0$ (orthogonal, no similarity)

**Advantages**:
- Simple and fast to compute
- Works well when magnitude doesn't matter (only direction)

**Disadvantages**:
- Doesn't account for different rating scales
- Two users who both rate everything 5 have similarity 1, even if not informative

**Example**:
```
User A: [5, 3, 4]
User B: [4, 2, 3]

Dot product: 5×4 + 3×2 + 4×3 = 20 + 6 + 12 = 38
||A|| = √(25 + 9 + 16) = √50 ≈ 7.07
||B|| = √(16 + 4 + 9) = √29 ≈ 5.39

cosine = 38 / (7.07 × 5.39) ≈ 0.998
```

---

### 3. Adjusted Cosine Similarity

**Cosine similarity with mean-centering**.

$$\text{adj\_cosine}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}$$

**Note**: This is equivalent to Pearson correlation!

**Why "Adjusted"?**
- Original cosine doesn't account for rating scale differences
- Subtracting mean addresses this issue

---

### 4. Jaccard Similarity (for Binary Data)

**For implicit feedback** (watched/not watched, clicked/not clicked).

$$\text{Jaccard}(u, v) = \frac{|I_u \cap I_v|}{|I_u \cup I_v|}$$

where:
- $I_u$ = set of items user $u$ interacted with
- $I_v$ = set of items user $v$ interacted with

**Range**: $[0, 1]$

**Example**:
```
User A watched: {Movie1, Movie2, Movie3, Movie5}
User B watched: {Movie2, Movie3, Movie4, Movie5}

Intersection: {Movie2, Movie3, Movie5} → 3 movies
Union: {Movie1, Movie2, Movie3, Movie4, Movie5} → 5 movies

Jaccard = 3/5 = 0.6
```

**Advantages**:
- Simple for binary data
- No need for explicit ratings

**Disadvantages**:
- Ignores rating magnitudes
- Popularity bias (popular items dominate)

---

## k-Nearest Neighbors (kNN) Approach

### Algorithm

```
Algorithm: User-Based CF with kNN

Input:
  - User-item rating matrix R
  - Target user u
  - Target item i
  - Number of neighbors k
  - Similarity function sim()

Output:
  - Predicted rating r̂_ui

1. Compute similarity between u and all other users:
   similarities = {}
   for v in all_users:
       if v != u:
           similarities[v] = sim(u, v)

2. Find k nearest neighbors who rated item i:
   neighbors = top_k(similarities, k, rated_item_i=True)

3. Predict rating:
   numerator = 0
   denominator = 0
   for v in neighbors:
       numerator += similarities[v] × (r_vi - mean(v))
       denominator += |similarities[v]|

   r̂_ui = mean(u) + numerator / denominator

4. Return r̂_ui
```

### Choosing k

**Too small (k=1-5)**:
- High variance (overfitting to few neighbors)
- Sensitive to noise

**Too large (k=100+)**:
- Low variance but high bias
- Includes dissimilar users

**Typical values**: k = 20-50 for million-user systems

**Best practice**: Cross-validation to tune k

---

## Computational Complexity

### Analysis

**Step 1: Compute all pairwise similarities**
- For each pair of users: $O(|I|)$ (iterate over items)
- Total pairs: $\binom{|U|}{2} \approx |U|^2 / 2$
- **Complexity**: $O(|U|^2 \cdot |I|)$

**Step 2: Find k nearest neighbors**
- Sort similarities: $O(|U| \log |U|)$ per user
- For all users: $O(|U|^2 \log |U|)$

**Step 3: Predict rating**
- Aggregate k neighbors: $O(k)$
- For all (user, item) pairs: $O(|U| \cdot |I| \cdot k)$

**Total**: $O(|U|^2 \cdot |I|)$ dominated by similarity computation

### Scalability Issues

**Example: Netflix**
- $|U| = 260M$ users
- $|I| = 15K$ movies
- Complexity: $260M^2 \times 15K \approx 10^{21}$ operations

**Infeasible for real-time!**

---

## Optimizations and Tricks

### 1. Sparsity Exploitation

**Observation**: Most users share very few rated items.

**Solution**: Only compute similarity for user pairs with minimum overlap (e.g., ≥5 co-rated items).

**Benefit**: Reduces computation and improves reliability.

---

### 2. Significance Weighting

**Problem**: Similarity based on 1-2 co-rated items is unreliable.

**Solution**: Weight similarity by number of co-rated items.

$$\text{sim}_{weighted}(u, v) = \text{sim}(u, v) \cdot \min\left(1, \frac{|I_{uv}|}{threshold}\right)$$

**Example**: threshold = 50
- If $|I_{uv}| = 10$, multiply similarity by 10/50 = 0.2 (downweight)
- If $|I_{uv}| \geq 50$, use full similarity

---

### 3. Variance Weighting

**Problem**: Users who rate everything the same (e.g., always 5 stars) are not informative.

**Solution**: Weight by user's rating variance.

$$\text{sim}_{weighted}(u, v) = \text{sim}(u, v) \cdot \text{var}(u) \cdot \text{var}(v)$$

**Example**:
- User A: ratings = [5, 5, 5, 5, 5], var = 0 → similarity weighted to 0
- User B: ratings = [1, 3, 5, 2, 4], var = 2.5 → keeps similarity

---

### 4. Negative Similarity Handling

**Issue**: Negative Pearson correlation

**Options**:
1. **Ignore**: Only use positive similarities
2. **Absolute value**: $|\text{sim}(u,v)|$
3. **Include**: Can be informative (users with opposite tastes)

**Recommendation**: Use only positive similarities (negative correlations often spurious with few co-ratings)

---

## Example Walkthrough

### Given Data

```
         Movie1  Movie2  Movie3  Movie4  Movie5
Alice      5       3       4       ?       ?
Bob        3       1       2       2       4
Carol      4       3       5       3       ?
Dave       3       3       3       4       3
Eve        ?       ?       5       ?       3
```

**Task**: Predict Alice's rating for Movie4 using k=2 nearest neighbors (Pearson correlation).

---

### Step 1: Compute Similarities

**Alice vs. Bob**:
- Co-rated: Movie1, Movie2, Movie3
- Alice: [5, 3, 4], mean = 4
- Bob: [3, 1, 2], mean = 2

$$\text{Pearson} = \frac{(5-4)(3-2) + (3-4)(1-2) + (4-4)(2-2)}{\sqrt{(5-4)^2 + (3-4)^2 + (4-4)^2} \sqrt{(3-2)^2 + (1-2)^2 + (2-2)^2}}$$

$$= \frac{1 \cdot 1 + (-1) \cdot (-1) + 0 \cdot 0}{\sqrt{1 + 1 + 0} \sqrt{1 + 1 + 0}} = \frac{2}{\sqrt{2} \cdot \sqrt{2}} = 1.0$$

**Alice vs. Carol**:
- Co-rated: Movie1, Movie2, Movie3
- Alice: [5, 3, 4], mean = 4
- Carol: [4, 3, 5], mean = 4

$$\text{Pearson} = \frac{(5-4)(4-4) + (3-4)(3-4) + (4-4)(5-4)}{\sqrt{1 + 1 + 0} \sqrt{0 + 1 + 1}} = \frac{0 + 1 + 0}{\sqrt{2} \cdot \sqrt{2}} = 0.5$$

**Alice vs. Dave**:
- Co-rated: Movie1, Movie2, Movie3
- Alice: [5, 3, 4], mean = 4
- Dave: [3, 3, 3], mean = 3

$$\text{Pearson} = \frac{(5-4)(3-3) + (3-4)(3-3) + (4-4)(3-3)}{\sqrt{1 + 1 + 0} \sqrt{0 + 0 + 0}} = \frac{0}{\text{undefined}}$$

Dave's variance is 0 → **Pearson undefined** (or set to 0).

**Alice vs. Eve**:
- Co-rated: Movie3
- Only 1 co-rated item → **Unreliable, skip**

---

### Step 2: Select k=2 Neighbors Who Rated Movie4

**Similarities**:
- Bob: 1.0 ✓ (rated Movie4 = 2)
- Carol: 0.5 ✓ (rated Movie4 = 3)
- Dave: 0.0 ✓ (rated Movie4 = 4)
- Eve: N/A ✗ (didn't rate Movie4)

**Top 2**: Bob (1.0) and Carol (0.5)

---

### Step 3: Predict Rating

$$\hat{r}_{Alice, Movie4} = \bar{r}_{Alice} + \frac{\text{sim}(Alice, Bob) \cdot (r_{Bob, Movie4} - \bar{r}_{Bob}) + \text{sim}(Alice, Carol) \cdot (r_{Carol, Movie4} - \bar{r}_{Carol})}{\text{sim}(Alice, Bob) + \text{sim}(Alice, Carol)}$$

$$= 4 + \frac{1.0 \cdot (2 - 2) + 0.5 \cdot (3 - 4)}{1.0 + 0.5}$$

$$= 4 + \frac{0 + 0.5 \cdot (-1)}{1.5} = 4 + \frac{-0.5}{1.5} = 4 - 0.33 = 3.67$$

**Prediction**: Alice would rate Movie4 as **3.67** (likely between 3-4 stars).

---

## Advantages of User-Based CF

1. **Intuitive**: Easy to understand and explain
2. **No training phase**: Directly use rating matrix
3. **Handles new items**: If any user rated it, can recommend
4. **Captures subjective preferences**: Similar users defined by ratings

---

## Limitations of User-Based CF

### 1. **Scalability**
- $O(|U|^2 \cdot |I|)$ complexity
- Infeasible for millions of users
- **Solution**: Item-based CF or matrix factorization

### 2. **Sparsity**
- Few overlapping ratings → unreliable similarities
- **Example**: Two users each rated 10 out of 100K items → unlikely to overlap
- **Solution**: Use more data (implicit feedback), dimensionality reduction

### 3. **Cold Start**
- **New users**: No ratings → can't find similar users
- **Solution**: Ask for initial ratings, use demographics, popularity

### 4. **Popularity Bias**
- Popular items dominate overlaps
- Niche items rarely recommended
- **Solution**: Weighted sampling, diversity re-ranking

### 5. **Gray Sheep**
- Users with unique tastes have no similar neighbors
- **Example**: User loves both classical music and heavy metal
- **Solution**: Hybrid models

### 6. **Shilling Attacks**
- Malicious users can manipulate recommendations
- **Example**: Create fake profiles to promote an item
- **Solution**: Anomaly detection, trust models

---

## Comparison: User-Based vs. Item-Based CF

| Aspect | User-Based CF | Item-Based CF |
|--------|---------------|---------------|
| **Similarity** | Between users | Between items |
| **Scalability** | Poor ($|U|$ often > $|I|$) | Better (fewer items) |
| **Stability** | Changes as user preferences drift | More stable (item features static) |
| **Serendipity** | Higher (diverse users) | Lower (similar items) |
| **Cold Start** | Severe for new users | Handles new users better |

**Modern practice**: Item-based CF preferred for large-scale systems (covered in next section).

---

## Summary

**User-Based Collaborative Filtering**:
- Find users with similar rating patterns
- Aggregate their ratings for prediction
- Uses Pearson correlation or cosine similarity
- Complexity: $O(|U|^2 \cdot |I|)$ → doesn't scale

**Key Takeaway**: Intuitive but limited by scalability and sparsity. Modern systems use item-based CF or matrix factorization.

**Next**: See **item-based-cf.md** for a more scalable approach.

---

## References

1. Resnick, P., et al. (1994). "GroupLens: An open architecture for collaborative filtering". *CSCW*.
2. Breese, J. S., et al. (1998). "Empirical analysis of predictive algorithms for collaborative filtering". *UAI*.
3. Herlocker, J. L., et al. (1999). "An algorithmic framework for performing collaborative filtering". *SIGIR*.
