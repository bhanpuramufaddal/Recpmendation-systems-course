# Week 2: Limitations of Memory-Based Methods

## Overview

While memory-based collaborative filtering (user-based and item-based) methods are intuitive and effective, they suffer from several fundamental limitations that motivated the development of model-based approaches like matrix factorization.

This document explores these limitations in depth and sets the stage for Week 3's coverage of model-based methods.

---

## 1. Scalability Issues

### Problem Statement

Memory-based methods require storing the **entire user-item matrix** in memory and computing similarities **on-the-fly** at recommendation time.

**Computational Complexity:**

**User-Based CF:**
- Computing all user-user similarities: $O(n_u^2 \cdot n_i)$
  - $n_u$ = number of users
  - $n_i$ = number of items
- Finding top-K neighbors per user: $O(n_u \log K)$
- Total: $O(n_u^2 \cdot n_i)$ per model update

**Item-Based CF:**
- Computing all item-item similarities: $O(n_i^2 \cdot n_u)$
- Finding top-K similar items: $O(n_i \log K)$
- Total: $O(n_i^2 \cdot n_u)$ per model update

### Why This Is Problematic

**Example Scale:**
- Netflix: 200M+ users, 10K+ items → $O(10^{16})$ operations
- Amazon: 300M+ users, 100M+ items → Intractable

**Memory Requirements:**
- Storing full user-item matrix: $n_u \times n_i$ floats
- For Netflix: $200M \times 10K \times 4$ bytes = 8 TB (just for ratings!)
- For Amazon: $300M \times 100M \times 4$ bytes = 120 PB (infeasible)

**Real-Time Latency:**
- User-based: Must compute similarity with ALL users at prediction time
- Item-based: More practical (pre-compute item similarities), but still slow for large catalogs

---

## 2. Data Sparsity Sensitivity

### The Sparsity Problem

**Definition**: Most users interact with very few items.

**Typical Sparsity Levels:**
- MovieLens 1M: 95.8% sparse (users rate ~4% of movies)
- Netflix Prize: 98.8% sparse
- Amazon: 99.9%+ sparse

**Mathematical Sparsity:**
$$\text{Sparsity} = 1 - \frac{\text{number of ratings}}{n_u \times n_i}$$

### Impact on Memory-Based CF

**1. Insufficient Overlap for Similarity Computation**

**User-Based CF:**
- Two users may have rated **zero items in common**
- Similarity computation is unreliable or undefined
- Example: Cosine similarity with no overlap = undefined

```python
# Example: Two users with no overlap
user_A = [5, 4, ?, ?, ?, ?, ?]  # Rated 2 items
user_B = [?, ?, 3, 5, ?, ?, ?]  # Rated 2 different items

# Cosine similarity = undefined (no common items)
```

**Item-Based CF:**
- Two items may have **zero users in common** who rated both
- More robust than user-based (items accumulate ratings over time)
- But still problematic for long-tail items

**2. Unreliable Similarity Estimates**

Even when there's overlap, small overlap leads to noisy estimates:

```python
# Example: Users with minimal overlap
user_A = [5, ?, ?, ?, 4, ?, ...]  # 1000 ratings total
user_B = [4, ?, ?, ?, 5, ?, ...]  # 1000 ratings total

# Only 2 items in common → cosine similarity based on 2 items is unreliable
```

**Statistical Issue:**
- Correlation computed on small sample size has high variance
- Spurious correlations emerge
- No statistical significance testing

---

## 3. No Feature Learning

### The Problem

Memory-based methods **do not learn latent features** or **low-dimensional representations** of users/items.

**What's Missing:**
- **User embeddings**: Compact representation of user preferences
- **Item embeddings**: Compact representation of item characteristics
- **Latent factors**: Hidden patterns (e.g., genres, moods, themes)

### Why This Matters

**Example: Movie Recommendations**

**Memory-Based Approach:**
- Stores raw ratings: `User 123 rated Inception 5 stars`
- Computes similarity based on direct rating overlap
- **Cannot generalize** beyond observed ratings

**Model-Based Approach (Matrix Factorization):**
- Learns user vector: `User 123 → [0.8 (action), 0.6 (sci-fi), -0.2 (romance), ...]`
- Learns item vector: `Inception → [0.9 (action), 0.7 (sci-fi), 0.1 (romance), ...]`
- **Can generalize**: Recommend sci-fi action movies even if not directly co-rated

**Generalization Power:**
- Memory-based: Relies on exact co-occurrence
- Model-based: Infers latent structure, generalizes to unseen combinations

---

## 4. Limited Personalization Depth

### Surface-Level Patterns

Memory-based methods capture **surface-level co-occurrence patterns**:
- "Users who liked A also liked B"
- "Items A and B are often co-rated"

**What's Missing:**
- **Why** users like items (latent preferences)
- **Nuanced patterns** (e.g., user likes action movies but only sci-fi action, not military action)
- **Complex interactions** (e.g., user's preference for drama depends on mood/time)

### Example: User Preference Complexity

**User's True Preferences:**
- Loves: Sci-fi action (Inception, Matrix)
- Hates: Military action (Top Gun, Black Hawk Down)
- Likes: Psychological thrillers (Shutter Island, Memento)

**Memory-Based CF:**
- Finds users who rated Inception highly
- Recommends whatever those users liked (including military action)
- **Cannot distinguish** sci-fi action from military action

**Model-Based CF:**
- Learns latent factors: `sci-fi dimension`, `military dimension`, `psychological dimension`
- User has high `sci-fi` weight, low `military` weight
- **Can distinguish** and recommend appropriately

---

## 5. Cold Start Problem

### New User Cold Start

**Problem**: New user has **zero ratings**.

**Memory-Based CF Response:**
- User-based: Cannot find similar users (no ratings to compute similarity)
- Item-based: Cannot recommend (no user preferences known)

**Fallback Strategies:**
- Show popular items (non-personalized)
- Ask user to rate a few items (onboarding)
- Use demographic information (not part of pure CF)

**Limitation**: Memory-based CF has **no principled way** to handle new users without ratings.

### New Item Cold Start

**Problem**: New item has **zero ratings**.

**Memory-Based CF Response:**
- User-based: Can still recommend (based on user similarity), but won't recommend new item
- Item-based: Cannot recommend new item (no item similarity computed yet)

**Impact:**
- New movies, products, content get **zero exposure**
- Rich-get-richer effect (popular items stay popular)

---

## 6. No Incorporation of Side Information

### Limitation

Memory-based CF uses **only the user-item rating matrix**.

**What's Ignored:**
- **User features**: Age, gender, location, occupation
- **Item features**: Genre, director, actors, price, brand
- **Context**: Time, device, location, weather
- **Sequential patterns**: Order of interactions matters

### Example: Ignoring Context

**Scenario**: User watches comedies on weekends, documentaries on weekdays.

**Memory-Based CF:**
- Treats all ratings equally
- **Cannot capture** temporal or contextual patterns
- Recommends comedies on Monday (suboptimal)

**Context-Aware Approaches:**
- Tensor factorization: User × Item × Context
- Contextual bandits
- Sequential models (RNNs, Transformers)

---

## 7. Inability to Handle Implicit Feedback

### The Challenge

Memory-based CF was designed for **explicit feedback** (ratings: 1-5 stars).

**Implicit Feedback** (clicks, views, purchases):
- No negative examples (only positive or missing)
- Missing ≠ negative (user may not know item exists)
- Varying confidence (10 views vs. 1 view)

### Why Memory-Based Struggles

**Example: User clicks on 5 items**

```
User → [item_1: click, item_2: click, ..., item_100000: no click]
```

**Problem:**
- How to compute similarity with binary data?
- How to interpret "no click"? (unknown vs. dislike)

**Cosine similarity with binary data:**
- Only considers overlap of positive items
- Ignores magnitude of preference
- Treats all clicks equally (1 view = 100 views)

**Better Approaches:**
- Weighted matrix factorization (Hu et al., 2008)
- Bayesian Personalized Ranking (BPR)
- Neural CF with implicit feedback

---

## 8. Synonymy Problem

### Definition

**Synonymy**: Different items that are essentially the same.

**Examples:**
- Movies: "The Lord of the Rings: The Fellowship of the Ring" vs. "LOTR: Fellowship"
- Products: "iPhone 14 Pro 256GB Black" vs. "Apple iPhone 14 Pro (256GB, Midnight)"
- Music: Same song on different albums (original vs. live version)

### Impact on Memory-Based CF

**Item-Based CF:**
- Treats synonymous items as separate entities
- Dilutes similarity scores
- Wastes computational resources

**Example:**
```
User rates:
- "iPhone 14 Pro 256GB Black": 5 stars
- "iPhone 14 Pro 128GB White": 4 stars

Similar item: "Apple iPhone 14 Pro (256GB, Midnight)"
```

Memory-based CF **cannot recognize** these are the same product.

**Model-Based Solution:**
- Learn item embeddings
- Synonymous items get similar embeddings automatically
- Clustering and deduplication

---

## 9. Transparency vs. Accuracy Trade-off

### The Paradox

**Memory-Based CF:**
- ✅ **Transparent**: "Users who liked X also liked Y"
- ❌ **Less accurate**: Cannot capture complex patterns

**Model-Based CF:**
- ✅ **More accurate**: Learns latent factors
- ❌ **Less transparent**: "Black box" (what do latent factors mean?)

### Why This Matters

**Applications Requiring Explainability:**
- Healthcare recommendations: "Why was this treatment recommended?"
- Financial products: Regulatory compliance
- High-stakes decisions: Users need to trust recommendations

**Current State:**
- Memory-based methods are more explainable but less accurate
- Model-based methods are more accurate but less explainable
- Active research area: Explainable AI for recommendations

---

## 10. Popularity Bias

### The Problem

Memory-based CF suffers from **popularity bias**:
- Popular items get more ratings
- More ratings → more similar items → more recommendations
- Rich-get-richer feedback loop

### Example: MovieLens

**Popular Movie (e.g., "The Shawshank Redemption"):**
- 50,000 ratings
- High similarity with many items
- Frequently recommended

**Niche Movie (e.g., indie film):**
- 100 ratings
- Low similarity with most items (insufficient overlap)
- Rarely recommended

### Consequences

- **Filter bubble**: Users only see popular items
- **Long-tail items** get no exposure
- **Diversity** decreases
- **Serendipity** (surprising discoveries) is lost

**Mitigation Strategies:**
- Re-ranking with diversity objectives
- Exploration-exploitation (bandits)
- Debiasing techniques (inverse propensity scoring)

---

## 11. Shilling Attacks

### Vulnerability

Memory-based CF is vulnerable to **shilling attacks** (fake profiles).

**Attack Scenario:**
- Competitor creates fake user accounts
- Fake users give low ratings to competitor products
- Fake users give high ratings to own products

**Impact on Item-Based CF:**
- Fake ratings inflate item similarity
- Malicious items get recommended more
- Legitimate items get downranked

**Example:**
```python
# Attacker creates 1000 fake users
for fake_user in fake_users:
    rate(target_item, 5_stars)  # Boost target
    rate(competitor_item, 1_star)  # Hurt competitor
```

**Why Memory-Based Is Vulnerable:**
- No model validation or anomaly detection
- Ratings are directly used (not learned/filtered)
- No outlier rejection

**Model-Based Defenses:**
- Regularization (penalizes extreme patterns)
- Outlier detection
- Robust matrix factorization

---

## Summary Comparison: Memory-Based vs. Model-Based

| **Aspect** | **Memory-Based CF** | **Model-Based CF** |
|------------|---------------------|---------------------|
| **Scalability** | ❌ Poor ($O(n_u^2 \cdot n_i)$ or $O(n_i^2 \cdot n_u)$) | ✅ Good ($O(k \cdot (n_u + n_i))$, $k$ = latent dims) |
| **Sparsity** | ❌ Very sensitive | ✅ Robust (learns latent patterns) |
| **Feature Learning** | ❌ No | ✅ Yes (latent factors) |
| **Personalization** | ⚠️ Surface-level | ✅ Deep (nuanced preferences) |
| **Cold Start** | ❌ No solution | ⚠️ Partial (hybrid, meta-learning) |
| **Side Information** | ❌ Cannot use | ✅ Can incorporate |
| **Implicit Feedback** | ⚠️ Limited support | ✅ Designed for it |
| **Explainability** | ✅ High ("users who liked X") | ❌ Low (latent factors) |
| **Popularity Bias** | ❌ High | ⚠️ Moderate (can mitigate) |
| **Robustness to Attacks** | ❌ Vulnerable | ✅ More robust |
| **Training Time** | ✅ None (lazy learning) | ⚠️ Requires training |
| **Prediction Time** | ❌ Slow (compute on-the-fly) | ✅ Fast (precomputed embeddings) |

---

## When to Use Memory-Based CF

Despite limitations, memory-based CF is still useful in certain scenarios:

### ✅ Good Use Cases

1. **Small datasets** (<10K users, <10K items)
   - Computational cost is manageable
   - Sparsity is lower

2. **High-density matrices**
   - Users rate many items (e.g., internal employee tools)
   - Sparsity <80%

3. **Explainability is critical**
   - Medical, legal, financial domains
   - Users need to understand recommendations

4. **Rapidly changing catalogs**
   - News, events (items have short lifespan)
   - No time to retrain models

5. **Quick prototyping**
   - Baseline for comparison
   - No infrastructure for model training

---

## Transition to Model-Based Methods

The limitations discussed above motivated the development of **model-based collaborative filtering**, particularly:

1. **Matrix Factorization (Week 3)**
   - Learns latent factors
   - Scales to millions of users/items
   - Handles sparsity better

2. **Deep Learning (Weeks 5-8)**
   - Non-linear patterns
   - Incorporates side information
   - Handles implicit feedback

3. **Context-Aware & Sequential (Weeks 6, 10)**
   - Temporal dynamics
   - Contextual information
   - Session-based patterns

4. **Hybrid Methods (Week 4)**
   - Combines content-based + collaborative
   - Mitigates cold start
   - Uses item/user features

---

## Practice Problems

**Problem 1:** Calculate the computational complexity of user-based CF for a system with 1 million users and 100K items. How long would it take to compute all user-user similarities on a machine that can do 1 billion operations/second?

**Problem 2:** Given a user-item matrix with 99% sparsity, what percentage of user pairs have at least 5 co-rated items? (Assume uniform distribution of ratings)

**Problem 3:** Explain why item-based CF is more scalable than user-based CF for systems with many more users than items (e.g., Netflix, Amazon).

**Problem 4:** Design a hybrid approach that combines memory-based CF with content-based filtering to handle the cold start problem for new items.

**Problem 5:** Describe a shilling attack scenario for an e-commerce site. How would you detect and prevent such attacks?

---

## References

1. **Sarwar, B., et al. (2001)**. "Item-Based Collaborative Filtering Recommendation Algorithms". *WWW*.
   - Analysis of item-based CF scalability

2. **Herlocker, J. L., et al. (2004)**. "Evaluating Collaborative Filtering Recommender Systems". *ACM TOIS*.
   - Comprehensive coverage of CF limitations

3. **Koren, Y., et al. (2009)**. "Matrix Factorization Techniques for Recommender Systems". *IEEE Computer*.
   - Motivation for model-based approaches

4. **Su, X., & Khoshgoftaar, T. M. (2009)**. "A Survey of Collaborative Filtering Techniques". *Advances in Artificial Intelligence*.
   - Detailed comparison of memory-based vs. model-based

5. **Lam, S. K., & Riedl, J. (2004)**. "Shilling Recommender Systems for Fun and Profit". *WWW*.
   - Shilling attacks on memory-based CF

6. **Adomavicius, G., & Tuzhilin, A. (2005)**. "Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions". *IEEE TKDE*.
   - Context-aware and hybrid methods

---

**Next:** Week 3 introduces **Matrix Factorization**, which addresses many of these limitations by learning low-dimensional latent representations of users and items.
