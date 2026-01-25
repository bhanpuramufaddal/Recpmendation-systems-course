# Week 2: Item-Based Collaborative Filtering

## Learning Objectives

- Understand item-based collaborative filtering intuition
- Compute item-item similarity
- Recognize advantages over user-based CF
- Implement Amazon's item-to-item approach
- Master precomputation and storage strategies

---

## Intuition: "You Liked X, So You Might Like Y..."

### Core Idea

**Items that are liked by the same users are similar.**

**Example**:
- 1000 users loved both *The Matrix* and *Inception*
- 800 users loved both *The Matrix* and *Blade Runner 2049*
- These movies are similar
- User U loved *The Matrix* → Recommend *Inception* and *Blade Runner 2049*

**Key Insight**: Focus on item relationships, not user relationships.

---

## Why Item-Based Over User-Based?

### Scalability

**User-Based CF**:
- Complexity: $O(|U|^2 \cdot |I|)$
- Amazon in 2003: 20M+ customers, 1M+ items
- Computing all user-user similarities: **Infeasible**

**Item-Based CF**:
- Complexity: $O(|I|^2 \cdot |U|)$
- Usually $|I| \ll |U|$ (fewer items than users)
- **Much more tractable**

### Stability

**User preferences change**:
- User gets married → different movie tastes
- User has kids → starts buying baby products
- User similarities need constant recomputation

**Item similarities are stable**:
- *The Matrix* and *Inception* will always be similar
- Can precompute and cache similarities
- Recompute only when new items added

### Example: Amazon (2003)

| Metric | User-Based | Item-Based |
|--------|-----------|------------|
| Users | 20M | 20M |
| Items | 1M | 1M |
| Similarity computations | 200T (20M²) | 500B (1M²) |
| Relative complexity | **400x more** | Baseline |
| Update frequency needed | Daily | Weekly/Monthly |

---

## The Algorithm

### Step 1: Compute Item-Item Similarity

Given user-item rating matrix $R \in \mathbb{R}^{|U| \times |I|}$:

$$\text{sim}(i, j) = f(r_{\cdot i}, r_{\cdot j})$$

where $r_{\cdot i}$ and $r_{\cdot j}$ are rating vectors for items $i$ and $j$ (across all users).

### Step 2: Find k Most Similar Items

$$N_k(i) = \{j_1, j_2, \ldots, j_k\}$$

where $j_1, \ldots, j_k$ are the $k$ items most similar to $i$.

### Step 3: Predict Rating

**Weighted Average**:

$$\hat{r}_{ui} = \frac{\sum_{j \in N_k(i) \cap I_u} \text{sim}(i,j) \cdot r_{uj}}{\sum_{j \in N_k(i) \cap I_u} |\text{sim}(i,j)|}$$

where:
- $N_k(i)$ = k most similar items to $i$
- $I_u$ = items rated by user $u$
- $\text{sim}(i,j)$ = similarity between items $i$ and $j$
- $r_{uj}$ = user $u$'s rating of item $j$

**Interpretation**:
- Look at items similar to target item $i$
- Among those, find items user $u$ has rated
- Weight by similarity
- Average to get prediction

---

## Similarity Measures for Items

### 1. Cosine Similarity (Most Common)

**Formula**:

$$\text{cosine}(i, j) = \frac{\sum_{u \in U_{ij}} r_{ui} \cdot r_{uj}}{\sqrt{\sum_{u \in U_{ij}} r_{ui}^2} \cdot \sqrt{\sum_{u \in U_{ij}} r_{uj}^2}}$$

where $U_{ij}$ = users who rated both items $i$ and $j$.

**Example**:
```
Item A ratings (by 5 users): [5, 4, 3, 5, 4]
Item B ratings (by same users): [4, 3, 3, 4, 3]

Dot product: 5×4 + 4×3 + 3×3 + 5×4 + 4×3 = 20 + 12 + 9 + 20 + 12 = 73
||A|| = √(25 + 16 + 9 + 25 + 16) = √91 ≈ 9.54
||B|| = √(16 + 9 + 9 + 16 + 9) = √59 ≈ 7.68

cosine(A, B) = 73 / (9.54 × 7.68) ≈ 0.997
```

**High similarity!** Items A and B are very similar.

---

### 2. Adjusted Cosine Similarity

**Problem with basic cosine**: Doesn't account for user rating scales.
- User Alice always rates 4-5 (generous)
- User Bob always rates 1-2 (harsh)
- Both might like the same items but use different scales

**Solution**: Subtract user mean from each rating.

$$\text{adj\_cosine}(i, j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)(r_{uj} - \bar{r}_u)}{\sqrt{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)^2} \cdot \sqrt{\sum_{u \in U_{ij}} (r_{uj} - \bar{r}_u)^2}}$$

**Example**:
```
        User1   User2   User3   User1_mean  User2_mean  User3_mean
ItemA     5       3       2        4.5         2.5         3.0
ItemB     4       2       4        4.5         2.5         3.0

Adjusted ratings for ItemA: [5-4.5, 3-2.5, 2-3.0] = [0.5, 0.5, -1.0]
Adjusted ratings for ItemB: [4-4.5, 2-2.5, 4-3.0] = [-0.5, -0.5, 1.0]

Numerator: 0.5×(-0.5) + 0.5×(-0.5) + (-1.0)×1.0 = -0.25 - 0.25 - 1.0 = -1.5
Denominator: √(0.25 + 0.25 + 1.0) × √(0.25 + 0.25 + 1.0) = √1.5 × √1.5 = 1.5

adj_cosine = -1.5 / 1.5 = -1.0
```

**Perfect negative correlation** after adjustment (items are dissimilar).

---

### 3. Pearson Correlation

Same as adjusted cosine for item-based CF.

$$\text{Pearson}(i, j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_i)(r_{uj} - \bar{r}_j)}{\sqrt{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_i)^2} \cdot \sqrt{\sum_{u \in U_{ij}} (r_{uj} - \bar{r}_j)^2}}$$

where $\bar{r}_i$ = average rating for item $i$ (across users who rated it).

---

## Amazon's Item-to-Item Collaborative Filtering

### The 2003 Breakthrough

**Paper**: Linden, G., Smith, B., & York, J. (2003). "Amazon.com recommendations: Item-to-item collaborative filtering". *IEEE Internet Computing*.

**Problem**:
- 20M+ customers
- 1M+ products
- User-based CF too slow

**Solution**: Item-to-item CF with precomputation

### Algorithm

**Offline (Daily/Weekly)**:
1. Compute all item-item similarities
2. For each item, store top-N (N=20-100) most similar items
3. Store in fast lookup database

**Online (Real-time)**:
1. User views/purchases item $i$
2. Retrieve precomputed similar items to $i$
3. Rank by similarity × user's rating (if available)
4. Display top-K recommendations

**Latency**: $O(1)$ lookup! (constant time)

### "Customers Who Bought This Item Also Bought"

**Implementation**:
```python
def recommend_for_item(item_id, k=10):
    """
    Given an item, recommend similar items.

    Args:
        item_id: Target item
        k: Number of recommendations

    Returns:
        List of (item_id, similarity_score) tuples
    """
    # O(1) lookup from precomputed table
    similar_items = precomputed_similarities[item_id]

    # Return top-k
    return similar_items[:k]

def recommend_for_user(user_id, k=10):
    """
    Given a user, recommend items based on their history.

    Args:
        user_id: Target user
        k: Number of recommendations

    Returns:
        List of recommended item IDs
    """
    # Get user's purchase/rating history
    user_items = get_user_history(user_id)

    # For each item user interacted with
    recommendations = {}
    for item, rating in user_items.items():
        # Get similar items
        similar_items = precomputed_similarities[item]

        # Aggregate scores
        for sim_item, similarity in similar_items:
            if sim_item not in user_items:  # Don't recommend already-owned
                if sim_item not in recommendations:
                    recommendations[sim_item] = 0
                recommendations[sim_item] += similarity * rating

    # Sort and return top-k
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return [item for item, score in sorted_recs[:k]]
```

---

## Computational Complexity

### Offline Phase (Precomputation)

**Step 1: Compute all item-item similarities**
- For each pair of items: $O(|U|)$ (iterate over users)
- Total pairs: $\binom{|I|}{2} \approx |I|^2 / 2$
- **Complexity**: $O(|I|^2 \cdot |U|)$

**Example**:
- 1M items, 20M users
- $10^{12} \times 20 \times 10^6 = 2 \times 10^{19}$ operations
- With optimizations (sparsity exploitation): Tractable

**Step 2: Store top-N similarities**
- For each item: Sort similarities, keep top-N
- **Complexity**: $O(|I|^2 \log |I|)$ for sorting
- **Storage**: $O(|I| \times N)$ where $N \approx 100$

**Frequency**: Daily or weekly (items don't change rapidly)

---

### Online Phase (Serving)

**Retrieve precomputed similarities**: $O(1)$ or $O(\log N)$ lookup

**Aggregate for user**: $O(|I_u| \times N)$
- $|I_u|$ = items in user's history
- $N$ = number of similar items per item

**Example**:
- User purchased 50 items
- Top-20 similar items per purchased item
- $50 \times 20 = 1000$ candidate items
- Sort and return top-10: $O(1000 \log 1000) \approx O(10,000)$ operations

**Latency**: <10ms (extremely fast)

---

## Optimizations and Tricks

### 1. Sparsity Exploitation

**Observation**: Most item pairs have no co-raters.

**Solution**: Only compute similarity for item pairs with at least $\tau$ co-raters (e.g., $\tau = 5$).

**Benefit**:
- Reduces computation by 90%+
- Improves reliability (few co-raters → noisy similarity)

---

### 2. Significance Weighting

**Problem**: Similarity based on 1-2 co-raters is unreliable.

**Solution**: Weight similarity by number of co-raters.

$$\text{sim}_{weighted}(i, j) = \text{sim}(i, j) \cdot \min\left(1, \frac{|U_{ij}|}{\tau}\right)$$

**Example**: $\tau = 25$
- If $|U_{ij}| = 5$, multiply similarity by $5/25 = 0.2$ (downweight)
- If $|U_{ij}| \geq 25$, use full similarity

---

### 3. Normalization

**Problem**: Popular items dominate (high ratings from many users).

**Solution**: Normalize by item popularity.

$$\text{sim}_{normalized}(i, j) = \frac{\text{sim}(i, j)}{\sqrt{|U_i| \cdot |U_j|}}$$

where $|U_i|$ = number of users who rated item $i$.

---

### 4. Incremental Updates

**Problem**: Recomputing all similarities when new item added is expensive.

**Solution**: Only compute similarities for new item.
- New item $i_{new}$ arrives
- Compute $\text{sim}(i_{new}, j)$ for all existing items $j$
- **Complexity**: $O(|I| \cdot |U|)$ (linear in number of items)

---

## Example Walkthrough

### Given Data

```
         User1  User2  User3  User4  User5
Movie1     5      4      ?      5      3
Movie2     3      ?      4      ?      2
Movie3     4      3      5      4      ?
Movie4     ?      2      3      3      4
Movie5     5      ?      4      5      5
```

**Task**: Compute similarity between Movie1 and Movie3, then predict User3's rating for Movie1.

---

### Step 1: Compute Similarity (Cosine)

**Movie1 and Movie3**:
- Co-rated by: User1, User2, User4
- Movie1 ratings: [5, 4, 5]
- Movie3 ratings: [4, 3, 4]

$$\text{cosine}(Movie1, Movie3) = \frac{5 \times 4 + 4 \times 3 + 5 \times 4}{\sqrt{5^2 + 4^2 + 5^2} \times \sqrt{4^2 + 3^2 + 4^2}}$$

$$= \frac{20 + 12 + 20}{\sqrt{66} \times \sqrt{41}} = \frac{52}{\sqrt{2706}} \approx \frac{52}{52.02} \approx 0.9996$$

**Very high similarity!**

---

### Step 2: Predict User3's Rating for Movie1

**User3's ratings**:
- Movie2: 4
- Movie3: 5
- Movie4: 3

**Find items similar to Movie1 that User3 rated**:
- Movie3: similarity = 0.9996 ✓ (User3 rated it 5)
- (Assume we computed other similarities and found Movie2, Movie4 less similar)

**Using k=1 (only Movie3)**:

$$\hat{r}_{User3, Movie1} = \frac{\text{sim}(Movie1, Movie3) \times r_{User3, Movie3}}{\text{sim}(Movie1, Movie3)}$$

$$= \frac{0.9996 \times 5}{0.9996} = 5$$

**Prediction**: User3 would rate Movie1 as **5 stars**.

**Using k=2 (Movie3 and Movie2, assume sim(Movie1, Movie2) = 0.6)**:

$$\hat{r}_{User3, Movie1} = \frac{0.9996 \times 5 + 0.6 \times 4}{0.9996 + 0.6} = \frac{4.998 + 2.4}{1.5996} = \frac{7.398}{1.5996} \approx 4.62$$

**Prediction**: **4.6 stars** (still high, but tempered by less similar Movie2)

---

## Advantages of Item-Based CF

### 1. **Scalability**
- $O(|I|^2 \cdot |U|)$ vs. $O(|U|^2 \cdot |I|)$ for user-based
- Usually $|I| \ll |U|$ → much faster
- Precomputation makes online phase $O(1)$

### 2. **Stability**
- Item similarities don't change rapidly
- Can cache for days/weeks
- Reduces computational overhead

### 3. **Sparsity Handling**
- Items with many ratings (popular items) have reliable similarities
- Long-tail items can still be similar to popular items
- Better than user-based for sparse data

### 4. **Explainability**
- "You liked *The Matrix*, so you might like *Inception*"
- Clear, intuitive explanations
- Builds user trust

### 5. **Cold Start for New Users**
- New user rates a few items
- Immediately get recommendations based on those items
- No need to find similar users first

---

## Limitations of Item-Based CF

### 1. **Cold Start for New Items**
- New item has no co-ratings with existing items
- Can't compute similarities
- **Solution**: Content-based bootstrapping, hybrid methods

### 2. **Popularity Bias**
- Popular items dominate similarity computations
- Niche items underrepresented
- **Solution**: Normalization, diversity re-ranking

### 3. **Limited Serendipity**
- Recommends similar items to what user already likes
- Can create filter bubbles
- **Solution**: Occasional random recommendations, diversity objectives

### 4. **Scalability for Huge Catalogs**
- $O(|I|^2)$ storage for similarities
- 1M items → 1T similarity pairs
- **Solution**: Store only top-N, approximate methods

### 5. **Static Similarities**
- Doesn't adapt to temporal trends
- "Hot" items not prioritized
- **Solution**: Time-weighted similarities, trending boosts

---

## Comparison: User-Based vs. Item-Based

| Aspect | User-Based CF | Item-Based CF |
|--------|---------------|---------------|
| **Similarity** | Between users | Between items |
| **Complexity** | $O(\|U\|^2 \cdot \|I\|)$ | $O(\|I\|^2 \cdot \|U\|)$ |
| **Scalability** | Poor (many users) | Better (fewer items) |
| **Stability** | Low (user prefs change) | High (item features stable) |
| **Precomputation** | Hard (constantly changes) | Easy (cache for days) |
| **Serendipity** | Higher (diverse users) | Lower (similar items) |
| **Explainability** | Harder ("Users like you...") | Easier ("You liked X...") |
| **Cold Start (new user)** | Severe (no neighbors) | Better (rate few items → get recs) |
| **Cold Start (new item)** | Better (users can rate it) | Severe (no similarities) |
| **Best For** | Small user base, diverse tastes | Large user base, stable catalog |

---

## Production Example: Amazon (2003)

### System Architecture

```
┌──────────────────────────────────────┐
│      OFFLINE PROCESSING (Weekly)      │
│                                       │
│  1. Extract purchase/rating data     │
│  2. Compute item-item similarities   │
│  3. Store top-100 similar items/item │
│                                       │
│  Output: Similarity Database          │
└─────────────────┬────────────────────┘
                  │
┌─────────────────┴────────────────────┐
│    ONLINE SERVING (<10ms)             │
│                                       │
│  User views product P                 │
│      ↓                                │
│  Lookup similar items to P (O(1))     │
│      ↓                                │
│  Rank by similarity                   │
│      ↓                                │
│  Display "Customers who bought        │
│  this also bought..."                 │
└───────────────────────────────────────┘
```

### Results (from Amazon's 2003 paper)

**Metrics**:
- **Recommendation quality**: Better than user-based CF
- **Scalability**: 20M customers, 1M products handled easily
- **Latency**: <10ms per request
- **Business impact**: Significant increase in sales

**Quote from paper**:
> "Item-to-item collaborative filtering scales independently of the number of customers and number of items in the product catalog."

---

## Implementation: Production-Ready Item-Based CF

### Pseudocode

```python
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class ItemBasedCF:
    def __init__(self, k=20, min_support=5):
        """
        Item-based collaborative filtering.

        Args:
            k: Number of similar items to store per item
            min_support: Minimum co-ratings for similarity
        """
        self.k = k
        self.min_support = min_support
        self.item_similarities = {}

    def fit(self, user_item_matrix):
        """
        Compute item-item similarities.

        Args:
            user_item_matrix: Sparse matrix (users × items)
        """
        n_items = user_item_matrix.shape[1]

        # Compute cosine similarity (items × items)
        # Transpose to get items as rows
        item_user_matrix = user_item_matrix.T
        similarities = cosine_similarity(item_user_matrix, dense_output=False)

        # For each item, store top-k similar items
        for item_id in range(n_items):
            # Get similarities for this item
            sim_scores = similarities[item_id].toarray().flatten()

            # Filter by minimum support
            # (In production, you'd count co-ratings here)

            # Get top-k (excluding self)
            top_k_indices = np.argsort(sim_scores)[::-1][1:self.k+1]
            top_k_scores = sim_scores[top_k_indices]

            # Store
            self.item_similarities[item_id] = list(zip(top_k_indices, top_k_scores))

    def predict(self, user_id, item_id, user_item_matrix):
        """
        Predict rating for user-item pair.

        Args:
            user_id: User ID
            item_id: Item ID
            user_item_matrix: Sparse matrix (users × items)

        Returns:
            Predicted rating
        """
        # Get user's ratings
        user_ratings = user_item_matrix[user_id].toarray().flatten()

        # Get similar items to target item
        if item_id not in self.item_similarities:
            return np.mean(user_ratings[user_ratings > 0])  # Fallback: user's avg

        similar_items = self.item_similarities[item_id]

        # Compute weighted average
        numerator = 0
        denominator = 0

        for sim_item, similarity in similar_items:
            if user_ratings[sim_item] > 0:  # User rated this similar item
                numerator += similarity * user_ratings[sim_item]
                denominator += abs(similarity)

        if denominator == 0:
            return np.mean(user_ratings[user_ratings > 0])  # Fallback

        return numerator / denominator

    def recommend(self, user_id, user_item_matrix, n=10):
        """
        Recommend top-n items for user.

        Args:
            user_id: User ID
            user_item_matrix: Sparse matrix (users × items)
            n: Number of recommendations

        Returns:
            List of (item_id, predicted_rating) tuples
        """
        n_items = user_item_matrix.shape[1]
        user_ratings = user_item_matrix[user_id].toarray().flatten()

        predictions = []
        for item_id in range(n_items):
            if user_ratings[item_id] == 0:  # User hasn't rated this item
                pred = self.predict(user_id, item_id, user_item_matrix)
                predictions.append((item_id, pred))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n]
```

---

## Summary

**Item-Based Collaborative Filtering**:
- Compute item-item similarities (offline)
- Recommend items similar to what user liked
- $O(|I|^2 \cdot |U|)$ complexity → better than user-based for large user bases
- Stable similarities → precomputation and caching
- Powers Amazon's "Customers who bought this also bought"

**Key Advantages**:
- Scalability
- Stability
- Explainability
- Handles new users well

**Key Limitations**:
- Cold start for new items
- Popularity bias
- Limited serendipity

**Modern Usage**:
- Still used in production (Amazon, Netflix)
- Often combined with other methods (hybrid systems)
- Baseline for comparison with deep learning approaches

**Next**: See **similarity-measures.md** for detailed comparison of all similarity metrics.

---

## References

1. **Linden, G., Smith, B., & York, J. (2003)**. "Amazon.com recommendations: Item-to-item collaborative filtering recommendation algorithms". *IEEE Internet Computing*, 7(1), 76-80.
   - **Essential**: Original Amazon paper

2. **Sarwar, B., et al. (2001)**. "Item-based collaborative filtering recommendation algorithms". *WWW*.
   - Comprehensive comparison of item-based techniques

3. **Deshpande, M., & Karypis, G. (2004)**. "Item-based top-N recommendation algorithms". *ACM TOIS*.
   - Analysis of top-N recommendation with item-based CF
