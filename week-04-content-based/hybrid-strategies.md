# Week 4: Content-Based Filtering - Hybrid Strategies

## Overview

**Pure content-based filtering** has limitations: filter bubble, limited serendipity, requires good features.

**Pure collaborative filtering** has limitations: cold start for new items, sparsity, popularity bias.

**Solution**: **Hybrid systems** combine both approaches to get the best of both worlds.

This document covers hybrid recommendation strategies that power systems like Netflix, Spotify, and Amazon.

**Business impact**: Hybrid systems improve engagement by 10-30% compared to single-method systems (measured in A/B tests).

---

## Learning Objectives

By the end of this section, you will:
- Understand why hybrid systems outperform single-method systems
- Master different hybrid strategies (weighted, switching, cascade, etc.)
- Implement hybrid recommenders in production
- Apply lessons from Netflix, Spotify, and Amazon
- Optimize hybrid systems for specific use cases

---

## Why Hybrid?

### Limitations of Pure Approaches

**Content-Based Filtering**:
- ❌ **Filter bubble**: Only recommends similar items
- ❌ **Limited serendipity**: No unexpected discoveries
- ❌ **Feature engineering**: Requires domain expertise
- ✅ **New item problem**: Handles new items well
- ✅ **Interpretability**: Easy to explain recommendations

**Collaborative Filtering**:
- ❌ **Cold start**: Poor for new items and new users
- ❌ **Sparsity**: Needs many ratings
- ❌ **Popularity bias**: Recommends popular items too much
- ✅ **Serendipity**: Can discover unexpected items
- ✅ **No features needed**: Works with interactions only

**Hybrid approach**: Combine strengths, mitigate weaknesses.

---

### Performance Comparison

**Example** (hypothetical A/B test results):

| Approach | Engagement Rate | Coverage (items) | Serendipity Score |
|----------|----------------|------------------|-------------------|
| Content-Based | 12% | 45% | Low (2.1/5) |
| Collaborative | 15% | 30% | High (4.2/5) |
| **Hybrid** | **18%** | **60%** | **Medium-High (3.8/5)** |

**Takeaway**: Hybrid achieves best engagement, coverage, and balance.

---

## Taxonomy of Hybrid Strategies

**Burke (2002)** identified seven hybrid strategies:

1. **Weighted**: Combine scores from multiple recommenders
2. **Switching**: Choose one recommender based on situation
3. **Mixed**: Mix recommendations from multiple sources
4. **Feature Combination**: Use CF features in CB model (or vice versa)
5. **Cascade**: Refine recommendations in stages
6. **Feature Augmentation**: Output of one system as input to another
7. **Meta-level**: Learn model from one system, apply to another

We'll cover the most practical ones.

---

## 1. Weighted Hybrid

### Approach

**Combine scores** from content-based (CB) and collaborative filtering (CF) with weights.

$$\text{score}_{\text{hybrid}}(u, i) = \alpha \cdot \text{score}_{\text{CB}}(u, i) + (1 - \alpha) \cdot \text{score}_{\text{CF}}(u, i)$$

where $\alpha \in [0, 1]$ is the weight for content-based.

**Example**:
```
Content-based score for item 42: 0.85
Collaborative score for item 42: 0.72
Weight α = 0.6

Hybrid score = 0.6 × 0.85 + 0.4 × 0.72 = 0.798
```

---

### Implementation

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class WeightedHybrid:
    def __init__(self, alpha=0.6):
        """
        alpha: Weight for content-based (0-1)
               1.0 = pure content-based
               0.0 = pure collaborative
        """
        self.alpha = alpha

    def recommend(self, user_profile, item_features, cf_scores, top_k=10):
        """
        user_profile: (d,) - user preference vector
        item_features: (n_items, d) - item feature matrix
        cf_scores: (n_items,) - collaborative filtering scores
        """
        # Content-based scores
        cb_scores = cosine_similarity([user_profile], item_features)[0]

        # Normalize scores to [0, 1]
        cb_scores = (cb_scores - cb_scores.min()) / (cb_scores.max() - cb_scores.min())
        cf_scores = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min())

        # Hybrid scores
        hybrid_scores = self.alpha * cb_scores + (1 - self.alpha) * cf_scores

        # Top-K
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return top_indices, hybrid_scores[top_indices]


# Example
user_profile = np.array([0.9, 0.1, 0.5, 0.6])
item_features = np.random.rand(100, 4)  # 100 items
cf_scores = np.random.rand(100)  # From collaborative filtering

hybrid = WeightedHybrid(alpha=0.6)
recommendations, scores = hybrid.recommend(user_profile, item_features, cf_scores, top_k=10)

print(f"Top 10 recommendations: {recommendations}")
print(f"Scores: {scores}")
```

---

### Choosing α (Weight)

**Options**:

**1. Fixed weight** (e.g., α = 0.6)
- Simple, interpretable
- Doesn't adapt to user/item

**2. User-specific weight**
- New users: Higher α (rely on content)
- Established users: Lower α (rely on CF)

```python
def adaptive_alpha(user, num_interactions_threshold=20):
    """Adjust alpha based on user history."""
    if len(user.interactions) < num_interactions_threshold:
        return 0.8  # New user → content-based
    else:
        return 0.4  # Established user → collaborative
```

**3. Item-specific weight**
- New items: Higher α (CF has no data)
- Popular items: Lower α (CF works well)

```python
def item_alpha(item, num_ratings_threshold=50):
    """Adjust alpha based on item popularity."""
    if item.num_ratings < num_ratings_threshold:
        return 0.9  # New item → content-based
    else:
        return 0.3  # Popular item → collaborative
```

**4. Learned weight** (via validation set)
- Optimize α to maximize metric (e.g., CTR, engagement)
- Can use grid search or gradient-based optimization

```python
from sklearn.model_selection import GridSearchCV

alphas = np.linspace(0, 1, 11)  # [0.0, 0.1, ..., 1.0]
best_alpha = None
best_score = 0

for alpha in alphas:
    hybrid = WeightedHybrid(alpha=alpha)
    score = evaluate_on_validation_set(hybrid)  # Custom evaluation
    if score > best_score:
        best_score = score
        best_alpha = alpha

print(f"Best alpha: {best_alpha} (score: {best_score})")
```

---

## 2. Switching Hybrid

### Approach

**Choose one recommender** based on situation (user, item, context).

**Decision rule**:
$$\text{score}(u, i) = \begin{cases}
\text{score}_{\text{CB}}(u, i) & \text{if condition A} \\
\text{score}_{\text{CF}}(u, i) & \text{otherwise}
\end{cases}$$

**Example conditions**:
- **New user** (< 10 interactions) → Use content-based
- **New item** (< 5 ratings) → Use content-based
- **Established user + popular item** → Use collaborative

---

### Implementation

```python
class SwitchingHybrid:
    def __init__(self, user_threshold=10, item_threshold=5):
        self.user_threshold = user_threshold
        self.item_threshold = item_threshold

    def recommend(self, user, items, cb_recommender, cf_recommender):
        """
        Switch between CB and CF based on user/item characteristics.
        """
        recommendations = []

        for item in items:
            # Decide which recommender to use
            if len(user.interactions) < self.user_threshold or item.num_ratings < self.item_threshold:
                # Use content-based (cold start scenario)
                score = cb_recommender.score(user, item)
                source = "CB"
            else:
                # Use collaborative (sufficient data)
                score = cf_recommender.score(user, item)
                source = "CF"

            recommendations.append((item, score, source))

        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:10]  # Top-10
```

**Advantage**: Optimal method for each scenario.

**Disadvantage**: Hard cutoffs can be abrupt.

---

## 3. Mixed Hybrid

### Approach

**Present recommendations from multiple sources simultaneously**.

**Example** (Netflix homepage):
```
Popular on Netflix (CF)
───────────────────────
[Item 1] [Item 2] [Item 3] ...

Because You Watched "Inception" (CB)
───────────────────────────────────
[Item 4] [Item 5] [Item 6] ...

Trending Now (CF)
─────────────────
[Item 7] [Item 8] [Item 9] ...
```

**Implementation**:
```python
def mixed_hybrid(user, top_k=10):
    """Generate recommendations from multiple sources."""
    recommendations = []

    # Source 1: Content-based
    cb_recs = content_based_recommender.recommend(user, top_k=5)
    recommendations.extend([(item, "CB") for item in cb_recs])

    # Source 2: Collaborative filtering
    cf_recs = collaborative_recommender.recommend(user, top_k=5)
    recommendations.extend([(item, "CF") for item in cf_recs])

    # Deduplicate (if item appears in both)
    seen = set()
    unique_recs = []
    for item, source in recommendations:
        if item not in seen:
            unique_recs.append((item, source))
            seen.add(item)

    return unique_recs[:top_k]
```

**Advantage**: User sees diverse recommendations from different perspectives.

---

## 4. Feature Combination Hybrid

### Approach

**Use collaborative features as input to content-based model** (or vice versa).

**Example**: Augment item features with collaborative signals.

**Item features**:
- **Content**: Genre, director, year (from metadata)
- **Collaborative**: Average rating, number of ratings, user clusters who liked it

**Combined feature vector**:
$$\mathbf{f}_i^{\text{hybrid}} = [\mathbf{f}_i^{\text{content}}, \mathbf{f}_i^{\text{collab}}]$$

---

### Implementation

```python
import numpy as np

class FeatureCombinationHybrid:
    def __init__(self):
        pass

    def create_hybrid_features(self, item_content_features, item_cf_features):
        """
        Combine content and collaborative features.

        item_content_features: (n_items, d_content)
        item_cf_features: (n_items, d_cf)
        """
        # Concatenate
        hybrid_features = np.hstack([item_content_features, item_cf_features])
        return hybrid_features

    def recommend(self, user_profile, hybrid_features, top_k=10):
        """Recommend using hybrid features."""
        # Extend user profile to match hybrid feature dimension
        # (Here simplified: assume user profile matches or use learned projection)

        scores = cosine_similarity([user_profile], hybrid_features)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices, scores[top_indices]


# Example
# Content features: genre, director (4 dims)
content_features = np.random.rand(100, 4)

# CF features: avg rating, num ratings, user_cluster_affinity (3 dims)
cf_features = np.random.rand(100, 3)

# Combine
hybrid_model = FeatureCombinationHybrid()
hybrid_features = hybrid_model.create_hybrid_features(content_features, cf_features)  # (100, 7)

print(f"Hybrid feature shape: {hybrid_features.shape}")
```

**Advantage**: Richer representation, leverages both content and behavior.

---

## 5. Cascade Hybrid

### Approach

**Multi-stage filtering**: Use one recommender to generate candidates, another to refine.

**Example**:
1. **Stage 1 (CB)**: Retrieve 1000 candidates similar to user's profile
2. **Stage 2 (CF)**: Rank candidates using collaborative filtering
3. **Return top-10**

**Architecture**:
```
All Items (millions)
      ↓
Stage 1: Content-Based Retrieval (fast, coarse)
      ↓
Candidates (1000 items)
      ↓
Stage 2: Collaborative Ranking (accurate, expensive)
      ↓
Top-10 Recommendations
```

---

### Implementation

```python
class CascadeHybrid:
    def __init__(self, cb_retriever, cf_ranker, num_candidates=1000):
        self.cb_retriever = cb_retriever
        self.cf_ranker = cf_ranker
        self.num_candidates = num_candidates

    def recommend(self, user, top_k=10):
        """Two-stage recommendation."""
        # Stage 1: Content-based retrieval
        candidates = self.cb_retriever.retrieve(user, top_k=self.num_candidates)

        # Stage 2: Collaborative ranking
        ranked = self.cf_ranker.rank(user, candidates)

        # Return top-K
        return ranked[:top_k]


# Example (pseudo-code)
cb_retriever = ContentBasedRetriever()
cf_ranker = CollaborativeRanker()

cascade = CascadeHybrid(cb_retriever, cf_ranker, num_candidates=500)
recommendations = cascade.recommend(user, top_k=10)
```

**Advantage**: Efficient (fast retrieval) + accurate (CF ranking).

**Used by**: YouTube, Pinterest, Spotify.

---

## 6. Feature Augmentation

### Approach

**Output of one system becomes input to another**.

**Example**: Use collaborative filtering to generate latent factors, then use them as features in content-based model.

**Process**:
1. Train matrix factorization (CF) to get item embeddings $\mathbf{q}_i$
2. Concatenate with content features: $\mathbf{f}_i^{\text{aug}} = [\mathbf{f}_i^{\text{content}}, \mathbf{q}_i]$
3. Use augmented features in content-based model

---

### Implementation

```python
import numpy as np
from sklearn.decomposition import NMF

class FeatureAugmentationHybrid:
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.nmf = NMF(n_components=n_factors, init='random', random_state=0)

    def fit(self, interaction_matrix):
        """Learn latent factors from interactions."""
        # interaction_matrix: (n_users, n_items)
        W = self.nmf.fit_transform(interaction_matrix)  # (n_users, n_factors)
        H = self.nmf.components_  # (n_factors, n_items)
        self.item_factors = H.T  # (n_items, n_factors)

    def augment_features(self, content_features):
        """Augment content features with latent factors."""
        # content_features: (n_items, d_content)
        augmented = np.hstack([content_features, self.item_factors])
        return augmented


# Example
interaction_matrix = np.random.randint(0, 2, size=(500, 100))  # 500 users, 100 items
content_features = np.random.rand(100, 10)  # 100 items, 10 content features

model = FeatureAugmentationHybrid(n_factors=20)
model.fit(interaction_matrix)
augmented_features = model.augment_features(content_features)

print(f"Original features: {content_features.shape}")
print(f"Augmented features: {augmented_features.shape}")  # (100, 30)
```

**Advantage**: Latent factors capture collaborative patterns, enriching content features.

---

## Real-World Examples

### Netflix

**Strategy**: **Mixed + Cascade**

**Rows on homepage** (mixed):
- "Top Picks for You" (CF)
- "Because You Watched X" (CB)
- "Trending Now" (CF)
- "Award Winners" (CB/Editorial)

**Within each row** (cascade):
1. Retrieve candidates (fast)
2. Rank with deep neural network (accurate)

**Hybrid weight**: Varies by row (some more CB, some more CF).

---

### Spotify

**Strategy**: **Weighted + Feature Combination**

**Discover Weekly** (personalized playlist):
1. **Collaborative filtering**: Find similar users, get their liked songs
2. **Content-based**: Audio features (tempo, key, energy) to find similar songs
3. **Hybrid**: Combine CF and CB scores with learned weights

**Feature combination**: Song features = audio features + collaborative embeddings.

**Result**: 40M+ users engage with Discover Weekly (high success rate).

---

### Amazon

**Strategy**: **Mixed + Switching**

**Product page** (mixed):
- "Customers who bought this also bought" (CF)
- "Products related to this item" (CB - same category, brand)
- "Sponsored products" (Ads)

**Switching**: New products use CB (no ratings yet), popular products use CF.

---

## Choosing a Hybrid Strategy

| Strategy | Use When | Pros | Cons |
|----------|---------|------|------|
| **Weighted** | Have scores from both CB and CF | Simple, tuneable | Requires score normalization |
| **Switching** | Clear decision criteria (new user/item) | Optimal for scenario | Hard cutoffs |
| **Mixed** | Want to show diverse sources | Transparency, diversity | UI complexity |
| **Feature Combination** | Want single model with rich features | Unified model | Feature engineering |
| **Cascade** | Need fast retrieval + accurate ranking | Efficient, scalable | Two models to maintain |
| **Feature Augmentation** | Want to enrich features | Leverages CF patterns | Complex pipeline |

**Most common in production**: **Cascade** (fast + accurate) and **Weighted** (simple + effective).

---

## Production Considerations

### 1. Score Normalization

**Problem**: CF scores in [0, 1], CB scores in [0.5, 1] → weighted sum biased.

**Solution**: Normalize scores.

```python
def normalize_scores(scores):
    """Min-max normalization to [0, 1]."""
    return (scores - scores.min()) / (scores.max() - scores.min())

cb_scores_norm = normalize_scores(cb_scores)
cf_scores_norm = normalize_scores(cf_scores)
hybrid_scores = alpha * cb_scores_norm + (1 - alpha) * cf_scores_norm
```

---

### 2. Deduplication

**Problem**: Same item recommended by both CB and CF.

**Solution**: Deduplicate before presenting.

```python
def deduplicate(recommendations):
    """Remove duplicate items, keeping first occurrence."""
    seen = set()
    unique = []
    for item in recommendations:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique
```

---

### 3. Diversification

**Problem**: Hybrid can still have filter bubble if both CB and CF recommend similar items.

**Solution**: Inject diversity (see Week 11 on evaluation).

```python
def diversify(recommendations, diversity_weight=0.2):
    """Select diverse items using MMR (Maximal Marginal Relevance)."""
    selected = [recommendations[0]]  # Start with top item

    while len(selected) < 10:
        best_score = -float('inf')
        best_item = None

        for item in recommendations:
            if item in selected:
                continue

            # Relevance to user
            relevance = item.score

            # Diversity (min similarity to selected items)
            diversity = min([1 - similarity(item, s) for s in selected])

            # MMR score
            mmr_score = diversity_weight * diversity + (1 - diversity_weight) * relevance

            if mmr_score > best_score:
                best_score = mmr_score
                best_item = item

        selected.append(best_item)

    return selected
```

---

## Summary

**Key Takeaways**:
1. **Hybrid > single method**: Combines strengths, mitigates weaknesses
2. **Seven strategies**: Weighted, switching, mixed, feature combination, cascade, feature augmentation, meta-level
3. **Most practical**: Weighted (simple), cascade (efficient)
4. **Real-world**: Netflix (mixed+cascade), Spotify (weighted+feature combo), Amazon (mixed+switching)
5. **Production**: Normalize scores, deduplicate, diversify

**Best Practices**:
- Start with weighted hybrid (simple, effective)
- Use cascade for scale (fast retrieval + accurate ranking)
- Adapt weights based on user/item (new vs. established)
- A/B test to find optimal strategy

**Choosing Strategy**:
- **Cold start heavy** (e.g., news) → Switching or weighted (favor CB for new items)
- **Scale heavy** (e.g., YouTube) → Cascade (retrieve + rank)
- **Transparency needed** → Mixed (show diverse sources)

**Next**: Week 5: Neural Collaborative Filtering (deep learning for recommendations).

---

## References

1. **Burke, R. (2002)**. "Hybrid Recommender Systems: Survey and Experiments". *User Modeling and User-Adapted Interaction*.
   - **Taxonomy** of hybrid strategies

2. **Gomez-Uribe, C. A., & Hunt, N. (2016)**. "The Netflix Recommender System: Algorithms, Business Value, and Innovation". *ACM TMIS*.
   - Netflix's hybrid approach

3. **Jacobson, K., et al. (2016)**. "Music Personalization at Spotify". *RecSys*.
   - Spotify's Discover Weekly (hybrid)

4. **Linden, G., Smith, B., & York, J. (2003)**. "Amazon.com Recommendations: Item-to-Item Collaborative Filtering". *IEEE Internet Computing*.
   - Amazon's hybrid system

5. **Adomavicius, G., & Tuzhilin, A. (2005)**. "Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions". *IEEE TKDE*.
   - Overview of hybrid methods

---

## Practice Problems

### Problem 1: Weighted Hybrid

**Given**:
```
Content-based scores: [0.8, 0.6, 0.9, 0.5]
Collaborative scores: [0.7, 0.8, 0.6, 0.9]
Alpha = 0.6

Compute hybrid scores and rank items.
```

**Solution**:
```python
import numpy as np

cb_scores = np.array([0.8, 0.6, 0.9, 0.5])
cf_scores = np.array([0.7, 0.8, 0.6, 0.9])
alpha = 0.6

hybrid_scores = alpha * cb_scores + (1 - alpha) * cf_scores
print(f"Hybrid scores: {hybrid_scores}")
# [0.76 0.68 0.78 0.66]

ranked = np.argsort(hybrid_scores)[::-1]
print(f"Ranked items: {ranked}")
# [2 0 1 3]
```

**Answer**: Item 2 (score 0.78), Item 0 (0.76), Item 1 (0.68), Item 3 (0.66).

---

### Problem 2: Adaptive Alpha

**Given**:
```
User A: 5 interactions (new user)
User B: 50 interactions (established)

Rule: α = 0.8 if interactions < 20, else α = 0.4

What alpha for each user?
```

**Solution**:
```
User A: 5 < 20 → α = 0.8 (favor content-based)
User B: 50 ≥ 20 → α = 0.4 (favor collaborative)
```

---

### Problem 3: Cascade Hybrid

**Task**: Design a cascade hybrid for YouTube with 1B videos.

**Solution**:
```
Stage 1: Content-Based Retrieval
  - Input: User's watch history
  - Method: Two-tower model (user encoder + video encoder)
  - Output: 1000 candidate videos (fast, <50ms)

Stage 2: Collaborative Ranking
  - Input: 1000 candidates
  - Method: Deep neural network with rich features
    (user history, video metadata, engagement signals)
  - Output: Top-10 ranked videos (accurate, <50ms)

Total latency: <100ms
```
