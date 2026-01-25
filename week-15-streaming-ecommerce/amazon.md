# Week 15: Amazon Product Recommendations

## Overview

**Amazon**: World's largest e-commerce platform, billions of products.

**Recommendation types**:
1. **Item-to-item CF**: "Customers who bought this also bought..."
2. **Personalized homepage**: Custom product feed
3. **Search ranking**: Results ordered by relevance + personalization
4. **Post-purchase**: Replenishment, accessories
5. **Email**: Personalized product emails

**Revenue impact**: 35% of Amazon sales from recommendations.

---

## Item-to-Item Collaborative Filtering

### Origins (2003)

**Problem with user-based CF**:
- Users change preferences quickly
- Cold start for new users
- Scalability (millions of users)

**Solution**: **Item-to-item** CF (Linden et al., 2003).

---

### Algorithm

**Precompute similarity** between all item pairs.

**Similarity metric** (cosine):
$$\text{sim}(i, j) = \frac{\sum_{u} r_{ui} \cdot r_{uj}}{\sqrt{\sum_{u} r_{ui}^2} \cdot \sqrt{\sum_{u} r_{uj}^2}}$$

where $r_{ui}$ = 1 if user $u$ purchased item $i$, else 0.

**Recommendation**:
For user $u$ who bought items $\{i_1, i_2, \ldots\}$:
1. Find items similar to $i_1, i_2, \ldots$
2. Aggregate scores
3. Return top-N

**Benefits**:
- **Scalable**: Items change slower than users
- **Explainable**: "Bought X → recommend similar Y"
- **Real-time**: Precomputed similarities, fast lookup

---

### Implementation

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# User-item matrix (rows=users, cols=items)
R = np.array([
    [1, 1, 0, 0, 1],  # User 0 bought items 0, 1, 4
    [0, 1, 1, 0, 0],  # User 1 bought items 1, 2
    [1, 0, 0, 1, 1],  # User 2 bought items 0, 3, 4
])

# Compute item-item similarity
item_similarity = cosine_similarity(R.T)  # Transpose (items as rows)

print("Item-item similarity:")
print(item_similarity)

# Recommend for user who bought item 0
def recommend_for_item(item_id, top_k=3):
    scores = item_similarity[item_id]
    # Exclude self
    scores[item_id] = -1
    # Top-K
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return top_indices

recs = recommend_for_item(item_id=0, top_k=3)
print(f"Customers who bought item 0 also bought: {recs}")
```

---

## Personalized Homepage

### Inventory Optimization

**Challenge**: Billions of products, limited homepage slots.

**Approach**: Multi-stage funnel
1. **Candidate generation**: CF, trending, browsing history → 10K products
2. **Ranking**: Deep learning model → top 100
3. **Re-ranking**: Business rules (diversity, ads) → 20-50 shown

---

### Features

**User**:
- Purchase history
- Browsing history
- Search queries
- Demographics (inferred from behavior)

**Item**:
- Category, brand, price
- Ratings, review count
- Popularity (recent sales)

**Context**:
- Time of day, day of week
- Device (mobile, desktop)
- Seasonal (holidays, back-to-school)

**Cross**:
- User's affinity to category
- Price sensitivity

---

## Search Ranking

### Semantic Search

**Beyond keyword matching**: Understand query intent.

**Example**:
```
Query: "laptop for gaming"
Traditional: Match "laptop" in title
Semantic: Understand "gaming" → high GPU, fast CPU → rank accordingly
```

**BERT embeddings**: Encode query + product title → semantic similarity.

---

### Personalization in Search

**Same query, different users → different results**.

**Example**:
```
Query: "running shoes"
User A (marathon runner) → Professional running shoes
User B (casual jogger) → Comfortable daily trainers
```

**Personalization features**:
- User's past purchases in category
- Price range preferences
- Brand affinity

---

## Cross-Category Recommendations

### Problem

**User behavior spans categories**: Bought laptop → need mouse, bag.

**Opportunity**: Recommend complementary items.

---

### Association Rules

**Market basket analysis**: Find frequent item sets.

**Example**:
```
{Laptop, Mouse, Laptop Bag} appears in 1000 transactions
{Laptop} appears in 5000 transactions

Confidence: 1000 / 5000 = 20%
→ "Customers who bought laptop also bought mouse and bag (20%)"
```

---

### Session-Based

**Within-session patterns**: User browses laptop → mouse → bag.

**RNN/Transformer**: Model session sequence, predict next item.

---

## Replenishment Recommendations

### Subscribe & Save

**Goal**: Predict when user needs to reorder (diapers, coffee, pet food).

**Features**:
- Product type (consumable vs. durable)
- Estimated usage rate (quantity / time since purchase)
- Historical reorder patterns

**Trigger**:
When predicted reorder date approaches → send reminder email.

---

## Seasonal and Trending Products

### Seasonal Patterns

**Examples**:
- Back-to-school (August): Notebooks, backpacks
- Holidays (December): Gifts, decorations
- Summer: Outdoor gear, swimwear

**Time-aware recommendations**: Boost seasonal items at appropriate times.

---

### Trending Detection

**Real-time spikes**: Detect products with sudden popularity.

**Metrics**:
- Sales velocity (sales per hour)
- Search volume increase
- Social media mentions

**Action**: Promote trending items on homepage.

---

## A/B Testing at Scale

### Experimentation Platform

**Amazon runs**: 1000s of experiments simultaneously.

**Metrics**:
- Conversion rate (% visitors who purchase)
- Revenue per visitor (RPV)
- Add-to-cart rate
- Customer lifetime value (CLV)

**Sample size**: Millions of users per test.

**Duration**: 1-2 weeks for statistical power.

---

### Example

**Test**: New recommendation algorithm on homepage.

**Setup**:
- Control (50%): Existing algorithm
- Treatment (50%): New algorithm

**Results**:
- Control RPV: $10.50
- Treatment RPV: $11.00
- **Improvement**: +4.8%
- **p-value**: <0.001 (significant)

**Decision**: Deploy new algorithm.

---

## Summary

**Key Takeaways**:
1. **Item-to-item CF**: Pioneered scalable, interpretable recommendations
2. **Personalized homepage**: Multi-stage (candidates → ranking → re-ranking)
3. **Search ranking**: Semantic understanding + personalization
4. **Cross-category**: Association rules, session-based
5. **Replenishment**: Predict reorder timing
6. **A/B testing**: Thousands of experiments, millions of users

**Impact**: 35% of Amazon revenue attributed to recommendations.

---

## References

1. **Linden, G., Smith, B., & York, J. (2003)**. "Amazon.com Recommendations: Item-to-Item Collaborative Filtering". *IEEE Internet Computing*.
   - Seminal paper on item-to-item CF

2. **Smith, B., & Linden, G. (2017)**. "Two Decades of Recommender Systems at Amazon.com". *IEEE Internet Computing*.
   - Evolution of Amazon's RecSys

3. **Amatriain, X., & Pujol, J. M. (2015)**. "Data Mining Methods for Recommender Systems". *Recommender Systems Handbook*.
