# Week 11: Offline Evaluation Metrics

## The Opening Question: Why Can't We Just Measure Accuracy?

*"Professor, can't we just use accuracy to evaluate our recommendation model? It seems straightforward."*

**Let me show you why that thinking will lead you astray.**

Consider this scenario: You build a movie recommender for a streaming platform. Your model achieves **95% accuracy** in predicting whether users will like movies.

**Sounds great, right? Here's the problem.**

**Scenario**: 95% of movies in your catalog are unpopular. Your model learns a simple strategy - **always recommend popular movies**.

```
User Alice's actual preferences:
  - Loves: Indie films, documentaries, foreign cinema
  - Test set: 20 movies she'll watch (15 obscure, 5 popular)

Model prediction: "Always recommend popular movies"

Accuracy calculation:
  - Predicts Alice likes popular movies: 5/20 correct
  - Predicts Alice dislikes obscure movies: Many correct (but wrong!)
  - Overall accuracy: 95% (matches popular-movie bias in data)

Actual user experience:
  - Alice gets recommended Marvel movies and rom-coms
  - Alice wanted Kurosawa films and climate documentaries
  - Alice churns to a competitor
```

**The "accurate" model gave terrible recommendations.**

**Why?** Because accuracy doesn't capture:
1. **Ranking quality**: Position 1 matters more than position 100
2. **Relevance vs. coverage**: Are we finding ALL good items?
3. **User satisfaction**: Does the metric align with business goals?

**This is why we need a sophisticated toolkit of evaluation metrics.**

---

## Overview

**Offline evaluation** assesses recommendation quality using **historical data** before deploying to users.

**Advantages**:
- **Safe**: No risk to user experience
- **Fast**: Test many models quickly
- **Reproducible**: Same dataset gives consistent results

**Limitation**: Offline performance does not guarantee online performance (distribution shift, feedback loops).

This document covers offline metrics for recommendation systems, from first principles.

---

## Learning Objectives

By the end of this section, you will:
- Derive precision, recall, and their tradeoffs from first principles
- Understand why NDCG uses logarithmic discounting (attention decay model)
- Compute MAP and understand why it captures ranking quality
- Know when to use MRR vs. other metrics
- Calculate all metrics for the same example (complete walkthrough)
- Recognize what can go wrong with offline evaluation
- Select the right metric for your use case

---

## Accuracy Metrics

### Rating Prediction

**RMSE** (Root Mean Squared Error):
$$\text{RMSE} = \sqrt{\frac{1}{|T|} \sum_{(u,i) \in T} (r_{ui} - \hat{r}_{ui})^2}$$

where:
- $T$ = test set of (user, item) pairs
- $r_{ui}$ = actual rating user $u$ gave item $i$
- $\hat{r}_{ui}$ = predicted rating

**MAE** (Mean Absolute Error):
$$\text{MAE} = \frac{1}{|T|} \sum_{(u,i) \in T} |r_{ui} - \hat{r}_{ui}|$$

**When to use**: Rating prediction tasks (Netflix Prize era).

**RMSE vs MAE**: RMSE penalizes large errors more heavily (squared term). Use RMSE if big mistakes are especially bad.

```python
import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
```

---

## Ranking Metrics: The Precision-Recall Tradeoff

### The Fundamental Tension

*"Professor, can't I just maximize both precision and recall?"*

**No. They are fundamentally at odds. Let me show you why.**

**Setup**:
- User has **5 relevant items** in catalog of 1000 items
- You recommend **K items**

**The Tradeoff**:

| K | Best-case Precision | Best-case Recall | Tension |
|---|---------------------|------------------|---------|
| 1 | 1.0 (if relevant) | 0.2 (1/5) | High precision, low recall |
| 5 | 1.0 (all 5 relevant) | 1.0 (all 5 found) | Perfect (rare!) |
| 10 | 0.5 (only 5 relevant exist) | 1.0 (all found) | Lower precision, max recall |
| 50 | 0.1 (5/50) | 1.0 (all found) | Precision tanks |

**Key insight**: As you recommend more items:
- **Recall can only increase** (you find more relevant items)
- **Precision typically decreases** (you add irrelevant items)

---

### Precision@K: Derivation and Intuition

**Question**: Of the K items I showed the user, how many were relevant?

**Definition**:
$$\text{Precision@K} = \frac{|\text{relevant items} \cap \text{top-K recommended}|}{K}$$

**Derivation from first principles**:

Let's define indicator variables:
- $\text{rel}(i) = 1$ if item $i$ is relevant to user, else $0$
- Let $R_K = \{r_1, r_2, \ldots, r_K\}$ be top-K recommendations

Then:
$$\text{Precision@K} = \frac{\sum_{i \in R_K} \text{rel}(i)}{K} = \frac{\text{hits}}{K}$$

**Numerical Example** (K=10, 5 relevant items exist):

```
Recommended list:     [A, B, C, D, E, F, G, H, I, J]  (K=10)
Relevant items:       {A, C, F, X, Y}  (5 total, X and Y not in top-10)

Position:  1  2  3  4  5  6  7  8  9  10
Relevant:  1  0  1  0  0  1  0  0  0  0   (3 hits)

Precision@10 = 3/10 = 0.30
```

**Interpretation**: 30% of what we showed was relevant.

**Problem**: Says nothing about the 2 relevant items we missed (X, Y).

```python
def precision_at_k(recommended, relevant, k):
    """
    recommended: list of item IDs (ordered by predicted relevance)
    relevant: set of actually relevant item IDs
    k: number of recommendations to consider
    """
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k
```

---

### Recall@K: Derivation and Intuition

**Question**: Of all the relevant items that exist, how many did I find in top-K?

**Definition**:
$$\text{Recall@K} = \frac{|\text{relevant items} \cap \text{top-K recommended}|}{|\text{relevant items}|}$$

**Derivation**:

Using same notation:
$$\text{Recall@K} = \frac{\sum_{i \in R_K} \text{rel}(i)}{|\text{relevant}|} = \frac{\text{hits}}{\text{total relevant}}$$

**Same Example**:

```
Recommended list:     [A, B, C, D, E, F, G, H, I, J]  (K=10)
Relevant items:       {A, C, F, X, Y}  (5 total)

Hits in top-10: A, C, F = 3

Recall@10 = 3/5 = 0.60
```

**Interpretation**: We found 60% of all relevant items.

**Problem**: Says nothing about all the irrelevant items we showed.

```python
def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant) if len(relevant) > 0 else 0
```

---

### F1@K: Balancing the Tradeoff

**Question**: Can we combine precision and recall into one number?

**Harmonic mean** (not arithmetic!) of precision and recall:

$$\text{F1@K} = 2 \cdot \frac{\text{Precision@K} \cdot \text{Recall@K}}{\text{Precision@K} + \text{Recall@K}}$$

**Why harmonic mean?**

Arithmetic mean: $(0.3 + 0.6) / 2 = 0.45$

But this is misleading if one metric is terrible:
- Precision = 0.01, Recall = 0.99
- Arithmetic mean = 0.50 (seems okay!)
- Harmonic mean = 0.02 (reveals the problem)

**Harmonic mean penalizes extreme imbalances.**

**Our Example**:
$$\text{F1@10} = 2 \cdot \frac{0.3 \times 0.6}{0.3 + 0.6} = 2 \cdot \frac{0.18}{0.9} = 0.40$$

---

## NDCG: Derivation from First Principles

### The Attention Decay Model

*"Professor, why does NDCG use logarithms? Seems arbitrary."*

**It's not arbitrary. Let me derive it from a model of human attention.**

**Observation**: Users pay more attention to items at the top of a list.

**The question**: How does attention decay with position?

---

### Modeling User Attention

**Hypothesis**: Each position has a probability of being examined.

**Linear decay model**:
$$P(\text{examine position } i) = 1 - \frac{i-1}{K}$$

**Problem**: Position 10 gets 10% of position 1's attention? Too severe.

**Exponential decay model**:
$$P(\text{examine position } i) = e^{-\lambda(i-1)}$$

**Problem**: Decays too fast or too slow depending on $\lambda$.

**Logarithmic decay model** (Jarvelin & Kekalainen, 2002):
$$P(\text{examine position } i) \propto \frac{1}{\log_2(i+1)}$$

**Why logarithmic?**

| Position | Linear Decay | Exp Decay ($\lambda$=0.2) | Log Decay |
|----------|--------------|---------------------------|-----------|
| 1 | 1.00 | 1.00 | 1.00 |
| 2 | 0.90 | 0.82 | 0.63 |
| 3 | 0.80 | 0.67 | 0.50 |
| 5 | 0.60 | 0.45 | 0.39 |
| 10 | 0.10 | 0.17 | 0.30 |

**Logarithmic decay**:
- Sharp drop-off at top (positions 1-3 matter a lot)
- Gradual decline later (positions 5-10 still matter somewhat)
- Matches empirical click data from search engines

---

### DCG: Discounted Cumulative Gain

**Idea**: Weight each item's relevance by its position discount.

**Formula**:
$$\text{DCG@K} = \sum_{i=1}^K \frac{\text{rel}_i}{\log_2(i+1)}$$

where:
- $\text{rel}_i$ = relevance score of item at position $i$
- $\log_2(i+1)$ = position discount (larger for lower positions)

**Why $\log_2(i+1)$ instead of $\log_2(i)$?**

If we used $\log_2(i)$, position 1 would have $\log_2(1) = 0$ in denominator. Using $i+1$ gives:
- Position 1: $\log_2(2) = 1$ (full credit)
- Position 2: $\log_2(3) = 1.58$ (63% credit)
- Position 3: $\log_2(4) = 2$ (50% credit)

---

### IDCG: The Ideal Ranking

**Question**: What's the best possible DCG?

**Answer**: Sort all relevant items by relevance, place at top positions.

**Formula**:
$$\text{IDCG@K} = \sum_{i=1}^{\min(K, |\text{relevant}|)} \frac{\text{rel}^*_i}{\log_2(i+1)}$$

where $\text{rel}^*_i$ = relevance of $i$-th most relevant item.

---

### NDCG: Normalized DCG

**Problem**: DCG values aren't comparable across queries (different numbers of relevant items).

**Solution**: Normalize by ideal.

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

**Range**: 0 to 1 (1 = perfect ranking)

**Key Property**: NDCG = 1 if and only if all top-K items are relevant AND sorted by relevance.

---

### NDCG Numerical Example

**Scenario**: Binary relevance (relevant = 1, not relevant = 0)

```
Recommended:  [A, B, C, D, E]  (K=5)
Relevant:     {A, C, E}

Actual ranking:
  Position 1 (A): rel=1, discount=log2(2)=1.0  -> 1/1.0 = 1.000
  Position 2 (B): rel=0, discount=log2(3)=1.58 -> 0/1.58 = 0.000
  Position 3 (C): rel=1, discount=log2(4)=2.0  -> 1/2.0 = 0.500
  Position 4 (D): rel=0, discount=log2(5)=2.32 -> 0/2.32 = 0.000
  Position 5 (E): rel=1, discount=log2(6)=2.58 -> 1/2.58 = 0.387

DCG@5 = 1.000 + 0.000 + 0.500 + 0.000 + 0.387 = 1.887

Ideal ranking (all relevant at top): [A, C, E, ?, ?]
  Position 1: 1/1.0 = 1.000
  Position 2: 1/1.58 = 0.631
  Position 3: 1/2.0 = 0.500

IDCG@5 = 1.000 + 0.631 + 0.500 = 2.131

NDCG@5 = 1.887 / 2.131 = 0.886
```

**Interpretation**: Our ranking captures 88.6% of the ideal's gain.

```python
def dcg_at_k(relevances, k):
    """
    relevances: list of relevance scores (in ranking order)
    k: number of positions to consider
    """
    relevances = np.asarray(relevances)[:k]
    if relevances.size:
        # Positions 1, 2, ..., k have discounts log2(2), log2(3), ..., log2(k+1)
        discounts = np.log2(np.arange(2, relevances.size + 2))
        return np.sum(relevances / discounts)
    return 0.0

def ndcg_at_k(predicted_relevances, true_relevances, k):
    """
    predicted_relevances: relevances in predicted order
    true_relevances: all true relevance values (for computing IDCG)
    """
    dcg = dcg_at_k(predicted_relevances, k)
    # IDCG: sort relevances descending
    idcg = dcg_at_k(sorted(true_relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0
```

---

## MAP: Mean Average Precision - Full Derivation

### Why Average Precision?

*"Professor, Precision@K only looks at one cutoff. What if I want to capture ranking quality at all positions?"*

**Great question. That's exactly what Average Precision does.**

**Intuition**: Compute precision at every position where we find a relevant item, then average.

**Why only at relevant positions?** Because precision at non-relevant positions doesn't tell us anything new about how well we're ranking relevant items.

---

### AP Derivation

**Setup**:
- Ranked list: $[r_1, r_2, \ldots, r_N]$
- Relevant items: $\mathcal{R}$
- $\text{rel}(k) = 1$ if item at position $k$ is relevant, else $0$

**Average Precision**:
$$\text{AP} = \frac{1}{|\mathcal{R}|} \sum_{k=1}^N \text{Precision@k} \cdot \text{rel}(k)$$

**Expanding**:
$$\text{AP} = \frac{1}{|\mathcal{R}|} \sum_{k: \text{rel}(k)=1} \text{Precision@k}$$

**What this captures**:
- If relevant items appear early: Precision@k is high when we sum
- If relevant items appear late: Precision@k is low when we sum

---

### AP Step-by-Step Example

```
Recommended: [A, B, C, D, E, F, G, H, I, J]
Relevant:    {A, C, E, H}  (4 items)

Position-by-position analysis:

Position 1 (A): RELEVANT
  Precision@1 = 1/1 = 1.000
  Contribute to AP: 1.000

Position 2 (B): not relevant
  (Skip - doesn't contribute)

Position 3 (C): RELEVANT
  Precision@3 = 2/3 = 0.667
  Contribute to AP: 0.667

Position 4 (D): not relevant
  (Skip)

Position 5 (E): RELEVANT
  Precision@5 = 3/5 = 0.600
  Contribute to AP: 0.600

Position 6 (F): not relevant
Position 7 (G): not relevant

Position 8 (H): RELEVANT
  Precision@8 = 4/8 = 0.500
  Contribute to AP: 0.500

Positions 9, 10: not relevant

AP = (1.000 + 0.667 + 0.600 + 0.500) / 4 = 2.767 / 4 = 0.692
```

**Interpretation**: On average, when we found a relevant item, 69.2% of items above it were also relevant.

---

### Why AP Captures Ranking Quality

**Compare two rankings**:

**Ranking 1** (good): [A, C, E, H, B, D, F, G, I, J]
- All relevant items at top
- AP = (1/1 + 2/2 + 3/3 + 4/4) / 4 = 4/4 = **1.000**

**Ranking 2** (poor): [B, D, F, G, I, J, A, C, E, H]
- All relevant items at bottom
- AP = (1/7 + 2/8 + 3/9 + 4/10) / 4 = (0.143 + 0.250 + 0.333 + 0.400) / 4 = **0.282**

**Same items, different rankings, vastly different AP.**

---

### MAP: Aggregating Across Users

**Mean Average Precision**:
$$\text{MAP} = \frac{1}{|U|} \sum_{u \in U} \text{AP}_u$$

Simply average AP across all users.

```python
def average_precision(recommended, relevant):
    """
    Compute Average Precision for one user/query.
    """
    if len(relevant) == 0:
        return 0.0

    score = 0.0
    num_hits = 0.0

    for i, item in enumerate(recommended):
        if item in relevant:
            num_hits += 1
            precision_at_i = num_hits / (i + 1)
            score += precision_at_i

    return score / len(relevant)

def mean_average_precision(recommendations, relevances):
    """
    Compute MAP across multiple users/queries.

    recommendations: list of lists (one per user)
    relevances: list of sets (relevant items per user)
    """
    return np.mean([average_precision(rec, rel)
                    for rec, rel in zip(recommendations, relevances)])
```

---

## MRR: Mean Reciprocal Rank

### When to Use MRR

*"Professor, when do I use MRR instead of MAP or NDCG?"*

**MRR is for scenarios where only the FIRST relevant result matters.**

**Use cases**:
- **Question answering**: User wants ONE correct answer
- **Navigation queries**: "Facebook login page" - only need one result
- **Known-item search**: "Inception movie 2010" - user wants that specific item

---

### MRR Definition

**Reciprocal Rank**: 1 divided by position of first relevant item.

$$\text{RR} = \frac{1}{\text{rank of first relevant item}}$$

**If no relevant item found**: RR = 0

**MRR**: Average RR across all queries/users.

$$\text{MRR} = \frac{1}{|U|} \sum_{u=1}^{|U|} \frac{1}{\text{rank}_u}$$

---

### MRR Example

```
User 1: Recommended [A, B, C], Relevant {B, C}
  First relevant: B at position 2
  RR = 1/2 = 0.500

User 2: Recommended [A, B, C], Relevant {A}
  First relevant: A at position 1
  RR = 1/1 = 1.000

User 3: Recommended [A, B, C], Relevant {D}
  No relevant item found
  RR = 0

MRR = (0.500 + 1.000 + 0) / 3 = 0.500
```

---

### MRR vs. Other Metrics: Decision Guide

| Scenario | Best Metric | Why |
|----------|-------------|-----|
| User wants ONE answer | MRR | Only first relevant matters |
| User browses multiple items | NDCG | Position-weighted relevance |
| Binary relevance, full list | MAP | Captures all relevant positions |
| Graded relevance (1-5 stars) | NDCG | Handles non-binary relevance |
| Click prediction | Precision@K | Focus on top positions |
| Catalog discovery | Recall@K | Finding all relevant items |

```python
def reciprocal_rank(recommended, relevant):
    """
    Compute Reciprocal Rank for one query.
    """
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0

def mean_reciprocal_rank(recommendations, relevances):
    return np.mean([reciprocal_rank(rec, rel)
                    for rec, rel in zip(recommendations, relevances)])
```

---

## Complete Numerical Walkthrough: All Metrics for One User

**Let's compute EVERY metric for the same example.**

### Setup

```
Catalog: 20 items total
User's relevant items: {A, C, E, H, J}  (5 relevant)
Recommendations: [A, B, C, D, E, F, G, H, I, J]  (K=10)

Relevance pattern:
Position:  1   2   3   4   5   6   7   8   9   10
Item:      A   B   C   D   E   F   G   H   I   J
Relevant:  1   0   1   0   1   0   0   1   0   1
```

### Step 1: Precision@K

```
Hits in top-10: A, C, E, H, J = 5

Precision@5 = 3/5 = 0.600  (A, C, E in top-5)
Precision@10 = 5/10 = 0.500
```

### Step 2: Recall@K

```
Total relevant: 5

Recall@5 = 3/5 = 0.600  (found A, C, E)
Recall@10 = 5/5 = 1.000  (found all!)
```

### Step 3: F1@K

```
F1@5 = 2 * (0.6 * 0.6) / (0.6 + 0.6) = 0.600
F1@10 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 0.667
```

### Step 4: MAP (Average Precision)

```
Precision at each relevant position:
  Position 1 (A): P@1 = 1/1 = 1.000
  Position 3 (C): P@3 = 2/3 = 0.667
  Position 5 (E): P@5 = 3/5 = 0.600
  Position 8 (H): P@8 = 4/8 = 0.500
  Position 10 (J): P@10 = 5/10 = 0.500

AP = (1.000 + 0.667 + 0.600 + 0.500 + 0.500) / 5
AP = 3.267 / 5 = 0.653
```

### Step 5: NDCG@10

```
DCG calculation:
  Pos 1 (rel=1): 1/log2(2) = 1/1.000 = 1.000
  Pos 2 (rel=0): 0/log2(3) = 0.000
  Pos 3 (rel=1): 1/log2(4) = 1/2.000 = 0.500
  Pos 4 (rel=0): 0/log2(5) = 0.000
  Pos 5 (rel=1): 1/log2(6) = 1/2.585 = 0.387
  Pos 6 (rel=0): 0.000
  Pos 7 (rel=0): 0.000
  Pos 8 (rel=1): 1/log2(9) = 1/3.170 = 0.315
  Pos 9 (rel=0): 0.000
  Pos 10 (rel=1): 1/log2(11) = 1/3.459 = 0.289

DCG@10 = 1.000 + 0.500 + 0.387 + 0.315 + 0.289 = 2.491

IDCG calculation (all 5 relevant items at top):
  Pos 1: 1/1.000 = 1.000
  Pos 2: 1/1.585 = 0.631
  Pos 3: 1/2.000 = 0.500
  Pos 4: 1/2.322 = 0.431
  Pos 5: 1/2.585 = 0.387

IDCG@10 = 1.000 + 0.631 + 0.500 + 0.431 + 0.387 = 2.949

NDCG@10 = 2.491 / 2.949 = 0.845
```

### Step 6: MRR

```
First relevant item: A at position 1

RR = 1/1 = 1.000
```

### Summary Table

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Precision@5 | 0.600 | 60% of top-5 is relevant |
| Precision@10 | 0.500 | 50% of top-10 is relevant |
| Recall@5 | 0.600 | Found 60% of all relevant items in top-5 |
| Recall@10 | 1.000 | Found all relevant items in top-10 |
| F1@5 | 0.600 | Balanced precision/recall at K=5 |
| F1@10 | 0.667 | Balanced precision/recall at K=10 |
| AP | 0.653 | Average precision when finding relevant items |
| NDCG@10 | 0.845 | Captured 84.5% of ideal ranking gain |
| MRR | 1.000 | First relevant item at position 1 |

---

## Beyond-Accuracy Metrics

### Coverage

**Catalog Coverage**: Percentage of items recommended at least once.

$$\text{Coverage} = \frac{|\bigcup_{u \in U} R_u|}{|I|}$$

where $R_u$ = items recommended to user $u$.

**Goal**: High coverage indicates diverse recommendations (not just popular items).

```python
def catalog_coverage(all_recommendations, n_items):
    unique_items = set()
    for recs in all_recommendations:
        unique_items.update(recs)
    return len(unique_items) / n_items
```

---

### Diversity

**Intra-list diversity**: How different are items within a single recommendation list?

$$\text{Diversity} = \frac{1}{K(K-1)} \sum_{i \neq j} d(i, j)$$

where $d(i, j)$ = dissimilarity (e.g., 1 - cosine similarity).

```python
from sklearn.metrics.pairwise import cosine_similarity

def intra_list_diversity(recommended_items, item_features):
    """
    item_features: (n_items, n_features) embeddings
    """
    rec_features = item_features[recommended_items]
    similarity_matrix = cosine_similarity(rec_features)

    # Average pairwise dissimilarity
    n = len(recommended_items)
    diversity = 0
    for i in range(n):
        for j in range(i+1, n):
            diversity += (1 - similarity_matrix[i, j])

    return diversity / (n * (n-1) / 2) if n > 1 else 0
```

---

### Novelty

**Novelty**: How surprising are recommendations?

$$\text{Novelty} = -\frac{1}{K} \sum_{i \in R} \log_2 p(i)$$

where $p(i)$ = popularity (fraction of users who interacted with item $i$).

**High novelty**: Recommending niche items (low $p(i)$).

```python
def novelty(recommended_items, item_popularity):
    """
    item_popularity: dict {item_id: popularity score (0-1)}
    """
    scores = [-np.log2(item_popularity[i] + 1e-10)
              for i in recommended_items]
    return np.mean(scores)
```

---

### Serendipity

**Serendipity**: Relevant AND unexpected.

$$\text{Serendipity} = \text{relevant} \cap \text{unexpected}$$

**Unexpectedness**: Not predictable by simple baseline (e.g., not popular).

**Challenging to measure**: Requires defining "expected" recommendations.

---

## What Can Go Wrong: The Offline-Online Gap

### The Fundamental Problem

*"Professor, my model has amazing offline metrics but users hate it. What went wrong?"*

**This happens more often than you'd think. Let me explain why.**

---

### Problem 1: Selection Bias in Historical Data

**Issue**: You only observe feedback on items that were shown.

**Example**:
```
Historical data shows:
  - Users clicked on popular movies
  - Therefore, model learns to recommend popular movies

Reality:
  - Users were ONLY shown popular movies
  - They might love indie films, but we never showed them
  - Model perpetuates existing bias
```

**The Missing Data Problem**:
- Positive: User clicked item A
- Negative: User didn't click item B
- **Unknown**: Would user have clicked item C if we had shown it?

**Consequence**: Offline metrics measure performance on SHOWN items, not on what users actually want.

---

### Problem 2: Position Bias

**Issue**: Items shown in top positions get more clicks regardless of relevance.

**Example**:
```
Observed data:
  Position 1: 30% CTR
  Position 5: 5% CTR

Interpretation 1: Position 1 items are more relevant
Interpretation 2: Users click position 1 because it's first

Reality: Usually a mix of both, but your model can't distinguish.
```

**Impact on Evaluation**:
- Test set inherits position bias from training set
- Model that puts ANY item in position 1 looks good
- Doesn't reflect true relevance

---

### Problem 3: Popularity Bias in Evaluation

**Issue**: Popular items dominate both training and test sets.

**Example**:
```
Catalog: 1M items
Test set: 100K interactions

Distribution:
  Top 1% items: 50% of test interactions
  Bottom 50% items: 5% of test interactions

Consequence:
  Model that recommends only popular items
  → High precision (popular items in test set)
  → Users see same 100 items forever
  → Discovery and diversity suffer
```

**Solution**: Stratified evaluation (report metrics by item popularity buckets).

---

### Problem 4: Temporal Distribution Shift

**Issue**: User preferences change over time.

**Example**:
```
Training data: Jan 2024 - Oct 2024
Test data: Nov 2024

What changed:
  - New trending topics
  - Seasonal preferences (holidays)
  - New items added
  - User tastes evolved

Result: Model optimized for past, evaluated on past, deployed to future.
```

---

### Problem 5: Feedback Loops

**Issue**: Your model's recommendations influence future training data.

**Cycle**:
```
1. Model recommends popular items
2. Users interact with popular items (no choice)
3. Training data shows users prefer popular items
4. Model learns to recommend popular items more
5. Repeat → filter bubble / echo chamber
```

**Offline evaluation can't detect this** because it uses static historical data.

---

### How to Mitigate

**1. Counterfactual Evaluation**:
```python
# Weight samples by inverse propensity
# Items that were unlikely to be shown get higher weight
weight = 1.0 / propensity_score[item]
```

**2. Unbiased Datasets**:
- Randomly show some items (exploration)
- Use this unbiased data for evaluation

**3. Multiple Evaluation Sets**:
- Test on different time periods
- Test on different user segments
- Test on long-tail items separately

**4. A/B Testing**:
- The gold standard: test on real users
- Offline metrics for filtering, online metrics for decisions

---

## Metric Selection Guide: Decision Tree

### When to Use Each Metric

```
START: What's your recommendation scenario?
│
├── Rating prediction (1-5 stars)?
│   └── Use RMSE or MAE
│       - RMSE if large errors are costly
│       - MAE if errors are equally bad
│
├── Ranking with binary relevance (like/don't like)?
│   │
│   ├── User wants ONE result (search, QA)?
│   │   └── Use MRR
│   │
│   ├── User examines full list (browse)?
│   │   └── Use MAP or Recall@K
│   │       - MAP: care about ranking order
│   │       - Recall@K: care about finding all relevant
│   │
│   └── Focus on top positions (homepage)?
│       └── Use Precision@K or NDCG@K
│           - Precision@K: simpler, interpretable
│           - NDCG@K: position-weighted
│
├── Ranking with graded relevance (1-5 relevance)?
│   └── Use NDCG
│       - Handles non-binary relevance
│       - Position-weighted
│
└── Beyond accuracy concerns?
    │
    ├── Worried about filter bubbles?
    │   └── Add Diversity metric
    │
    ├── Worried about only popular items?
    │   └── Add Coverage and Novelty
    │
    └── Want surprising recommendations?
        └── Add Serendipity
```

### Quick Reference Table

| Use Case | Primary Metric | Secondary Metrics |
|----------|----------------|-------------------|
| E-commerce product ranking | NDCG@10 | Precision@10, Coverage |
| News article recommendation | CTR (Precision@1) | Diversity, Novelty |
| Movie recommendation | NDCG@K | Recall@K, Coverage |
| Music playlist generation | NDCG@K | Diversity, Novelty |
| Search engine | MRR | NDCG@10 |
| Social feed | CTR, Engagement time | Diversity |
| Job recommendations | Precision@K | Recall@K, Coverage |
| Course recommendations | Recall@K | Diversity |

---

## Combining Metrics

### Multi-Objective Evaluation

**Reality**: Trade-offs between metrics.

**Example**:
- High precision often leads to recommending only popular items
- High diversity often leads to recommending unpopular items

**Solution**: Combine metrics with weights:

$$\text{Score} = \alpha \cdot \text{NDCG} + \beta \cdot \text{Diversity} + \gamma \cdot \text{Coverage}$$

**Hyperparameters** ($\alpha, \beta, \gamma$): Domain-specific.

---

## Implementation: Complete Evaluation Suite

```python
class RecommenderEvaluator:
    def __init__(self, item_features=None):
        self.item_features = item_features

    def evaluate(self, recommendations, ground_truth, k=10):
        """
        Comprehensive evaluation of a recommender system.

        recommendations: List[List[item_ids]] (one list per user)
        ground_truth: List[Set[item_ids]] (relevant items per user)

        Returns dict with all metrics.
        """
        metrics = {}
        n_users = len(recommendations)

        # Basic metrics
        precisions, recalls, aps, mrrs = [], [], [], []
        ndcgs = []

        for rec, gt in zip(recommendations, ground_truth):
            rec_k = rec[:k]
            gt_set = set(gt)

            # Precision and Recall
            hits = len(set(rec_k) & gt_set)
            precisions.append(hits / k)
            recalls.append(hits / len(gt_set) if gt_set else 0)

            # AP
            aps.append(average_precision(rec, gt_set))

            # MRR
            mrrs.append(reciprocal_rank(rec, gt_set))

            # NDCG
            relevances = [1 if item in gt_set else 0 for item in rec_k]
            if gt_set:
                ndcgs.append(ndcg_at_k(relevances, [1]*len(gt_set), k))
            else:
                ndcgs.append(0)

        metrics[f'Precision@{k}'] = np.mean(precisions)
        metrics[f'Recall@{k}'] = np.mean(recalls)
        metrics['MAP'] = np.mean(aps)
        metrics['MRR'] = np.mean(mrrs)
        metrics[f'NDCG@{k}'] = np.mean(ndcgs)

        # F1
        p, r = metrics[f'Precision@{k}'], metrics[f'Recall@{k}']
        metrics[f'F1@{k}'] = 2 * p * r / (p + r) if (p + r) > 0 else 0

        # Coverage
        all_items_rec = set()
        for rec in recommendations:
            all_items_rec.update(rec[:k])
        # Note: need total catalog size for actual coverage
        metrics['Items_Recommended'] = len(all_items_rec)

        # Diversity (if features available)
        if self.item_features is not None:
            diversities = [intra_list_diversity(rec[:k], self.item_features)
                          for rec in recommendations]
            metrics['Diversity'] = np.mean(diversities)

        return metrics

    def evaluate_by_popularity(self, recommendations, ground_truth,
                               item_popularity, k=10):
        """
        Stratified evaluation by item popularity.

        Returns metrics for popular vs. long-tail items.
        """
        # Split items into popularity buckets
        pop_threshold = np.percentile(list(item_popularity.values()), 80)

        popular_gt = []
        longtail_gt = []

        for gt in ground_truth:
            pop_items = {i for i in gt if item_popularity.get(i, 0) >= pop_threshold}
            tail_items = {i for i in gt if item_popularity.get(i, 0) < pop_threshold}
            popular_gt.append(pop_items)
            longtail_gt.append(tail_items)

        results = {
            'overall': self.evaluate(recommendations, ground_truth, k),
            'popular': self.evaluate(recommendations, popular_gt, k),
            'long_tail': self.evaluate(recommendations, longtail_gt, k)
        }

        return results


# Example usage
if __name__ == "__main__":
    # Sample data
    recommendations = [
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        ['X', 'Y', 'A', 'Z', 'W', 'V', 'U', 'T', 'S', 'R']
    ]
    ground_truth = [
        {'A', 'C', 'E', 'H', 'J'},
        {'A', 'B', 'Q'}
    ]

    evaluator = RecommenderEvaluator()
    results = evaluator.evaluate(recommendations, ground_truth, k=10)

    print("Evaluation Results:")
    print("-" * 40)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
```

---

## Summary

### Key Takeaways

1. **Accuracy alone is dangerous**: High accuracy can mean biased, unhelpful recommendations

2. **Precision vs. Recall tradeoff**: Fundamental tension - as K increases, recall goes up, precision typically goes down

3. **NDCG uses logarithmic discounting** because it models attention decay (steep at top, gradual later)

4. **MAP captures ranking quality** by averaging precision at each relevant item position

5. **MRR for single-answer scenarios**: When only the first relevant result matters

6. **Offline metrics have blind spots**: Selection bias, position bias, popularity bias, temporal shift

7. **Always use multiple metrics**: No single metric captures everything

### Best Practices

- **Start with NDCG** for graded relevance, **MAP** for binary relevance
- **Always report coverage and diversity** to catch filter bubbles
- **Stratify by popularity**: Report metrics for popular and long-tail separately
- **Stratify by user type**: New users, power users, different demographics
- **Use offline metrics for filtering**, online A/B tests for final decisions
- **Track metric trends over time**, not just point estimates

### The Hierarchy of Evaluation

```
                    A/B Testing (Gold Standard)
                         ↑
                    Interleaving
                         ↑
              Counterfactual Evaluation
                         ↑
         Offline Metrics (Necessary but insufficient)
```

**Next**: Experimental design and A/B testing.

---

## References

1. **Herlocker, J. L., et al. (2004)**. "Evaluating Collaborative Filtering Recommender Systems". *ACM TOIS*.
   - Comprehensive framework for RecSys evaluation

2. **Shani, G., & Gunawardana, A. (2011)**. "Evaluating Recommendation Systems". *Recommender Systems Handbook*.
   - Definitive guide to offline and online evaluation

3. **Jarvelin, K., & Kekalainen, J. (2002)**. "Cumulated Gain-Based Evaluation of IR Techniques". *ACM TOIS*.
   - Original NDCG paper with logarithmic discounting derivation

4. **Vargas, S., & Castells, P. (2011)**. "Rank and Relevance in Novelty and Diversity Metrics for Recommender Systems". *RecSys*.
   - Beyond-accuracy metrics

5. **Schnabel, T., et al. (2016)**. "Recommendations as Treatments: Debiasing Learning and Evaluation". *ICML*.
   - Counterfactual evaluation and propensity scoring

6. **Joachims, T., et al. (2017)**. "Unbiased Learning-to-Rank with Biased Feedback". *WSDM*.
   - Position bias and unbiased evaluation

---

## Practice Problems

### Problem 1: Precision-Recall Tradeoff

**Given**:
```
User has 8 relevant items in catalog of 1000.
You can recommend K items.

Compute best-case Precision@K and Recall@K for:
a) K = 4
b) K = 8
c) K = 16
```

**Solution**:
```
a) K = 4:
   Best case: All 4 recommendations are relevant
   Precision@4 = 4/4 = 1.0
   Recall@4 = 4/8 = 0.5

b) K = 8:
   Best case: All 8 recommendations are relevant (all found!)
   Precision@8 = 8/8 = 1.0
   Recall@8 = 8/8 = 1.0

c) K = 16:
   Best case: 8 relevant + 8 irrelevant (only 8 relevant exist)
   Precision@16 = 8/16 = 0.5
   Recall@16 = 8/8 = 1.0

Observation: After K > |relevant|, precision drops, recall stays at 1.0.
```

---

### Problem 2: NDCG Calculation

**Given**:
```
Recommendations: [D, A, B, C, E]
Relevant items: {A, C, E} (binary relevance)

Compute NDCG@5.
```

**Solution**:
```
Relevance vector: [0, 1, 0, 1, 1] (D=0, A=1, B=0, C=1, E=1)

DCG@5:
  = 0/log2(2) + 1/log2(3) + 0/log2(4) + 1/log2(5) + 1/log2(6)
  = 0 + 0.631 + 0 + 0.431 + 0.387
  = 1.449

IDCG@5 (best ordering [A, C, E, ?, ?]):
  = 1/log2(2) + 1/log2(3) + 1/log2(4)
  = 1.0 + 0.631 + 0.5
  = 2.131

NDCG@5 = 1.449 / 2.131 = 0.680
```

---

### Problem 3: MAP Calculation

**Given**:
```
Recommendations: [A, B, C, D, E, F, G, H, I, J]
Relevant: {B, D, F, H, J}
```

**Solution**:
```
Relevant items found at positions: 2, 4, 6, 8, 10

Precision at each relevant position:
  P@2 = 1/2 = 0.500 (B found)
  P@4 = 2/4 = 0.500 (D found)
  P@6 = 3/6 = 0.500 (F found)
  P@8 = 4/8 = 0.500 (H found)
  P@10 = 5/10 = 0.500 (J found)

AP = (0.5 + 0.5 + 0.5 + 0.5 + 0.5) / 5 = 0.500

Note: Even spacing of relevant items leads to constant precision.
Compare to relevant items at positions 1,2,3,4,5: AP = 1.0
```

---

### Problem 4: Metric Selection

**Scenario**: You're building a job recommendation system for LinkedIn.

**Question**: Which metrics would you prioritize and why?

**Solution**:
```
Primary: Recall@K
  - Users need to see ALL relevant job postings
  - Missing a good job opportunity is costly
  - K should be reasonably large (20-50)

Secondary: NDCG@K
  - Best jobs should appear first
  - Users have limited time to browse

Also track:
  - Coverage: Are we showing diverse job types?
  - Diversity: Are recommendations too similar?
  - Novelty: Are we only showing well-known companies?

Business metrics to correlate with:
  - Application rate
  - Interview rate
  - Hire rate (ultimate success metric)
```

---

### Problem 5: Identifying Bias

**Scenario**: Your model achieves NDCG@10 = 0.85 offline but users complain about seeing the same recommendations.

**Diagnose**: What metrics would reveal the problem?

**Solution**:
```
1. Coverage: Likely very low
   - Model recommending same popular items to everyone

2. Diversity: Likely low
   - Each user's list contains similar items

3. Stratified NDCG:
   - Compute NDCG separately for popular vs. long-tail items
   - Likely high on popular, low on long-tail

4. Per-user item overlap:
   - Compute Jaccard similarity between users' recommendations
   - If high, everyone sees the same things

The high NDCG is inflated by popularity bias in the test set.
```
