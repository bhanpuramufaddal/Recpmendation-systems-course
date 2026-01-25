# Week 11: Evaluation Methodologies - Practice Problems

## Overview
Master offline metrics (RMSE, NDCG, MAP), experimental design, A/B testing, and evaluation challenges in recommendation systems.

---

## Problem 1: NDCG Calculation
**Difficulty:** Medium

**Recommendations:** [Item A, Item B, Item C, Item D, Item E]
**Relevance:** [3, 2, 3, 0, 1] (0-3 scale)

Calculate NDCG@5:
1. Compute DCG@5
2. Compute IDCG@5 (ideal ordering)
3. NDCG@5 = DCG/IDCG

**Formula:** $DCG@K = \sum_{i=1}^K \frac{2^{rel_i} - 1}{\log_2(i+1)}$

**Learning Outcomes:** Compute ranking metrics, understand position discounting

---

## Problem 2: Train-Test Splitting Strategies
**Difficulty:** Medium

**Dataset:** MovieLens with timestamps

**Strategies:**
1. **Random split:** 80% train, 20% test
2. **Temporal split:** Train on 2015-2019, test on 2020
3. **User-based:** 80% of users for train, 20% for test
4. **Leave-one-out:** For each user, last item = test

**Questions:**
1. Which avoids data leakage for production simulation?
2. Which tests cold-start users?
3. What are the pros/cons of each?
4. Design a splitting strategy for session-based recommendations

**Learning Outcomes:** Choose appropriate splits, avoid leakage, simulate production

---

## Problem 3: A/B Test Design
**Difficulty:** Hard

**Scenario:** Test new recommendation algorithm

**Metrics:**
- **Primary:** Click-through rate (CTR)
- **Guardrails:** Session duration, bounce rate, purchases

**Design:**
1. Sample size calculation (1% minimum detectable effect, 95% confidence)
2. Duration (1 week? 2 weeks?)
3. Segment (all users? new users only?)
4. Success criteria (when to launch?)

**Learning Outcomes:** Design A/B tests, calculate sample size, set success criteria

---

## Problem 4: Precision vs. Recall Trade-off
**Difficulty:** Easy

**Scenario:** Recommend 10 items to a user who liked 20 items in test set

| k | TP | FP | Precision@k | Recall@k |
|---|----|----|-------------|----------|
| 5 | 4  | 1  | 0.80        | 0.20     |
| 10| 7  | 3  | 0.70        | 0.35     |
| 20| 12 | 8  | 0.60        | 0.60     |

**Questions:**
1. Plot precision-recall curve
2. Which k would you choose for production?
3. How does k affect user experience?
4. Calculate F1@10

**Learning Outcomes:** Interpret precision/recall, choose k, balance metrics

---

## Problem 5: Offline-Online Metric Gap
**Difficulty:** Hard

**Observation:** Model A has better offline NDCG, but Model B wins in online A/B test (higher CTR, revenue)

**Possible reasons:**
1. Position bias in offline data
2. Different user behavior online
3. Diversity matters online (not captured by NDCG)
4. Freshness/novelty effects

**Tasks:**
1. Diagnose why the gap exists
2. Design better offline metrics that correlate with online success
3. Propose evaluation framework combining offline + online

**Learning Outcomes:** Understand offline-online gap, design correlated metrics, combine evaluations

---

## Programming Exercises

### Exercise 1: Implement Evaluation Metrics

```python
import numpy as np

def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k

def recall_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant) if len(relevant) > 0 else 0

def ndcg_at_k(recommended, relevant_scores, k=10):
    recommended_k = recommended[:k]
    dcg = sum([relevant_scores.get(item, 0) / np.log2(i + 2) for i, item in enumerate(recommended_k)])
    ideal_scores = sorted(relevant_scores.values(), reverse=True)[:k]
    idcg = sum([score / np.log2(i + 2) for i, score in enumerate(ideal_scores)])
    return dcg / idcg if idcg > 0 else 0

def map_at_k(recommended, relevant, k=10):
    # Mean Average Precision
    precisions = []
    hits = 0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            precisions.append(hits / (i + 1))
    return np.mean(precisions) if precisions else 0
```

---

### Exercise 2: Temporal Cross-Validation

```python
def temporal_cv(data, n_splits=5):
    data = data.sort_values('timestamp')
    fold_size = len(data) // (n_splits + 1)

    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = fold_size * (i + 2)

        train = data.iloc[:train_end]
        test = data.iloc[train_end:test_end]

        yield train, test

# Evaluate
for train, test in temporal_cv(ratings):
    model.fit(train)
    metrics = evaluate(model, test)
    print(metrics)
```

---

### Exercise 3: A/B Test Analysis

```python
from scipy import stats

def ab_test_analysis(control_ctr, treatment_ctr, control_n, treatment_n):
    # Two-proportion z-test
    p_control = control_ctr
    p_treatment = treatment_ctr

    p_pooled = (control_ctr * control_n + treatment_ctr * treatment_n) / (control_n + treatment_n)

    z = (p_treatment - p_control) / np.sqrt(p_pooled * (1 - p_pooled) * (1/control_n + 1/treatment_n))
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    lift = (p_treatment - p_control) / p_control

    return {'z_score': z, 'p_value': p_value, 'lift': lift}

# Example
result = ab_test_analysis(control_ctr=0.05, treatment_ctr=0.052, control_n=10000, treatment_n=10000)
print(f"Lift: {result['lift']:.2%}, p-value: {result['p_value']:.4f}")
```

---

### Exercise 4: Diversity and Coverage Metrics

```python
def diversity(recommendations):
    # Intra-list diversity (average pairwise distance)
    from itertools import combinations
    pairs = list(combinations(recommendations, 2))
    distances = [distance(item1, item2) for item1, item2 in pairs]
    return np.mean(distances)

def coverage(all_recommendations, catalog):
    # % of catalog recommended at least once
    recommended_items = set()
    for rec_list in all_recommendations:
        recommended_items.update(rec_list)
    return len(recommended_items) / len(catalog)

def novelty(recommendations, item_popularity):
    # Average self-information
    novelty_scores = [-np.log2(item_popularity[item]) for item in recommendations]
    return np.mean(novelty_scores)
```

---

## Discussion Questions

1. **Metric Choice:** RMSE for rating prediction, NDCG for ranking. When would you use each?
2. **Cold Start Evaluation:** How do you evaluate on cold-start users/items?
3. **Bias in Evaluation:** Test set has position bias (top items clicked more). How to debias?
4. **Long-Term Metrics:** How do you measure user satisfaction over weeks/months?
5. **Multi-Objective:** Optimize for CTR and diversity. How do you combine into single metric?

---

## References
1. Shani, G., & Gunawardana, A. (2011). "Evaluating recommendation systems". Recommender Systems Handbook.
2. Kohavi, R., et al. (2009). "Controlled experiments on the web: survey and practical guide". Data Mining and Knowledge Discovery.

---

*Return to [Week 11 Main Page](README.md)*
