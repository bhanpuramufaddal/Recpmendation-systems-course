# Week 11: Experimental Design for RecSys

## Overview

**Experimental design** ensures evaluation is **valid, reliable, and unbiased**.

**Key questions**:
- How to split data (train/test)?
- How to handle temporal dynamics?
- How to ensure statistical significance?

This document covers experimental design best practices.

---

## Train-Test Splitting

### Random Splitting

**Method**: Randomly split interactions 80/20.

**Pros**: Simple, standard in ML.
**Cons**: **Data leakage** - test set may contain earlier interactions than train (violates causality).

```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(interactions, test_size=0.2, random_state=42)
```

---

### Temporal Splitting

**Method**: Split by time (last 20% of interactions → test).

**Benefit**: Realistic - predict future from past.

**Example** (Netflix):
- Train: Ratings before Dec 1, 2024
- Test: Ratings after Dec 1, 2024

```python
def temporal_split(df, test_ratio=0.2):
    df = df.sort_values('timestamp')
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test
```

---

### Leave-One-Out

**Method**: For each user, hold out 1 interaction (usually last) for testing.

**Use case**: Very sparse data.

**Variants**:
- **Leave-last-one-out**: Hold out most recent interaction
- **Leave-k-out**: Hold out last k interactions

```python
def leave_one_out(user_interactions):
    """
    user_interactions: dict {user_id: [item1, item2, ...]}
    """
    train = {}
    test = {}

    for user, items in user_interactions.items():
        if len(items) > 1:
            train[user] = items[:-1]
            test[user] = [items[-1]]

    return train, test
```

---

## Cross-Validation

### K-Fold Cross-Validation

**Problem**: Random k-fold violates temporal order.

**Solution**: **Time-series CV** - fold by time windows.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(interactions):
    train = interactions.iloc[train_idx]
    test = interactions.iloc[test_idx]
    # Train and evaluate
```

---

### User-Based Splitting

**Goal**: Ensure no user in test set has all interactions in training.

**Method**: Hold out subset of each user's interactions.

---

## Negative Sampling

### Problem

**Implicit feedback**: Only observe positives (clicks, purchases) → no explicit negatives.

**Challenge**: How to evaluate ranking?

**Solution**: **Negative sampling** - sample non-interactions as negatives.

---

### Strategies

**1. Random Negatives**:
```python
def sample_negatives(user, positive_items, all_items, n_negatives=99):
    negatives = list(set(all_items) - set(positive_items))
    return random.sample(negatives, min(n_negatives, len(negatives)))
```

**2. Popularity-biased**:
- Sample negatives proportional to popularity
- Harder task (distinguish from popular items)

**3. Hard negatives**:
- Items user almost clicked but didn't
- Requires logged data (impressions)

---

## Statistical Significance

### Hypothesis Testing

**Null hypothesis** ($H_0$): Model A = Model B (no difference).

**Alternative** ($H_1$): Model A ≠ Model B.

**Test**: Paired t-test (compare metrics per user).

```python
from scipy.stats import ttest_rel

model_a_scores = [0.85, 0.78, 0.92, ...]  # NDCG per user
model_b_scores = [0.87, 0.80, 0.89, ...]

t_stat, p_value = ttest_rel(model_a_scores, model_b_scores)

if p_value < 0.05:
    print("Statistically significant difference!")
else:
    print("No significant difference.")
```

---

### Effect Size

**Problem**: Statistical significance ≠ practical significance.

**Example**: Improvement of 0.001 NDCG may be significant but not meaningful.

**Solution**: Report **effect size** (Cohen's d).

$$d = \frac{\bar{x}_A - \bar{x}_B}{s_{\text{pooled}}}$$

**Interpretation**:
- $|d| < 0.2$: Small effect
- $0.2 \leq |d| < 0.5$: Medium
- $|d| \geq 0.5$: Large

---

## Cold Start Evaluation

### Problem

**New users/items**: No interactions → hard to recommend.

**Evaluation challenge**: How to measure performance on cold starts?

---

### Strategies

**1. Cold User Evaluation**:
- Hold out first k interactions
- Evaluate on new users with <k interactions

**2. Cold Item Evaluation**:
- Hold out new items (recent releases)
- Evaluate how well model recommends new items

**3. Stratified Reporting**:
```
Cold users (0-5 interactions): NDCG = 0.45
Medium users (6-20): NDCG = 0.62
Active users (>20): NDCG = 0.78
```

---

## Summary

**Key Takeaways**:
1. **Temporal split**: Most realistic for time-series data
2. **Negative sampling**: Required for implicit feedback evaluation
3. **Statistical tests**: Ensure differences are significant
4. **Effect size**: Check practical significance
5. **Stratify**: Report metrics by user activity level

**Best Practices**:
- Use temporal split for production-like evaluation
- Sample hard negatives when possible
- Always test statistical significance (p < 0.05)
- Report results for cold start separately

**Next**: Evaluation challenges (bias, position effects).

---

## References

1. **Bellogin, A., et al. (2017)**. "Statistical Biases in Information Retrieval Metrics for Recommender Systems". *Information Retrieval Journal*.
2. **Cremonesi, P., et al. (2010)**. "Performance of Recommender Algorithms on Top-N Recommendation Tasks". *RecSys*.
