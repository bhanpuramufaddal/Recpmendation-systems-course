# Week 12: Debiasing Techniques

## Overview

**Debiasing** removes or reduces biases in training data and models.

**Approaches**:
1. **Data reweighting**: Inverse propensity scoring
2. **Causal inference**: Estimate counterfactuals
3. **Regularization**: Penalize biased predictions
4. **Re-ranking**: Post-process to debias

---

## Inverse Propensity Scoring (IPS)

### Method

**Idea**: Reweight observations by inverse of selection probability.

**Weight**:
$$w_{ui} = \frac{1}{p(i | u)}$$

**Loss**:
$$\mathcal{L}_{\text{IPS}} = \sum_{(u,i) \in D} w_{ui} \cdot \text{loss}(r_{ui}, \hat{r}_{ui})$$

**Effect**: Down-weight over-represented items (popular), up-weight under-represented.

---

### Implementation

```python
def ips_loss(predictions, targets, propensities):
    """
    propensities: p(item shown | user)
    """
    weights = 1.0 / (propensities + 1e-8)
    weighted_mse = weights * (predictions - targets) ** 2
    return weighted_mse.mean()
```

---

### Variance Reduction

**Problem**: IPS has high variance (especially for rare items with low $p$).

**Solution**: **Capped IPS** - cap weights.

$$w_{ui} = \min\left(\frac{1}{p(i|u)}, M\right)$$

where $M$ = max weight (e.g., 100).

---

## Doubly Robust Estimation

### Method

**Combine** IPS + imputation.

$$\hat{r}_{ui} = \hat{r}_{\text{model}}(u, i) + \frac{1}{p(i|u)} \cdot (r_{ui} - \hat{r}_{\text{model}}(u, i))$$

**Benefits**:
- If model accurate → low variance
- If propensities accurate → unbiased

**Doubly robust**: Unbiased if **either** model or propensities correct.

---

## Causal Inference

### Counterfactual Reasoning

**Question**: What would user rating be if we showed different item?

**Notation**:
- $r_{ui}^{obs}$: Observed rating (item $i$ shown)
- $r_{uj}^{cf}$: Counterfactual (item $j$ not shown)

**Challenge**: Can't observe $r_{uj}^{cf}$ → must estimate.

---

### Propensity-based Debiasing

**Estimate** $r_{uj}^{cf}$ using similar users who saw item $j$:

$$\hat{r}_{uj}^{cf} = \mathbb{E}[r_{u'j} | u' \sim u, (u', j) \in D]$$

---

## Debiasing Popularity

### Calibration

**Goal**: Match recommendation distribution to user's true preference distribution.

**Method**: Re-rank to match popularity distribution of user's history.

```python
def calibrate_recommendations(rec_items, user_history_popularity, item_popularity):
    """
    Re-rank to match user's historical popularity distribution.
    """
    target_dist = np.histogram(user_history_popularity, bins=10)[0]
    target_dist = target_dist / target_dist.sum()

    # Score items by how well they match target distribution
    scores = []
    for item in rec_items:
        pop_bin = int(item_popularity[item] * 10)
        scores.append(target_dist[min(pop_bin, 9)])

    # Re-rank
    reranked = [x for _, x in sorted(zip(scores, rec_items), reverse=True)]
    return reranked
```

---

### Regularization

**Penalize** popular items during training.

$$\mathcal{L} = \mathcal{L}_{\text{base}} + \lambda \sum_{i} \log(1 + \text{popularity}(i)) \cdot \hat{r}_{ui}^2$$

**Effect**: Discourage recommending already-popular items.

---

## Debiasing Position

**Randomization**: Randomly shuffle top results occasionally.

**Inverse rank weighting**: Weight clicks by $1 / \text{position}$.

```python
def position_debiased_ctr(clicks, positions):
    weights = 1.0 / np.array(positions)
    return (clicks * weights).sum() / weights.sum()
```

---

## Exposure Fairness

### Equal Exposure

**Goal**: All items/creators get fair exposure.

**Method**: **Exposure redistribution**

```python
def exposure_fair_rerank(scored_items, current_exposure, target_exposure_per_item, k=10):
    """
    Re-rank to equalize exposure across items.
    """
    # Compute exposure deficit
    deficit = {item: target_exposure_per_item - current_exposure.get(item, 0)
               for item, score in scored_items}

    # Re-score: base score + deficit bonus
    adjusted_scores = {item: score + 0.1 * deficit[item]
                       for item, score in scored_items}

    # Re-rank
    reranked = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, score in reranked[:k]]
```

---

## Summary

**Key Takeaways**:
1. **IPS**: Reweight by inverse propensity (handle selection bias)
2. **Doubly robust**: Combine IPS + imputation (lower variance)
3. **Causal inference**: Estimate counterfactuals
4. **Calibration**: Match recommendation to user preference distribution
5. **Exposure fairness**: Equalize item/creator exposure

**Trade-offs**: Accuracy vs. fairness (often inversely correlated).

**Next**: Fairness definitions and methods.

---

## References

1. **Schnabel, T., et al. (2016)**. "Recommendations as Treatments". *ICML*.
2. **Wang, X., et al. (2019)**. "Doubly Robust Joint Learning for Recommendation on Data Missing Not at Random". *ICML*.
3. **Steck, H. (2018)**. "Calibrated Recommendations". *RecSys*.
