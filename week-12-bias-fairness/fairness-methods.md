# Week 12: Fairness in Recommendations

## Overview

**Fairness**: Ensuring recommendations don't discriminate against users or items based on protected attributes.

**Stakeholders**:
1. **User fairness**: Equal service quality across demographics
2. **Item/provider fairness**: Equal exposure for content creators

**Fairness definitions** (often conflicting):
- Demographic parity
- Equal opportunity
- Calibration

---

## User Fairness

### Demographic Parity

**Definition**: Recommendation rate independent of protected attribute.

$$P(\hat{y} = 1 | A = 0) = P(\hat{y} = 1 | A = 1)$$

where $A$ = protected attribute (e.g., gender).

**Example**: Same % of recommendations for men and women.

**Limitation**: Ignores if groups have different preferences.

---

### Equal Opportunity

**Definition**: True positive rate equal across groups.

$$P(\hat{y} = 1 | y = 1, A = 0) = P(\hat{y} = 1 | y = 1, A = 1)$$

**Example**: Among users who like action movies, recommend action equally regardless of gender.

**Better than** demographic parity (accounts for true preferences).

---

### Calibration

**Definition**: Predicted probability matches observed frequency.

$$P(y = 1 | \hat{p} = p, A = a) = p \quad \forall a$$

**Example**: If model predicts 80% chance user likes item, 80% of users should like it (for all groups).

---

### Implementation

```python
def demographic_parity(predictions, protected_attribute):
    """Check if recommendation rates are equal"""
    rate_group_0 = predictions[protected_attribute == 0].mean()
    rate_group_1 = predictions[protected_attribute == 1].mean()
    return abs(rate_group_0 - rate_group_1)

def equal_opportunity(predictions, labels, protected_attribute):
    """Check if TPR is equal"""
    group_0_tpr = predictions[(labels == 1) & (protected_attribute == 0)].mean()
    group_1_tpr = predictions[(labels == 1) & (protected_attribute == 1)].mean()
    return abs(group_0_tpr - group_1_tpr)

# Fairness constraint during training
def fair_loss(predictions, labels, protected_attribute, lambda_fair=0.1):
    """Combine accuracy loss + fairness penalty"""
    accuracy_loss = ((predictions - labels) ** 2).mean()
    fairness_penalty = demographic_parity(predictions, protected_attribute)
    return accuracy_loss + lambda_fair * fairness_penalty
```

---

## Provider Fairness

### Exposure Fairness

**Definition**: Each item/creator gets exposure proportional to quality/merit.

**Metric**: **Exposure gap**

$$\text{Gap}_i = \frac{\text{Expected exposure}_i}{\text{Actual exposure}_i}$$

---

### Envy-Freeness

**Definition**: No item prefers another's allocation.

**Formulation**: Item $i$ doesn't envy item $j$ if:

$$\text{utility}_i(\text{allocation}_i) \geq \text{utility}_i(\text{allocation}_j)$$

---

## Multi-Stakeholder Fairness

### Trade-offs

**Challenge**: User fairness vs. provider fairness often conflict.

**Example**:
- User fairness → show diverse content to all users
- Provider fairness → promote small creators
- **Conflict**: Popular items may be highest quality for users

---

### Pareto Optimization

**Goal**: Find recommendations on Pareto frontier (can't improve one without harming other).

**Method**: Multi-objective optimization

$$\max_{\theta} [\text{user\_utility}(\theta), \text{provider\_utility}(\theta)]$$

```python
# Weighted combination
def multi_stakeholder_loss(predictions, user_labels, provider_exposure, alpha=0.7):
    user_loss = ((predictions - user_labels) ** 2).mean()
    provider_loss = -provider_exposure.mean()  # Maximize exposure
    return alpha * user_loss + (1 - alpha) * provider_loss
```

---

## Fairness-Aware Ranking

### Re-ranking for Fairness

**Greedy algorithm**:
1. Score all items by relevance
2. Iteratively add items to ranking:
   - Pick item maximizing: $\text{relevance} + \lambda \cdot \text{fairness\_boost}$

```python
def fair_rerank(scored_items, fairness_scores, lambda_fair=0.3, k=10):
    """
    scored_items: List of (item, relevance_score)
    fairness_scores: Dict {item: fairness_score}
    """
    ranking = []
    remaining = list(scored_items)

    for _ in range(k):
        best_item = None
        best_score = -float('inf')

        for item, rel_score in remaining:
            fair_score = fairness_scores.get(item, 0)
            combined = rel_score + lambda_fair * fair_score

            if combined > best_score:
                best_score = combined
                best_item = (item, rel_score)

        ranking.append(best_item[0])
        remaining.remove(best_item)

    return ranking
```

---

### FA*IR Algorithm

**Fairness-Aware Ranking** (Zehlike et al.):

**Idea**: Ensure minimum representation of minority groups at each rank.

**Constraint**: At position $k$, at least $p \cdot k$ items from minority group.

---

## Summary

**Key Takeaways**:
1. **User fairness**: Demographic parity, equal opportunity, calibration
2. **Provider fairness**: Exposure fairness, envy-freeness
3. **Multi-stakeholder**: Trade-offs, Pareto optimization
4. **Fair ranking**: Re-ranking algorithms (FA*IR)

**Trade-offs**: Fairness often reduces accuracy (short-term).

**Next**: Ethical considerations.

---

## References

1. **Zehlike, M., et al. (2017)**. "FA*IR: A Fair Top-k Ranking Algorithm". *CIKM*.
2. **Burke, R., et al. (2017)**. "Balanced Neighborhoods for Multi-sided Fairness in Recommendation". *FATML*.
3. **Biega, A., et al. (2018)**. "Equity of Attention: Amortizing Individual Fairness in Rankings". *SIGIR*.
