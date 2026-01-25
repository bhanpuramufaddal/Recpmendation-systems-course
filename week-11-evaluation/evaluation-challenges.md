# Week 11: Evaluation Challenges

## Overview

**Evaluation challenges** arise from **biases** in data and metrics that don't reflect real user behavior.

**Key challenges**:
1. **Position bias**: Users click top results regardless of relevance
2. **Popularity bias**: Metrics favor popular items
3. **Selection bias**: Only observe ratings for shown items
4. **Feedback loops**: Recommendations create self-fulfilling prophecies

This document covers these challenges and mitigation strategies.

---

## Position Bias

### Problem

**Observation**: Users more likely to click items shown at top, even if less relevant.

**Example** (Search/RecSys):
- Item at position 1: 40% CTR
- Same item at position 5: 10% CTR

**Impact on evaluation**: Metrics overestimate performance of systems that rank popular items first.

---

### Examination Hypothesis

**Model**: User examines item at position $k$ with probability $p(k)$, then clicks if relevant.

$$P(\text{click} | k) = P(\text{examine} | k) \cdot P(\text{relevant})$$

**Position decay**:
$$P(\text{examine} | k) = \frac{1}{\log_2(k+1)}$$

---

### Mitigation: Inverse Propensity Weighting

**Idea**: Reweight clicks by inverse examination probability.

$$\text{adjusted\_click}(k) = \frac{\text{click}(k)}{p(k)}$$

**Example**:
```
Position 1: click=1, p(1)=0.8 → adjusted=1/0.8=1.25
Position 5: click=1, p(5)=0.3 → adjusted=1/0.3=3.33
```

**Effect**: Down-weight top positions, up-weight lower positions.

---

### Implementation

```python
def position_bias_weights(max_position=10):
    """Compute position bias weights (1/log2(k+1))"""
    return {k: 1.0 / np.log2(k + 1) for k in range(1, max_position + 1)}

def debiased_evaluation(clicks, positions, weights):
    """
    clicks: List of binary clicks
    positions: List of positions where items shown
    weights: Dict {position: weight}
    """
    adjusted_clicks = [
        click / weights[pos] if click else 0
        for click, pos in zip(clicks, positions)
    ]
    return np.mean(adjusted_clicks)


# Example
clicks = [1, 0, 1, 0, 0]
positions = [1, 2, 3, 4, 5]
weights = position_bias_weights()

biased_ctr = np.mean(clicks)
debiased_ctr = debiased_evaluation(clicks, positions, weights)

print(f"Biased CTR: {biased_ctr:.3f}")
print(f"Debiased CTR: {debiased_ctr:.3f}")
```

---

## Popularity Bias

### Problem

**Observation**: Metrics favor recommending popular items.

**Example**:
- Recommend top-10 popular items → High precision (many users like them)
- But: No personalization, poor long-tail coverage

**Impact**: Systems optimize for popular items, ignore niche preferences.

---

### Popularity-Stratified Evaluation

**Idea**: Report metrics separately for popular vs. niche items.

**Strata**:
- Head (top 20%): Very popular items
- Torso (20-80%): Moderately popular
- Tail (bottom 20%): Niche items

```python
def stratified_metrics(recommended, relevant, item_popularity, strata_thresholds=[0.2, 0.8]):
    """
    item_popularity: dict {item: popularity score (0-1)}
    """
    # Categorize items
    head_items = {i for i, p in item_popularity.items() if p > strata_thresholds[1]}
    tail_items = {i for i, p in item_popularity.items() if p < strata_thresholds[0]}

    # Metrics for each stratum
    head_relevant = set(relevant) & head_items
    tail_relevant = set(relevant) & tail_items

    head_rec = [i for i in recommended if i in head_items]
    tail_rec = [i for i in recommended if i in tail_items]

    metrics = {
        'head_precision': len(set(head_rec) & head_relevant) / len(head_rec) if head_rec else 0,
        'tail_precision': len(set(tail_rec) & tail_relevant) / len(tail_rec) if tail_rec else 0
    }

    return metrics
```

---

### Novelty-Adjusted Metrics

**Idea**: Weight items by inverse popularity (reward recommending niche items).

$$\text{Novelty-NDCG} = \sum_{k=1}^K \frac{(1 - p(i_k)) \cdot \text{rel}_k}{\log_2(k+1)}$$

---

## Selection Bias

### Problem

**Observation**: Only see ratings for items users were shown (by previous system).

**Example**:
- Popular items shown often → many ratings
- Niche items rarely shown → few ratings

**Impact**: Can't evaluate on unseen (user, item) pairs → biased metrics.

---

### Missing Not At Random (MNAR)

**Assumption**: Ratings are **not missing at random** - missingness depends on true rating.

**Example**: Users don't rate movies they dislike (quit watching).

**Solution**: **Unbiased estimators** (inverse propensity scoring).

---

### Propensity Scores

**Propensity**: Probability item was shown to user.

$$p(i | u) = P(\text{item } i \text{ shown to user } u)$$

**Unbiased loss**:
$$\mathcal{L}_{\text{unbiased}} = \frac{1}{N} \sum_{(u,i) \in \text{observed}} \frac{1}{p(i|u)} (r_{ui} - \hat{r}_{ui})^2$$

**Challenge**: Estimating $p(i|u)$ requires logged data (which items were considered).

---

## Feedback Loops

### Problem

**Observation**: Recommendations influence future user behavior → **self-fulfilling prophecy**.

**Example**:
- System recommends popular items
- Users interact with popular items
- Train new model on this data → recommends same popular items
- **Filter bubble**: Users never see diverse content

---

### Echo Chamber Effect

**Mechanism**:
1. System shows sports content (user clicked once)
2. User clicks sports again (no other options)
3. System learns: user loves sports
4. Only shows sports
5. User preferences narrow

**Impact**: Reduced diversity, user dissatisfaction long-term.

---

### Mitigation

**1. Exploration**: Inject random/diverse recommendations (10-20% of slots).

**2. Causal inference**: Estimate counterfactual - "What if we showed item X?"

**3. Online evaluation**: A/B test to break feedback loop.

---

## Metric Gaming

### Problem

**Goodhart's Law**: "When a measure becomes a target, it ceases to be a good measure."

**Example**:
- Optimize for CTR → Show clickbait (high clicks, low satisfaction)
- Optimize for watch time → Autoplay next video (addictive, not desired)

---

### Solution: Multiple Objectives

**Combine metrics**:
$$\text{Score} = \alpha \cdot \text{CTR} + \beta \cdot \text{Satisfaction} - \gamma \cdot \text{Regret}$$

**Satisfaction**: Explicit feedback (thumbs up/down)
**Regret**: User searches for different content after watching

---

## Correlation vs. Causation

### Problem

**Offline metrics** (NDCG, MAP) correlate with online metrics (user engagement), but **causation unclear**.

**Example**:
- Offline NDCG improves 5%
- Online CTR improves 0.5% (not 5%!)

**Reason**: Distribution shift, feedback loops, user behavior changes.

---

### Solution

**Always validate offline gains with online A/B testing**.

**Process**:
1. Offline: NDCG@10 = 0.75 (baseline), 0.78 (new model)
2. A/B test: Deploy to 5% users
3. Measure: CTR, session length, retention
4. If online metrics improve → roll out

---

## Summary

**Key Takeaways**:
1. **Position bias**: Users click top results → inverse propensity weighting
2. **Popularity bias**: Metrics favor popular items → stratify by popularity
3. **Selection bias**: Only observe shown items → propensity scoring
4. **Feedback loops**: Recommendations shape preferences → exploration
5. **Metric gaming**: Optimizing single metric harms others → multi-objective

**Best Practices**:
- **Debias metrics**: Adjust for position and popularity
- **Stratified reporting**: Separate metrics for head/tail items
- **Online validation**: A/B test before deploying
- **Multiple objectives**: Balance relevance, diversity, satisfaction

**Next**: Online A/B testing.

---

## References

1. **Joachims, T., et al. (2017)**. "Unbiased Learning-to-Rank with Biased Feedback". *WSDM*.
   - Position bias correction

2. **Schnabel, T., et al. (2016)**. "Recommendations as Treatments: Debiasing Learning and Evaluation". *ICML*.
   - Selection bias, propensity scoring

3. **Jiang, R., et al. (2019)**. "Degenerate Feedback Loops in Recommender Systems". *AIES*.
   - Feedback loops analysis

4. **Abdollahpouri, H., et al. (2019)**. "Managing Popularity Bias in Recommender Systems with Personalized Re-ranking". *FLAIRS*.
   - Popularity bias mitigation
