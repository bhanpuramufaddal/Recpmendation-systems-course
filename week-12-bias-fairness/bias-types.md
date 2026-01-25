# Week 12: Types of Bias in Recommendations

## Overview

**Bias** in recommender systems leads to unfair outcomes, filter bubbles, and reduced diversity.

**Types**:
1. **Popularity bias**: Over-recommend popular items
2. **Selection bias**: Training data from biased system
3. **Position bias**: Users click top results
4. **Conformity bias**: Users follow crowd ratings
5. **Demographic bias**: Unfair treatment by user demographics

---

## Popularity Bias

### Mechanism

**Matthew Effect**: "Rich get richer" - popular items recommended more → become more popular.

**Impact**:
- **Head items**: Dominate recommendations (top 1% items get 50% traffic)
- **Tail items**: Rarely recommended (99% items get 50% traffic)

**Example** (Netflix):
- Popular movie shown 1000x/day → 100 clicks → more popular
- Niche film shown 10x/day → 1 click → remains niche

---

### Measurement

**Gini coefficient**: Inequality in item exposure

$$G = \frac{\sum_{i=1}^n \sum_{j=1}^n |x_i - x_j|}{2n^2 \bar{x}}$$

where $x_i$ = exposure of item $i$.

- $G = 0$: Perfect equality
- $G = 1$: Maximal inequality

```python
def gini_coefficient(exposures):
    """
    exposures: Array of item exposure counts
    """
    n = len(exposures)
    exposures = np.sort(exposures)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * exposures)) / (n * np.sum(exposures)) - (n + 1) / n
```

---

## Selection Bias

### Problem

**Observation**: Training data from previous system → biased toward what old system recommended.

**Example**:
- Old system recommended action movies
- Dataset: 80% action ratings, 20% other
- New model learns: "users like action" (but haven't tried others!)

---

### MNAR (Missing Not At Random)

**Assumption**: Ratings missing because users didn't like item (quit watching).

**Impact**: Training data overrepresents positives.

**Example**: Movie ratings
- Users rate movies they liked (4-5 stars) → abundant
- Users don't rate bad movies (quit) → sparse

---

### Solutions

**1. Inverse Propensity Scoring**:
$$\mathcal{L} = \sum_{(u,i) \in D} \frac{1}{p(i|u)} \cdot \text{loss}(r_{ui}, \hat{r}_{ui})$$

**2. Doubly Robust**:
- Combines propensity weighting + imputation
- Unbiased even if one component wrong

---

## Position Bias

**Already covered in evaluation-challenges.md**

**Summary**: Users click top results → overestimate relevance of highly-ranked items.

**Solution**: Inverse propensity weighting by position.

---

## Conformity Bias

### Problem

**Observation**: Users rate items similar to existing ratings (social pressure).

**Example**:
- Movie has 4.5⭐ average → new user rates 4⭐ (influenced by average)
- Same movie with 2.5⭐ average → user rates 2⭐

**Impact**: Ratings converge, reducing signal diversity.

---

### Herding Effect

**Mechanism**: Early ratings disproportionately influence later ratings.

**Example**: First 10 reviews are 5⭐ → subsequent reviews skew positive.

**Mitigation**: Hide average rating initially, collect unbiased early ratings.

---

## Demographic Bias

### Problem

**Observation**: System performs differently for different demographic groups.

**Example**:
- Accuracy for young users (20-30): NDCG=0.75
- Accuracy for older users (60+): NDCG=0.50

**Causes**:
1. **Data imbalance**: Fewer interactions from minority groups
2. **Feature bias**: Features correlated with protected attributes
3. **Model bias**: Algorithm favors majority patterns

---

### Protected Attributes

**Sensitive attributes**:
- Age, gender, race, religion, sexual orientation, disability

**Fair ML requirement**: Don't discriminate based on protected attributes.

**Legal**: GDPR (EU), Fair Housing Act (US)

---

### Measurement

**Disparate Impact**:

$$\text{DI} = \frac{P(\hat{y}=1 | A=0)}{P(\hat{y}=1 | A=1)}$$

where $A$ = protected attribute (e.g., gender).

**Fair if**: $0.8 \leq \text{DI} \leq 1.25$ (80% rule).

```python
def disparate_impact(predictions, protected_attribute):
    """
    predictions: Binary predictions
    protected_attribute: Binary (0=minority, 1=majority)
    """
    group_0_rate = predictions[protected_attribute == 0].mean()
    group_1_rate = predictions[protected_attribute == 1].mean()
    return group_0_rate / group_1_rate if group_1_rate > 0 else 0
```

---

## Exposure Bias

### Problem

**Observation**: Some items/creators get disproportionate exposure.

**Example** (YouTube):
- Verified channels: 80% of recommendations
- Small creators: 20% of recommendations
- Even if both have similar quality

**Impact**: Platform concentration, reduced creator diversity.

---

## Summary

**Key Takeaways**:
1. **Popularity bias**: Matthew effect, Gini coefficient
2. **Selection bias**: MNAR, propensity scoring
3. **Conformity bias**: Herding, social influence
4. **Demographic bias**: Disparate impact, protected attributes
5. **Exposure bias**: Creator fairness

**Next**: Debiasing techniques.

---

## References

1. **Abdollahpouri, H., et al. (2019)**. "Managing Popularity Bias in Recommender Systems". *User Modeling and User-Adapted Interaction*.
2. **Schnabel, T., et al. (2016)**. "Recommendations as Treatments". *ICML*.
3. **Mehrotra, R., et al. (2018)**. "Towards a Fair Marketplace: Counterfactual Evaluation of the trade-off between Relevance, Fairness & Satisfaction". *CIKM*.
