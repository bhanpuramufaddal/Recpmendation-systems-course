# Week 12: Bias, Fairness, and Ethics - Practice Problems

## Overview
Understand popularity bias, selection bias, debiasing techniques (IPS), fairness frameworks, and ethical considerations in recommendation systems.

---

## Problem 1: Popularity Bias
**Difficulty:** Easy

**Data:** 80% of interactions are with top-20% popular items (Pareto principle)

**Questions:**
1. Why does collaborative filtering amplify popularity bias?
2. Calculate the Gini coefficient for item popularity distribution
3. Propose three debiasing strategies
4. Trade-off: Reducing bias may hurt accuracy. How do you balance?

**Learning Outcomes:** Quantify popularity bias, design debiasing strategies, balance fairness and accuracy

---

## Problem 2: Inverse Propensity Scoring (IPS)
**Difficulty:** Hard

**Problem:** Training data has position bias (top items clicked more)

**Click data:**
- Position 1: 1000 impressions, 100 clicks (10%)
- Position 10: 1000 impressions, 20 clicks (2%)

**IPS reweighting:** $L_{IPS} = \frac{1}{N} \sum \frac{1}{p(observe)} \cdot loss$

**Tasks:**
1. Estimate position propensities p(observe | position)
2. Apply IPS to reweight training examples
3. What happens if propensity estimates are wrong?
4. Implement doubly robust estimator (IPS + prediction)

**Learning Outcomes:** Implement IPS, correct observational biases, understand causal inference

---

## Problem 3: Demographic Parity
**Difficulty:** Medium

**Scenario:** Movie recommendations for different age groups

**Metric:** P(recommended | age=young) vs. P(recommended | age=old)

**Data:**
- Young users: 60% get action movies recommended
- Old users: 30% get action movies recommended

**Questions:**
1. Is this demographic parity violation?
2. Should we enforce parity? What are the trade-offs?
3. Design a fair recommendation policy
4. What if young users genuinely prefer action more?

**Learning Outcomes:** Measure demographic parity, design fair policies, understand trade-offs

---

## Problem 4: Provider Fairness
**Difficulty:** Hard

**Scenario:** Music streaming (Spotify-like)

**Problem:** Popular artists get 80% of plays, long-tail artists starve

**Fairness goals:**
1. **Exposure fairness:** Each artist gets plays proportional to quality
2. **Diversity:** Users exposed to variety
3. **Revenue:** Platform maximizes subscriptions

**Tasks:**
1. Design a fair exposure allocation algorithm
2. Formulate as multi-stakeholder optimization
3. Simulate impact on artists and users
4. Propose metrics to measure provider fairness

**Learning Outcomes:** Balance stakeholder interests, design fair systems, measure multi-sided fairness

---

## Problem 5: Filter Bubbles
**Difficulty:** Medium

**Observation:** User watches only action movies → system only recommends action → user never discovers documentaries

**Questions:**
1. How do you detect filter bubbles?
2. Design an intervention to increase diversity
3. Trade-off: Diversity may reduce short-term engagement
4. How do you measure "healthy" recommendation diversity?

**Learning Outcomes:** Detect filter bubbles, design interventions, balance diversity and relevance

---

## Programming Exercises

### Exercise 1: Measure Popularity Bias

```python
def gini_coefficient(item_counts):
    # Measure inequality in item exposure
    sorted_counts = np.sort(item_counts)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n

# Measure before/after debiasing
item_exposure_before = count_recommendations(model_biased)
item_exposure_after = count_recommendations(model_debiased)

print(f"Gini before: {gini_coefficient(item_exposure_before):.3f}")
print(f"Gini after: {gini_coefficient(item_exposure_after):.3f}")
```

---

### Exercise 2: Implement IPS Reweighting

```python
def estimate_propensities(click_data):
    # Estimate P(click | position, relevance)
    position_clicks = click_data.groupby('position')['click'].mean()
    return position_clicks

def ips_loss(predictions, targets, positions, propensities):
    weights = 1.0 / propensities[positions]
    weighted_loss = weights * (predictions - targets) ** 2
    return weighted_loss.mean()

# Train with IPS
propensities = estimate_propensities(training_data)
for batch in dataloader:
    predictions = model(batch['features'])
    loss = ips_loss(predictions, batch['targets'], batch['positions'], propensities)
    loss.backward()
    optimizer.step()
```

---

### Exercise 3: Fair Ranking Algorithm

```python
def fair_ranking(scores, item_groups, alpha=0.5):
    # Balance score-based ranking with group fairness
    ranked = []
    remaining = list(range(len(scores)))

    for position in range(len(scores)):
        # Determine target group for this position
        target_group = position % len(item_groups)

        # Among remaining items from target group, pick highest score
        group_items = [i for i in remaining if item_groups[i] == target_group]

        if group_items:
            best = max(group_items, key=lambda i: scores[i])
        else:
            best = max(remaining, key=lambda i: scores[i])

        ranked.append(best)
        remaining.remove(best)

    return ranked
```

---

### Exercise 4: Diversity-Aware Reranking

```python
def mmr_reranking(candidates, scores, diversity_weight=0.3):
    # Maximal Marginal Relevance
    selected = []
    remaining = set(candidates)

    # Select first item (highest score)
    first = max(remaining, key=lambda x: scores[x])
    selected.append(first)
    remaining.remove(first)

    while remaining:
        mmr_scores = {}
        for item in remaining:
            relevance = scores[item]
            max_similarity = max([similarity(item, s) for s in selected])
            mmr_scores[item] = (1 - diversity_weight) * relevance - diversity_weight * max_similarity

        next_item = max(mmr_scores, key=mmr_scores.get)
        selected.append(next_item)
        remaining.remove(next_item)

    return selected
```

---

## Discussion Questions

1. **Ethics:** Is it ethical to explore (show suboptimal recs) to reduce bias?
2. **Trade-offs:** Fairness often hurts accuracy. How much accuracy loss is acceptable?
3. **Who Defines Fair?** Different stakeholders have different fairness notions. How to reconcile?
4. **Measurement:** How do you measure if your system is fair?
5. **Regulation:** GDPR, AI Act require explainability and fairness. How to comply?
6. **Unintended Consequences:** Debiasing can create new biases. Example?

---

## References
1. Schnabel, T., et al. (2016). "Recommendations as treatments: Debiasing learning and evaluation". ICML.
2. Steck, H. (2018). "Calibrated recommendations". RecSys.
3. Mehrotra, R., et al. (2018). "Towards a fair marketplace: Counterfactual evaluation of the trade-off between relevance, fairness and satisfaction in recommendation systems". CIKM.

---

*Return to [Week 12 Main Page](README.md)*
