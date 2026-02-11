# Week 12: Types of Bias in Recommendations

## Learning Objectives

By the end of this lecture, you will:
- Identify the major types of bias in recommender systems
- Trace bias sources through the data collection, training, and inference pipeline
- Mathematically formulate popularity bias and its long-tail problem
- Quantify position bias through feedback loop analysis
- Understand why missing-not-at-random data breaks standard ML assumptions
- Recognize the societal implications of biased recommendations

---

## Opening Problem: The Feedback Loop Trap

*"Professor, my recommendation model keeps suggesting the same popular items. The more I train, the worse it gets!"*

**The scenario**: You deploy a movie recommender. After three months, you notice:
- Top 1% of movies receive 60% of all recommendations
- New releases never get recommended (no training data)
- User satisfaction scores are declining
- Niche movie lovers are churning

**What went wrong?**

Let me trace through what happened:

```
┌─────────────────────────────────────────────────────────────────┐
│  ROUND 1: Deploy model trained on historical data               │
│  → Popular movies have more ratings → model learns "popular=good"│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  ROUND 2: Model recommends popular movies                       │
│  → Users click on what's shown → more interactions with popular │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  ROUND 3: Retrain on new data                                   │
│  → Even more biased toward popular → niche items invisible      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  ROUND N: Complete concentration                                │
│  → 99% of traffic to 1% of items → filter bubble               │
└─────────────────────────────────────────────────────────────────┘
```

**The fundamental question**: If users only see what you recommend, how do you learn about what you did not recommend?

This is the **exploration-exploitation dilemma** manifesting as **systemic bias**. Today, we will systematically understand every type of bias that can corrupt your recommender.

---

## Bias Taxonomy: From Data to Deployment

### The Recommendation Pipeline

To understand bias, we must first understand where it enters the system.

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECOMMENDATION PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐             │
│  │   DATA     │ → │  TRAINING  │ → │ INFERENCE  │ → Users      │
│  │ COLLECTION │    │            │    │            │     ↓       │
│  └────────────┘    └────────────┘    └────────────┘     ↓       │
│        ↑                                                │       │
│        └────────────────────────────────────────────────┘       │
│                         FEEDBACK LOOP                            │
└─────────────────────────────────────────────────────────────────┘
```

**Bias can enter at every stage:**

| Stage | Bias Type | Mechanism |
|-------|-----------|-----------|
| **Data Collection** | Selection bias | Only observe interactions with shown items |
| **Data Collection** | Position bias | Users click top positions regardless of relevance |
| **Data Collection** | Conformity bias | Ratings influenced by visible averages |
| **Training** | Popularity bias | Model overfits to frequent items |
| **Training** | Demographic bias | Model performs worse for minority groups |
| **Inference** | Exposure bias | Some items/creators never get recommended |
| **Feedback** | All biases amplify | Each round reinforces existing patterns |

**Key insight**: Bias is not a bug - it is an emergent property of how recommendation systems interact with users. The system is optimizing for what it can observe, which is fundamentally different from what users actually want.

---

## Popularity Bias: The Matthew Effect

### The Problem Statement

**The Matthew Effect**: "For to every one who has will more be given, and he will have abundance; but from him who has not, even what he has will be taken away." (Matthew 25:29)

In recommendations: **Rich get richer. Popular items get more popular.**

**Observed pattern** (Netflix, Spotify, YouTube):
- **Head items** (top 1%): Receive 50% of all traffic
- **Tail items** (bottom 80%): Receive 20% of traffic
- The remaining 19% of items receive 30%

*"Professor, but isn't this just reflecting that popular items are genuinely better?"*

**The counterfactual question**: If we gave equal exposure to all items, would the distribution look the same?

**Research finding** (Abdollahpouri et al., 2019): When items are shown equally, the interaction distribution is much flatter. Popularity is partly genuine quality, but largely **exposure-driven**.

---

### Mathematical Formulation

**Why does popularity bias emerge?**

Consider a simple model where recommendations are based on estimated item quality:

$$\hat{r}_i = \frac{\sum_{u} r_{ui}}{n_i}$$

where $n_i$ is the number of ratings for item $i$.

**Problem 1: Variance**

The variance of this estimate is:

$$\text{Var}(\hat{r}_i) = \frac{\sigma^2}{n_i}$$

**Long-tail items** (small $n_i$) have high variance. Even if their true quality is high, their estimated quality is unreliable.

**Problem 2: The self-reinforcing loop**

Let $P(\text{recommend } i)$ be proportional to $\hat{r}_i$.

At time $t$:
- Popular item ($n_i$ large): Low variance, stable estimate, consistent recommendations
- Niche item ($n_j$ small): High variance, unstable estimate, occasionally recommended

Over many rounds:

$$n_i^{(t+1)} = n_i^{(t)} + P(\text{recommend } i) \cdot P(\text{interact})$$

**The rich-get-richer dynamics**:

$$\frac{d n_i}{d t} \propto n_i^\alpha \quad (\alpha > 0)$$

This is a **preferential attachment** process. Solution:

$$n_i(t) \propto n_i(0)^\alpha \cdot t$$

Items that start popular grow polynomially. Items that start unknown remain unknown.

---

### Long-Tail Distribution Derivation

**Observation**: Item popularity follows a power law:

$$P(\text{item has } k \text{ interactions}) \propto k^{-\gamma}$$

where $\gamma \approx 2$ for most platforms.

**Why power law?** The preferential attachment process we derived above generates power-law distributions.

**Proof sketch** (Barabasi-Albert model):
1. New interactions are distributed proportional to existing interactions
2. Probability item $i$ receives next interaction: $p_i = \frac{n_i}{\sum_j n_j}$
3. Solving the master equation yields: $P(k) \propto k^{-3}$

**Implication**: No matter how long you train, the bottom 80% of items will have insufficient data for reliable recommendations.

---

### Measurement: Gini Coefficient

**Gini coefficient**: Measures inequality in item exposure.

$$G = \frac{\sum_{i=1}^n \sum_{j=1}^n |x_i - x_j|}{2n^2 \bar{x}}$$

where $x_i$ is the exposure (number of recommendations) of item $i$.

**Interpretation**:
- $G = 0$: Perfect equality (all items shown equally)
- $G = 1$: Maximal inequality (one item gets all recommendations)

**Typical values**:
- Random recommendations: $G \approx 0$
- Collaborative filtering: $G \approx 0.7$
- Popularity-based: $G \approx 0.9$

```python
def gini_coefficient(exposures):
    """
    Compute Gini coefficient for item exposures.

    exposures: Array of exposure counts per item
    Returns: Gini coefficient (0 = equal, 1 = maximal inequality)
    """
    n = len(exposures)
    exposures = np.sort(exposures)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * exposures)) / (n * np.sum(exposures)) - (n + 1) / n


# Example: Compare recommender fairness
import numpy as np

# Scenario 1: Equal exposure
equal_exposure = np.ones(1000) * 100
print(f"Equal exposure Gini: {gini_coefficient(equal_exposure):.3f}")  # ~0

# Scenario 2: Power-law distribution (typical RecSys)
power_law = np.random.pareto(2, 1000) * 100
print(f"Power-law Gini: {gini_coefficient(power_law):.3f}")  # ~0.7-0.8
```

---

## Position Bias: The Slot Machine Effect

### The Problem

*"Professor, I noticed users always click the first result. Is that because my ranking is perfect?"*

**Position bias**: Users are more likely to examine and click items shown at top positions, **regardless of relevance**.

**The examination hypothesis**:

$$P(\text{click} | \text{item } i, \text{position } k) = P(\text{examine} | k) \cdot P(\text{relevant} | i)$$

**Typical examination probabilities**:

| Position | Examination Probability |
|----------|------------------------|
| 1 | 0.80 |
| 2 | 0.60 |
| 3 | 0.45 |
| 4 | 0.35 |
| 5 | 0.25 |
| 6+ | < 0.20 |

---

### Numerical Example: 5-Round Feedback Loop

**Setup**: Two items A and B, both with true relevance 0.5.

**Position bias model**:
- Position 1: Examination probability = 0.50
- Position 2: Examination probability = 0.25

**Round 1**: Random ranking (A at position 1, B at position 2)

| Item | Position | P(examine) | P(relevant) | P(click) | Expected clicks (1000 users) |
|------|----------|------------|-------------|----------|------------------------------|
| A | 1 | 0.50 | 0.50 | 0.25 | 250 |
| B | 2 | 0.25 | 0.50 | 0.125 | 125 |

**After Round 1**: A has 2x the clicks. Model learns: "A is better than B."

**Round 2**: Model ranks A at position 1 (because more clicks)

| Item | Position | P(click) | Expected clicks |
|------|----------|----------|-----------------|
| A | 1 | 0.25 | 250 |
| B | 2 | 0.125 | 125 |

**Cumulative after Round 2**: A=500, B=250

**Round 3-5**: Pattern reinforces.

| Round | A Cumulative | B Cumulative | A:B Ratio |
|-------|--------------|--------------|-----------|
| 1 | 250 | 125 | 2.0 |
| 2 | 500 | 250 | 2.0 |
| 3 | 750 | 375 | 2.0 |
| 4 | 1000 | 500 | 2.0 |
| 5 | 1250 | 625 | 2.0 |

**Conclusion**: Despite **identical true relevance**, item A has **2x the training signal** purely due to position.

**The vicious cycle**:
```
A shown first → A clicked more → model thinks A better → A shown first → ...
```

*"So professor, my model is learning position preference, not item preference?"*

**Exactly.** Without debiasing, your model learns a mixture of:
- True item quality (what you want)
- Position effect (confounding noise)
- Popularity effect (more confounding noise)

---

### The Counterfactual Question

**Socratic question**: How would you know if item B is actually better than A?

**Answer**: You need **randomized exposure** or **position-aware models**.

**Solution 1: Inverse Propensity Weighting**

$$\text{adjusted\_click}_k = \frac{\text{click}_k}{P(\text{examine} | k)}$$

**Solution 2: Position-aware models** (YouTube, Google)

Learn separate parameters for position effect and item quality:

$$P(\text{click}) = \sigma(\underbrace{w_{\text{pos}}^T x_{\text{pos}}}_{\text{position effect}} + \underbrace{w_{\text{item}}^T x_{\text{item}}}_{\text{item quality}})$$

---

## Selection Bias: The Missing Data Problem

### Missing Not At Random (MNAR)

**Standard ML assumption**: Data is Missing At Random (MAR) or Missing Completely At Random (MCAR).

**Reality in recommendations**: Data is **Missing Not At Random (MNAR)**.

**Why?** Users only rate items they:
1. Were shown (previous system's bias)
2. Chose to interact with (self-selection)
3. Felt strongly about (extreme ratings bias)

---

### How MNAR Breaks Standard Assumptions

**Standard approach**: Train on observed (user, item, rating) triples.

$$\hat{\theta} = \arg\min_\theta \sum_{(u,i) \in \text{observed}} (r_{ui} - f_\theta(u,i))^2$$

**Implicit assumption**: Observed data is representative of all (user, item) pairs.

**MNAR violation**: Observed data is systematically different.

**Example**: Movie rating dataset
- Users rate movies they watched
- Users watch movies they expect to like
- Observed ratings skew positive

| True Rating | P(Rating Observed) |
|-------------|-------------------|
| 5 stars | 0.60 |
| 4 stars | 0.50 |
| 3 stars | 0.30 |
| 2 stars | 0.15 |
| 1 star | 0.10 |

**Result**: Model trained on observed data will:
- Overestimate average ratings
- Fail to predict low ratings accurately
- Recommend items users will dislike

---

### Mathematical Formulation

Let $O_{ui}$ indicate whether rating $(u,i)$ is observed.

**True objective** (what we want to minimize):

$$\mathcal{L}_{\text{true}} = \frac{1}{|U| \cdot |I|} \sum_{u,i} (r_{ui} - \hat{r}_{ui})^2$$

**Observed objective** (what we actually minimize):

$$\mathcal{L}_{\text{observed}} = \frac{1}{|O|} \sum_{(u,i) \in O} (r_{ui} - \hat{r}_{ui})^2$$

**Bias**:

$$\mathbb{E}[\mathcal{L}_{\text{observed}}] \neq \mathcal{L}_{\text{true}}$$

unless $P(O_{ui} = 1)$ is independent of $r_{ui}$ (MAR assumption).

---

### Propensity Score Correction

**Solution**: Weight by inverse propensity (probability of observation).

$$\mathcal{L}_{\text{unbiased}} = \sum_{(u,i) \in O} \frac{1}{p(O_{ui}=1)} \cdot (r_{ui} - \hat{r}_{ui})^2$$

**Intuition**: Up-weight rare observations, down-weight common ones.

**Challenge**: Estimating $p(O_{ui}=1)$ is itself a chicken-and-egg problem.

**Practical approaches**:
1. **Naive Bayes**: $p(O_{ui}=1) \approx p_u \cdot p_i$ (user activity times item popularity)
2. **Logistic regression**: Model observation probability from features
3. **Self-normalized estimator**: Use ratio estimator for stability

```python
def ips_loss(predictions, ratings, propensities):
    """
    Inverse Propensity Scoring loss for debiased training.

    predictions: Model predictions for observed pairs
    ratings: True ratings for observed pairs
    propensities: P(observed) for each pair
    """
    weights = 1.0 / np.clip(propensities, 0.01, 1.0)  # Clip for stability
    squared_errors = (predictions - ratings) ** 2
    return np.mean(weights * squared_errors)
```

---

## Exposure Bias: The Unobservable Problem

### The Fundamental Challenge

**Exposure bias**: Items never shown can never be evaluated.

*"Professor, how do I know if my model would have recommended item X?"*

**You cannot.** This is a counterfactual question. You only observe:
- What you showed
- How users responded

You never observe:
- What you did not show
- How users would have responded to items they never saw

---

### Formal Derivation

Let $E_i^{(t)}$ be the exposure of item $i$ at time $t$ (probability of being shown).

**The learning update**:

$$\theta_i^{(t+1)} = \theta_i^{(t)} + \eta \cdot E_i^{(t)} \cdot \nabla_{\theta_i} \mathcal{L}$$

**Key observation**: If $E_i^{(t)} = 0$, then:

$$\theta_i^{(t+1)} = \theta_i^{(t)}$$

**The item never learns.** Its parameters remain at initialization.

**The vicious cycle**:

1. Item $i$ has poor initial estimate (random initialization)
2. Poor estimate leads to low predicted relevance
3. Low predicted relevance leads to no exposure: $E_i = 0$
4. No exposure means no gradient: $\nabla_{\theta_i} = 0$
5. No gradient means no learning: $\theta_i$ unchanged
6. Return to step 2

**Mathematically**:

$$\lim_{t \to \infty} E_i^{(t)} = 0 \quad \text{for items with } E_i^{(0)} < \epsilon$$

Items below the initial exposure threshold are **permanently invisible**.

---

### Creator Fairness Perspective

**Example** (YouTube, TikTok):
- Verified channels: 80% of recommendations
- Small creators: 20% of recommendations
- Even if content quality is similar

**Impact**:
- New creators cannot grow
- Platform becomes concentrated
- User diversity decreases
- Small creators leave

**The Socratic question**: If you never show a creator's content, how do you know it's not good?

**Answer**: You don't. Your model is making a **Type II error** - failing to identify good content because it never tests the hypothesis.

---

## Conformity Bias: Social Influence

### The Herding Effect

**Observation**: Users rate items similar to existing ratings.

**Experiment** (Muchnik et al., 2013, Science):
- Randomly upvoted some comments
- Result: 25% increase in final rating
- "Social proof" influences judgment

**In recommendations**:
- Movie shows 4.5 stars average
- New user rates 4 (influenced by displayed average)
- Same movie with 2.5 average
- Same user might rate 3

---

### Mechanism

$$r_{ui}^{\text{observed}} = r_{ui}^{\text{true}} + \alpha \cdot (\bar{r}_i - r_{ui}^{\text{true}})$$

where $\alpha$ is the conformity coefficient (typically 0.2-0.4).

**Impact**:
- Ratings converge toward mean
- Signal diversity decreases
- Hard to distinguish good items from mediocre-with-social-proof

---

## Demographic Bias: Fairness Across Groups

### The Problem

**Observation**: Recommender systems perform differently for different demographic groups.

| User Group | NDCG@10 |
|------------|---------|
| Age 20-30 | 0.75 |
| Age 30-50 | 0.72 |
| Age 50+ | 0.58 |

**Why?**
1. **Data imbalance**: Fewer interactions from older users
2. **Feature bias**: Features correlate with protected attributes
3. **Item pool bias**: Catalog skews toward majority preferences

---

### Disparate Impact

**Measurement**:

$$\text{DI} = \frac{P(\hat{y}=1 | A=0)}{P(\hat{y}=1 | A=1)}$$

where $A$ is the protected attribute (e.g., age group, gender).

**80% rule**: Fair if $0.8 \leq \text{DI} \leq 1.25$.

```python
def disparate_impact(recommendations, protected_attribute):
    """
    Compute disparate impact ratio.

    recommendations: Binary array (1=recommended, 0=not)
    protected_attribute: Binary array (0=minority, 1=majority)

    Returns: DI ratio (1.0 = perfect fairness)
    """
    minority_rate = recommendations[protected_attribute == 0].mean()
    majority_rate = recommendations[protected_attribute == 1].mean()

    return minority_rate / majority_rate if majority_rate > 0 else 0
```

---

## What Can Go Wrong: Real-World Consequences

### Rich-Get-Richer Dynamics

**Platform concentration**:
- 1% of creators receive 90% of engagement
- New creators cannot break through
- Quality content goes unseen

**Economic impact**:
- Winner-take-all markets
- Reduced innovation (no incentive if you can't get discovered)
- Platform dependency (creators have no alternative)

---

### Filter Bubbles

**Definition**: Users only see content that confirms existing preferences.

**Mechanism**:
1. User clicks on political news from one perspective
2. System recommends more from same perspective
3. User sees increasingly narrow worldview
4. User believes this represents reality

**Research finding** (Pariser, 2011): Two users searching the same term on Google get completely different results.

**Democratic implications**:
- Polarization
- Epistemic fragmentation
- Difficulty finding common ground

---

### Demographic Disparities

**Example 1**: Job recommendations
- System trained on historical hiring data
- Historical data reflects past discrimination
- System perpetuates: fewer women in tech roles recommended
- **Legal issue**: Potential EEOC violation

**Example 2**: Credit/loan recommendations
- Historical data: minorities had fewer loans approved
- System learns: minorities are higher risk
- System perpetuates: fewer loan offers to minorities
- **Legal issue**: Fair Housing Act violation

**Example 3**: Healthcare recommendations
- System trained on healthcare spending data
- Black patients had less spent on them (due to access issues)
- System learns: Black patients need fewer interventions
- **Real harm**: Documented cases of algorithmic discrimination in healthcare

---

## Socratic Reflection

*"If users only see what you recommend, how do you learn about what you didn't recommend?"*

**The core insight**: Recommendation systems are not passive observers - they actively shape the data they will be trained on.

**This creates**:
1. **Feedback loops**: Current recommendations influence future data
2. **Counterfactual blindness**: Cannot evaluate unchosen alternatives
3. **Exploration-exploitation tradeoff**: Must balance learning vs. performance

**Solutions require**:
- **Exploration**: Occasionally show items outside the model's preferences
- **Causal reasoning**: Estimate what would have happened under different policies
- **Fairness constraints**: Explicitly enforce exposure for all groups/items
- **Online learning**: Continuously adapt instead of batch retraining

*"So professor, is it impossible to build a fair recommender?"*

**Not impossible, but it requires intentionality.** Default ML training optimizes for observed outcomes, which bakes in historical biases. Fair recommendations require:
1. Recognizing bias exists
2. Measuring it explicitly
3. Designing interventions
4. Monitoring ongoing impact

---

## Summary

**Key Types of Bias**:

| Bias Type | Source | Consequence |
|-----------|--------|-------------|
| **Popularity bias** | Matthew effect, preferential attachment | Long-tail items invisible |
| **Position bias** | User examination patterns | Model learns position, not quality |
| **Selection bias** | MNAR data | Model learns from biased sample |
| **Exposure bias** | Items never shown | Cannot learn about unexposed items |
| **Conformity bias** | Social influence | Ratings converge, signal loss |
| **Demographic bias** | Data imbalance, historical discrimination | Unfair treatment by group |

**Key Formulas**:

$$\text{Gini coefficient: } G = \frac{\sum_{i=1}^n \sum_{j=1}^n |x_i - x_j|}{2n^2 \bar{x}}$$

$$\text{IPS correction: } \mathcal{L} = \sum_{(u,i) \in D} \frac{1}{p(i|u)} \cdot \text{loss}(r_{ui}, \hat{r}_{ui})$$

$$\text{Disparate Impact: } \text{DI} = \frac{P(\hat{y}=1 | A=0)}{P(\hat{y}=1 | A=1)}$$

**Key Insights**:
1. **Bias is emergent**: It arises from how systems interact with users
2. **Feedback loops amplify**: Each training round reinforces existing biases
3. **Counterfactuals are unobservable**: Cannot evaluate what was never shown
4. **Fairness requires intentionality**: Default optimization perpetuates bias

**Next**: Debiasing techniques - how to mitigate these biases in practice.

---

## References

1. **Abdollahpouri, H., et al. (2019)**. "Managing Popularity Bias in Recommender Systems". *User Modeling and User-Adapted Interaction*.
   - Popularity bias measurement and mitigation

2. **Schnabel, T., et al. (2016)**. "Recommendations as Treatments". *ICML*.
   - Selection bias, propensity scoring, causal framework

3. **Mehrotra, R., et al. (2018)**. "Towards a Fair Marketplace: Counterfactual Evaluation of the trade-off between Relevance, Fairness & Satisfaction". *CIKM*.
   - Fairness-relevance tradeoffs

4. **Joachims, T., et al. (2017)**. "Unbiased Learning-to-Rank with Biased Feedback". *WSDM*.
   - Position bias in learning-to-rank

5. **Pariser, E. (2011)**. *The Filter Bubble: What the Internet Is Hiding from You*. Penguin.
   - Filter bubbles and personalization

6. **Barabasi, A.-L., & Albert, R. (1999)**. "Emergence of Scaling in Random Networks". *Science*.
   - Preferential attachment and power laws

7. **Muchnik, L., et al. (2013)**. "Social Influence Bias: A Randomized Experiment". *Science*.
   - Conformity bias in online ratings
