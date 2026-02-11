# Week 12: Debiasing Techniques

## Learning Objectives

By the end of this section, you will:
- Understand why training on logged data produces biased recommendations
- Derive Inverse Propensity Scoring (IPS) from first principles
- Implement variance reduction techniques (Capped IPS, SNIPS)
- Master doubly robust estimation and understand why it's "doubly" robust
- Identify failure modes and pitfalls when applying debiasing in practice

---

## The Opening Problem: Why Does Training on Logged Data Give Biased Recommendations?

*"Let me start with a question that every recommendation system practitioner eventually faces..."*

### The Exposure Bias Paradox

**The deceptively simple question**: You have millions of user ratings. You train a collaborative filtering model. Why doesn't it work well?

*"The answer lies in a subtle but devastating bias: you only observe ratings for items you showed."*

**Consider this scenario**:

You run a movie streaming service with 10,000 movies. Your current recommendation system (the "logging policy") shows users what it thinks they'll like. Over 6 months, you collect 5 million ratings.

**Pause and think**: What's the problem with this data?

*"Here's the catch: your model can only learn from what users saw. But what about the 80% of your catalog that most users never encountered?"*

### Numerical Reality: True Preferences vs. Observed Data

Let's make this concrete with a worked example.

**Setup**: 1,000 users, 100 movies

| Metric | Value |
|--------|-------|
| Total possible ratings | 1,000 x 100 = 100,000 |
| Actual ratings collected | 8,000 |
| Coverage | 8% |

**But the coverage isn't uniform!**

| Movie Popularity Tier | Movies | Ratings Received | Coverage |
|----------------------|--------|------------------|----------|
| Top 10 (popular) | 10 | 4,000 | 40% of users rated each |
| Middle 40 | 40 | 3,500 | ~9% of users rated each |
| Long tail 50 | 50 | 500 | ~1% of users rated each |

*"The popular movies get 40x more coverage than long-tail movies. Now here's the key insight..."*

### How the Model Learns to Recommend Already-Popular Items

**Step-by-step propagation of bias**:

**Round 1 (Data Collection)**:
- Logging policy shows popular movies more often
- Users rate what they see
- Popular movies get more ratings

**Round 2 (Model Training)**:
- Model fits to observed data
- More data for popular items = better predictions for popular items
- Less data for long-tail = poor/uncertain predictions

**Round 3 (Recommendation)**:
- Model recommends items it's confident about
- Confident about popular items (lots of data)
- Uncertain about long-tail (little data)

**Round 4 (The Feedback Loop)**:
- Model recommends popular items
- Users rate popular items
- *Even more* data for popular items
- Loop continues...

*"Can you see how popular items dominate? The model never gets a chance to learn that users might actually prefer long-tail items if they were shown them!"*

### The Mathematical Formulation

**What we want to estimate**:
$$\theta^* = \arg\min_\theta \mathbb{E}_{(u,i) \sim \text{Uniform}}[\text{loss}(r_{ui}, \hat{r}_{ui}(\theta))]$$

where the expectation is over ALL user-item pairs uniformly.

**What we actually estimate**:
$$\hat{\theta} = \arg\min_\theta \sum_{(u,i) \in D} \text{loss}(r_{ui}, \hat{r}_{ui}(\theta))$$

where $D$ is our observed data (which is NOT uniform).

**The gap**: Our optimization target is biased toward frequently-observed (popular) items!

---

## Inverse Propensity Scoring (IPS) from First Principles

*"Now that we understand the problem, let's derive the solution. This is one of the most elegant ideas in causal inference."*

### What is Propensity?

**Definition**: Propensity is the probability that an item was shown to a user.

$$p(i | u) = P(\text{item } i \text{ shown to user } u)$$

*"Think of it as asking: 'Given our logging policy, how likely was it that this particular user saw this particular item?'"*

**Examples of propensities**:

| Item | Propensity $p(i|u)$ | Interpretation |
|------|---------------------|----------------|
| Avengers (popular) | 0.80 | Shown to 80% of users |
| Indie film | 0.05 | Shown to only 5% of users |
| New release | 0.30 | Moderate exposure |

**Key insight**: Propensities depend on the logging policy (your previous recommendation system).

---

### Why Reweight? Correcting for Sampling Bias

*"Here's the intuition: if an indie film lover rated that 5% exposure film, their opinion is more valuable for understanding the general population's preferences."*

**Analogy**: Political polling

Imagine you're polling voters, but your polling method reaches:
- 80% urban voters
- 20% rural voters

But the actual population is:
- 50% urban
- 50% rural

**Solution**: Reweight responses so each rural response counts 2.5x more.

$$\text{Weight}_{\text{rural}} = \frac{0.50}{0.20} = 2.5$$

**Same principle for recommendations**:

$$w_{ui} = \frac{1}{p(i | u)}$$

A rating for an item with 5% exposure gets weight $\frac{1}{0.05} = 20$, while a rating for an 80% exposure item gets weight $\frac{1}{0.80} = 1.25$.

---

### The IPS Estimator: Full Derivation with Expectation Proof

*"Let's prove that IPS actually gives us an unbiased estimate. This is beautiful mathematics."*

**Setup**:
- True expected loss: $L = \mathbb{E}_{(u,i) \sim \text{Uniform}}[\ell(u,i)]$
- Observed data $D$ sampled according to propensities $p(i|u)$

**The IPS estimator**:
$$\hat{L}_{\text{IPS}} = \frac{1}{|D|} \sum_{(u,i) \in D} \frac{\ell(u,i)}{p(i|u)}$$

**Proof of unbiasedness**:

**Step 1**: Write the expectation over observed data

$$\mathbb{E}_{D}[\hat{L}_{\text{IPS}}] = \mathbb{E}_{D}\left[\frac{1}{|D|} \sum_{(u,i) \in D} \frac{\ell(u,i)}{p(i|u)}\right]$$

**Step 2**: For a single observation $(u,i)$ sampled with probability $p(i|u)$

$$\mathbb{E}\left[\frac{\ell(u,i)}{p(i|u)}\right] = \sum_{i} p(i|u) \cdot \frac{\ell(u,i)}{p(i|u)} = \sum_{i} \ell(u,i)$$

*"The propensity in the numerator cancels with the propensity weighting in the denominator!"*

**Step 3**: Averaging over all user-item pairs

$$\mathbb{E}[\hat{L}_{\text{IPS}}] = \frac{1}{|U| \cdot |I|} \sum_{u} \sum_{i} \ell(u,i) = L$$

**Conclusion**: $\hat{L}_{\text{IPS}}$ is an unbiased estimator of the true loss!

---

### IPS Implementation

```python
def ips_loss(predictions, targets, propensities, epsilon=1e-8):
    """
    Compute IPS-weighted loss.

    Args:
        predictions: Model predictions for observed (u, i) pairs
        targets: True ratings for observed (u, i) pairs
        propensities: p(item shown | user) for each observation
        epsilon: Small constant for numerical stability

    Returns:
        IPS-weighted mean squared error
    """
    # Compute IPS weights
    weights = 1.0 / (propensities + epsilon)

    # Weighted squared error
    squared_errors = (predictions - targets) ** 2
    weighted_loss = weights * squared_errors

    return weighted_loss.mean()
```

---

## The Variance Problem: Why IPS Can Blow Up

*"IPS is unbiased, which sounds great. But there's a catch that will bite you in practice..."*

### Numerical Example: When Propensities Are Small

**Scenario**: 5 users rate 10 items, but with highly non-uniform exposure.

| Observation | User | Item | Rating | Propensity $p$ | IPS Weight $1/p$ |
|-------------|------|------|--------|----------------|------------------|
| 1 | Alice | Avengers | 4 | 0.80 | 1.25 |
| 2 | Bob | Avengers | 5 | 0.80 | 1.25 |
| 3 | Carol | Spider-Man | 3 | 0.60 | 1.67 |
| 4 | Dave | Indie Film | 5 | 0.02 | **50.0** |
| 5 | Eve | Art House | 4 | 0.01 | **100.0** |

*"Stop and look at those weights. Eve's single rating has weight 100, while Alice's has weight 1.25. Is this reasonable?"*

**Computing the IPS-weighted loss**:

Standard MSE loss (assuming predictions all = 4):
$$L_{\text{standard}} = \frac{1}{5}[(4-4)^2 + (5-4)^2 + (3-4)^2 + (5-4)^2 + (4-4)^2] = \frac{3}{5} = 0.6$$

IPS-weighted loss:
$$L_{\text{IPS}} = \frac{1}{5}[1.25 \cdot 0 + 1.25 \cdot 1 + 1.67 \cdot 1 + 50 \cdot 1 + 100 \cdot 0]$$
$$L_{\text{IPS}} = \frac{1}{5}[0 + 1.25 + 1.67 + 50 + 0] = \frac{52.92}{5} = 10.58$$

**The problem**: Dave's single rating dominates the entire loss!

### Variance Analysis

*"What happens when propensity approaches zero?"*

For the IPS estimator of a single sample:
$$\text{Var}\left[\frac{\ell(u,i)}{p(i|u)}\right] = \frac{\text{Var}[\ell(u,i)]}{p(i|u)^2}$$

**As $p \to 0$, variance $\to \infty$!**

**Numerical demonstration**:

| Propensity | Weight | Variance multiplier |
|------------|--------|---------------------|
| 0.80 | 1.25 | 1.56x |
| 0.20 | 5.0 | 25x |
| 0.05 | 20.0 | 400x |
| 0.01 | 100.0 | 10,000x |
| 0.001 | 1000.0 | 1,000,000x |

*"With propensity 0.001, a single observation contributes a million times more variance than a uniformly sampled observation. This makes training extremely unstable."*

---

## Variance Reduction: Capped IPS and SNIPS

### Capped IPS: Simple but Effective

**Idea**: Cap the maximum weight to prevent extreme values.

$$w_{ui}^{\text{capped}} = \min\left(\frac{1}{p(i|u)}, M\right)$$

where $M$ is the maximum allowed weight (e.g., $M = 100$).

**Trade-off**:
- **Pro**: Dramatically reduces variance
- **Con**: Introduces bias (no longer unbiased estimator)

**Choosing $M$**:

| $M$ value | Effect |
|-----------|--------|
| Very small (e.g., 5) | Low variance, high bias |
| Medium (e.g., 50-100) | Balanced |
| Very large (e.g., 10000) | Low bias, high variance |

*"In practice, M = 100 works well for most recommendation settings."*

```python
def capped_ips_loss(predictions, targets, propensities, max_weight=100.0):
    """
    Capped IPS loss to reduce variance.
    """
    weights = 1.0 / (propensities + 1e-8)
    capped_weights = np.minimum(weights, max_weight)

    squared_errors = (predictions - targets) ** 2
    weighted_loss = capped_weights * squared_errors

    return weighted_loss.mean()
```

---

### Self-Normalized IPS (SNIPS): Derivation and Why It Helps

*"SNIPS is a clever normalization that reduces variance while maintaining approximate unbiasedness."*

**The SNIPS estimator**:

$$\hat{L}_{\text{SNIPS}} = \frac{\sum_{(u,i) \in D} \frac{\ell(u,i)}{p(i|u)}}{\sum_{(u,i) \in D} \frac{1}{p(i|u)}}$$

**Why this helps**:

**Standard IPS**: Sums weighted losses, then divides by $|D|$
$$\hat{L}_{\text{IPS}} = \frac{\sum w_{ui} \cdot \ell_{ui}}{|D|}$$

**SNIPS**: Divides by sum of weights instead
$$\hat{L}_{\text{SNIPS}} = \frac{\sum w_{ui} \cdot \ell_{ui}}{\sum w_{ui}}$$

**The key insight**:
- If all propensities are equal, SNIPS = IPS = standard average
- If propensities vary, SNIPS normalizes by the "effective sample size"

**Variance reduction intuition**:

When propensities are highly variable:
- Standard IPS: $\sum w_{ui}$ can be very large or small depending on sample
- SNIPS: The ratio is more stable because numerator and denominator are correlated

**Numerical example**:

Sample 1: Observe $(u_1, i_1)$ with $p = 0.01$, $\ell = 1$
Sample 2: Observe $(u_2, i_2)$ with $p = 0.80$, $\ell = 1$

**IPS estimate**:
$$\hat{L}_{\text{IPS}} = \frac{1}{2}\left(\frac{1}{0.01} + \frac{1}{0.80}\right) = \frac{1}{2}(100 + 1.25) = 50.625$$

**SNIPS estimate**:
$$\hat{L}_{\text{SNIPS}} = \frac{100 \cdot 1 + 1.25 \cdot 1}{100 + 1.25} = \frac{101.25}{101.25} = 1.0$$

*"SNIPS gives a much more reasonable estimate! The extreme weight is normalized out."*

```python
def snips_loss(predictions, targets, propensities):
    """
    Self-Normalized IPS loss.
    """
    weights = 1.0 / (propensities + 1e-8)

    squared_errors = (predictions - targets) ** 2
    weighted_loss = weights * squared_errors

    # Self-normalize
    return weighted_loss.sum() / weights.sum()
```

---

## Doubly Robust Estimation: The Best of Both Worlds

### Why "Doubly" Robust?

*"Doubly robust estimation combines two approaches, and here's the magic: it's unbiased if EITHER approach is correct."*

**Two approaches being combined**:
1. **Direct method (imputation)**: Use a model to predict missing ratings
2. **IPS**: Reweight observed ratings

**Each has weaknesses**:
- Direct method: Biased if model is wrong
- IPS: High variance if propensities are small

**Doubly robust**: Hedges between both!

---

### Derivation of the Doubly Robust Estimator

**Setup**:
- $\hat{r}(u,i)$: Imputed (predicted) rating from a model
- $r_{ui}$: Observed rating (when available)
- $p(i|u)$: Propensity
- $O_{ui}$: Indicator = 1 if $(u,i)$ observed, 0 otherwise

**The DR estimator for expected rating**:

$$\hat{R}_{\text{DR}} = \frac{1}{|U||I|} \sum_{u,i} \left[ \hat{r}(u,i) + O_{ui} \cdot \frac{r_{ui} - \hat{r}(u,i)}{p(i|u)} \right]$$

*"Let's parse this formula piece by piece..."*

**Term 1**: $\hat{r}(u,i)$ - Start with imputed rating for ALL pairs

**Term 2**: $O_{ui} \cdot \frac{r_{ui} - \hat{r}(u,i)}{p(i|u)}$ - Correction term
- Only applies when we observe the rating ($O_{ui} = 1$)
- Corrects the imputation error $(r_{ui} - \hat{r}(u,i))$
- Weighted by inverse propensity

---

### Proof: Why It's "Doubly" Robust

**Scenario 1: Imputation model is perfect**

If $\hat{r}(u,i) = \mathbb{E}[r_{ui}]$ for all $(u,i)$:

The correction term has expected value 0:
$$\mathbb{E}\left[O_{ui} \cdot \frac{r_{ui} - \hat{r}(u,i)}{p(i|u)}\right] = \mathbb{E}\left[\frac{r_{ui} - \mathbb{E}[r_{ui}]}{p(i|u)} \cdot p(i|u)\right] = 0$$

So $\hat{R}_{\text{DR}} = \frac{1}{|U||I|} \sum_{u,i} \hat{r}(u,i)$ which is unbiased.

**Scenario 2: Propensities are exactly correct**

Even if $\hat{r}(u,i)$ is wrong, the IPS correction fixes it:

$$\mathbb{E}[\hat{R}_{\text{DR}}] = \mathbb{E}\left[\hat{r}(u,i)\right] + \mathbb{E}\left[\frac{r_{ui} - \hat{r}(u,i)}{p(i|u)} \cdot p(i|u)\right]$$
$$= \mathbb{E}[\hat{r}(u,i)] + \mathbb{E}[r_{ui}] - \mathbb{E}[\hat{r}(u,i)] = \mathbb{E}[r_{ui}]$$

**The magic**: Even if $\hat{r}(u,i)$ is arbitrarily wrong, we still get an unbiased estimate!

*"This is why it's 'doubly' robust: it works if either the model is good OR the propensities are good. In practice, both are approximate, and DR is more robust than either alone."*

---

### Implementation

```python
class DoublyRobustEstimator:
    def __init__(self, imputation_model, propensity_model):
        """
        Doubly robust estimator combining imputation and IPS.

        imputation_model: Predicts ratings for any (u, i) pair
        propensity_model: Estimates p(i | u)
        """
        self.imputation_model = imputation_model
        self.propensity_model = propensity_model

    def estimate_loss(self, observed_ratings, all_pairs):
        """
        Compute DR loss estimate.

        observed_ratings: List of (user, item, rating) tuples
        all_pairs: All (user, item) pairs to estimate over
        """
        total = 0.0

        # Create lookup for observed ratings
        observed_dict = {(u, i): r for u, i, r in observed_ratings}

        for u, i in all_pairs:
            # Imputed rating
            r_hat = self.imputation_model.predict(u, i)

            if (u, i) in observed_dict:
                # Observed: add correction term
                r_true = observed_dict[(u, i)]
                propensity = self.propensity_model.get_propensity(u, i)

                correction = (r_true - r_hat) / (propensity + 1e-8)
                total += r_hat + correction
            else:
                # Not observed: use imputation only
                total += r_hat

        return total / len(all_pairs)

    def dr_loss_for_training(self, predictions, targets, imputations, propensities):
        """
        DR loss function for gradient-based training.
        """
        weights = 1.0 / (propensities + 1e-8)

        # Residual from imputation
        residuals = targets - imputations

        # IPS-weighted residual loss
        ips_term = weights * (predictions - targets) ** 2

        # Variance reduction term
        control_variate = weights * (predictions - imputations) ** 2

        return (ips_term - control_variate + (predictions - imputations) ** 2).mean()
```

---

## Complete Numerical Walkthrough: Standard vs. IPS-Weighted Training

*"Let's trace through a concrete example to see exactly how IPS changes the learning process."*

### Setup

**5 users, 10 items, 15 observed ratings** (not uniformly distributed)

**True user preferences** (unknown to us):

|  | Item 1 | Item 2 | Item 3 | Item 4 | Item 5 | Item 6 | Item 7 | Item 8 | Item 9 | Item 10 |
|--|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|
| User A | 4 | 5 | 3 | 2 | 4 | 5 | 3 | 4 | 2 | 5 |
| User B | 5 | 4 | 4 | 3 | 5 | 4 | 2 | 3 | 4 | 4 |
| User C | 3 | 5 | 5 | 4 | 3 | 5 | 4 | 5 | 3 | 4 |
| User D | 4 | 3 | 4 | 5 | 4 | 3 | 5 | 4 | 4 | 3 |
| User E | 5 | 4 | 3 | 4 | 5 | 4 | 4 | 3 | 5 | 4 |

**Exposure log** (what was shown):

| User | Items Shown | Items NOT Shown |
|------|-------------|-----------------|
| A | 1, 2, 3 | 4, 5, 6, 7, 8, 9, 10 |
| B | 1, 2, 4 | 3, 5, 6, 7, 8, 9, 10 |
| C | 1, 2, 5 | 3, 4, 6, 7, 8, 9, 10 |
| D | 1, 2, 6 | 3, 4, 5, 7, 8, 9, 10 |
| E | 1, 2, 7 | 3, 4, 5, 6, 8, 9, 10 |

**Observed ratings** (15 total):

| Observation | User | Item | Rating |
|-------------|------|------|--------|
| 1 | A | 1 | 4 |
| 2 | A | 2 | 5 |
| 3 | A | 3 | 3 |
| 4 | B | 1 | 5 |
| 5 | B | 2 | 4 |
| 6 | B | 4 | 3 |
| 7 | C | 1 | 3 |
| 8 | C | 2 | 5 |
| 9 | C | 5 | 3 |
| 10 | D | 1 | 4 |
| 11 | D | 2 | 3 |
| 12 | D | 6 | 3 |
| 13 | E | 1 | 5 |
| 14 | E | 2 | 4 |
| 15 | E | 7 | 4 |

### Step 1: Compute Propensities from Exposure Log

**Item exposure counts**:

| Item | Times Shown | Propensity $p(i)$ |
|------|-------------|-------------------|
| 1 | 5 | 5/5 = 1.00 |
| 2 | 5 | 5/5 = 1.00 |
| 3 | 1 | 1/5 = 0.20 |
| 4 | 1 | 1/5 = 0.20 |
| 5 | 1 | 1/5 = 0.20 |
| 6 | 1 | 1/5 = 0.20 |
| 7 | 1 | 1/5 = 0.20 |
| 8-10 | 0 | 0/5 = 0.00 |

*"Notice the severe imbalance: Items 1 and 2 have 100% exposure, Items 3-7 have only 20%, and Items 8-10 were never shown!"*

### Step 2: Standard Loss vs. IPS-Weighted Loss

**Model prediction**: Assume simple model predicts global average = 3.93 for all pairs.

**Standard MSE Loss** (treating all observations equally):

$$L_{\text{standard}} = \frac{1}{15} \sum_{i=1}^{15} (r_i - 3.93)^2$$

| Obs | Rating | Prediction | Error$^2$ |
|-----|--------|------------|-----------|
| 1 | 4 | 3.93 | 0.005 |
| 2 | 5 | 3.93 | 1.145 |
| 3 | 3 | 3.93 | 0.865 |
| ... | ... | ... | ... |

$$L_{\text{standard}} = \frac{1}{15}(0.005 + 1.145 + 0.865 + ... ) = 0.729$$

**IPS-Weighted Loss**:

| Obs | Rating | Propensity | Weight | Weighted Error$^2$ |
|-----|--------|------------|--------|-------------------|
| 1 (Item 1) | 4 | 1.00 | 1.0 | 0.005 |
| 2 (Item 2) | 5 | 1.00 | 1.0 | 1.145 |
| 3 (Item 3) | 3 | 0.20 | **5.0** | **4.325** |
| 4 (Item 1) | 5 | 1.00 | 1.0 | 1.145 |
| 5 (Item 2) | 4 | 1.00 | 1.0 | 0.005 |
| 6 (Item 4) | 3 | 0.20 | **5.0** | **4.325** |
| 7 (Item 1) | 3 | 1.00 | 1.0 | 0.865 |
| 8 (Item 2) | 5 | 1.00 | 1.0 | 1.145 |
| 9 (Item 5) | 3 | 0.20 | **5.0** | **4.325** |
| 10 (Item 1) | 4 | 1.00 | 1.0 | 0.005 |
| 11 (Item 2) | 3 | 1.00 | 1.0 | 0.865 |
| 12 (Item 6) | 3 | 0.20 | **5.0** | **4.325** |
| 13 (Item 1) | 5 | 1.00 | 1.0 | 1.145 |
| 14 (Item 2) | 4 | 1.00 | 1.0 | 0.005 |
| 15 (Item 7) | 4 | 0.20 | **5.0** | **0.025** |

$$L_{\text{IPS}} = \frac{1}{15}(0.005 + 1.145 + 4.325 + ... ) = 1.51$$

### Step 3: How IPS Changes Gradient Updates

**Key observation**: Items 3-7 now contribute 5x more to the loss!

**Effect on learning**:

| Item | Standard Contribution | IPS Contribution | Change |
|------|----------------------|------------------|--------|
| Item 1 (popular) | 5 observations | 5 x 1.0 = 5 | Same |
| Item 2 (popular) | 5 observations | 5 x 1.0 = 5 | Same |
| Item 3 (rare) | 1 observation | 1 x 5.0 = 5 | **5x more** |
| Item 4 (rare) | 1 observation | 1 x 5.0 = 5 | **5x more** |
| ... | ... | ... | ... |

*"IPS upweights rare item interactions so they have equal influence on the model. This prevents the model from being dominated by popular items!"*

---

## Causal Inference: Counterfactual Reasoning

### The Fundamental Question

**What would the user's rating be if we showed a different item?**

**Notation**:
- $r_{ui}^{\text{obs}}$: Observed rating (item $i$ was shown)
- $r_{uj}^{\text{cf}}$: Counterfactual rating (item $j$ was NOT shown)

**Challenge**: We can never observe $r_{uj}^{\text{cf}}$ - it's fundamentally unknowable!

*"This is the fundamental problem of causal inference. We can only observe one potential outcome per user-item pair."*

### Propensity-based Counterfactual Estimation

**Idea**: Estimate counterfactuals using similar users who DID see the item.

$$\hat{r}_{uj}^{\text{cf}} = \mathbb{E}[r_{u'j} | u' \sim u, (u', j) \in D]$$

where $u' \sim u$ means "user $u'$ is similar to user $u$".

**This is essentially collaborative filtering, but reframed causally!**

---

## Debiasing Popularity

### Calibration: Matching Recommendation Distribution

**Goal**: Match recommendation distribution to user's true preference distribution.

**Problem**: If a user watches 80% indie films, but your model recommends 80% blockbusters, there's a mismatch.

**Method**: Re-rank to match popularity distribution of user's history.

```python
def calibrate_recommendations(rec_items, user_history_popularity, item_popularity, n_bins=10):
    """
    Re-rank recommendations to match user's historical popularity preferences.

    Args:
        rec_items: List of recommended item IDs
        user_history_popularity: Popularity scores of items in user's history
        item_popularity: Dict mapping item_id -> popularity score
        n_bins: Number of popularity bins

    Returns:
        Calibrated list of item IDs
    """
    # Compute target distribution from user history
    target_hist, bin_edges = np.histogram(user_history_popularity, bins=n_bins, range=(0, 1))
    target_dist = target_hist / target_hist.sum()

    # Score items by how well they match target distribution
    scores = []
    for item in rec_items:
        pop = item_popularity[item]
        bin_idx = min(int(pop * n_bins), n_bins - 1)
        scores.append(target_dist[bin_idx])

    # Re-rank by calibration score
    reranked = [x for _, x in sorted(zip(scores, rec_items), reverse=True)]
    return reranked
```

### Popularity Regularization

**Penalize** popular items during training to prevent over-recommendation.

$$\mathcal{L} = \mathcal{L}_{\text{base}} + \lambda \sum_{i} \log(1 + \text{popularity}(i)) \cdot \hat{r}_{ui}^2$$

**Effect**: Discourage high predictions for already-popular items.

---

## Debiasing Position

### The Position Bias Problem

*"Not all slots are equal. The first result gets 10x more clicks than the fifth, regardless of relevance."*

**Methods**:

**1. Randomization**: Occasionally shuffle top results to collect unbiased data.

**2. Inverse Rank Weighting**: Weight clicks by position.

```python
def position_debiased_ctr(clicks, positions):
    """
    Compute position-debiased CTR.

    Higher positions (worse) get higher weights to account
    for the fact that they're less likely to be seen.
    """
    position_weights = 1.0 / np.array(positions)
    return (clicks * position_weights).sum() / position_weights.sum()
```

---

## Exposure Fairness

### Equal Exposure Across Items/Creators

**Goal**: All items/creators get fair exposure opportunity.

**Method**: Exposure redistribution during re-ranking.

```python
def exposure_fair_rerank(scored_items, current_exposure, target_exposure, k=10, boost_factor=0.1):
    """
    Re-rank to equalize exposure across items.

    Args:
        scored_items: List of (item_id, relevance_score) tuples
        current_exposure: Dict mapping item_id -> exposure count
        target_exposure: Target exposure per item
        k: Number of items to return
        boost_factor: How much to boost under-exposed items

    Returns:
        List of k item IDs with exposure-adjusted ranking
    """
    # Compute exposure deficit for each item
    deficit = {item: target_exposure - current_exposure.get(item, 0)
               for item, score in scored_items}

    # Adjust scores: boost under-exposed items
    adjusted_scores = {item: score + boost_factor * max(deficit[item], 0)
                       for item, score in scored_items}

    # Re-rank by adjusted score
    reranked = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, score in reranked[:k]]
```

---

## What Can Go Wrong: Failure Modes and Pitfalls

*"Debiasing isn't magic. Let me walk you through the ways it can fail catastrophically..."*

### Failure Mode 1: Propensity Estimation is Hard

**The Problem**: You need to know $p(i|u)$, but how do you estimate it?

**Challenge**: You need access to the **logging policy** - the system that decided what to show.

**Scenarios where this fails**:

| Scenario | Why Propensity is Unknown |
|----------|---------------------------|
| Legacy system | No one documented how recommendations were made |
| Multiple policies | Different algorithms over time, A/B tests |
| User-driven | User searched for items (not recommended) |
| External factors | Items shown based on inventory, licensing |

**Numerical example of estimation error**:

True propensity: $p = 0.05$

Estimated propensity: $\hat{p} = 0.10$ (2x overestimate)

True IPS weight: $1/0.05 = 20$

Estimated weight: $1/0.10 = 10$ (50% underweight!)

*"Even a 2x error in propensity estimation leads to a 50% error in weights. This can severely bias your estimator."*

**Mitigation**:
- Use doubly robust estimation (less sensitive to propensity errors)
- Carefully log your recommendation policy
- Use propensity clipping to reduce sensitivity

---

### Failure Mode 2: Extreme Weights Blow Up Variance

**The Problem**: When propensities are very small, weights explode.

**Numerical demonstration**:

| Propensity | IPS Weight | Variance Multiplier |
|------------|------------|---------------------|
| 0.50 | 2 | 4x |
| 0.10 | 10 | 100x |
| 0.01 | 100 | 10,000x |
| 0.001 | 1,000 | 1,000,000x |

**Practical consequence**:

Imagine training with batch size 32. One observation has propensity 0.001 (weight 1000).

Effective contribution of that one sample:
$$\frac{1000}{1000 + 31 \times 1} \approx 97\%$$

*"A single sample dominates your entire batch! Gradients become extremely noisy."*

**Symptoms in training**:
- Loss oscillates wildly
- Model performance varies dramatically between epochs
- "Lucky" batches with no extreme weights do well; others don't

**Mitigation**:
- Use capped IPS (max weight = 100)
- Use SNIPS for self-normalization
- Filter out observations with propensity below threshold

---

### Failure Mode 3: Position Bias Conflated with Relevance Bias

**The Problem**: Users click more on top positions. Is this because:
1. Top items are more relevant?
2. Top positions are more visible?

**Both!** But standard IPS only corrects for item exposure, not position.

**Example**:

| Position | Clicks | Position CTR | Item Relevance |
|----------|--------|--------------|----------------|
| 1 | 100 | 10% | 8% (true) |
| 2 | 50 | 5% | 6% (true) |
| 3 | 20 | 2% | 4% (true) |
| 4 | 5 | 0.5% | 3% (true) |

*"Item in position 1 appears 25% more relevant than it actually is due to position bias!"*

**Why IPS doesn't fix this**: IPS corrects for WHICH items are shown, not WHERE they're shown.

**Mitigation**:
- Model position bias separately
- Use examination models: $P(\text{click}) = P(\text{examine}) \times P(\text{click}|\text{examine})$
- Run randomization experiments to estimate position effects

---

### Failure Mode 4: Overcorrection Hurts Popular Items Too Much

**The Problem**: After debiasing, popular items get systematically under-recommended.

**Why this happens**:

Popular items have high propensity $\to$ low IPS weight $\to$ contribute less to loss $\to$ model underfits them.

**Numerical example**:

Before debiasing:
- Popular items: RMSE = 0.8 (good predictions, lots of data)
- Long-tail items: RMSE = 1.5 (poor predictions, little data)

After aggressive IPS:
- Popular items: RMSE = 1.2 (worse! model ignores them)
- Long-tail items: RMSE = 1.1 (better, upweighted)

**Total RMSE might improve, but popular item quality suffers!**

*"This matters because popular items still generate most of your revenue/engagement. Over-correcting can hurt business metrics."*

**Mitigation**:
- Use moderate weight caps (not too aggressive)
- Monitor per-popularity-tier performance
- Use hybrid approaches (some standard loss + some IPS loss)

---

### Failure Mode 5: Debiasing Without Enough Data

**The Problem**: Long-tail items have few observations. Even with high IPS weights, estimates are noisy.

**Statistics reminder**:

Confidence interval width $\propto \frac{1}{\sqrt{n}}$

| Observations | 95% CI Width (for mean) |
|--------------|-------------------------|
| 100 | $\pm 0.20$ |
| 10 | $\pm 0.62$ |
| 1 | Undefined! |

*"IPS can make a single observation count as 100, but it's still just ONE observation. The variance is enormous."*

**Mitigation**:
- Require minimum observation count before IPS weighting
- Use shrinkage/regularization for low-count items
- Combine with content-based features for cold items

---

### Quick Diagnostic Checklist

*"Before deploying debiasing in production, verify:"*

| Check | What to Look For |
|-------|------------------|
| Propensity distribution | Any items with $p < 0.01$? Cap or filter |
| Weight distribution | Max weight < 100? If not, cap |
| Per-tier metrics | Popular items not significantly degraded? |
| Training stability | Loss not oscillating wildly? |
| Logging policy known? | Can you compute propensities accurately? |
| Position effects modeled? | Separate position from relevance bias? |

---

## Summary

**Key Takeaways**:

1. **Exposure bias**: Training on logged data creates feedback loops favoring popular items

2. **IPS**: Reweight observations by $1/p(i|u)$ to correct for selection bias
   - Unbiased but high variance

3. **Variance reduction**:
   - Capped IPS: $\min(1/p, M)$ - introduces bias, reduces variance
   - SNIPS: Self-normalized - approximately unbiased, lower variance

4. **Doubly robust**: Combine IPS + imputation - unbiased if EITHER is correct

5. **Failure modes**: Propensity estimation, extreme weights, position bias, overcorrection

**Trade-offs**: There's always a bias-variance trade-off. More aggressive debiasing reduces popularity bias but increases variance.

**Practical recommendations**:
- Start with SNIPS or capped IPS (M=100)
- Use doubly robust if you have a decent imputation model
- Monitor per-popularity-tier metrics
- Be skeptical of very small propensities

**Next**: Fairness definitions and methods.

---

## Reflection Questions

*"Before moving on, make sure you can answer these:"*

1. Why does training on logged data give biased recommendations toward popular items?
2. Derive the IPS estimator and prove it's unbiased.
3. What happens to IPS variance when propensity approaches zero?
4. Why is doubly robust estimation "doubly" robust? Prove it.
5. When would you prefer capped IPS over SNIPS?
6. How does position bias differ from exposure bias, and why doesn't standard IPS fix it?

---

## References

1. **Schnabel, T., et al. (2016)**. "Recommendations as Treatments: Debiasing Learning and Evaluation". *ICML*.
   - **IPS for recommendations**

2. **Wang, X., et al. (2019)**. "Doubly Robust Joint Learning for Recommendation on Data Missing Not at Random". *ICML*.
   - **Doubly robust estimation**

3. **Steck, H. (2018)**. "Calibrated Recommendations". *RecSys*.
   - **Calibration methods**

4. **Swaminathan, A., & Joachims, T. (2015)**. "The Self-Normalized Estimator for Counterfactual Learning". *NeurIPS*.
   - **SNIPS derivation**

5. **Joachims, T., et al. (2017)**. "Unbiased Learning-to-Rank with Biased Feedback". *WSDM*.
   - **Position bias correction**
