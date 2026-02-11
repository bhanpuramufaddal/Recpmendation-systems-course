# Week 11: Experimental Design for Recommendation Systems

## The Failure That Launched a Career in Proper Evaluation

*Before we dive into experimental design, let me tell you about a mistake I see all the time.*

**The scenario**: A data scientist builds a recommendation model. They split their data randomly 80/20, train on 80%, test on 20%. The model gets **95% accuracy** on the test set. They present to leadership: "Our model is amazing!"

**Three months later**: The model is deployed. Click-through rate drops 15%. Users complain about irrelevant recommendations. What went wrong?

*Here's what happened*: The random split created a **time machine**. Some test set interactions happened *before* training interactions. The model was literally predicting the past, which is easy but useless.

**The causality violation**:
- User clicks item A on January 1st (ended up in test set)
- User clicks item B on January 15th (ended up in training set)
- Model "predicts" January 1st click using January 15th data
- This is like predicting yesterday's weather using today's forecast!

*Why is this a problem?* In production, you can only use past data to predict future behavior. A model that "predicts the past" tells you nothing about how it will perform on tomorrow's users.

---

## Learning Objectives

- Understand why random splitting causes data leakage in time-series data
- Master temporal split strategies with mathematical justification
- Derive when to use leave-one-out vs. cross-validation
- Calculate sample sizes for A/B tests
- Recognize and avoid the 5+ common experimental failures

---

## The Core Problem: Why Can't We Just Split Randomly?

### A Numerical Demonstration

*Let me show you exactly what goes wrong. Follow along with these numbers.*

**Dataset**: 100 user-item interactions with timestamps

| Interaction ID | User | Item | Timestamp | Action |
|---------------|------|------|-----------|--------|
| 1 | Alice | Movie A | Day 1 | Click |
| 2 | Bob | Movie B | Day 2 | Click |
| 3 | Alice | Movie C | Day 3 | Click |
| ... | ... | ... | ... | ... |
| 50 | Carol | Movie X | Day 50 | Click |
| ... | ... | ... | ... | ... |
| 100 | Alice | Movie Z | Day 100 | Click |

**Random 80/20 Split** (using random seed 42):

Training set (80 interactions): Includes interactions from Days 5, 12, 23, 45, 67, 78, 89, 95...

Test set (20 interactions): Includes interactions from Days 3, 8, 15, 41, 55, 62, 73, 88...

*Do you see the problem?*

**Causality violations in this test set**:
- Day 3 interaction in test, but Days 5, 12, 23... in training
- Model predicts Day 3 using information from Day 5, 12, 23...
- **12 out of 20 test interactions have at least one "future" interaction in training!**

---

### The Mathematical Consequence

*Let's compute what happens to our metrics.*

**Scenario**: Collaborative filtering model predicting user preferences.

**With Random Split** (data leakage):
- Test interaction: Alice clicks Movie C on Day 3
- Training includes: Alice clicked Movies D, E, F on Days 5, 10, 15
- Model learns: "Alice likes sci-fi" (from Days 5, 10, 15)
- Predicts: Alice will like Movie C (which is sci-fi)
- Result: **Correct prediction, but using future information!**

**Computed Metrics (Random Split)**:
$$\text{Precision@10} = 0.82, \quad \text{Recall@10} = 0.71, \quad \text{NDCG@10} = 0.89$$

**With Temporal Split** (correct):
- Training: All interactions before Day 80
- Test: All interactions on/after Day 80
- Model predicts Day 80+ using only Day 0-79 data
- No causality violations

**Computed Metrics (Temporal Split)**:
$$\text{Precision@10} = 0.45, \quad \text{Recall@10} = 0.38, \quad \text{NDCG@10} = 0.52$$

*Look at the difference!* The random split inflated NDCG by **71%** (0.89 vs 0.52). This isn't a better model - it's a deceptive evaluation.

---

### Why Does Data Leakage Inflate Metrics?

*Here's the intuition.*

**User behavior has temporal patterns**:
1. **Preference drift**: Alice liked action movies in January, but shifted to dramas by March
2. **Exploration then exploitation**: Users try many items early, then settle on favorites
3. **Seasonal effects**: Holiday shopping patterns differ from regular browsing

**With random split**: Future behavior "leaks" into training, making predictions easier:
- Model learns Alice's March preferences
- Predicts Alice's January behavior using March knowledge
- Gets it "right" because March Alice still somewhat resembles January Alice

**With temporal split**: Model must genuinely extrapolate:
- Model only knows Alice's behavior up to cutoff
- Must predict future behavior without seeing it
- Much harder, but this is the real task!

---

## Temporal Splitting: The Right Approach

### Mathematical Framework

*Let's formalize what we mean by temporal split.*

**Definition**: Given interactions $\mathcal{D} = \{(u, i, t)\}$ where $t$ is timestamp:

$$\mathcal{D}_{\text{train}} = \{(u, i, t) \in \mathcal{D} : t \leq T\}$$

where $T$ = the temporal cutoff point

$$\mathcal{D}_{\text{test}} = \{(u, i, t) \in \mathcal{D} : t > T\}$$

where $T$ = the temporal cutoff point

**Key property**: For all $(u_1, i_1, t_1) \in \mathcal{D}_{\text{train}}$ and $(u_2, i_2, t_2) \in \mathcal{D}_{\text{test}}$:

$$t_1 \leq T < t_2$$

This guarantees **no temporal leakage**.

---

### Choosing the Cutoff Point T

*How do we pick T?*

**Method 1: Percentage-based** (e.g., 80/20 by time)

```python
def temporal_split_percentage(df, test_ratio=0.2):
    """Split by time: last test_ratio% of interactions become test set."""
    df = df.sort_values('timestamp')
    split_idx = int(len(df) * (1 - test_ratio))

    # T = timestamp of the split_idx-th interaction
    T = df.iloc[split_idx]['timestamp']

    train = df[df['timestamp'] <= T]
    test = df[df['timestamp'] > T]

    return train, test, T
```

**Method 2: Calendar-based** (e.g., test = last month)

```python
def temporal_split_calendar(df, test_start_date='2024-12-01'):
    """Split by calendar date."""
    T = pd.to_datetime(test_start_date)

    train = df[df['timestamp'] < T]
    test = df[df['timestamp'] >= T]

    return train, test, T
```

---

### Numerical Walkthrough: Correct Temporal Split

*Let's redo our 100-interaction example properly.*

**Dataset**: 100 interactions over 100 days

**Temporal Split** (T = Day 80):

| Set | Interactions | Days | Count |
|-----|--------------|------|-------|
| Training | 1-80 | Day 1-80 | 80 |
| Test | 81-100 | Day 81-100 | 20 |

**Verification (no causality violations)**:
- Every test interaction (Days 81-100) is **after** every training interaction (Days 1-80)
- Model cannot use future information

**What the model sees**:
```
Training: User Alice -> [Movie A (Day 1), Movie C (Day 3), Movie F (Day 25), ...]
Test query: What will Alice click on Day 85?
Model answer: Based on Days 1-80 only
```

*This is realistic!* In production, you only have past data.

---

## Leave-One-Out Evaluation

### When to Use Leave-One-Out

*Random splits and even temporal splits have a problem with sparse data. Let me show you.*

**Problem**: User has only 3 interactions total.
- Temporal split: 2 in training, 1 in test
- But with only 2 training interactions, can we really learn this user's preferences?

**Leave-One-Out (LOO)** addresses this by:
1. For each user, hold out only their **last** interaction for testing
2. Train on **all other** interactions (including from other users)

---

### Mathematical Formulation

**For each user $u$**:

$$\mathcal{D}_{\text{train}}^{(u)} = \{(u', i, t) : u' \neq u\} \cup \{(u, i, t) : t < t_{\max}^{(u)}\}$$

where $t_{\max}^{(u)} = \max\{t : (u, i, t) \in \mathcal{D}\}$ is the timestamp of user $u$'s last interaction

$$\mathcal{D}_{\text{test}}^{(u)} = \{(u, i, t_{\max}^{(u)})\}$$

where $t_{\max}^{(u)} = \max\{t : (u, i, t) \in \mathcal{D}\}$ is the timestamp of user $u$'s last interaction (just the single last interaction)

**Key insight**: This is **per-user temporal split**. We're still respecting causality!

---

### Statistical Properties of LOO

*Why is LOO statistically sound?*

**Property 1: Maximum Training Data**
- Each user gets (almost) all their history for training
- Better for sparse datasets where every interaction matters

**Property 2: One Test Sample Per User**
- Prevents users with many interactions from dominating metrics
- Each user contributes equally to the final score

**Property 3: Variance Considerations**

With LOO, we get one sample per user. The variance of our metric estimate is:

$$\text{Var}(\hat{\mu}) = \frac{\sigma^2}{n}$$

where:
- $\sigma^2$ = variance of per-user scores
- $n$ = number of users

*Question: Is one sample per user enough?*

**Answer**: For ranking metrics (Hit@K, NDCG@K), one sample is often sufficient because:
- We're asking: "Is the held-out item in top K?"
- This is a Bernoulli trial with clear outcome
- Across many users, we get a reliable estimate

---

### Leave-K-Out Variant

**When K > 1**: Hold out last K interactions per user

**Trade-off**:
- More test samples per user (lower variance)
- Less training data per user (potentially worse model)

**Guideline**:
- Sparse data (< 10 interactions/user): LOO (K=1)
- Dense data (> 50 interactions/user): Leave-5-out or Leave-10-out
- Very dense (> 100 interactions/user): Temporal split is fine

```python
def leave_k_out(user_interactions, k=1):
    """
    Hold out last k interactions per user.

    Args:
        user_interactions: dict {user_id: [(item, timestamp), ...]}
        k: number of interactions to hold out

    Returns:
        train, test dictionaries
    """
    train = {}
    test = {}

    for user, interactions in user_interactions.items():
        # Sort by timestamp
        sorted_interactions = sorted(interactions, key=lambda x: x[1])

        if len(sorted_interactions) > k:
            train[user] = sorted_interactions[:-k]
            test[user] = sorted_interactions[-k:]
        # else: skip user (not enough interactions)

    return train, test
```

---

## Cross-Validation for Time Series

### The Problem with Standard K-Fold

*Why can't we just use scikit-learn's KFold?*

**Standard K-Fold**:
```
Fold 1: Train on [2,3,4,5], Test on [1]
Fold 2: Train on [1,3,4,5], Test on [2]
...
```

**Problem**: In Fold 1, we train on data from time periods 2-5 and test on period 1.
- We're predicting the past from the future!
- Same data leakage problem as random split

---

### Expanding Window Cross-Validation

*The correct approach for time series.*

**Setup**: Data spans time periods [1, 2, 3, 4, 5]

**Expanding Window**:
```
Fold 1: Train on [1],       Test on [2]
Fold 2: Train on [1,2],     Test on [3]
Fold 3: Train on [1,2,3],   Test on [4]
Fold 4: Train on [1,2,3,4], Test on [5]
```

**Properties**:
- Training window **expands** over time
- Test is always **after** training
- Mimics production: more data accumulates over time

---

### Sliding Window Cross-Validation

**Alternative approach**: Fixed-size training window

**Sliding Window** (window size = 2):
```
Fold 1: Train on [1,2],   Test on [3]
Fold 2: Train on [2,3],   Test on [4]
Fold 3: Train on [3,4],   Test on [5]
```

**When to use Sliding vs. Expanding**:

| Aspect | Expanding Window | Sliding Window |
|--------|-----------------|----------------|
| Training data | Grows over time | Fixed size |
| Recency bias | Less (uses all history) | More (recent data only) |
| Concept drift | May miss drift | Adapts to drift |
| Computation | Later folds slower | Constant time |

**Guideline**:
- Stable preferences: Expanding window
- Fast-changing preferences (fashion, news): Sliding window

---

### Implementation

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(df, n_splits=5, method='expanding'):
    """
    Time series cross-validation.

    Args:
        df: DataFrame sorted by timestamp
        n_splits: Number of folds
        method: 'expanding' or 'sliding'
    """
    df = df.sort_values('timestamp')

    if method == 'expanding':
        # sklearn's TimeSeriesSplit does expanding window
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, test_idx in tscv.split(df):
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]
            yield train, test

    elif method == 'sliding':
        # Custom sliding window
        total_size = len(df)
        test_size = total_size // (n_splits + 1)
        train_size = test_size * 2  # Fixed training window

        for i in range(n_splits):
            train_start = i * test_size
            train_end = train_start + train_size
            test_end = train_end + test_size

            if test_end > total_size:
                break

            train = df.iloc[train_start:train_end]
            test = df.iloc[train_end:test_end]
            yield train, test
```

---

## Statistical Significance Testing

### Why Paired t-Test?

*Why do we use a paired t-test instead of a regular t-test?*

**Scenario**: Comparing Model A vs Model B on NDCG@10

**Unpaired approach** (wrong for our setting):
- Compute mean NDCG for Model A: $\bar{x}_A = 0.52$
- Compute mean NDCG for Model B: $\bar{x}_B = 0.54$
- Standard t-test: "Are these means different?"

**Problem**: High variance from user differences masks model differences!
- User 1 is easy to predict (NDCG = 0.9 for both models)
- User 2 is hard to predict (NDCG = 0.2 for both models)
- This user-to-user variance swamps the model difference

---

### The Paired Difference Approach

**Paired approach** (correct):

For each user $u$, compute the **difference**:
$$d_u = \text{NDCG}_A(u) - \text{NDCG}_B(u)$$

Now test: "Is the mean difference different from zero?"

**Why this works**: The user-specific difficulty **cancels out**!
- User 1: $d_1 = 0.90 - 0.88 = 0.02$
- User 2: $d_2 = 0.22 - 0.20 = 0.02$
- Both show Model A is 0.02 better, despite different difficulty levels

---

### Mathematical Derivation

**Null Hypothesis**: $H_0: \mu_d = 0$ (no difference between models)

**Alternative**: $H_1: \mu_d \neq 0$ (models differ)

**Test statistic**:

$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

where:
- $\bar{d} = \frac{1}{n}\sum_{u=1}^n d_u$ (mean difference)
- $s_d = \sqrt{\frac{1}{n-1}\sum_{u=1}^n (d_u - \bar{d})^2}$ (standard deviation of differences)
- $n$ = number of users

**Decision**: Reject $H_0$ if $|t| > t_{\alpha/2, n-1}$ or equivalently if $p < \alpha$

---

### Numerical Example

*Let's work through a complete example.*

**Data**: 10 users, NDCG@10 for two models

| User | Model A | Model B | Difference $d_u$ |
|------|---------|---------|------------------|
| 1 | 0.85 | 0.82 | +0.03 |
| 2 | 0.72 | 0.70 | +0.02 |
| 3 | 0.68 | 0.71 | -0.03 |
| 4 | 0.91 | 0.88 | +0.03 |
| 5 | 0.55 | 0.52 | +0.03 |
| 6 | 0.78 | 0.79 | -0.01 |
| 7 | 0.62 | 0.58 | +0.04 |
| 8 | 0.88 | 0.85 | +0.03 |
| 9 | 0.45 | 0.44 | +0.01 |
| 10 | 0.73 | 0.71 | +0.02 |

**Step 1: Compute mean difference**
$$\bar{d} = \frac{0.03 + 0.02 + (-0.03) + 0.03 + 0.03 + (-0.01) + 0.04 + 0.03 + 0.01 + 0.02}{10} = \frac{0.17}{10} = 0.017$$

**Step 2: Compute standard deviation of differences**
$$s_d = \sqrt{\frac{\sum(d_u - 0.017)^2}{9}} = \sqrt{\frac{0.00461}{9}} = 0.0226$$

**Step 3: Compute t-statistic**
$$t = \frac{0.017}{0.0226 / \sqrt{10}} = \frac{0.017}{0.00715} = 2.38$$

**Step 4: Find p-value**
For $t = 2.38$ with df = 9, $p \approx 0.041$

**Conclusion**: $p < 0.05$, so we reject $H_0$. Model A is significantly better than Model B.

```python
from scipy.stats import ttest_rel

model_a_ndcg = [0.85, 0.72, 0.68, 0.91, 0.55, 0.78, 0.62, 0.88, 0.45, 0.73]
model_b_ndcg = [0.82, 0.70, 0.71, 0.88, 0.52, 0.79, 0.58, 0.85, 0.44, 0.71]

t_stat, p_value = ttest_rel(model_a_ndcg, model_b_ndcg)
print(f"t-statistic: {t_stat:.3f}")  # 2.378
print(f"p-value: {p_value:.4f}")     # 0.0413

if p_value < 0.05:
    print("Significant difference at alpha=0.05")
```

---

## A/B Testing Design

### Sample Size Calculation

*How many users do we need for a reliable A/B test?*

**The fundamental trade-off**:
- Too few users: Can't detect real differences (low power)
- Too many users: Waste resources, delay decisions

**Required inputs**:
1. **Baseline metric** ($p_0$): Current conversion rate (e.g., 5%)
2. **Minimum detectable effect** ($\delta$): Smallest improvement worth detecting (e.g., 0.5%)
3. **Significance level** ($\alpha$): False positive rate (typically 0.05)
4. **Power** ($1 - \beta$): True positive rate (typically 0.80)

---

### The Sample Size Formula

**For comparing two proportions** (e.g., conversion rates):

$$n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\bar{p}(1-\bar{p})}{\delta^2}$$

where:
- $z_{\alpha/2}$ = z-score for significance level (1.96 for $\alpha = 0.05$)
- $z_\beta$ = z-score for power (0.84 for power = 0.80)
- $\bar{p} = (p_0 + p_1)/2$ = average of baseline and expected new rate
- $\delta = p_1 - p_0$ = minimum detectable effect

---

### Numerical Example: Sample Size Calculation

*Let's calculate sample size for a realistic scenario.*

**Scenario**: Testing a new recommendation algorithm
- Current click-through rate: $p_0 = 5\%$ = 0.05
- Want to detect: 10% relative improvement (0.5% absolute)
- So $p_1 = 5.5\%$ = 0.055, $\delta = 0.005$
- $\alpha = 0.05$ (5% false positive rate)
- Power = 0.80 (80% chance of detecting true effect)

**Step 1: Look up z-scores**
- $z_{\alpha/2} = z_{0.025} = 1.96$
- $z_\beta = z_{0.20} = 0.84$

**Step 2: Compute average proportion**
$$\bar{p} = \frac{0.05 + 0.055}{2} = 0.0525$$

**Step 3: Apply formula**
$$n = \frac{(1.96 + 0.84)^2 \times 2 \times 0.0525 \times (1 - 0.0525)}{0.005^2}$$

$$n = \frac{7.84 \times 2 \times 0.0525 \times 0.9475}{0.000025}$$

$$n = \frac{7.84 \times 0.0995}{0.000025} = \frac{0.780}{0.000025} = 31,200$$

**Result**: Need ~31,200 users **per group** (62,400 total) to detect a 0.5% improvement in CTR with 80% power.

*This is why A/B tests take so long!* A 10% relative improvement in CTR requires tens of thousands of users.

```python
def sample_size_proportions(p0, delta, alpha=0.05, power=0.80):
    """
    Calculate required sample size per group for A/B test.

    Args:
        p0: Baseline proportion (e.g., 0.05 for 5% CTR)
        delta: Minimum detectable effect (absolute, e.g., 0.005)
        alpha: Significance level
        power: Statistical power

    Returns:
        n: Required sample size per group
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha/2)  # Two-tailed
    z_beta = norm.ppf(power)

    p1 = p0 + delta
    p_bar = (p0 + p1) / 2

    n = ((z_alpha + z_beta)**2 * 2 * p_bar * (1 - p_bar)) / (delta**2)

    return int(np.ceil(n))

# Example
n = sample_size_proportions(p0=0.05, delta=0.005)
print(f"Required sample size per group: {n:,}")  # ~31,200
```

---

### Randomization Unit: A Critical Choice

*What should we randomize: users, sessions, or individual requests?*

**Option 1: User-level randomization**
- Each user is assigned to control or treatment for all their sessions
- Pros: No within-user contamination, simpler analysis
- Cons: Need more users, slower to reach sample size

**Option 2: Session-level randomization**
- Each session is independently assigned
- Same user might see control in morning, treatment in evening
- Pros: More samples per day
- Cons: Carryover effects (user behavior influenced by previous session)

**Option 3: Request-level randomization**
- Each page view gets independent assignment
- Pros: Maximum samples
- Cons: Inconsistent user experience, confusing for users

**Recommendation**: **User-level** for most recommendation systems.
- User experience should be consistent
- Recommendation quality builds over time (needs consistent treatment)
- Statistical analysis is cleaner

---

### Multiple Testing Correction

*What if we're testing 10 different metrics?*

**The problem**: If we test 10 metrics at $\alpha = 0.05$, expected false positives:
$$\text{Expected false positives} = 10 \times 0.05 = 0.5$$

With 50% chance of at least one false positive, we'll often "discover" spurious effects!

---

**Solution 1: Bonferroni Correction**

Divide $\alpha$ by number of tests:
$$\alpha_{\text{adjusted}} = \frac{\alpha}{m} = \frac{0.05}{10} = 0.005$$

**Pros**: Simple, controls family-wise error rate (FWER)
**Cons**: Very conservative, may miss real effects

---

**Solution 2: False Discovery Rate (FDR) - Benjamini-Hochberg**

Control the expected proportion of false discoveries among rejected hypotheses.

**Procedure**:
1. Sort p-values: $p_{(1)} \leq p_{(2)} \leq ... \leq p_{(m)}$
2. Find largest $k$ where $p_{(k)} \leq \frac{k}{m} \cdot \alpha$
3. Reject all hypotheses with $p \leq p_{(k)}$

**Example**:
- 10 metrics with p-values: 0.001, 0.008, 0.015, 0.04, 0.05, 0.12, 0.25, 0.38, 0.56, 0.89
- With FDR = 0.10:
  - $p_{(1)} = 0.001 \leq \frac{1}{10} \times 0.10 = 0.01$ (reject)
  - $p_{(2)} = 0.008 \leq \frac{2}{10} \times 0.10 = 0.02$ (reject)
  - $p_{(3)} = 0.015 \leq \frac{3}{10} \times 0.10 = 0.03$ (reject)
  - $p_{(4)} = 0.04 \leq \frac{4}{10} \times 0.10 = 0.04$ (reject)
  - $p_{(5)} = 0.05 > \frac{5}{10} \times 0.10 = 0.05$ (don't reject)

**Result**: Reject 4 hypotheses (vs. 1 with Bonferroni at $\alpha = 0.005$)

**Guideline**:
- Bonferroni: When false positives are very costly (medical decisions)
- FDR: When some false positives are acceptable (exploratory analysis)

---

## Negative Sampling

### The Problem with Implicit Feedback

*In explicit feedback systems (ratings), we know what users dislike. But what about implicit feedback (clicks)?*

**Observed**: User clicked items A, B, C
**Not observed**: User didn't click items D, E, F, G, H, I, J, K, ...

*Does "not clicked" mean "dislike"?*

**No!** The user might:
- Not have seen the item (never shown)
- Have seen but not noticed
- Plan to click later
- Like it but not in the mood right now

Yet for evaluation, we need **negatives** to compute ranking metrics.

---

### Sampling Strategies

**Strategy 1: Random Negatives**

Sample uniformly from items user didn't interact with.

```python
def random_negatives(user_positives, all_items, n_negatives=99):
    """Sample random items as negatives."""
    candidates = list(set(all_items) - set(user_positives))
    return random.sample(candidates, min(n_negatives, len(candidates)))
```

**Pros**: Simple, fast
**Cons**: Most negatives are "obviously wrong" (e.g., sampling baby products for a teenager)

---

**Strategy 2: Popularity-Biased Negatives**

Sample proportional to item popularity.

```python
def popularity_negatives(user_positives, item_counts, n_negatives=99):
    """Sample negatives proportional to popularity."""
    candidates = {i: c for i, c in item_counts.items()
                  if i not in user_positives}

    items = list(candidates.keys())
    probs = np.array(list(candidates.values()))
    probs = probs / probs.sum()

    return np.random.choice(items, size=n_negatives, p=probs, replace=False)
```

**Pros**: Harder task (distinguishing from popular items)
**Cons**: May over-penalize models that recommend popular items

---

**Strategy 3: Hard Negatives**

Sample items the user almost clicked (seen but not clicked).

**Requires**: Impression logs (which items were shown to user)

```python
def hard_negatives(user_positives, user_impressions, n_negatives=99):
    """Sample from impressions that weren't clicked."""
    hard_negs = list(set(user_impressions) - set(user_positives))
    return random.sample(hard_negs, min(n_negatives, len(hard_negs)))
```

**Pros**: Most realistic evaluation
**Cons**: Requires impression data (not always available)

**Guideline**: Use hard negatives when available; otherwise use random with many negatives (99 or 999).

---

## What Can Go Wrong: Failure Mode Checklist

### Failure Mode 1: Data Leakage

*We've discussed this, but let me give you a detection checklist.*

**Detection Checklist**:

- [ ] Are any test timestamps before the earliest training timestamp?
- [ ] Is there user-level contamination? (User's test items used as training features)
- [ ] Are features computed on the full dataset? (e.g., item popularity from all data)
- [ ] Is target encoding done before the split?
- [ ] Are embeddings pre-trained on data that includes test period?

**Symptoms**:
- Offline metrics much better than expected
- Large gap between offline and online performance
- Model performs well on "historical" predictions, poorly on "future" ones

**Fix**: Re-run evaluation with strict temporal separation. If metrics drop significantly, you had leakage.

---

### Failure Mode 2: Simpson's Paradox

*What if the overall metric improves, but every segment gets worse?*

**Example**: New algorithm tested

**Overall Results**:
| Model | Overall CTR |
|-------|-------------|
| Control | 4.0% |
| Treatment | 4.2% |

**Treatment wins!** Right?

**Segmented Results**:
| Model | Mobile CTR | Desktop CTR |
|-------|------------|-------------|
| Control | 5.0% | 3.0% |
| Treatment | 4.8% | 2.8% |

**Treatment is worse in BOTH segments!** How?

**The explanation**: Treatment shifted traffic composition
- Control: 50% mobile, 50% desktop
- Treatment: 70% mobile, 30% desktop
- Mobile has higher CTR baseline
- Treatment's higher mobile share inflates overall CTR

**Lesson**: Always check segment-level results, not just aggregates.

---

### Failure Mode 3: Novelty Effects

*New things get clicks just because they're new.*

**Scenario**: Launch new recommendation UI
- Week 1: CTR jumps 20%!
- Week 2: CTR up 15%
- Week 4: CTR up 5%
- Week 8: CTR same as before

**What happened?** Users clicked to explore the new UI, not because recommendations were better.

**Detection**:
- Monitor metrics over time (not just first few days)
- Segment by user tenure (new vs. existing users)
- Check if effect is on recommendations or elsewhere (UI exploration)

**Mitigation**:
- Run experiments longer (2+ weeks minimum)
- Use "holdout" groups that never see treatment
- Compare to previous feature launches

---

### Failure Mode 4: Network Effects

*Treatment affects control through social connections.*

**Scenario**: Testing a "share with friends" feature
- Treatment users can share recommendations
- They share with control users
- Control users' behavior changes!

**The problem**: Control is no longer a true control. Treatment effect "leaks" through the social network.

**Called**: Spillover effects, interference, SUTVA violation

**Detection**:
- Check if control users connected to many treatment users behave differently
- Look for geographic clusters (if treatment spreads locally)

**Mitigation**:
- **Cluster randomization**: Randomize at the community/cluster level
- **Ego-network designs**: Keep user and their friends in same group
- **Geographic randomization**: Different cities get different treatments

---

### Failure Mode 5: Survivorship Bias

*Only evaluating on users who stayed.*

**Scenario**: Testing a new onboarding recommendation flow
- Run A/B test for 4 weeks
- Measure engagement metrics at week 4
- Treatment shows 10% higher engagement!

**The problem**: We only measured users who stayed 4 weeks.
- Maybe treatment drove away users in week 1
- Remaining treatment users are more engaged (selection effect)
- We're comparing survivors to survivors, not original populations

**Detection**:
- Track user counts over time (is there dropout difference?)
- Intent-to-treat analysis (include all users randomized, not just survivors)

**Mitigation**:
- Always track user retention as a primary metric
- Use intent-to-treat analysis (analyze all randomized users)
- Include early dropouts with their partial metrics

---

### Failure Mode 6: Peeking Problem

*Checking results before the experiment is complete.*

**Scenario**:
- Day 3: Treatment is winning! (p = 0.02)
- Day 5: Treatment still ahead (p = 0.03)
- Day 7: Treatment barely ahead (p = 0.06)
- Day 10: Treatment loses (p = 0.15)

**If you stopped at Day 3**, you'd declare victory. But it was just random fluctuation!

**The math**: Each peek is a hypothesis test. If you peek 10 times at $\alpha = 0.05$, your actual false positive rate is much higher (~19%).

**Mitigation**:
- Pre-register experiment duration
- Use sequential testing methods (SPRT, group sequential designs) if you must peek
- Adjust alpha for number of planned interim analyses

---

## Complete Numerical Example: Random vs. Temporal Split

*Let's walk through a comprehensive example showing the difference.*

### Dataset Setup

**100 interactions across 100 days**:

```
Day 1:  Alice -> ItemA (sci-fi)
Day 2:  Bob -> ItemB (romance)
Day 3:  Alice -> ItemC (sci-fi)
Day 5:  Carol -> ItemD (action)
...
Day 45: Alice -> ItemX (sci-fi)
Day 46: Bob -> ItemY (romance)
...
Day 80: Alice -> ItemZ (sci-fi)
Day 81: Carol -> ItemA (action)
...
Day 100: Bob -> ItemM (drama)
```

### Random Split Results

**Random 80/20 split** (seed=42):

Training interactions: 80 (from various days)
Test interactions: 20 (from various days)

**Causality violations identified**:
- Test: Alice->ItemC (Day 3), Train includes Alice->ItemX (Day 45)
- Test: Bob->ItemY (Day 46), Train includes Bob->ItemM (Day 100)
- **12 of 20 test interactions have future training data!**

**Model Performance (Random Split)**:

```python
# Simulated results showing inflated metrics
random_split_metrics = {
    'Precision@10': 0.78,
    'Recall@10': 0.65,
    'NDCG@10': 0.82,
    'Hit@10': 0.91
}
```

### Temporal Split Results

**Temporal split at Day 80**:

Training: All interactions Days 1-80 (80 interactions)
Test: All interactions Days 81-100 (20 interactions)

**Causality violations**: 0 (by construction)

**Model Performance (Temporal Split)**:

```python
# Realistic metrics without data leakage
temporal_split_metrics = {
    'Precision@10': 0.42,
    'Recall@10': 0.35,
    'NDCG@10': 0.48,
    'Hit@10': 0.62
}
```

### The Comparison

| Metric | Random Split | Temporal Split | Inflation |
|--------|--------------|----------------|-----------|
| Precision@10 | 0.78 | 0.42 | **86%** |
| Recall@10 | 0.65 | 0.35 | **86%** |
| NDCG@10 | 0.82 | 0.48 | **71%** |
| Hit@10 | 0.91 | 0.62 | **47%** |

*The random split inflated metrics by 47-86%!* A model that appears excellent with random split is actually mediocre.

**The lesson**: Always use temporal split for recommendation systems evaluation. Random split is a lie.

---

## Summary

**Core Principle**: Experimental design in recommendation systems must respect the **temporal nature** of the prediction task.

### Train-Test Splitting

| Method | When to Use | Watch Out For |
|--------|-------------|---------------|
| **Temporal Split** | Default choice | Users with no test data |
| **Leave-One-Out** | Sparse data (<10 items/user) | High variance |
| **Leave-K-Out** | Dense data | Reduced training |
| **Time Series CV** | Need multiple folds | Computational cost |

### A/B Testing Essentials

1. **Sample Size**: Use the formula, expect thousands of users
2. **Randomization**: User-level for consistent experience
3. **Multiple Testing**: Apply Bonferroni (conservative) or FDR (balanced)
4. **Duration**: Run 2+ weeks to avoid novelty effects

### Critical Failure Modes

1. **Data Leakage**: Future information in training (detection checklist provided)
2. **Simpson's Paradox**: Aggregate improves, segments worsen
3. **Novelty Effects**: New = clicks regardless of quality
4. **Network Effects**: Treatment contaminates control
5. **Survivorship Bias**: Only measuring users who stayed
6. **Peeking**: Checking results early inflates false positives

### Golden Rules

1. **Never use random split** for time-series data
2. **Always respect causality**: Train on past, test on future
3. **Report per-segment metrics**, not just aggregates
4. **Pre-register experiments** to avoid p-hacking
5. **Run experiments long enough** to avoid novelty effects

---

## References

1. **Bellogin, A., et al. (2017)**. "Statistical Biases in Information Retrieval Metrics for Recommender Systems". *Information Retrieval Journal*.

2. **Cremonesi, P., et al. (2010)**. "Performance of Recommender Algorithms on Top-N Recommendation Tasks". *RecSys*.

3. **Kohavi, R., et al. (2020)**. "Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing". *Cambridge University Press*.

4. **Ji, Y., et al. (2020)**. "A Critical Study on Data Leakage in Recommender System Offline Evaluation". *arXiv preprint*.

5. **Gilotte, A., et al. (2018)**. "Offline A/B Testing for Recommender Systems". *WSDM*.
