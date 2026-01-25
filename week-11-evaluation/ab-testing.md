# Week 11: A/B Testing for Recommendation Systems

## Learning Objectives

- Design and run A/B tests for recommendations
- Choose appropriate metrics and sample sizes
- Interpret results with statistical rigor
- Avoid common pitfalls in online evaluation

---

## Why A/B Testing?

### The Offline-Online Gap

**Offline evaluation** (RMSE, Precision@K on historical data):
- ✅ Fast, cheap, reproducible
- ❌ May not reflect real user behavior
- ❌ Doesn't capture business metrics (revenue, retention)

**Example**:
- Model A: RMSE = 0.85
- Model B: RMSE = 0.87
- **Offline**: Model A wins
- **Online A/B test**: Model B increases watch time by 15%!

**Why?** RMSE optimizes rating accuracy. Users care about discovering great content.

**Lesson**: **Always A/B test before full deployment.**

---

## A/B Testing Fundamentals

### What is an A/B Test?

**Randomized controlled experiment**:
1. **Split users** into groups (typically 50/50)
2. **Group A (Control)**: Existing recommendation algorithm
3. **Group B (Treatment)**: New recommendation algorithm
4. **Measure** key metrics for each group
5. **Compare** to determine if treatment is better

**Goal**: Isolate causal effect of algorithm change on user behavior.

---

### The A/B Test Process

```
┌─────────────────────────────────────────┐
│   1. FORMULATE HYPOTHESIS                │
│   "New model will increase watch time"   │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────┴──────────────────────┐
│   2. CHOOSE METRICS                      │
│   Primary: Watch time/user              │
│   Secondary: CTR, retention, revenue     │
│   Guardrail: Latency, error rate        │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────┴──────────────────────┐
│   3. DETERMINE SAMPLE SIZE               │
│   Power analysis: Need N users          │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────┴──────────────────────┐
│   4. RANDOMIZE USERS                     │
│   50% Control, 50% Treatment             │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────┴──────────────────────┐
│   5. RUN EXPERIMENT                      │
│   Duration: 1-4 weeks (typically)        │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────┴──────────────────────┐
│   6. ANALYZE RESULTS                     │
│   Statistical significance testing       │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────┴──────────────────────┐
│   7. DECIDE                              │
│   Ship, Iterate, or Kill                 │
└──────────────────────────────────────────┘
```

---

## Step 1: Formulate Hypothesis

### Good Hypotheses

**Specific and measurable**:
- ✅ "New model will increase average watch time by at least 5%"
- ❌ "New model is better"

**Tied to business goals**:
- ✅ "Personalized thumbnails will increase CTR by 10%"
- ❌ "Matrix factorization is cool, let's use it"

**Realistic**:
- ✅ "Reducing latency from 200ms to 100ms will increase engagement"
- ❌ "This will 10x our revenue" (probably not)

---

## Step 2: Choose Metrics

### Metric Types

#### 1. **Primary Metrics** (Decision Criteria)

**What you're optimizing for**. Examples:

**E-Commerce**:
- Revenue per user
- Conversion rate
- Items purchased per visit

**Streaming** (Netflix, YouTube):
- Watch time per session
- Videos watched per session
- Session length

**Social Media**:
- Time spent on platform
- Posts liked/shared
- Comments per session

**Choose ONE primary metric** to avoid ambiguity.

---

#### 2. **Secondary Metrics** (Supporting Evidence)

**Additional signals** to understand impact:
- Click-through rate (CTR)
- Like rate
- Share rate
- Completion rate (for videos)
- Diversity of content consumed

**Not used for decision**, but provide context.

---

#### 3. **Guardrail Metrics** (Safety Checks)

**Must not degrade**:
- **Latency**: Page load time, recommendation response time
- **Error rate**: 4xx/5xx errors
- **User complaints**: Support tickets, "not interested" clicks
- **Revenue**: Can't tank revenue for engagement

**If guardrail violated → don't ship**, even if primary metric improves.

---

### Example Metric Framework (YouTube)

| Type | Metric | Target |
|------|--------|--------|
| **Primary** | Watch time per user | +2% vs. control |
| **Secondary** | CTR | Maintain or improve |
| **Secondary** | Videos per session | +5% |
| **Guardrail** | p99 latency | <200ms |
| **Guardrail** | Revenue per user | No degradation |

---

## Step 3: Sample Size Calculation

### Why Sample Size Matters

**Too small**: Can't detect real effects (underpowered)

**Too large**: Wastes resources, slows iteration

**Goal**: Minimum sample size to detect meaningful effect with confidence.

---

### Power Analysis Formula

For comparing two proportions (e.g., CTR):

$$n = \frac{(z_{\alpha/2} + z_{\beta})^2 \cdot 2\bar{p}(1-\bar{p})}{\delta^2}$$

where:
- $n$ = sample size per group
- $\alpha$ = significance level (typically 0.05 for 95% confidence)
- $\beta$ = desired power (typically 0.2 for 80% power)
- $z_{\alpha/2}$ = 1.96 (for $\alpha = 0.05$)
- $z_{\beta}$ = 0.84 (for 80% power)
- $\bar{p}$ = baseline proportion (e.g., current CTR = 0.10)
- $\delta$ = minimum detectable effect (e.g., 0.01 for 1% absolute increase)

---

### Example Calculation

**Scenario**: Test if new recommendation algorithm increases CTR.

**Given**:
- Current CTR ($\bar{p}$): 10% = 0.10
- Want to detect: 1% absolute increase (0.10 → 0.11)
- $\delta = 0.01$
- $\alpha = 0.05$ (95% confidence)
- $\beta = 0.2$ (80% power)

**Calculate**:
$$n = \frac{(1.96 + 0.84)^2 \cdot 2 \cdot 0.10 \cdot 0.90}{(0.01)^2}$$

$$= \frac{7.84 \cdot 0.18}{0.0001} = \frac{1.41}{0.0001} = 14,112$$

**Result**: Need **~14,000 users per group** (28,000 total).

---

### Sample Size for Continuous Metrics (Watch Time, Revenue)

$$n = \frac{2(z_{\alpha/2} + z_{\beta})^2 \cdot \sigma^2}{\delta^2}$$

where:
- $\sigma$ = standard deviation of metric
- $\delta$ = minimum detectable difference (in same units as $\sigma$)

**Example**:
- Metric: Watch time per session
- Mean: 20 minutes, Std Dev: 15 minutes
- Want to detect: 1 minute increase
- $\alpha = 0.05$, power = 80%

$$n = \frac{2(1.96 + 0.84)^2 \cdot 15^2}{1^2} = \frac{2 \cdot 7.84 \cdot 225}{1} = 3,528$$

**Result**: ~3,500 users per group.

---

### Online Calculators

**Tools**:
- Evan Miller's A/B test calculator: https://www.evanmiller.org/ab-testing/
- Optimizely's calculator
- Google Optimize

**Input**: Baseline rate, desired lift, confidence, power → Get sample size.

---

## Step 4: Randomization

### User-Level Randomization

**Most common**: Assign users to control or treatment.

**Process**:
```python
import hashlib

def get_variant(user_id, experiment_id, split=0.5):
    """
    Deterministic randomization via hashing.

    Args:
        user_id: User identifier
        experiment_id: Experiment name (e.g., "rec_model_v2")
        split: % of users in treatment (default 0.5 = 50%)

    Returns:
        'control' or 'treatment'
    """
    hash_input = f"{user_id}_{experiment_id}".encode()
    hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
    probability = (hash_value % 10000) / 10000  # Normalize to [0, 1]

    return 'treatment' if probability < split else 'control'
```

**Why hashing?**
- **Deterministic**: Same user always gets same variant
- **Uniform**: Evenly distributes users
- **Independent**: Different experiments independent

---

### Session-Level vs. User-Level

**Session-level**:
- User might see control in one session, treatment in another
- **Pros**: More data points
- **Cons**: Confusing user experience, cross-contamination

**User-level** (recommended):
- User always sees same variant
- **Pros**: Consistent experience, clearer causality
- **Cons**: Fewer data points (but more reliable)

---

### Stratification (Advanced)

**Ensure balance** across important dimensions:
- Geographic region
- Device type (mobile vs. desktop)
- User tenure (new vs. existing)

**Why?** Prevent imbalance that could confound results.

**Example**: If treatment gets more mobile users (who watch less), might incorrectly conclude treatment is worse.

---

## Step 5: Run the Experiment

### Duration

**How long to run?**

**Factors**:
1. **Sample size**: Until you reach calculated $n$
2. **Weekly cycles**: Run for full weeks (avoid day-of-week effects)
3. **Novelty effects**: Allow time for users to adapt (1-2 weeks)

**Typical duration**: 1-4 weeks

**Example**:
- Need 10,000 users per group
- Daily active users: 5,000
- 50/50 split → 2,500 per day per group
- **Duration**: $10,000 / 2,500 = 4$ days → Run for 1 week (full cycle)

---

### Common Mistakes to Avoid

#### 1. **Peeking** (Early Stopping)

**Problem**: Checking results multiple times, stopping when significant.

**Why bad?** Inflates false positive rate (Type I error).

**Solution**:
- Pre-commit to sample size and duration
- Use sequential testing methods (e.g., always-valid p-values) if must peek

---

#### 2. **Changing Traffic Allocation Mid-Test**

**Problem**: Increase treatment to 75% after seeing good results.

**Why bad?** Violates randomization, introduces bias.

**Solution**: Commit to allocation, don't change during test.

---

#### 3. **Not Accounting for Seasonality**

**Problem**: Run test during holiday season, conclude effect is permanent.

**Solution**: Run for multiple weeks, include typical periods.

---

## Step 6: Analyze Results

### Statistical Tests

#### 1. **Two-Sample t-test** (Continuous Metrics)

**Use case**: Compare means (watch time, revenue, session length).

**Hypotheses**:
- $H_0$: $\mu_{treatment} = \mu_{control}$ (no difference)
- $H_1$: $\mu_{treatment} \neq \mu_{control}$ (difference exists)

**Test statistic**:
$$t = \frac{\bar{x}_{treatment} - \bar{x}_{control}}{\sqrt{\frac{s^2_{treatment}}{n_{treatment}} + \frac{s^2_{control}}{n_{control}}}}$$

**Decision**:
- If $p < 0.05$ (95% confidence): Reject $H_0$ (significant difference)
- Else: Fail to reject (no significant difference)

**Python**:
```python
from scipy import stats

control_data = [...]  # Watch times for control group
treatment_data = [...]  # Watch times for treatment group

t_stat, p_value = stats.ttest_ind(treatment_data, control_data)

if p_value < 0.05:
    print(f"Significant difference (p={p_value:.4f})")
else:
    print(f"No significant difference (p={p_value:.4f})")
```

---

#### 2. **Two-Proportion Z-test** (Binary Metrics)

**Use case**: Compare proportions (CTR, conversion rate).

**Test statistic**:
$$z = \frac{\hat{p}_{treatment} - \hat{p}_{control}}{\sqrt{\bar{p}(1-\bar{p})(\frac{1}{n_{treatment}} + \frac{1}{n_{control}})}}$$

where $\bar{p} = \frac{x_{treatment} + x_{control}}{n_{treatment} + n_{control}}$ (pooled proportion).

**Python**:
```python
from statsmodels.stats.proportion import proportions_ztest

count_control = 100  # # of clicks in control
count_treatment = 120  # # of clicks in treatment
nobs_control = 1000  # # of users in control
nobs_treatment = 1000  # # of users in treatment

z_stat, p_value = proportions_ztest(
    [count_treatment, count_control],
    [nobs_treatment, nobs_control]
)

print(f"CTR Control: {count_control/nobs_control:.2%}")
print(f"CTR Treatment: {count_treatment/nobs_treatment:.2%}")
print(f"p-value: {p_value:.4f}")
```

---

#### 3. **Bootstrap** (Robust Alternative)

**When to use**: Non-normal distributions, small samples.

**Process**:
1. Resample with replacement from control and treatment
2. Compute difference in means
3. Repeat 10,000 times
4. 95% CI: 2.5th and 97.5th percentiles

**Python**:
```python
import numpy as np

def bootstrap_diff(control, treatment, n_bootstrap=10000):
    diffs = []
    for _ in range(n_bootstrap):
        control_sample = np.random.choice(control, len(control), replace=True)
        treatment_sample = np.random.choice(treatment, len(treatment), replace=True)
        diffs.append(treatment_sample.mean() - control_sample.mean())

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return ci_low, ci_high

ci = bootstrap_diff(control_data, treatment_data)
print(f"95% CI for difference: ({ci[0]:.2f}, {ci[1]:.2f})")

if ci[0] > 0:
    print("Treatment is significantly better (CI excludes 0)")
```

---

### Effect Size

**Statistical significance ≠ Practical significance**

**Example**:
- Treatment increases CTR from 10.0% to 10.05%
- With 1M users, this is statistically significant (p < 0.001)
- But **business impact**: 0.05% increase → negligible

**Measure effect size**:
- **Absolute difference**: 10.05% - 10.0% = 0.05%
- **Relative lift**: $(10.05 - 10.0) / 10.0 = 0.5\%$

**Decision**: Is 0.5% lift worth the engineering effort?

---

## Step 7: Interpret and Decide

### Decision Matrix

| Scenario | Primary Metric | Guardrails | Decision |
|----------|---------------|-----------|----------|
| 1 | ✅ Significant improvement | ✅ All pass | **Ship it!** |
| 2 | ✅ Significant improvement | ❌ Latency increased | Optimize latency, then retest |
| 3 | ❌ No significant change | ✅ All pass | Don't ship |
| 4 | ❌ No significant change | ❌ Some fail | **Kill the experiment** |
| 5 | ❌ Significantly worse | - | **Kill immediately** |

---

### Real-World Example: Netflix

**Experiment**: Personalized artwork (different thumbnails per user).

**Hypothesis**: Personalized thumbnails will increase CTR and watch time.

**Results**:
- **CTR**: +20% (significant)
- **Watch time**: +3% (significant)
- **Latency**: No degradation
- **User surveys**: Positive feedback

**Decision**: Ship to 100% of users.

**Impact**: One of Netflix's most successful experiments.

---

## Advanced Topics

### 1. **Multi-Armed Bandits** (Adaptive A/B Testing)

**Problem with fixed A/B tests**: Equal traffic to control and (likely worse) treatment.

**Solution**: Multi-armed bandits
- Start with even split
- Gradually shift traffic to better variant
- Minimize "regret" (time spent on worse variant)

**Algorithms**:
- **Thompson Sampling**: Bayesian approach
- **UCB** (Upper Confidence Bound): Frequentist

**When to use**: High traffic, many variants, willing to trade off learning speed for regret.

---

### 2. **Interleaving**

**Combine rankings** from control and treatment in single result list.

**Example**:
- Control recommends: [A, B, C, D]
- Treatment recommends: [E, F, G, H]
- Show user: [A, E, B, F, C, G, D, H] (interleaved)
- Track which items clicked

**Advantage**: More sensitive than side-by-side A/B test.

**Used by**: Google Search, Bing.

---

### 3. **Heterogeneous Treatment Effects**

**Question**: Does treatment work better for some user segments?

**Analysis**:
- Segment by: New vs. existing users, mobile vs. desktop, region, etc.
- Run separate tests or analyze subgroups

**Example**:
- Overall: +2% watch time
- New users: +10% watch time
- Existing users: +0% watch time

**Insight**: Treatment works great for acquisition, not for retention.

---

## Common Pitfalls and How to Avoid

### 1. **Simpson's Paradox**

**Phenomenon**: Treatment wins in every subgroup but loses overall (or vice versa).

**Cause**: Confounding variables (e.g., user mix differs between groups).

**Solution**: Proper randomization, check for balance across key dimensions.

---

### 2. **Network Effects**

**Problem**: Users in control affected by users in treatment.

**Example**: Social network
- Treatment users see better recommendations → post more
- Control users see treatment users' posts → benefit indirectly

**Solution**: Cluster randomization (randomize by geography, network, etc.).

---

### 3. **Survivorship Bias**

**Problem**: Only measuring active users, ignoring those who left.

**Example**:
- Treatment annoys some users → they quit
- Remaining treatment users are engaged → metrics look good
- But overall: treatment caused churn

**Solution**: Include all users assigned to treatment, even if they churned.

---

## Summary

**A/B Testing for RecSys**:

1. **Formulate hypothesis**: Specific, measurable, business-aligned
2. **Choose metrics**: Primary (decision), secondary (context), guardrails (safety)
3. **Sample size**: Power analysis to detect meaningful effects
4. **Randomize**: User-level, deterministic (hashing)
5. **Run**: 1-4 weeks, avoid peeking and mid-test changes
6. **Analyze**: t-tests, z-tests, bootstrap, effect size
7. **Decide**: Ship, iterate, or kill based on results

**Key Principles**:
- Offline metrics guide, online tests decide
- Statistical significance ≠ practical significance
- Guard against latency, revenue degradation
- Account for seasonality, novelty effects

**Advanced**: Bandits, interleaving, heterogeneous effects

**Remember**: A/B testing is how you learn what actually works in production, not what works on last year's data.

---

## References

1. **Kohavi, R., et al. (2009)**. "Controlled experiments on the web: Survey and practical guide". *Data Mining and Knowledge Discovery*.
2. **Kohavi, R., & Longbotham, R. (2017)**. "Online Controlled Experiments and A/B Testing". *Encyclopedia of Machine Learning and Data Mining*.
3. **Netflix Tech Blog** (2016). "It's All A/About Testing: The Netflix Experimentation Platform".
