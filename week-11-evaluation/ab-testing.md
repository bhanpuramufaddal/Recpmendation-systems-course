# Week 11: A/B Testing for Recommendation Systems

## Learning Objectives

By the end of this lecture, you will be able to:
- Design and run A/B tests for recommendations
- Choose appropriate metrics and sample sizes
- Interpret results with statistical rigor
- Avoid common pitfalls in online evaluation
- Derive sample size requirements from first principles
- Understand and correct for the multiple testing problem

---

## The Opening Problem: Why Can't We Just Deploy and See What Happens?

*"Professor, why all this complexity? Why not just deploy the new algorithm and watch the numbers?"*

This is the most dangerous question in data science. Let me show you exactly why.

### The Uncontrolled Deployment Disaster

**Scenario**: Your team builds a new recommendation model. You deploy it Monday morning. By Friday, revenue is up 8%. Victory?

**Not so fast.** Consider what else happened that week:
- Marketing launched a 20% discount campaign
- A competitor's site went down for 2 days
- It was the week before Black Friday
- App store featured your app on Wednesday

**The fundamental problem**: You cannot isolate the effect of your change.

$$\text{Observed Change} = \underbrace{\text{Algorithm Effect}}_{\text{What you want}} + \underbrace{\text{Marketing Effect} + \text{Competitor Effect} + \text{Seasonality} + \text{App Store Effect} + \epsilon}_{\text{Confounders you can't separate}}$$

**Real case study**: A major e-commerce company deployed a "better" recommendation engine. Revenue increased 12%. Executives celebrated. Two months later, they discovered the increase happened because they accidentally showed more discount items. The algorithm was actually worse at personalization.

**The brutal truth**: Without a control group running simultaneously, you have no idea what caused any observed change.

### What the Control Group Actually Does

*"Okay, so we need comparison. But why random assignment?"*

Think about this carefully. Suppose you said: "Let's show the new algorithm to users who signed up this week, and keep the old one for existing users."

**Problem 1 - Selection bias**: New users behave differently than existing users. They're exploring, less loyal, different demographics. Any difference you observe might be new-vs-old users, not algorithm A vs B.

**Problem 2 - Temporal confounds**: Next week's users experience different world events than this week's users. Different competing products, different news cycles, different weather.

**The only solution**: Random assignment ensures that, on average, the two groups are identical in every way except the treatment. This is the *only* way to establish causation.

$$\mathbb{E}[\text{Confounders} | \text{Treatment}] = \mathbb{E}[\text{Confounders} | \text{Control}]$$

When this holds, any difference in outcomes must be caused by the treatment.

---

## Statistical Foundations: Why We Need A/B Tests

### First Principles Derivation

Let me build up the statistical framework from scratch.

**Setup**: We have a population of users. For each user $i$:
- $Y_i(1)$ = outcome if they receive treatment (new algorithm)
- $Y_i(0)$ = outcome if they receive control (old algorithm)

**The fundamental problem of causal inference**: We can only observe one of these for each user. We never see both.

**Definition - Treatment Effect for user $i$**:
$$\tau_i = Y_i(1) - Y_i(0)$$

**Definition - Average Treatment Effect (ATE)**:
$$\tau = \mathbb{E}[Y_i(1) - Y_i(0)] = \mathbb{E}[Y_i(1)] - \mathbb{E}[Y_i(0)]$$

**The key insight**: If we randomly assign users to treatment or control, then:

$$\mathbb{E}[Y_i | \text{Treatment Group}] = \mathbb{E}[Y_i(1)]$$
$$\mathbb{E}[Y_i | \text{Control Group}] = \mathbb{E}[Y_i(0)]$$

Therefore:
$$\hat{\tau} = \bar{Y}_{\text{treatment}} - \bar{Y}_{\text{control}}$$

is an unbiased estimator of the true ATE.

### Why Random Assignment is Essential

*"Can't we just match users on observable characteristics instead?"*

**The problem with matching**: You can only match on what you observe. There are always unobserved confounders.

**Example**: Match users by age, gender, location, past purchases. But you miss:
- Current mood
- Whether they just got paid
- If their kid is sick
- Their actual preferences (not revealed in data)

**Random assignment breaks all correlations**:
$$P(\text{Treatment} | X_{\text{observed}}, X_{\text{unobserved}}) = P(\text{Treatment}) = 0.5$$

This is why A/B tests are the "gold standard" for causal inference.

---

## Why A/B Testing? The Offline-Online Gap

### The Fundamental Problem

**Offline evaluation** (RMSE, Precision@K on historical data):
- Fast, cheap, reproducible
- May not reflect real user behavior
- Doesn't capture business metrics (revenue, retention)

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
+------------------------------------------+
|   1. FORMULATE HYPOTHESIS                 |
|   "New model will increase watch time"    |
+---------------------+--------------------+
                      |
+---------------------v--------------------+
|   2. CHOOSE METRICS                       |
|   Primary: Watch time/user               |
|   Secondary: CTR, retention, revenue      |
|   Guardrail: Latency, error rate         |
+---------------------+--------------------+
                      |
+---------------------v--------------------+
|   3. DETERMINE SAMPLE SIZE                |
|   Power analysis: Need N users           |
+---------------------+--------------------+
                      |
+---------------------v--------------------+
|   4. RANDOMIZE USERS                      |
|   50% Control, 50% Treatment              |
+---------------------+--------------------+
                      |
+---------------------v--------------------+
|   5. RUN EXPERIMENT                       |
|   Duration: 1-4 weeks (typically)         |
+---------------------+--------------------+
                      |
+---------------------v--------------------+
|   6. ANALYZE RESULTS                      |
|   Statistical significance testing        |
+---------------------+--------------------+
                      |
+---------------------v--------------------+
|   7. DECIDE                               |
|   Ship, Iterate, or Kill                  |
+------------------------------------------+
```

---

## Step 1: Formulate Hypothesis

### Good Hypotheses

**Specific and measurable**:
- "New model will increase average watch time by at least 5%"
- "Personalized thumbnails will increase CTR by 10%"

**Bad hypotheses**:
- "New model is better"
- "Matrix factorization is cool, let's use it"

**Realistic expectations**:
- "Reducing latency from 200ms to 100ms will increase engagement"
- Not: "This will 10x our revenue" (probably not)

---

## Step 2: Metric Selection - A Deep Dive

### Understanding Metric Types

#### 1. **Primary Metrics** (Decision Criteria)

**What you're optimizing for**. This is the ONE metric that determines ship/no-ship.

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

**Critical rule: Choose ONE primary metric** to avoid ambiguity and goal post moving.

---

#### 2. **Secondary Metrics** (Supporting Evidence)

**Additional signals** to understand impact:
- Click-through rate (CTR)
- Like rate
- Share rate
- Completion rate (for videos)
- Diversity of content consumed

**Not used for decision**, but provide context and help diagnose.

---

#### 3. **Guardrail Metrics** (Safety Checks)

**Must not degrade**:
- **Latency**: Page load time, recommendation response time
- **Error rate**: 4xx/5xx errors
- **User complaints**: Support tickets, "not interested" clicks
- **Revenue**: Can't tank revenue for engagement

**If guardrail violated, do not ship**, even if primary metric improves.

---

### The Overall Evaluation Criterion (OEC)

*"Professor, what if we have multiple metrics that matter equally? Revenue AND engagement?"*

**The OEC** is a single metric that combines multiple business objectives into one number.

**Design principles**:
1. **Must be measurable during experiment** (short-term proxy for long-term goals)
2. **Must be sensitive** (moves when you make real changes)
3. **Must be robust** (not easily gamed, not too noisy)

**Example OEC for streaming service**:
$$\text{OEC} = w_1 \cdot \text{Watch Time} + w_2 \cdot \text{Sessions/Week} - w_3 \cdot \text{Unsubscribes}$$

Where weights $w_i$ are determined by business value analysis.

**Example OEC for e-commerce**:
$$\text{OEC} = \text{Revenue} - \text{Refunds} + 0.1 \cdot \text{Customer Lifetime Value Prediction}$$

**The hard part**: Choosing weights. This requires deep understanding of business trade-offs.

**Netflix example**: Their OEC includes:
- Retention probability (will they stay subscribed?)
- Engagement (hours watched)
- Diversity (are they discovering new content?)

Weights determined by modeling how each affects long-term subscription value.

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

## Step 3: Sample Size Calculation - Complete Derivation

### Why Sample Size Matters

**Too small**: Can't detect real effects (underpowered)

**Too large**: Wastes resources, slows iteration

**Goal**: Minimum sample size to detect meaningful effect with confidence.

---

### Power Analysis: First Principles Derivation

*"Where does the sample size formula come from?"*

Let me derive it step by step.

**Setup for comparing two proportions** (e.g., CTR):
- Control proportion: $p_c$ (baseline CTR)
- Treatment proportion: $p_t = p_c + \delta$ (if treatment works)
- Null hypothesis: $H_0: p_t = p_c$
- Alternative: $H_1: p_t \neq p_c$

**Step 1: The test statistic**

Under $H_0$, the difference in sample proportions is approximately normal:
$$\hat{p}_t - \hat{p}_c \sim N\left(0, \sqrt{\frac{p_c(1-p_c)}{n} + \frac{p_c(1-p_c)}{n}}\right) = N\left(0, \sqrt{\frac{2p_c(1-p_c)}{n}}\right)$$

**Step 2: Type I error (false positive)**

We reject $H_0$ when:
$$|\hat{p}_t - \hat{p}_c| > z_{\alpha/2} \cdot \sqrt{\frac{2\bar{p}(1-\bar{p})}{n}}$$

where $\bar{p} = (p_t + p_c)/2 \approx p_c$ and $\alpha$ is significance level.

**Step 3: Type II error (false negative)**

Under $H_1$ (true effect $\delta$), the difference has distribution:
$$\hat{p}_t - \hat{p}_c \sim N\left(\delta, \sqrt{\frac{2p(1-p)}{n}}\right)$$

We fail to detect when the observed difference is below threshold:
$$P(\text{Type II Error}) = P\left(\hat{p}_t - \hat{p}_c < z_{\alpha/2} \cdot \text{SE} \mid H_1\right) = \beta$$

**Step 4: Solve for n**

At the critical point:
$$\delta = z_{\alpha/2} \cdot \sqrt{\frac{2\bar{p}(1-\bar{p})}{n}} + z_\beta \cdot \sqrt{\frac{2\bar{p}(1-\bar{p})}{n}}$$

Solving for $n$:
$$n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\bar{p}(1-\bar{p})}{\delta^2}$$

**This is our fundamental formula!**

---

### Understanding the Components

| Term | Value (typical) | Meaning |
|------|-----------------|---------|
| $z_{\alpha/2}$ | 1.96 ($\alpha=0.05$) | How sure we need to be it's not noise |
| $z_\beta$ | 0.84 ($\beta=0.20$, power=80%) | How sure we need to be we'll detect real effect |
| $\bar{p}$ | e.g., 0.05 | Baseline metric value |
| $\delta$ | e.g., 0.005 | Minimum effect we care about |

**Key insight**: Sample size is proportional to $1/\delta^2$. Detecting half the effect size requires 4x the sample!

---

### Complete Numerical Example: A/B Test Walkthrough

**Scenario**: You work at a content platform. You've built a new recommendation algorithm and want to test if it improves click-through rate.

**Given**:
- Daily active users: 100,000
- Current CTR (baseline): 5% = 0.05
- Want to detect: 10% relative lift (i.e., CTR from 5% to 5.5%)
- Significance level: $\alpha = 0.05$ (95% confidence)
- Power: 80% ($\beta = 0.20$)

**Step 1: Calculate minimum detectable effect**
$$\delta = 0.055 - 0.05 = 0.005 \text{ (absolute difference)}$$

**Step 2: Determine pooled proportion**
$$\bar{p} = \frac{0.05 + 0.055}{2} = 0.0525 \approx 0.05$$

**Step 3: Apply sample size formula**
$$n = \frac{(1.96 + 0.84)^2 \cdot 2 \cdot 0.05 \cdot 0.95}{(0.005)^2}$$

$$n = \frac{(2.80)^2 \cdot 2 \cdot 0.0475}{0.000025}$$

$$n = \frac{7.84 \cdot 0.095}{0.000025}$$

$$n = \frac{0.7448}{0.000025} = 29,792$$

**Result**: Need approximately **30,000 users per group** (60,000 total).

**Step 4: Calculate experiment duration**
- Total users needed: 60,000
- 50/50 split means all DAU can participate
- With 100,000 DAU: Need 60,000/100,000 = 0.6 days minimum
- **But**: Account for weekly cycles, run at least 1-2 full weeks
- **Recommended duration**: 7-14 days

**Step 5: Simulate the experiment**

Let's walk through what happens with 10,000 users per group:

```python
import numpy as np
from scipy import stats

# Simulation parameters
n_control = 10000
n_treatment = 10000
p_control = 0.05      # 5% baseline CTR
p_treatment = 0.055   # 5.5% treatment CTR (10% relative lift)

# Simulate one experiment
np.random.seed(42)
clicks_control = np.random.binomial(n_control, p_control)
clicks_treatment = np.random.binomial(n_treatment, p_treatment)

# Observed CTRs
ctr_control = clicks_control / n_control      # 0.0508
ctr_treatment = clicks_treatment / n_treatment  # 0.0549

# Z-test
pooled_p = (clicks_control + clicks_treatment) / (n_control + n_treatment)
se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n_control + 1/n_treatment))
z = (ctr_treatment - ctr_control) / se  # z = 1.34
p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # p = 0.18

# Result: NOT significant! We're underpowered with only 10,000 per group
```

**The punchline**: With only 10,000 users per group (underpowered), even when the treatment truly works, we often get p > 0.05!

**With 30,000 users per group** (properly powered):
```python
n_control = 30000
n_treatment = 30000

clicks_control = np.random.binomial(n_control, p_control)     # ~1500
clicks_treatment = np.random.binomial(n_treatment, p_treatment) # ~1650

# Now z ≈ 2.3, p ≈ 0.02 - Significant!
```

---

### Sample Size for Continuous Metrics (Watch Time, Revenue)

$$n = \frac{2(z_{\alpha/2} + z_\beta)^2 \cdot \sigma^2}{\delta^2}$$

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

**Input**: Baseline rate, desired lift, confidence, power -> Get sample size.

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
- 50/50 split: 2,500 per day per group
- **Duration**: $10,000 / 2,500 = 4$ days -> Run for 1 week (full cycle)

---

## The Multiple Testing Problem: A Critical Pitfall

### Derivation: Why 20 Tests Gives 64% False Positive Rate

*"Professor, we run lots of experiments. What's the problem?"*

**Setup**: You have 20 different metrics. You run A/B test, declare "significant" if p < 0.05.

**Question**: If there's NO real effect (null is true), what's the probability of at least one false positive?

**Derivation**:

For each test, under the null:
$$P(\text{false positive on test } i) = \alpha = 0.05$$

For 20 independent tests:
$$P(\text{no false positives}) = (1 - \alpha)^{20} = (0.95)^{20} = 0.358$$

Therefore:
$$P(\text{at least one false positive}) = 1 - 0.358 = 0.642$$

**Result**: 64.2% chance of false positive!

This is called the **Family-Wise Error Rate (FWER)**.

**General formula**:
$$\text{FWER} = 1 - (1 - \alpha)^m$$

where $m$ = number of tests.

| Number of Tests | FWER (with $\alpha = 0.05$) |
|-----------------|------------------------------|
| 1 | 5.0% |
| 5 | 22.6% |
| 10 | 40.1% |
| 20 | 64.2% |
| 50 | 92.3% |
| 100 | 99.4% |

---

### Bonferroni Correction

**The simplest solution**: Divide $\alpha$ by number of tests.

$$\alpha_{\text{adjusted}} = \frac{\alpha}{m}$$

**Example**: 20 tests, want FWER = 0.05
$$\alpha_{\text{adjusted}} = \frac{0.05}{20} = 0.0025$$

Only declare significant if p < 0.0025.

**Verification**:
$$\text{FWER} \leq m \cdot \alpha_{\text{adjusted}} = 20 \cdot 0.0025 = 0.05$$ (checkmark)

**Problem**: Very conservative! Hard to detect real effects.

---

### Better Approaches

**1. Benjamini-Hochberg (False Discovery Rate)**

Controls expected proportion of false discoveries, not probability of any false discovery.

**Procedure**:
1. Order p-values: $p_{(1)} \leq p_{(2)} \leq ... \leq p_{(m)}$
2. Find largest $k$ where $p_{(k)} \leq \frac{k}{m} \alpha$
3. Reject all hypotheses $1, 2, ..., k$

**2. Pre-registration**

Specify primary metric in advance. Multiple secondary metrics are exploratory.

**3. Hierarchical testing**

Only test secondary metrics if primary is significant.

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

**Statistical significance does not equal practical significance**

**Example**:
- Treatment increases CTR from 10.0% to 10.05%
- With 1M users, this is statistically significant (p < 0.001)
- But **business impact**: 0.05% increase -> negligible

**Measure effect size**:
- **Absolute difference**: 10.05% - 10.0% = 0.05%
- **Relative lift**: $(10.05 - 10.0) / 10.0 = 0.5\%$

**Decision**: Is 0.5% lift worth the engineering effort?

---

## Step 7: Interpret and Decide

### Decision Matrix

| Scenario | Primary Metric | Guardrails | Decision |
|----------|---------------|-----------|----------|
| 1 | Significant improvement | All pass | **Ship it!** |
| 2 | Significant improvement | Latency increased | Optimize latency, then retest |
| 3 | No significant change | All pass | Don't ship |
| 4 | No significant change | Some fail | **Kill the experiment** |
| 5 | Significantly worse | - | **Kill immediately** |

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

## What Can Go Wrong: A Comprehensive Guide to A/B Test Failures

### 1. Simpson's Paradox

**Definition**: A phenomenon where a trend appears in different groups of data but disappears or reverses when groups are combined.

**Mathematical setup**:
- Segment 1: Treatment wins (60% vs 50%)
- Segment 2: Treatment wins (90% vs 80%)
- Combined: Treatment loses (65% vs 70%)

**How is this possible?**

| Segment | Treatment | Control |
|---------|-----------|---------|
| Segment 1 (Low converters) | 600/1000 = 60% | 50/100 = 50% |
| Segment 2 (High converters) | 90/100 = 90% | 800/1000 = 80% |
| **Combined** | 690/1100 = 62.7% | 850/1100 = 77.3% |

**What happened**: Treatment group had more low-converters. The group composition changed!

**Root cause**: Randomization failed or there was selection bias.

**Solution**:
1. Check balance on key covariates before analysis
2. Stratified randomization
3. Report segment-level results

---

### 2. Novelty Effects

**Definition**: Users engage more with new features simply because they're new, not because they're better.

**Pattern**:
- Week 1: Treatment +15%
- Week 2: Treatment +8%
- Week 3: Treatment +2%
- Week 4: Treatment +0%

**Mechanism**:
- New UI attracts attention (curiosity)
- Users explore the new feature
- Once familiar, behavior reverts to baseline

**The danger**: If you stop experiment early, you ship something that provides no long-term value.

**Solutions**:
1. Run experiments for 2-4 weeks minimum
2. Analyze by user cohort (day of first exposure)
3. Wait for metrics to stabilize before concluding

---

### 3. Primacy Effects (Opposite of Novelty)

**Definition**: Users resist change initially but eventually prefer the new experience.

**Pattern**:
- Week 1: Treatment -10% (users hate change)
- Week 2: Treatment -5%
- Week 3: Treatment +2%
- Week 4: Treatment +8%

**The danger**: If you stop experiment early, you kill something that would have been successful.

**Solutions**:
1. Segment by user tenure with feature
2. Long-term holdout groups
3. Patience!

---

### 4. Interference Effects (SUTVA Violation)

**Definition**: Users in one group affect users in another group.

**The assumption we violate**: Stable Unit Treatment Value Assumption (SUTVA) - a user's outcome depends only on their own treatment assignment.

**Example - Social network**:
- Treatment users see better recommendations
- Treatment users post more content
- Control users see treatment users' posts
- Control users benefit from treatment!

**Result**: Underestimate treatment effect (both groups improve).

**Example - Marketplace**:
- Treatment sellers get better recommendations
- Treatment sellers capture more buyers
- Control sellers lose market share

**Result**: Overestimate treatment effect (zero-sum dynamics).

**Solutions**:
1. Cluster randomization (randomize by geography, network component)
2. Geo experiments
3. Synthetic control methods

---

### 5. Peeking (Early Stopping)

**The crime**: Checking results repeatedly and stopping when p < 0.05.

**Why it's bad**: Each peek is a statistical test. You're doing multiple testing!

**Simulation**:

```python
import numpy as np
from scipy import stats

def simulate_peeking(n_users=10000, n_peeks=10, true_effect=0):
    """
    Simulate an A/B test with peeking.
    true_effect=0 means null is true.
    """
    control = np.random.normal(100, 20, n_users)
    treatment = np.random.normal(100 + true_effect, 20, n_users)

    peek_points = np.linspace(100, n_users, n_peeks, dtype=int)

    for n in peek_points:
        _, p = stats.ttest_ind(treatment[:n], control[:n])
        if p < 0.05:
            return True  # "Significant" (but maybe false positive!)

    return False

# Run 1000 simulated experiments with NO real effect
false_positive_rate = np.mean([simulate_peeking(true_effect=0) for _ in range(1000)])
# Result: ~25% false positive rate instead of 5%!
```

**Solutions**:
1. Pre-commit to sample size, don't peek
2. Use sequential testing (O'Brien-Fleming bounds, always-valid p-values)
3. Bayesian methods with continuous monitoring

---

### 6. P-Hacking

**The crimes**:
1. Testing many metrics, reporting only significant ones
2. Trying many subgroups, reporting only significant ones
3. Removing "outliers" until p < 0.05
4. Transforming data until p < 0.05
5. Adding more data until p < 0.05

**Real example from published research**:

*"We found that listening to 'When I'm Sixty-Four' by the Beatles makes people younger!"*

(This was a satirical paper demonstrating p-hacking. They actually got p < 0.05 through data manipulation.)

**How to detect p-hacking**:
- P-values cluster just below 0.05
- Unusual subgroup analyses
- Post-hoc removal of data
- No pre-registration

**Solutions**:
1. Pre-registration of analysis plan
2. Report all metrics tested
3. Require replication
4. Adjust for multiple comparisons

---

### 7. Selection Bias in Experiment Entry

**The problem**: Who enters the experiment may not be random.

**Example**:
- Experiment triggers when user visits recommendations page
- Treatment changes what's recommended
- Some users never visit recommendations (never enter experiment)
- Treatment might cause MORE users to visit recommendations
- Those new users are different from existing ones

**Result**: Treatment group composition changes over time.

**Solutions**:
1. Intent-to-treat analysis (analyze all randomized users)
2. Instrument variable approaches
3. Careful triggering logic

---

## Socratic Deep Dive: Understanding Statistical Power

### "If Treatment is Truly Better, Why Might We Still See p > 0.05?"

*This is one of the most important questions in experimental design.*

**Let's think through this together.**

**Setting**: Treatment truly increases CTR from 5% to 5.5% (a real 10% relative lift).

**Question**: Why might our experiment fail to detect this?

**Reason 1: Insufficient sample size (low power)**

```
With n = 1,000 per group:
- Standard error of difference = sqrt(2 * 0.05 * 0.95 / 1000) = 0.0097
- True effect = 0.005
- Signal-to-noise ratio = 0.005 / 0.0097 = 0.52
- Expected z-score ≈ 0.52
- P(z > 1.96) ≈ 2%

Result: Only 2% chance of detecting real effect!
```

**Reason 2: High variance in metric**

If individual user behavior is highly variable, the noise drowns out the signal.

$$\text{Power} \propto \frac{\delta}{\sigma/\sqrt{n}}$$

High $\sigma$ requires larger $n$.

**Reason 3: Effect is real but smaller than expected**

You planned to detect 10% lift, but true lift is only 3%. Your experiment is underpowered for the actual effect.

**Reason 4: Heterogeneous effects**

Treatment helps some users (+20%), hurts others (-15%). Average effect is small, but aggregate hides important patterns.

**Reason 5: Bad luck**

Even with 80% power, you have 20% chance of Type II error. One in five properly-powered experiments will miss real effects!

**The fundamental trade-off**:

$$\text{Power} = 1 - \beta = \Phi\left(\frac{\delta\sqrt{n}}{\sigma\sqrt{2}} - z_{\alpha/2}\right)$$

To increase power:
- Increase sample size $n$ (expensive)
- Accept larger minimum detectable effect $\delta$ (less sensitive)
- Increase $\alpha$ (more false positives)
- Reduce variance $\sigma$ (better measurement)

---

### "Why Not Just Use a Very Large Sample Size to Guarantee Detection?"

**First answer**: Resources. Every user in experiment could be seeing a worse experience.

**Second answer**: You'll detect everything, including meaningless effects.

**Example**: With 10 million users per group:
- Minimum detectable effect < 0.1%
- You'll find "significant" differences that don't matter
- Every tiny implementation detail becomes "significant"

**Third answer**: Time. Large samples take longer to collect.

**The art of A/B testing**: Choose sample size to detect the smallest effect that would change your decision.

---

### "What If Our Assumptions Are Wrong?"

**Assumption 1**: Users are independent.

*When violated*: Social networks, marketplaces, shared accounts

*Solution*: Cluster-robust standard errors, cluster randomization

**Assumption 2**: Metric is normally distributed.

*When violated*: Revenue (heavy-tailed), counts (Poisson), time (zero-inflated)

*Solution*: Bootstrap, transform data, use appropriate distribution

**Assumption 3**: No time effects.

*When violated*: Seasonality, news events, product launches

*Solution*: Run longer, use difference-in-differences

**Assumption 4**: Stable treatment effect.

*When violated*: Learning effects, novelty/primacy effects

*Solution*: Segment by exposure time, long-term holdouts

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

## Summary

**A/B Testing for RecSys**:

1. **Formulate hypothesis**: Specific, measurable, business-aligned
2. **Choose metrics**: Primary (decision), secondary (context), guardrails (safety), OEC (unified)
3. **Sample size**: Power analysis to detect meaningful effects
4. **Randomize**: User-level, deterministic (hashing)
5. **Run**: 1-4 weeks, avoid peeking and mid-test changes
6. **Analyze**: t-tests, z-tests, bootstrap, effect size
7. **Decide**: Ship, iterate, or kill based on results

**Key Principles**:
- Offline metrics guide, online tests decide
- Statistical significance does not equal practical significance
- Guard against latency, revenue degradation
- Account for seasonality, novelty effects
- Correct for multiple testing

**What Can Go Wrong**:
- Simpson's paradox - check segment balance
- Novelty/primacy effects - run longer experiments
- Interference effects - cluster randomization
- Peeking - pre-commit to analysis plan
- P-hacking - pre-register, report everything

**Advanced**: Bandits, interleaving, heterogeneous effects

**Remember**: A/B testing is how you learn what actually works in production, not what works on last year's data.

---

## Key Takeaways for Your Career

1. **Always A/B test** before shipping to production
2. **Pre-register** your analysis plan to avoid p-hacking
3. **Calculate sample size** before starting - don't just run until significant
4. **One primary metric** - don't move goalposts
5. **Watch for interference** - especially in social/marketplace products
6. **Statistical significance is necessary but not sufficient** - effect size matters
7. **Run long enough** - novelty effects are real
8. **Correct for multiple testing** - or you will fool yourself

---

## References

1. **Kohavi, R., et al. (2009)**. "Controlled experiments on the web: Survey and practical guide". *Data Mining and Knowledge Discovery*.
2. **Kohavi, R., & Longbotham, R. (2017)**. "Online Controlled Experiments and A/B Testing". *Encyclopedia of Machine Learning and Data Mining*.
3. **Netflix Tech Blog** (2016). "It's All A/About Testing: The Netflix Experimentation Platform".
4. **Kohavi, R., Tang, D., & Xu, Y. (2020)**. "Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing". Cambridge University Press.
5. **Deng, A., et al. (2013)**. "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data". *WSDM*.
