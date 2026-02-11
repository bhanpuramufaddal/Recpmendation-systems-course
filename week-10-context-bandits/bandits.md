# Week 10: Multi-Armed Bandits

## Overview

**Multi-Armed Bandits (MAB)** address the **exploration-exploitation dilemma**:
- **Exploit**: Show items we know users like - maximize immediate reward
- **Explore**: Try new items to learn preferences - maximize long-term reward

**Bandit analogy**: Slot machines (one-armed bandits) with unknown payout rates.

**Goal**: Maximize cumulative reward over time while learning which "arms" (items) are best.

**Applications**:
- A/B testing (which version performs better?)
- Content recommendations (which article to show?)
- Ad placement (which ad gets highest CTR?)

This document covers multi-armed bandit algorithms for recommendations.

---

## Learning Objectives

By the end of this section, you will:
- Understand exploration-exploitation tradeoff
- Implement epsilon-greedy, UCB, Thompson Sampling
- Apply bandits to recommendation systems
- Measure regret and performance
- Choose appropriate bandit algorithms

---

## The Opening Problem: Why Pure Exploitation Fails

*"Let me start with a question that seems obvious at first, but will reveal something profound..."*

### The Deceptively Simple Question

**Why doesn't always showing the best-so-far work?**

*"Imagine you run a news website. You have 10 articles to show on your homepage. After the first day, you've collected some data:"*

| Article | Impressions | Clicks | CTR |
|---------|-------------|--------|-----|
| A | 100 | 8 | 8.0% |
| B | 50 | 6 | 12.0% |
| C | 20 | 3 | 15.0% |
| D | 10 | 2 | 20.0% |
| E | 5 | 1 | 20.0% |
| F | 5 | 0 | 0.0% |
| G | 5 | 0 | 0.0% |
| H | 3 | 1 | 33.3% |
| I | 1 | 0 | 0.0% |
| J | 1 | 1 | 100.0% |

*"A pure exploitation strategy says: 'Show Article J! It has 100% CTR!' But wait..."*

**Stop and think**: What's wrong with this reasoning?

*"Article J was shown exactly once, and that one person clicked. Does this mean it's really 100% likely to be clicked? Of course not! We have almost no data."*

**The Core Insight**: We're confusing **estimated reward** with **true reward**. With limited samples, our estimates are noisy.

*"Article H looks great at 33.3% CTR, but it only has 3 impressions. Meanwhile, Article A has 100 impressions and 8% CTR - we're much more confident about that number."*

**The Exploitation Trap**: If we only show Article J going forward, we:
1. Never learn if Articles B, C, D, or H might actually be better
2. Base our entire strategy on a single lucky (or unlucky) observation
3. Miss articles that truly deserve more traffic

*"This is exactly the exploration-exploitation dilemma. Today, we'll develop principled solutions."*

---

## The Exploration-Exploitation Dilemma

### Problem Setup

**Scenario**: News website with 10 articles. Which to show users?

**Naive approach**: Show most-clicked article - **exploit**.

**Problem**: What if there's a better article you haven't tried?

**Better approach**: Sometimes try less-popular articles - **explore**.

---

## Regret: Quantifying the Cost of Learning

### The Intuition Behind Regret

*"Before we dive into algorithms, we need a way to measure how well we're doing. This is where regret comes in."*

**Thought experiment**: Imagine an oracle that knows the true click-through rate of every article. The oracle would always show the best article and get the maximum possible clicks.

*"We don't have that oracle. We have to learn while we earn. Regret measures the price we pay for this learning."*

### Step-by-Step Derivation of the Regret Formula

**Step 1: Define the optimal reward**

If we knew the best arm had reward rate $\mu^* = 0.7$ (70% CTR), then over $T$ rounds, the optimal cumulative reward is:

$$\text{Optimal Reward} = T \cdot \mu^*$$

*"With perfect knowledge over 100 rounds: $100 \times 0.7 = 70$ expected clicks."*

**Step 2: Define our actual reward**

We don't know which arm is best, so we explore and sometimes pick suboptimal arms. Our actual cumulative reward is:

$$\text{Actual Reward} = \sum_{t=1}^T r_t$$

where $r_t$ is the reward we actually received at round $t$.

**Step 3: Regret is the difference**

$$\text{Regret}(T) = T \cdot \mu^* - \sum_{t=1}^T r_t$$

*"Regret answers: 'How many clicks did we lose by not knowing the best article from the start?'"*

### Numerical Example: Tracking Regret Over 100 Rounds

*"Let's make this concrete with a worked example."*

**Setup**:
- 3 articles with true CTRs: A = 0.3, B = 0.5, C = 0.7 (C is best, so $\mu^* = 0.7$)
- We don't know these values initially

| Round | Action | True CTR | Reward | Optimal Reward | Cumulative Regret |
|-------|--------|----------|--------|----------------|-------------------|
| 1 | Try A | 0.3 | 0.3 | 0.7 | 0.4 |
| 2 | Try B | 0.5 | 0.5 | 0.7 | 0.6 |
| 3 | Try C | 0.7 | 0.7 | 0.7 | 0.6 |
| 4 | Try A | 0.3 | 0.3 | 0.7 | 1.0 |
| 5 | Pick C | 0.7 | 0.7 | 0.7 | 1.0 |
| ... | ... | ... | ... | ... | ... |
| 50 | Pick C | 0.7 | 0.7 | 0.7 | ~5.0 |
| 100 | Pick C | 0.7 | 0.7 | 0.7 | ~5.0 |

*"Notice how regret accumulates quickly during exploration (rounds 1-4) but then flattens once we've learned C is best. A good algorithm has regret that grows slowly - ideally sublinearly in T."*

**Key Insight**: We want $\text{Regret}(T) = o(T)$, meaning regret grows slower than linearly. If regret grows linearly, we're doing no better than random guessing!

---

## epsilon-Greedy

### Algorithm

**Idea**: With probability $\varepsilon$, explore (random arm); otherwise, exploit (best known arm).

**Pseudo-code**:
```
For each round t:
  With probability epsilon:
    Choose random arm (explore)
  With probability 1-epsilon:
    Choose arm with highest estimated reward (exploit)

  Observe reward r_t
  Update reward estimates
```

**Parameters**:
- $\varepsilon = 0$: Pure exploitation (greedy)
- $\varepsilon = 1$: Pure exploration (random)
- $\varepsilon = 0.1$: Typical value (10% exploration)

---

### Why epsilon-Greedy Works: Expected Regret Analysis

*"Let's derive the expected regret of epsilon-greedy to understand its behavior."*

**Setup**: Assume the best arm has reward $\mu^*$ and the gap to the second-best arm is $\Delta$.

**Per-round expected regret**:
- With probability $1 - \varepsilon$: we exploit (ideally picking best arm) - regret $\approx 0$
- With probability $\varepsilon$: we explore uniformly - expected regret $\approx \frac{\varepsilon \cdot \Delta}{2}$ (on average)

**Over T rounds**:

$$\mathbb{E}[\text{Regret}(T)] \approx \varepsilon \cdot T \cdot \bar{\Delta}$$

where $\bar{\Delta}$ is the average suboptimality gap.

*"Here's the problem: this is O(T)! Regret grows linearly with time."*

**Why O(T)?** Even after 1 million rounds, we're still exploring 10% of the time. We keep paying the exploration cost forever.

**The Fix**: Decay epsilon over time!

With $\varepsilon_t = 1/t$:

$$\mathbb{E}[\text{Regret}(T)] = O(\log T)$$

*"Now regret only grows logarithmically - much better! After 100 rounds we have about 4.6 regret, after 1000 rounds only about 6.9 regret."*

---

### Implementation

```python
import numpy as np

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        """
        epsilon-greedy bandit algorithm.

        n_arms: Number of arms (items)
        epsilon: Exploration probability
        """
        self.n_arms = n_arms
        self.epsilon = epsilon

        # Estimated rewards (average)
        self.estimates = np.zeros(n_arms)

        # Counts (how many times each arm pulled)
        self.counts = np.zeros(n_arms)

    def select_arm(self):
        """
        Select an arm using epsilon-greedy policy.
        """
        if np.random.rand() < self.epsilon:
            # Explore: random arm
            return np.random.randint(self.n_arms)
        else:
            # Exploit: best arm
            return np.argmax(self.estimates)

    def update(self, arm, reward):
        """
        Update estimates after observing reward.
        """
        self.counts[arm] += 1
        n = self.counts[arm]

        # Incremental average
        self.estimates[arm] += (reward - self.estimates[arm]) / n


# Example
np.random.seed(42)
n_arms = 5
true_rewards = [0.2, 0.5, 0.3, 0.7, 0.4]  # Unknown to algorithm

bandit = EpsilonGreedy(n_arms, epsilon=0.1)

# Simulate 1000 rounds
total_reward = 0
for t in range(1000):
    arm = bandit.select_arm()

    # Simulate reward (Bernoulli)
    reward = 1 if np.random.rand() < true_rewards[arm] else 0

    bandit.update(arm, reward)
    total_reward += reward

print(f"Total reward: {total_reward}")
print(f"Estimated rewards: {bandit.estimates}")
print(f"True rewards: {true_rewards}")
print(f"Arm counts: {bandit.counts}")
```

---

### Decaying epsilon

**Problem**: Fixed $\varepsilon$ - always explores (even after learning).

**Solution**: Decay $\varepsilon$ over time.

$$\varepsilon_t = \frac{1}{1 + t}$$

or

$$\varepsilon_t = \varepsilon_0 \cdot e^{-\lambda t}$$

**Effect**: Explore more early, exploit more later.

```python
class DecayingEpsilonGreedy(EpsilonGreedy):
    def __init__(self, n_arms, epsilon_init=1.0, decay_rate=0.01):
        super().__init__(n_arms, epsilon=epsilon_init)
        self.epsilon_init = epsilon_init
        self.decay_rate = decay_rate
        self.t = 0

    def select_arm(self):
        # Update epsilon
        self.epsilon = self.epsilon_init * np.exp(-self.decay_rate * self.t)
        self.t += 1

        return super().select_arm()
```

---

## Upper Confidence Bound (UCB)

### A Socratic Introduction

*"Before I show you the UCB algorithm, let me ask YOU a question..."*

**Pause and think**: You have estimated rewards for each arm. epsilon-greedy just adds randomness. But randomness is wasteful - you might re-explore arms you already know are bad.

**What would YOU add to the estimated mean to decide which arm to try next?**

*"Think about what information you have: the estimated mean, and how many times you've pulled each arm..."*

*"If you said 'something that reflects uncertainty' or 'a bonus for less-tried arms' - you've got it! That's exactly the insight behind UCB."*

---

### Intuition

**epsilon-greedy problem**: Explores randomly - wastes time on obviously bad arms.

**UCB idea**: Explore arms with **high uncertainty** (less-tried arms).

**Principle**: "Optimism in the face of uncertainty" - assume arms could be good until proven otherwise.

---

### UCB1 Algorithm: Full Derivation from Hoeffding's Inequality

*"Let's derive UCB from first principles. This is beautiful mathematics that leads to a practical algorithm."*

**Step 1: The Concentration Problem**

We observe rewards $r_1, r_2, ..., r_n$ from an arm. The sample mean is:

$$\hat{\mu} = \frac{1}{n}\sum_{i=1}^n r_i$$

*"Question: How close is the sample mean to the true mean mu?"*

**Step 2: Hoeffding's Inequality**

For bounded random variables $r_i \in [0, 1]$:

$$P(|\hat{\mu} - \mu| \geq \epsilon) \leq 2e^{-2n\epsilon^2}$$

*"This says: the probability that our estimate is far from the truth decreases exponentially as we collect more samples."*

**Step 3: Inverting for a Confidence Bound**

We want a bound that holds with high probability. Set:

$$2e^{-2n\epsilon^2} = \delta$$

Solving for $\epsilon$:

$$\epsilon = \sqrt{\frac{\ln(2/\delta)}{2n}}$$

*"With probability at least 1 - delta, the true mean is within epsilon of our estimate."*

**Step 4: Making it Time-Dependent**

We want the bound to hold across all time steps. Set $\delta = 1/t^2$ (this choice makes the sum of failure probabilities finite):

$$\epsilon = \sqrt{\frac{\ln(2t^2)}{2n}} = \sqrt{\frac{2\ln t + \ln 2}{2n}} \approx \sqrt{\frac{2\ln t}{n}}$$

**Step 5: The UCB Formula**

The upper confidence bound for arm $i$ at time $t$ is:

$$\text{UCB}_i(t) = \hat{\mu}_i + \sqrt{\frac{2\ln t}{n_i}}$$

**Selection rule**:
$$\text{arm}_t = \arg\max_{i} \left[ \hat{\mu}_i + \sqrt{\frac{2 \ln t}{n_i}} \right]$$

where:
- $\hat{\mu}_i$ = estimated reward of arm $i$
- $t$ = total rounds so far
- $n_i$ = times arm $i$ pulled
- $\sqrt{\frac{2 \ln t}{n_i}}$ = **confidence bonus** (higher for less-tried arms)

**Interpretation**: Choose arm with highest **upper confidence bound**.

*"The genius of UCB: it automatically balances exploration and exploitation. Arms with high uncertainty get explored. Arms with high estimated reward get exploited. Arms that are confidently bad get ignored."*

---

### UCB Regret Bound

**Theorem**: UCB1 achieves regret:

$$\text{Regret}(T) = O\left(\sqrt{KT\ln T}\right)$$

where $K$ is the number of arms.

*"This is much better than epsilon-greedy's O(T)! UCB gets roughly sqrt(T) regret, which means the per-round regret vanishes over time."*

---

### Implementation

```python
class UCB1:
    def __init__(self, n_arms):
        """
        UCB1 bandit algorithm.
        """
        self.n_arms = n_arms
        self.estimates = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        self.t = 0

    def select_arm(self):
        """
        Select arm using UCB1.
        """
        self.t += 1

        # Initially, try each arm once
        if self.t <= self.n_arms:
            return self.t - 1

        # UCB values
        ucb_values = self.estimates + np.sqrt(2 * np.log(self.t) / (self.counts + 1e-8))

        return np.argmax(ucb_values)

    def update(self, arm, reward):
        """
        Update estimates.
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        self.estimates[arm] += (reward - self.estimates[arm]) / n


# Example
bandit = UCB1(n_arms=5)

total_reward = 0
for t in range(1000):
    arm = bandit.select_arm()
    reward = 1 if np.random.rand() < true_rewards[arm] else 0
    bandit.update(arm, reward)
    total_reward += reward

print(f"UCB1 total reward: {total_reward}")
print(f"UCB1 estimates: {bandit.estimates}")
```

---

## Thompson Sampling

### Bayesian Approach

**Idea**: Maintain **probability distribution** over each arm's reward.

**Procedure**:
1. For each arm, maintain belief about reward (e.g., Beta distribution)
2. Sample from each arm's distribution
3. Choose arm with highest sample
4. Update belief based on observed reward

**Advantage**: Naturally balances exploration-exploitation via uncertainty.

---

### Why Sampling from the Posterior is Optimal: Probability Matching

*"Thompson Sampling seems almost too simple to work. Why does randomly sampling give optimal behavior?"*

**The Probability Matching Principle**

Thompson Sampling selects each arm with probability equal to the probability that arm is optimal:

$$P(\text{select arm } i) = P(\text{arm } i \text{ is best} | \text{data})$$

*"Think about what this means: if we're 70% sure arm 1 is best and 30% sure arm 2 is best, we pick arm 1 about 70% of the time."*

**Why This is Optimal: The Bayesian Argument**

1. **If we're very confident arm 1 is best** (say 99%), we exploit it 99% of the time. Good!

2. **If we're uncertain** (50-50 between arms 1 and 2), we explore both equally. This is exactly what we should do - gather information.

3. **As we gather data**, our posterior concentrates on the true best arm, so we automatically transition from exploration to exploitation.

**The Mathematical Magic**

For any arm $i$ with true reward $\mu_i$, Thompson Sampling selects it with probability:

$$P(\text{select } i) = \int P(\theta_i > \max_{j \neq i} \theta_j | \text{data}) \, d\theta$$

This integral weighs exploration against exploitation perfectly!

*"Thompson Sampling is 'embarrassingly simple' but theoretically optimal. It was proposed in 1933 but largely ignored until the 2010s when people realized how well it works in practice."*

---

### Beta Distribution

**For binary rewards** (click/no-click):

**Prior**: Beta($\alpha$, $\beta$)
- $\alpha$ = successes + 1
- $\beta$ = failures + 1

**Uniform prior**: Beta(1, 1)

**Update**: After observing reward $r \in \{0, 1\}$:
- If $r = 1$: $\alpha \leftarrow \alpha + 1$
- If $r = 0$: $\beta \leftarrow \beta + 1$

---

### Implementation

```python
class ThompsonSampling:
    def __init__(self, n_arms):
        """
        Thompson Sampling for Bernoulli bandits.
        """
        self.n_arms = n_arms

        # Beta distribution parameters
        self.alphas = np.ones(n_arms)  # Successes + 1
        self.betas = np.ones(n_arms)   # Failures + 1

    def select_arm(self):
        """
        Sample from Beta distributions and choose highest.
        """
        # Sample from each arm's Beta distribution
        samples = np.random.beta(self.alphas, self.betas)

        return np.argmax(samples)

    def update(self, arm, reward):
        """
        Update Beta distribution for arm.
        """
        if reward > 0:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1


# Example
bandit = ThompsonSampling(n_arms=5)

total_reward = 0
for t in range(1000):
    arm = bandit.select_arm()
    reward = 1 if np.random.rand() < true_rewards[arm] else 0
    bandit.update(arm, reward)
    total_reward += reward

print(f"Thompson Sampling total reward: {total_reward}")
print(f"Estimated rewards (alpha/(alpha+beta)): {bandit.alphas / (bandit.alphas + bandit.betas)}")
```

---

## Complete Numerical Walkthrough: Three Algorithms Side-by-Side

*"Let's trace through all three algorithms on the same problem to see how they behave differently."*

### Setup

**Arms**: 3 arms with true CTRs: A = 0.3, B = 0.5, C = 0.7 (unknown to algorithms)

**epsilon-greedy**: epsilon = 0.3 (30% exploration)

**UCB**: Using UCB1 formula

**Thompson Sampling**: Using Beta(1,1) priors

### Round-by-Round Comparison (20 Rounds)

| Round | epsilon-Greedy | UCB1 | Thompson Sampling |
|-------|---------------|------|-------------------|
| **1** | Random: A | Initialize: A | Sample: A (uniform priors) |
| | Reward: 0 | Reward: 1 | Reward: 0 |
| | Est: A=0.0 | Est: A=1.0 | Beta: A(1,2), B(1,1), C(1,1) |
| **2** | Exploit: A (0.0) | Initialize: B | Sample: C wins (0.71) |
| | Reward: 1 | Reward: 0 | Reward: 1 |
| | Est: A=0.5 | Est: A=1.0, B=0.0 | Beta: A(1,2), B(1,1), C(2,1) |
| **3** | Explore: B | Initialize: C | Sample: C wins (0.82) |
| | Reward: 0 | Reward: 1 | Reward: 0 |
| | Est: A=0.5, B=0.0 | Est: C=1.0 | Beta: A(1,2), B(1,1), C(2,2) |
| **4** | Exploit: A | UCB: A wins | Sample: B wins (0.65) |
| | UCB: A=1+1.67, B=0+1.67, C=1+1.67 | | |
| | Reward: 0 | Reward: 0 | Reward: 1 |
| | Est: A=0.33 | Est: A=0.5 | Beta: B(2,1) |
| **5** | Exploit: A | UCB: C=1+1.18 wins | Sample: B wins (0.73) |
| | Reward: 1 | Reward: 1 | Reward: 0 |
| **...**| | | |
| **10** | Est: A=0.4, B=0.3, C=0.5 | Est: A=0.4, B=0.4, C=0.67 | Beta: A(2,4), B(4,3), C(5,3) |
| | Counts: A=5, B=3, C=2 | Counts: A=4, B=3, C=3 | Counts: A=4, B=5, C=6 |
| **...**| | | |
| **20** | Est: A=0.33, B=0.5, C=0.57 | Est: A=0.33, B=0.5, C=0.64 | Beta: A(3,6), B(5,4), C(9,4) |
| | Counts: A=7, B=6, C=7 | Counts: A=4, B=5, C=11 | Counts: A=7, B=7, C=11 |
| | **Total Reward: 9** | **Total Reward: 11** | **Total Reward: 12** |

### Analysis of Behavior

**epsilon-Greedy**:
- Wastes pulls on arm A even after learning it's worst
- Random exploration doesn't target uncertainty
- Still pulling A 7 times after 20 rounds

**UCB**:
- Quickly focuses on arm C once confidence bounds separate
- Efficiently stops pulling arm A after a few tries
- Deterministic - no randomness in selection

**Thompson Sampling**:
- Similar outcome to UCB but stochastic
- Naturally concentrates on best arm
- Posterior uncertainty drives exploration

*"Notice how UCB and Thompson Sampling both converge to arm C faster than epsilon-greedy. They're 'smarter' about where to explore."*

---

## Comparison of Algorithms

### Regret Bounds

**epsilon-greedy**: $O(T)$ with fixed epsilon (linear - bad!)

**epsilon-greedy (decaying)**: $O(\log T)$ (much better)

**UCB1**: $O(\sqrt{KT \ln T})$ (near-optimal)

**Thompson Sampling**: $O(\sqrt{KT})$ (optimal!)

---

### Empirical Comparison

```python
def evaluate_bandit(bandit_class, n_arms, true_rewards, n_rounds=1000, n_trials=100):
    """
    Evaluate bandit algorithm across multiple trials.
    """
    total_rewards = []

    for _ in range(n_trials):
        if bandit_class == EpsilonGreedy:
            bandit = bandit_class(n_arms, epsilon=0.1)
        else:
            bandit = bandit_class(n_arms)

        reward_sum = 0
        for t in range(n_rounds):
            arm = bandit.select_arm()
            reward = 1 if np.random.rand() < true_rewards[arm] else 0
            bandit.update(arm, reward)
            reward_sum += reward

        total_rewards.append(reward_sum)

    return np.mean(total_rewards), np.std(total_rewards)


# Compare algorithms
algorithms = [
    ("epsilon-greedy", EpsilonGreedy),
    ("UCB1", UCB1),
    ("Thompson Sampling", ThompsonSampling)
]

for name, alg_class in algorithms:
    mean_reward, std_reward = evaluate_bandit(alg_class, 5, true_rewards, n_rounds=1000, n_trials=100)
    print(f"{name}: {mean_reward:.1f} +/- {std_reward:.1f}")
```

---

## What Can Go Wrong: Failure Modes and Limitations

*"Now let's discuss when bandits fail. Understanding limitations is as important as understanding capabilities."*

### 1. Non-Stationary Rewards

**The Problem**: Article popularity changes over time. An article about "2024 election results" has high CTR in November 2024, but low CTR in December.

**Why Standard Bandits Fail**: They weight all historical observations equally. Old data about the election article still influences decisions.

**Symptoms**:
- Algorithm locked onto previously-good arm
- Regret grows linearly instead of sublinearly
- Estimates lag behind true reward changes

**Solutions**:
- Sliding window: Only use last N observations
- Discounting: Weight recent observations more heavily
- Change detection: Reset when drift detected

```python
# Discounted Thompson Sampling
class DiscountedThompsonSampling(ThompsonSampling):
    def __init__(self, n_arms, discount=0.99):
        super().__init__(n_arms)
        self.discount = discount

    def update(self, arm, reward):
        # Discount previous beliefs (forgetting)
        self.alphas = 1 + (self.alphas - 1) * self.discount
        self.betas = 1 + (self.betas - 1) * self.discount

        if reward > 0:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1
```

### 2. Delayed Feedback

**The Problem**: User clicks on article, but you don't know if they actually read it until later (time on page, scroll depth). Or: user clicks ad but converts days later.

**Why Standard Bandits Fail**: They assume immediate feedback. If you pull arm A 10 times before getting any feedback, you can't update properly.

**Symptoms**:
- Over-exploration of recently-pulled arms
- Reward estimates based on stale data
- Poor convergence

**Solutions**:
- Batch updates: Wait for feedback before next decision
- Imputation: Estimate delayed rewards
- Credit assignment: Model delay distribution

### 3. Many Arms (Large Action Space)

**The Problem**: Netflix has 10,000+ movies. Each movie is an "arm." You can't try each one even once!

**Why Standard Bandits Fail**: Regret bounds are O(sqrt(KT)) where K is number of arms. With K=10,000, regret is huge.

**Symptoms**:
- Never converge - always exploring
- Prohibitive regret during learning phase
- Most arms never pulled

**Solutions**:
- Contextual bandits: Use features to generalize across arms
- Clustering: Group similar arms, share information
- Linear UCB: Assume reward is linear in features

*"This is why we'll study contextual bandits next! Plain multi-armed bandits don't scale to large action spaces."*

### 4. Multiple Simultaneous Decisions

**The Problem**: You show 5 articles on the homepage, not just 1. How do you credit clicks?

**Why Standard Bandits Fail**: They assume one arm per round. With 5 slots, position matters (slot 1 gets more clicks than slot 5).

**Solutions**:
- Cascading bandits: Model position effects
- Combinatorial bandits: Actions are subsets of arms
- Factored rewards: Decompose reward by position and content

### 5. Adversarial Environments

**The Problem**: Competitors actively try to make your algorithm fail. Or: fraudulent clicks on ads.

**Why Standard Bandits Fail**: Stochastic bandits assume rewards are random but honest. Adversarial attacks can exploit exploration.

**Solutions**:
- Adversarial bandits (EXP3): Worst-case guarantees
- Robust statistics: Down-weight outliers
- Anomaly detection: Flag suspicious patterns

### 6. Cold Start with New Arms

**The Problem**: New article published. No data at all. How much should we explore it?

**Why It's Tricky**: Pure bandit algorithms start with optimistic priors (UCB) or uniform priors (TS). This might over-explore bad new items.

**Solutions**:
- Content-based priors: Use article features to estimate initial CTR
- Contextual bandits: Transfer knowledge from similar items
- Warm-starting: Use editorial judgment for initial estimates

### Quick Diagnostic Checklist

*"When deploying bandits in production, ask yourself:"*

| Question | If Yes... |
|----------|-----------|
| Do rewards change over time? | Use discounting or sliding windows |
| Is feedback delayed by hours/days? | Use batch updates or delay modeling |
| Are there thousands of arms? | Use contextual bandits |
| Do you show multiple items per page? | Use cascading or combinatorial bandits |
| Could rewards be adversarial? | Use EXP3 or robust methods |
| Do new items arrive frequently? | Use content-based priors |

---

## Application to Recommendations

### Content Recommendation

**Scenario**: News website with new articles daily.

**Challenge**: Which articles to show on homepage?

**Bandit formulation**:
- **Arms**: Articles
- **Reward**: Click (1) or no click (0)
- **Goal**: Maximize total clicks

```python
class ArticleRecommender:
    def __init__(self, article_ids):
        """
        Bandit-based article recommender.
        """
        self.article_ids = article_ids
        self.n_articles = len(article_ids)

        # Use Thompson Sampling
        self.bandit = ThompsonSampling(self.n_articles)

    def recommend(self):
        """
        Recommend an article.
        """
        arm = self.bandit.select_arm()
        return self.article_ids[arm]

    def update_feedback(self, article_id, clicked):
        """
        Update model based on user feedback.

        clicked: 1 if user clicked, 0 otherwise
        """
        arm = self.article_ids.index(article_id)
        self.bandit.update(arm, clicked)


# Example
articles = ["article_1", "article_2", "article_3", "article_4", "article_5"]
recommender = ArticleRecommender(articles)

# Simulate user interactions
for _ in range(100):
    article = recommender.recommend()
    clicked = np.random.rand() < 0.3  # Simulate 30% CTR
    recommender.update_feedback(article, clicked)

print("Estimated CTRs:")
for i, article in enumerate(articles):
    alpha = recommender.bandit.alphas[i]
    beta = recommender.bandit.betas[i]
    ctr = alpha / (alpha + beta)
    print(f"  {article}: {ctr:.3f}")
```

---

### Cold Start with Bandits

**Problem**: New items have no interaction history.

**Solution**: Bandits naturally explore new items (high uncertainty).

**Comparison**:
- **Collaborative filtering**: Can't recommend new items
- **Bandits**: Thompson Sampling gives new items high probability initially

---

## Advanced Topics

### Restless Bandits

**Problem**: Arm rewards change over time (news article popularity decays).

**Solution**: **Discounted** or **sliding window** updates.

```python
class DiscountedThompsonSampling(ThompsonSampling):
    def __init__(self, n_arms, discount=0.99):
        super().__init__(n_arms)
        self.discount = discount

    def update(self, arm, reward):
        # Discount previous beliefs
        self.alphas *= self.discount
        self.betas *= self.discount

        # Add new observation
        if reward > 0:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1
```

---

### Contextual Bandits (Preview)

**Limitation of MAB**: Doesn't use context (user features, time, etc.).

**Contextual bandits**: Choose arm based on context - see next section.

---

## Summary

**Key Takeaways**:
1. **Exploration-exploitation**: Balance trying new items vs. showing known good items
2. **epsilon-greedy**: Simple, random exploration (O(T) regret with fixed epsilon)
3. **UCB**: Optimistic exploration derived from Hoeffding's inequality (O(sqrt(T log T)) regret)
4. **Thompson Sampling**: Bayesian, probabilistic exploration via posterior sampling (O(sqrt(T)) regret)
5. **Regret**: Measures the price of learning - good algorithms have sublinear regret

**Best Practices**:
- **Start simple**: epsilon-greedy good baseline
- **Thompson Sampling**: Best overall performance
- **Decaying epsilon**: Explore less over time
- **Restless bandits**: For dynamic environments

**When to use**:
- **Cold start**: New items, no interaction history
- **A/B testing**: Which variant performs better?
- **Dynamic content**: News, trending topics
- **Limited feedback**: Only observe clicks on shown items

**When NOT to use (or use with modifications)**:
- Non-stationary rewards - use discounting
- Delayed feedback - use batch updates
- Thousands of arms - use contextual bandits
- Multiple slots - use cascading bandits

**Next**: Contextual bandits with user/item features.

---

## Reflection Questions

*"Before moving on, make sure you can answer these:"*

1. Why does pure exploitation (always showing the best-so-far) fail?
2. Derive the regret formula and explain what each term means.
3. Why does epsilon-greedy have O(T) regret with fixed epsilon?
4. How does the UCB confidence bonus come from Hoeffding's inequality?
5. Why is Thompson Sampling equivalent to probability matching?
6. When would you NOT want to use standard multi-armed bandits?

---

## References

1. **Auer, P., et al. (2002)**. "Finite-time Analysis of the Multiarmed Bandit Problem". *Machine Learning*.
   - **UCB1** algorithm

2. **Thompson, W. R. (1933)**. "On the Likelihood that One Unknown Probability Exceeds Another". *Biometrika*.
   - **Thompson Sampling**

3. **Agrawal, S., & Goyal, N. (2012)**. "Analysis of Thompson Sampling for the Multi-armed Bandit Problem". *COLT*.
   - **Theoretical analysis of TS**

4. **Chapelle, O., & Li, L. (2011)**. "An Empirical Evaluation of Thompson Sampling". *NeurIPS*.
   - **Empirical comparison**

5. **Li, L., et al. (2010)**. "A Contextual-Bandit Approach to Personalized News Article Recommendation". *WWW*.
   - **Yahoo! news recommendation**
