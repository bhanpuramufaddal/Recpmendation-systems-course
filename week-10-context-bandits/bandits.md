# Week 10: Multi-Armed Bandits

## Overview

**Multi-Armed Bandits (MAB)** address the **exploration-exploitation dilemma**:
- **Exploit**: Show items we know users like → maximize immediate reward
- **Explore**: Try new items to learn preferences → maximize long-term reward

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
- Implement ε-greedy, UCB, Thompson Sampling
- Apply bandits to recommendation systems
- Measure regret and performance
- Choose appropriate bandit algorithms

---

## The Exploration-Exploitation Dilemma

### Problem Setup

**Scenario**: News website with 10 articles. Which to show users?

**Naive approach**: Show most-clicked article → **exploit**.

**Problem**: What if there's a better article you haven't tried?

**Better approach**: Sometimes try less-popular articles → **explore**.

---

### Regret

**Optimal strategy**: Always choose best arm (with perfect knowledge).

**Actual strategy**: Learn while choosing (imperfect knowledge).

**Regret**: Difference between optimal and actual cumulative reward.

$$\text{Regret}(T) = T \cdot \mu^* - \sum_{t=1}^T r_t$$

where:
- $T$ = total rounds
- $\mu^*$ = reward of best arm
- $r_t$ = reward at round $t$

**Goal**: Minimize regret.

---

## ε-Greedy

### Algorithm

**Idea**: With probability $\varepsilon$, explore (random arm); otherwise, exploit (best known arm).

**Pseudo-code**:
```
For each round t:
  With probability ε:
    Choose random arm (explore)
  With probability 1-ε:
    Choose arm with highest estimated reward (exploit)

  Observe reward r_t
  Update reward estimates
```

**Parameters**:
- $\varepsilon = 0$: Pure exploitation (greedy)
- $\varepsilon = 1$: Pure exploration (random)
- $\varepsilon = 0.1$: Typical value (10% exploration)

---

### Implementation

```python
import numpy as np

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        """
        ε-greedy bandit algorithm.

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
        Select an arm using ε-greedy policy.
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

### Decaying ε

**Problem**: Fixed $\varepsilon$ → always explores (even after learning).

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

### Intuition

**ε-greedy problem**: Explores randomly → wastes time on obviously bad arms.

**UCB idea**: Explore arms with **high uncertainty** (less-tried arms).

**Principle**: "Optimism in the face of uncertainty" → assume arms could be good until proven otherwise.

---

### UCB1 Algorithm

**Selection rule**:
$$\text{arm}_t = \arg\max_{i} \left[ \hat{\mu}_i + \sqrt{\frac{2 \ln t}{n_i}} \right]$$

where:
- $\hat{\mu}_i$ = estimated reward of arm $i$
- $t$ = total rounds so far
- $n_i$ = times arm $i$ pulled
- $\sqrt{\frac{2 \ln t}{n_i}}$ = **confidence bonus** (higher for less-tried arms)

**Interpretation**: Choose arm with highest **upper confidence bound**.

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
print(f"Estimated rewards (α/(α+β)): {bandit.alphas / (bandit.alphas + bandit.betas)}")
```

---

## Comparison of Algorithms

### Regret Bounds

**ε-greedy**: $O(T^{2/3})$ (sublinear, but slow)

**UCB1**: $O(\sqrt{T \ln T})$ (better theoretical guarantee)

**Thompson Sampling**: $O(\sqrt{T})$ (best theoretical, also best empirical)

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
    ("ε-greedy", EpsilonGreedy),
    ("UCB1", UCB1),
    ("Thompson Sampling", ThompsonSampling)
]

for name, alg_class in algorithms:
    mean_reward, std_reward = evaluate_bandit(alg_class, 5, true_rewards, n_rounds=1000, n_trials=100)
    print(f"{name}: {mean_reward:.1f} ± {std_reward:.1f}")
```

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

**Contextual bandits**: Choose arm based on context → see next section.

---

## Summary

**Key Takeaways**:
1. **Exploration-exploitation**: Balance trying new items vs. showing known good items
2. **ε-greedy**: Simple, random exploration
3. **UCB**: Optimistic exploration (favor uncertain arms)
4. **Thompson Sampling**: Bayesian, probabilistic exploration
5. **Regret**: Measure of suboptimality

**Best Practices**:
- **Start simple**: ε-greedy good baseline
- **Thompson Sampling**: Best overall performance
- **Decaying ε**: Explore less over time
- **Restless bandits**: For dynamic environments

**When to use**:
- **Cold start**: New items, no interaction history
- **A/B testing**: Which variant performs better?
- **Dynamic content**: News, trending topics
- **Limited feedback**: Only observe clicks on shown items

**Next**: Contextual bandits with user/item features.

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
