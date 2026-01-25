# Week 10: Contextual Bandits

## Overview

**Contextual bandits** extend multi-armed bandits by incorporating **context** (features) when choosing actions.

**Key difference from MAB**:
- **MAB**: One best arm for all users
- **Contextual**: Best arm depends on context (user features, time, etc.)

**Example** (News):
- **MAB**: Article A is best → show to everyone
- **Contextual**: Article A best for sports fans, Article B best for tech enthusiasts

**Formulation**:
- Observe context $\mathbf{x}_t$ (user features, item features)
- Choose arm $a_t$ based on context
- Receive reward $r_t$
- Update policy

This document covers contextual bandit algorithms for personalized recommendations.

---

## Learning Objectives

By the end of this section, you will:
- Understand contextual bandits vs. MAB
- Implement LinUCB for linear contextual bandits
- Apply neural contextual bandits
- Use Thompson Sampling with context
- Deploy contextual bandits in production

---

## Problem Formulation

### Context-Dependent Rewards

**Setup**: At each round $t$:
1. Observe context $\mathbf{x}_t \in \mathbb{R}^d$ (user features, item features, time, etc.)
2. Choose arm $a_t \in \{1, \ldots, K\}$
3. Receive reward $r_t = f(\mathbf{x}_t, a_t) + \epsilon_t$

**Goal**: Learn policy $\pi(\mathbf{x}) \rightarrow a$ that maximizes expected reward.

---

### Linear Reward Model

**Assumption**: Reward is linear in context.

$$r_t = \mathbf{x}_t^T \theta_a + \epsilon_t$$

where $\theta_a \in \mathbb{R}^d$ = parameters for arm $a$.

**Interpretation**: Each arm has different feature weights.

**Example** (Movie recommendations):
```
Context x = [is_action_fan, is_comedy_fan, weekend, evening]
            [1, 0, 1, 1]

Arm 1 (Action movie): θ₁ = [0.8, -0.2, 0.3, 0.1]
  → r₁ = 0.8*1 + (-0.2)*0 + 0.3*1 + 0.1*1 = 1.2

Arm 2 (Comedy): θ₂ = [-0.1, 0.9, 0.2, -0.1]
  → r₂ = (-0.1)*1 + 0.9*0 + 0.2*1 + (-0.1)*1 = 0.0

Choose Arm 1 (higher expected reward for this context)
```

---

## LinUCB

### Algorithm

**LinUCB** (Linear Upper Confidence Bound): UCB for linear contextual bandits.

**Key idea**: Maintain confidence ellipsoid around $\hat{\theta}_a$.

**Selection rule**:
$$a_t = \arg\max_a \left[ \hat{\theta}_a^T \mathbf{x}_t + \alpha \sqrt{\mathbf{x}_t^T A_a^{-1} \mathbf{x}_t} \right]$$

where:
- $\hat{\theta}_a$ = estimated parameters for arm $a$
- $A_a$ = precision matrix (inverse covariance)
- $\alpha$ = exploration parameter (typically 0.1-1.0)
- $\sqrt{\mathbf{x}_t^T A_a^{-1} \mathbf{x}_t}$ = confidence bonus

**Updates**:
$$A_a = \sum_{t: a_t=a} \mathbf{x}_t \mathbf{x}_t^T + I$$
$$\mathbf{b}_a = \sum_{t: a_t=a} r_t \mathbf{x}_t$$
$$\hat{\theta}_a = A_a^{-1} \mathbf{b}_a$$

---

### Implementation

```python
import numpy as np

class LinUCB:
    def __init__(self, n_arms, n_features, alpha=1.0):
        """
        Linear UCB for contextual bandits.

        n_arms: Number of arms (actions)
        n_features: Dimension of context vector
        alpha: Exploration parameter
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha

        # Initialize for each arm
        self.A = [np.identity(n_features) for _ in range(n_arms)]  # Precision matrices
        self.b = [np.zeros(n_features) for _ in range(n_arms)]     # Accumulated rewards

    def select_arm(self, context):
        """
        Select arm given context.

        context: (n_features,) numpy array
        """
        ucb_values = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            # Estimate theta
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]

            # UCB value
            mean = theta_hat.T @ context
            std = np.sqrt(context.T @ A_inv @ context)
            ucb_values[a] = mean + self.alpha * std

        return np.argmax(ucb_values)

    def update(self, arm, context, reward):
        """
        Update model after observing reward.

        arm: Chosen arm
        context: (n_features,) numpy array
        reward: Observed reward
        """
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context


# Example
n_arms = 3
n_features = 5
bandit = LinUCB(n_arms, n_features, alpha=0.5)

# Simulate interactions
for t in range(100):
    # Random context
    context = np.random.randn(n_features)

    # Select arm
    arm = bandit.select_arm(context)

    # Simulate reward (true params unknown to algorithm)
    true_theta = np.random.randn(n_features)  # Different for each arm
    reward = context.T @ true_theta + 0.1 * np.random.randn()

    # Update
    bandit.update(arm, context, reward)

print("Learned parameters (theta) for each arm:")
for a in range(n_arms):
    theta_hat = np.linalg.inv(bandit.A[a]) @ bandit.b[a]
    print(f"Arm {a}: {theta_hat}")
```

---

## Neural Contextual Bandits

### Motivation

**LinUCB limitation**: Assumes linear rewards.

**Reality**: Rewards often non-linear (e.g., interactions between features).

**Solution**: **Neural networks** to model $f(\mathbf{x}, a)$.

---

### Neural ε-Greedy

**Simplest approach**: Use neural network + ε-greedy exploration.

**Architecture**:
```
Context x ──→ Neural Network ──→ Q(x, a₁), Q(x, a₂), ..., Q(x, aₖ)
```

**Selection**:
- With probability $\varepsilon$: Random arm
- Otherwise: $\arg\max_a Q(\mathbf{x}, a)$

**Update**: Gradient descent on prediction error.

---

### Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralContextualBandit:
    def __init__(self, n_arms, n_features, hidden_dim=64, epsilon=0.1, lr=0.01):
        """
        Neural contextual bandit with ε-greedy.
        """
        self.n_arms = n_arms
        self.epsilon = epsilon

        # Neural network
        self.model = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_arms)  # Output: Q-values for each arm
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_arm(self, context):
        """
        Select arm using ε-greedy.

        context: (n_features,) numpy array
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)

        with torch.no_grad():
            context_tensor = torch.FloatTensor(context).unsqueeze(0)
            q_values = self.model(context_tensor).squeeze()
            return torch.argmax(q_values).item()

    def update(self, arm, context, reward):
        """
        Update neural network.
        """
        context_tensor = torch.FloatTensor(context).unsqueeze(0)
        q_values = self.model(context_tensor).squeeze()

        # Target: observed reward for chosen arm, keep others as predicted
        target = q_values.clone().detach()
        target[arm] = reward

        # Loss and backprop
        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Example
bandit = NeuralContextualBandit(n_arms=5, n_features=10, epsilon=0.1)

for t in range(1000):
    context = np.random.randn(10)
    arm = bandit.select_arm(context)

    # Simulate reward
    true_value = np.sum(context[:5]) * (arm / 5.0)  # Non-linear
    reward = true_value + 0.1 * np.random.randn()

    bandit.update(arm, context, reward)
```

---

## Thompson Sampling for Contextual Bandits

### Bayesian Linear Regression

**Model**: $r = \mathbf{x}^T \theta + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

**Prior**: $\theta \sim \mathcal{N}(\mathbf{m}_0, \Sigma_0)$

**Posterior** (after observing data):
$$\theta | \mathcal{D} \sim \mathcal{N}(\mathbf{m}_a, \Sigma_a)$$

where:
$$\Sigma_a = \left( \Sigma_0^{-1} + \frac{1}{\sigma^2} \sum_t \mathbf{x}_t \mathbf{x}_t^T \right)^{-1}$$
$$\mathbf{m}_a = \Sigma_a \left( \Sigma_0^{-1} \mathbf{m}_0 + \frac{1}{\sigma^2} \sum_t r_t \mathbf{x}_t \right)$$

---

### Thompson Sampling Algorithm

**Selection**:
1. For each arm $a$, sample $\tilde{\theta}_a \sim \mathcal{N}(\mathbf{m}_a, \Sigma_a)$
2. Compute $\tilde{r}_a = \mathbf{x}^T \tilde{\theta}_a$
3. Choose arm with highest sampled reward

**Update**: Update posterior with observed $(x, a, r)$.

---

### Implementation

```python
class LinearThompsonSampling:
    def __init__(self, n_arms, n_features, sigma=1.0):
        """
        Thompson Sampling for linear contextual bandits.
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.sigma = sigma

        # Posterior parameters for each arm
        self.mu = [np.zeros(n_features) for _ in range(n_arms)]
        self.cov = [np.identity(n_features) for _ in range(n_arms)]

    def select_arm(self, context):
        """
        Sample theta from posterior and choose best arm.
        """
        sampled_rewards = []

        for a in range(self.n_arms):
            # Sample theta from posterior
            theta_sample = np.random.multivariate_normal(self.mu[a], self.cov[a])

            # Compute predicted reward
            reward_sample = context.T @ theta_sample
            sampled_rewards.append(reward_sample)

        return np.argmax(sampled_rewards)

    def update(self, arm, context, reward):
        """
        Bayesian update of posterior.
        """
        # Precision (inverse covariance)
        precision = np.linalg.inv(self.cov[arm])

        # Update precision
        precision_new = precision + (1 / self.sigma**2) * np.outer(context, context)

        # Update covariance
        self.cov[arm] = np.linalg.inv(precision_new)

        # Update mean
        precision_times_mu = precision @ self.mu[arm]
        precision_times_mu_new = precision_times_mu + (reward / self.sigma**2) * context

        self.mu[arm] = self.cov[arm] @ precision_times_mu_new


# Example
bandit = LinearThompsonSampling(n_arms=3, n_features=5, sigma=0.5)

for t in range(500):
    context = np.random.randn(5)
    arm = bandit.select_arm(context)

    # Simulate reward
    true_theta = np.array([1, -0.5, 0.3, 0, 0.2])
    reward = context.T @ true_theta + 0.1 * np.random.randn()

    bandit.update(arm, context, reward)

print("Learned theta:")
for a in range(3):
    print(f"Arm {a}: {bandit.mu[a]}")
```

---

## Off-Policy Evaluation

### Problem

**Online learning**: Test new policy on real users → risky (bad policy = lost revenue).

**Offline evaluation**: Estimate new policy performance from **logged data** (past interactions).

**Challenge**: Logged data collected by old policy $\pi_0$, want to evaluate new policy $\pi_1$.

---

### Inverse Propensity Scoring (IPS)

**Idea**: Reweight logged rewards by probability ratio.

**Estimator**:
$$\hat{V}(\pi_1) = \frac{1}{T} \sum_{t=1}^T \frac{\mathbb{1}[a_t = \pi_1(\mathbf{x}_t)]}{\pi_0(a_t | \mathbf{x}_t)} r_t$$

where:
- $\pi_0(a | \mathbf{x})$ = probability old policy chose arm $a$ given context $\mathbf{x}$
- $\pi_1(\mathbf{x})$ = new policy's choice

**Interpretation**: If new policy would've chosen same arm, upweight reward by inverse probability.

---

### Doubly Robust Estimator

**Improvement over IPS**: Combine IPS with reward model.

$$\hat{V}_{\text{DR}}(\pi_1) = \frac{1}{T} \sum_{t=1}^T \left[ \hat{r}(\mathbf{x}_t, \pi_1(\mathbf{x}_t)) + \frac{\mathbb{1}[a_t = \pi_1(\mathbf{x}_t)]}{\pi_0(a_t | \mathbf{x}_t)} (r_t - \hat{r}(\mathbf{x}_t, a_t)) \right]$$

**Benefit**: Lower variance than IPS, unbiased even if reward model inaccurate.

---

## Production Deployment

### Batched Updates

**Challenge**: Real-time updates expensive.

**Solution**: Batch updates (hourly, daily).

**Trade-off**: Slower learning vs. lower compute cost.

---

### Warm Start

**Problem**: New bandit starts with no data (random exploration).

**Solution**: Initialize with **supervised learning** on historical data.

```python
# Pre-train on historical data
X_train, y_train, arms_train = load_historical_data()

for i in range(len(X_train)):
    context = X_train[i]
    arm = arms_train[i]
    reward = y_train[i]

    bandit.update(arm, context, reward)

# Now deploy for live traffic
```

---

### A/B Testing Bandits

**Scenario**: Compare bandit vs. existing system.

**Setup**:
- 90% traffic → existing system (control)
- 10% traffic → bandit (treatment)

**Metrics**:
- CTR, revenue, user engagement
- Statistical significance (t-test)

**Decision**: If bandit wins, gradually increase traffic (10% → 50% → 100%).

---

## Case Study: Yahoo! News

### Problem

**Yahoo! Front Page**: Show 1 article from pool of ~20.

**Challenge**: Which article to show each user?

---

### Solution: LinUCB

**Context**:
- User features: demographics, browsing history (1000+ dims)
- Article features: category, keywords (100+ dims)

**Arms**: 20 articles

**Reward**: Click (1) or no click (0)

**Results**:
- **12.5% CTR improvement** over baseline
- Serves 600M+ users/month
- Real-time updates (batch hourly)

---

## Summary

**Key Takeaways**:
1. **Contextual bandits**: Personalize arm selection based on context
2. **LinUCB**: Linear rewards, confidence-based exploration
3. **Neural**: Non-linear rewards with deep networks
4. **Thompson Sampling**: Bayesian approach, sample from posterior
5. **Off-policy evaluation**: IPS, doubly robust for safe testing

**Best Practices**:
- **Start with LinUCB**: Strong baseline, interpretable
- **Use neural for complex patterns**: User-item interactions
- **Warm start**: Pre-train on historical data
- **Off-policy eval**: Test new policies safely
- **Batch updates**: Balance learning speed vs. cost

**When to use**:
- **Personalization**: Different users prefer different items
- **Rich features**: User demographics, item attributes
- **Real-time**: Online learning from user feedback
- **Exploration**: Need to try new items

**Next**: Reinforcement learning for long-term optimization.

---

## References

1. **Li, L., et al. (2010)**. "A Contextual-Bandit Approach to Personalized News Article Recommendation". *WWW*.
   - **LinUCB**, Yahoo! News

2. **Agrawal, S., & Goyal, N. (2013)**. "Thompson Sampling for Contextual Bandits with Linear Payoffs". *ICML*.
   - **Thompson Sampling for linear contextual bandits**

3. **Dudík, M., et al. (2014)**. "Doubly Robust Policy Evaluation and Optimization". *Statistical Science*.
   - **Doubly robust estimator**

4. **Riquelme, C., et al. (2018)**. "Deep Bayesian Bandits Showdown". *ICLR*.
   - **Neural contextual bandits comparison**

5. **Zhou, D., et al. (2020)**. "Neural Contextual Bandits with UCB-based Exploration". *ICML*.
   - **Neural UCB**
