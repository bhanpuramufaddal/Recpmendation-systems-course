# Week 10: Context-Aware and Bandit-Based Recommendations

## Overview

Context-aware systems adapt recommendations based on time, location, device, and other contextual factors. Bandits and reinforcement learning address the exploration-exploitation dilemma for long-term optimization.

## Topics

### [1. Context-Aware Recommendation](context-aware.md)
- Contextual dimensions (time, location, device)
- Tensor factorization
- Context-aware matrix factorization
- Feature-based models

### [2. Multi-Armed Bandits](bandits.md)
- Exploration vs. exploitation
- **ε-greedy**, **UCB**, **Thompson Sampling**
- Regret bounds
- Practical implementation

### [3. Contextual Bandits](contextual-bandits.md)
- **LinUCB**: Linear UCB with context
- Neural bandits
- Policy learning
- Off-policy evaluation

**Application**: Netflix artwork personalization

### [4. Reinforcement Learning](reinforcement-learning.md)
- MDP formulation
- Reward shaping
- Policy gradient (REINFORCE, A3C)
- Offline RL (batch RL)

**Case Study**: Cursor AI's online RL (Week 16)

## Key Algorithms

### Thompson Sampling
```python
for t in range(T):
    for arm in arms:
        sample_reward[arm] = beta.sample(alpha[arm], beta[arm])
    chosen_arm = argmax(sample_reward)
    reward = pull(chosen_arm)
    update_posterior(chosen_arm, reward)
```

### LinUCB
```python
for t in range(T):
    for arm in arms:
        theta = inv(A[arm]) @ b[arm]
        ucb[arm] = theta.T @ x + alpha * sqrt(x.T @ inv(A[arm]) @ x)
    chosen_arm = argmax(ucb)
```

*Return to [Main Course Page](../README.md)*
