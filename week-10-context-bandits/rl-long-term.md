# Week 10: Reinforcement Learning for Long-Term Optimization

## Overview

**Reinforcement Learning (RL)** optimizes for **long-term cumulative reward**, not just immediate clicks.

**Key insight**: Today's recommendation affects tomorrow's preferences.

**Example** (YouTube):
- **Immediate reward**: User clicks video → +1 reward
- **Long-term**: User watches low-quality clickbait → loses trust → churns
- **RL goal**: Maximize watch time over weeks/months, not just next click

**Difference from bandits**:
- **Bandits**: Each action independent, optimize immediate reward
- **RL**: Actions affect future state, optimize cumulative reward

This document covers RL for recommendation systems.

---

## Learning Objectives

By the end of this section, you will:
- Understand MDP formulation for RecSys
- Implement Q-learning and policy gradients
- Apply reward shaping for user satisfaction
- Use offline RL for recommendations
- Deploy RL systems in production

---

## Markov Decision Process (MDP)

### Formulation

**MDP**: Tuple $(S, A, P, R, \gamma)$

- **State** $s \in S$: User's current context (history, preferences)
- **Action** $a \in A$: Recommendation (which item to show)
- **Transition** $P(s' | s, a)$: Next state given current state and action
- **Reward** $R(s, a)$: Immediate reward (click, watch time)
- **Discount** $\gamma \in [0, 1]$: How much to value future rewards

**Goal**: Find policy $\pi(a | s)$ maximizing expected cumulative reward:

$$J(\pi) = \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \right]$$

---

### Example: Movie Recommendations

**State**: User's watch history (last 5 movies)
```
s = [Action1, Action2, Comedy1, Action3, SciFi1]
```

**Action**: Recommend movie from catalog (1000 movies)

**Reward**:
- Watch time (minutes): $r = \text{watch\_time} / \text{duration}$
- Engagement: $r = 1$ if finished, 0 if abandoned

**Transition**: User's next state depends on current movie
```
If recommend Action4:
  s' = [Action2, Comedy1, Action3, SciFi1, Action4]
```

**Policy**: $\pi(a | s)$ → probability of recommending movie $a$ given state $s$

---

## Q-Learning

### Q-Function

**Q-function**: Expected cumulative reward from taking action $a$ in state $s$:

$$Q^\pi(s, a) = \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \Big| s_0=s, a_0=a, \pi \right]$$

**Optimal Q-function**: $Q^*(s, a) = \max_\pi Q^\pi(s, a)$

**Optimal policy**: $\pi^*(s) = \arg\max_a Q^*(s, a)$

---

### Bellman Equation

**Recursive definition**:

$$Q^*(s, a) = R(s, a) + \gamma \mathbb{E}_{s'} \left[ \max_{a'} Q^*(s', a') \right]$$

**Interpretation**: Current reward + discounted future reward

---

### Q-Learning Algorithm

**Update rule** (after observing $(s, a, r, s')$):

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

where $\alpha$ = learning rate.

**Exploration**: Use ε-greedy on Q-values.

---

### Implementation

```python
import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=0.1):
        """
        Q-learning for discrete state-action spaces.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        """
        ε-greedy action selection.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        """
        Q-learning update.

        done: True if episode terminated
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        # TD error
        td_error = target - self.Q[state, action]

        # Update Q-value
        self.Q[state, action] += self.lr * td_error


# Example: Grid world navigation (simplified RecSys)
n_states = 100  # User states
n_actions = 10  # Items to recommend

agent = QLearning(n_states, n_actions, lr=0.1, gamma=0.95, epsilon=0.1)

# Simulate episodes
for episode in range(1000):
    state = np.random.randint(n_states)  # Initial state
    done = False

    for step in range(50):
        action = agent.select_action(state)

        # Simulate environment (placeholder)
        reward = np.random.rand()  # Replace with actual reward
        next_state = np.random.randint(n_states)
        done = (step == 49)

        agent.update(state, action, reward, next_state, done)

        state = next_state
        if done:
            break

print("Learned Q-values (first 5 states):")
print(agent.Q[:5])
```

---

## Deep Q-Networks (DQN)

### Motivation

**Problem**: Q-table doesn't scale to large state/action spaces.

**Example**: State = user's last 100 interactions → $10^{100}$ states!

**Solution**: **Function approximation** - neural network to estimate Q.

$$Q(s, a; \theta) \approx Q^*(s, a)$$

---

### DQN Architecture

```
State s ──→ Neural Network ──→ Q(s, a₁), Q(s, a₂), ..., Q(s, aₖ)
           (parameters θ)
```

**Loss**: Mean squared TD error

$$\mathcal{L}(\theta) = \mathbb{E}\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

where $\theta^-$ = target network (updated slowly for stability).

---

### DQN Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99, epsilon=0.1):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-network and target network
        self.q_net = DQN(state_dim, n_actions)
        self.target_net = DQN(state_dim, n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Experience replay buffer
        self.buffer = deque(maxlen=10000)

    def select_action(self, state):
        """ε-greedy"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store in replay buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        """Train on batch from replay buffer"""
        if len(self.buffer) < batch_size:
            return

        # Sample batch
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current Q-values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss
        loss = nn.MSELoss()(q_values, targets)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Copy Q-network weights to target network"""
        self.target_net.load_state_dict(self.q_net.state_dict())


# Example
state_dim = 20  # User state embedding
n_actions = 100  # Items

agent = DQNAgent(state_dim, n_actions)

# Training loop
for episode in range(100):
    state = np.random.randn(state_dim)

    for step in range(50):
        action = agent.select_action(state)

        # Simulate environment
        reward = np.random.rand()
        next_state = np.random.randn(state_dim)
        done = (step == 49)

        agent.store_transition(state, action, reward, next_state, done)
        agent.train(batch_size=32)

        state = next_state

    # Update target network periodically
    if episode % 10 == 0:
        agent.update_target_network()
```

---

## Policy Gradient Methods

### REINFORCE Algorithm

**Idea**: Directly optimize policy parameters to maximize expected reward.

**Policy**: $\pi(a | s; \theta)$ (neural network)

**Objective**: $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t r_t \right]$

**Gradient** (policy gradient theorem):

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R_t \right]$$

where $R_t = \sum_{t'=t}^T \gamma^{t'-t} r_{t'}$ (return from time $t$).

---

### REINFORCE Implementation

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)


class REINFORCE:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def select_action(self, state):
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_tensor).squeeze()

        action = torch.multinomial(probs, 1).item()

        # Store for training
        self.episode_states.append(state)
        self.episode_actions.append(action)

        return action

    def store_reward(self, reward):
        self.episode_rewards.append(reward)

    def train_episode(self):
        """Train on completed episode"""
        T = len(self.episode_rewards)

        # Compute returns
        returns = []
        G = 0
        for r in reversed(self.episode_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)

        # Normalize returns (reduce variance)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute loss
        states = torch.FloatTensor(self.episode_states)
        actions = torch.LongTensor(self.episode_actions)

        probs = self.policy(states)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze())

        # Policy gradient loss
        loss = -(log_probs * returns).mean()

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []


# Example
agent = REINFORCE(state_dim=20, n_actions=100, lr=1e-3)

for episode in range(500):
    state = np.random.randn(20)

    for step in range(50):
        action = agent.select_action(state)

        # Simulate
        reward = np.random.rand()
        next_state = np.random.randn(20)

        agent.store_reward(reward)
        state = next_state

    agent.train_episode()
```

---

## Reward Shaping

### Problem

**Sparse rewards**: User only provides feedback occasionally (rating after finishing movie).

**Delayed rewards**: Effect of recommendation seen days later (user retention).

**Solution**: **Reward shaping** - design intermediate rewards to guide learning.

---

### Reward Design for RecSys

**1. Immediate engagement**:
- Click: +0.1
- Watch time: +0.01 per second
- Finish video: +1.0

**2. Long-term satisfaction**:
- Return next day: +5.0
- Subscribe: +100.0
- Churn: -50.0

**3. Diversity**:
- Recommend from new category: +0.5
- Avoid filter bubble penalty: -0.2 if showing same genre 5 times in a row

**Combined**:
$$r = r_{\text{click}} + r_{\text{watch}} + r_{\text{diversity}} + r_{\text{retention}}$$

---

### Example: YouTube Reward

**YouTube's RL system** (Chen et al., 2019):

$$r = \alpha \cdot \text{watch\_time} + \beta \cdot \text{user\_satisfaction} - \gamma \cdot \text{regret}$$

where:
- Watch time: Minutes watched
- Satisfaction: Explicit feedback (thumbs up/down)
- Regret: User returns and searches for different content

**Hyperparameters** ($\alpha, \beta, \gamma$) tuned via A/B testing.

---

## Offline RL

### Problem

**Online RL**: Interact with users → risky (bad policy hurts users).

**Offline RL**: Learn from **logged data** (past interactions) → safe.

---

### Batch RL

**Scenario**: Historical dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}$ collected by behavior policy $\pi_b$.

**Goal**: Learn better policy $\pi$ without further interaction.

**Challenge**: **Distribution shift** - logged data doesn't cover all states policy $\pi$ might visit.

---

### Conservative Q-Learning (CQL)

**Idea**: Penalize Q-values for unseen actions → be conservative.

**Objective**:

$$\min_Q \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp(Q(s, a)) - \mathbb{E}_{a \sim \pi_b} [Q(s, a)] \right] + \text{Bellman error}$$

**Effect**: Q-values for in-distribution actions are accurate, out-of-distribution actions underestimated.

---

### Importance Sampling

**Idea**: Reweight logged data to match new policy.

**Weight**:
$$w_t = \frac{\pi(a_t | s_t)}{\pi_b(a_t | s_t)}$$

**Policy gradient** (off-policy):
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N w_i \cdot \nabla_\theta \log \pi_\theta(a_i | s_i) \cdot R_i$$

**Problem**: High variance if policies differ.

---

## Production Considerations

### Simulation Environment

**Challenge**: Can't train RL directly on users (unsafe).

**Solution**: Build **simulator** from logged data.

**Approach**:
1. Model user behavior: $P(s' | s, a)$ and $P(r | s, a)$
2. Train simulator on historical data
3. Train RL policy in simulator
4. Deploy to small % of users, iterate

---

### A/B Testing RL Policies

**Setup**:
- **Control**: Current system (e.g., supervised ranking)
- **Treatment**: RL policy

**Metrics**:
- Immediate: CTR, watch time
- Long-term: User retention (7-day, 30-day)

**Ramp-up**: 1% → 10% → 50% → 100% if RL wins.

---

## Case Study: YouTube

### Problem

Maximize **long-term watch time**, not just next video click.

**Observation**: Clickbait gets clicks but reduces long-term engagement.

---

### Solution: REINFORCE with Baseline

**State**: User's watch history (50 recent videos, embeddings)

**Action**: Recommend video from top-K candidates (from two-tower retrieval)

**Reward**:
- Watch time (hours)
- User return rate (next day)

**Policy**: Neural network (MLP)

**Training**: Offline RL on logged data + online fine-tuning

**Results**:
- **+0.5% watch time** vs. supervised baseline
- **Reduced clickbait** (user satisfaction scores improved)

---

## Summary

**Key Takeaways**:
1. **MDP formulation**: States (user history), actions (recommendations), rewards (engagement)
2. **Q-learning**: Learn action-value function, ε-greedy exploration
3. **DQN**: Scale Q-learning with neural networks
4. **Policy gradients**: Directly optimize policy (REINFORCE)
5. **Offline RL**: Learn from logged data, conservative approaches

**Best Practices**:
- **Reward shaping**: Balance immediate + long-term objectives
- **Offline first**: Train on historical data before deploying
- **Simulation**: Build user simulator for safe experimentation
- **A/B test**: Validate RL gains vs. baseline

**When to use**:
- **Long-term goals**: User retention, lifetime value
- **Exploration**: Discover content users wouldn't find otherwise
- **Complex dynamics**: Today's rec affects tomorrow's preferences

**Next**: Evaluation methodologies for RecSys.

---

## References

1. **Sutton, R. S., & Barto, A. G. (2018)**. *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
   - **RL fundamentals**

2. **Mnih, V., et al. (2015)**. "Human-level control through deep reinforcement learning". *Nature*.
   - **DQN**

3. **Chen, M., et al. (2019)**. "Top-K Off-Policy Correction for a REINFORCE Recommender System". *WSDM*.
   - **YouTube RL system**

4. **Ie, E., et al. (2019)**. "RecSim: A Configurable Simulation Platform for Recommender Systems". *arXiv*.
   - **Simulation for RecSys RL**

5. **Kumar, A., et al. (2020)**. "Conservative Q-Learning for Offline Reinforcement Learning". *NeurIPS*.
   - **CQL for offline RL**
