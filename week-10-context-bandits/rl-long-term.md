# Week 10: Reinforcement Learning for Long-Term Optimization

## The Opening Question

*"Why does maximizing immediate clicks destroy long-term engagement?"*

Today we're going to challenge a fundamental assumption that most recommendation systems make - and show why it leads to platform decay.

---

## The Failure That Started It All

### The Clickbait Trap

Imagine you're optimizing a news recommendation system. Your model learns:

| Article Type | Click-Through Rate (CTR) |
|--------------|--------------------------|
| "You Won't BELIEVE What Happened Next..." | 15% |
| "Comprehensive Analysis of Economic Policy" | 3% |
| "10 Celebrities Who Look TERRIBLE Without Makeup" | 18% |
| "Climate Science Research Summary" | 4% |

**Your CTR-maximizing model recommends**: Clickbait, clickbait, and more clickbait.

**Day 1 results**: Amazing! CTR is up 40%! Ship it!

**Day 30 results**: User engagement collapsed. Here's why.

### The Long-Term Destruction

Let's follow two users over 30 days.

**Greedy (CTR-maximizing) approach**:

| Day | Recommendation | Clicked? | User Feeling | Returns Tomorrow? |
|-----|----------------|----------|--------------|-------------------|
| 1 | Clickbait | Yes | Disappointed | Yes (habit) |
| 2 | Clickbait | Yes | Frustrated | Yes (checking) |
| 3 | Clickbait | No | Annoyed | Maybe |
| 5 | Clickbait | No | Fed up | Barely |
| 10 | Clickbait | No | - | **Churned** |

**Total value over 30 days**:
- Days 1-4: 4 clicks = 4 points
- Days 5-30: Churned = 0 points
- **Total: 4 points**

**Long-term (RL) approach**:

| Day | Recommendation | Clicked? | User Feeling | Returns Tomorrow? |
|-----|----------------|----------|--------------|-------------------|
| 1 | Quality article | No | Neutral | Yes |
| 2 | Quality article | Yes | Satisfied | Yes |
| 3 | Mix (explore) | Yes | Interested | Yes |
| 5 | Personalized quality | Yes | Happy | Yes |
| 10 | Personalized quality | Yes | Loyal | Yes |
| 30 | Personalized quality | Yes | Advocate | Yes |

**Total value over 30 days**:
- Days 1-30: ~20 clicks + user retained + potential referrals
- **Total: 20+ points**

### The Math of Short-Term Thinking

**Greedy value** = Immediate reward only

$$V_{greedy} = r_1 = 0.15 \text{ (15% CTR on clickbait)}$$

**RL value** = Discounted cumulative reward

$$V_{RL} = r_1 + \gamma r_2 + \gamma^2 r_3 + ... = \sum_{t=0}^{\infty} \gamma^t r_t$$

With $\gamma = 0.95$ and user staying 30 days:

$$V_{RL} = 0.03 + 0.95(0.05) + 0.95^2(0.06) + ... \approx 0.8$$

**Even though** daily CTR is lower (3-6% vs 15%), cumulative value is **5x higher**.

### The Core Insight

> **Today's recommendation affects tomorrow's user state.** Optimizing for immediate reward ignores the **state transition** caused by each action.

This is exactly what Markov Decision Processes (MDPs) model.

---

## Learning Objectives

By the end of this lecture, you will:
- Understand the explore-exploit tradeoff mathematically
- Derive Upper Confidence Bound (UCB) from first principles
- Implement Thompson Sampling with Beta distributions
- Derive and implement LinUCB for contextual bandits
- Calculate regret and understand its significance
- Recognize the pitfalls of RL in production

---

## Multi-Armed Bandits: The Foundation

### The Setup

Imagine a casino with $K$ slot machines (arms). Each arm has an unknown reward probability.

**Your goal**: Maximize total reward over $T$ pulls.

**The dilemma**:
- **Exploit**: Pull the arm that looks best so far
- **Explore**: Try other arms to learn their true values

### Why This Matters for Recommendations

Each "arm" is a recommendation strategy:
- Arm 1: Recommend trending items
- Arm 2: Recommend based on user history
- Arm 3: Recommend diverse items
- Arm 4: Recommend new items (cold start)

You don't know which works best for this user. You must **learn while recommending**.

---

## Exploration vs Exploitation: The Mathematical Tradeoff

### The Problem with Pure Exploitation

**Scenario**: You've tried 3 arms with these results:

| Arm | Pulls | Successes | Observed Rate |
|-----|-------|-----------|---------------|
| A | 100 | 15 | 15% |
| B | 2 | 1 | **50%** |
| C | 5 | 0 | 0% |

**Pure exploitation says**: Always pull Arm B (50% rate)!

**But wait**: Arm B only has 2 observations. That 50% could be noise!

**True rates** (unknown to us):
- Arm A: 18%
- Arm B: 12%
- Arm C: 25%

By exploiting early estimates, we **miss the best arm** (C) entirely!

### The Problem with Pure Exploration

**Pure exploration**: Pull each arm equally, regardless of results.

After 300 pulls (100 each):

| Arm | Pulls | Expected Reward |
|-----|-------|-----------------|
| A | 100 | 100 × 0.18 = 18 |
| B | 100 | 100 × 0.12 = 12 |
| C | 100 | 100 × 0.25 = 25 |
| **Total** | 300 | **55** |

But if we **knew** C was best and always pulled it:

$$300 \times 0.25 = 75$$

We lost 20 rewards to exploration!

### The Balance: Regret

**Regret** = What we lost compared to always pulling the best arm

$$\text{Regret}_T = T \cdot \mu^* - \sum_{t=1}^T r_t$$

where $\mu^*$ = best arm's true mean.

**Good algorithms**: $O(\log T)$ regret (grows slowly)
**Bad algorithms**: $O(T)$ regret (linear growth)

---

## Upper Confidence Bound (UCB): Derivation

### The Intuition

**Problem with sample mean**: Doesn't account for uncertainty.

**UCB idea**: Be optimistic in the face of uncertainty.

$$\text{UCB}_a = \underbrace{\hat{\mu}_a}_{\text{sample mean}} + \underbrace{c \sqrt{\frac{\ln t}{n_a}}}_{\text{uncertainty bonus}}$$

- $\hat{\mu}_a$ = average reward from arm $a$
- $n_a$ = number of times we pulled arm $a$
- $t$ = total pulls so far
- $c$ = exploration constant (often $\sqrt{2}$)

### Why This Formula? (Derivation)

**Step 1: Concentration Inequality**

By Hoeffding's inequality, with probability $1 - \delta$:

$$|\hat{\mu}_a - \mu_a| \leq \sqrt{\frac{\ln(1/\delta)}{2n_a}}$$

**Step 2: Set Confidence Level**

We want our confidence to increase with time. Set $\delta = t^{-4}$:

$$|\hat{\mu}_a - \mu_a| \leq \sqrt{\frac{\ln(t^4)}{2n_a}} = \sqrt{\frac{4\ln t}{2n_a}} = \sqrt{\frac{2\ln t}{n_a}}$$

**Step 3: Upper Bound**

The true mean is below the upper confidence bound with high probability:

$$\mu_a \leq \hat{\mu}_a + \sqrt{\frac{2\ln t}{n_a}} = \text{UCB}_a$$

**Step 4: Optimism**

By always picking $\arg\max_a \text{UCB}_a$, we either:
1. Exploit arms with high estimated reward, OR
2. Explore arms with high uncertainty (low $n_a$)

### Why Optimism Works

If $\text{UCB}_a$ is too high (overestimate):
- We pull arm $a$ frequently
- $n_a$ increases
- Uncertainty term shrinks
- $\text{UCB}_a$ converges to true $\mu_a$

The algorithm is **self-correcting**!

---

## Complete Numerical Walkthrough: UCB Over 10 Rounds

### Setup

**3 Arms** with true (unknown) probabilities:
- Arm 1: $\mu_1 = 0.3$
- Arm 2: $\mu_2 = 0.5$ (best!)
- Arm 3: $\mu_3 = 0.2$

**UCB Formula**: $\text{UCB}_a = \hat{\mu}_a + \sqrt{\frac{2\ln t}{n_a}}$

### Round-by-Round Execution

**Round 1**: Pull each arm once (initialization)
- Pull Arm 1: Reward = 0 (unlucky)
- Pull Arm 2: Reward = 1 (lucky)
- Pull Arm 3: Reward = 0 (expected)

| Arm | $n_a$ | Successes | $\hat{\mu}_a$ |
|-----|-------|-----------|---------------|
| 1 | 1 | 0 | 0.0 |
| 2 | 1 | 1 | 1.0 |
| 3 | 1 | 0 | 0.0 |

**Round 4** (t=4): Calculate UCB

$$\text{UCB}_1 = 0.0 + \sqrt{\frac{2\ln 4}{1}} = 0.0 + 1.67 = 1.67$$
$$\text{UCB}_2 = 1.0 + \sqrt{\frac{2\ln 4}{1}} = 1.0 + 1.67 = 2.67$$
$$\text{UCB}_3 = 0.0 + \sqrt{\frac{2\ln 4}{1}} = 0.0 + 1.67 = 1.67$$

**Decision**: Pull Arm 2 (highest UCB = 2.67)

**Result**: Reward = 1

| Arm | $n_a$ | Successes | $\hat{\mu}_a$ |
|-----|-------|-----------|---------------|
| 1 | 1 | 0 | 0.0 |
| 2 | 2 | 2 | 1.0 |
| 3 | 1 | 0 | 0.0 |

**Round 5** (t=5):

$$\text{UCB}_1 = 0.0 + \sqrt{\frac{2\ln 5}{1}} = 0.0 + 1.79 = 1.79$$
$$\text{UCB}_2 = 1.0 + \sqrt{\frac{2\ln 5}{2}} = 1.0 + 1.27 = 2.27$$
$$\text{UCB}_3 = 0.0 + \sqrt{\frac{2\ln 5}{1}} = 0.0 + 1.79 = 1.79$$

**Decision**: Pull Arm 2 (highest = 2.27)

**Result**: Reward = 0 (unlucky)

| Arm | $n_a$ | Successes | $\hat{\mu}_a$ |
|-----|-------|-----------|---------------|
| 1 | 1 | 0 | 0.0 |
| 2 | 3 | 2 | 0.67 |
| 3 | 1 | 0 | 0.0 |

**Round 6** (t=6):

$$\text{UCB}_1 = 0.0 + \sqrt{\frac{2\ln 6}{1}} = 0.0 + 1.89 = 1.89$$
$$\text{UCB}_2 = 0.67 + \sqrt{\frac{2\ln 6}{3}} = 0.67 + 1.09 = 1.76$$
$$\text{UCB}_3 = 0.0 + \sqrt{\frac{2\ln 6}{1}} = 0.0 + 1.89 = 1.89$$

**Decision**: Tie between Arms 1 and 3 (both 1.89). Pull Arm 1.

**Result**: Reward = 1

| Arm | $n_a$ | Successes | $\hat{\mu}_a$ |
|-----|-------|-----------|---------------|
| 1 | 2 | 1 | 0.5 |
| 2 | 3 | 2 | 0.67 |
| 3 | 1 | 0 | 0.0 |

**Rounds 7-10**: Continue similarly...

**Final State after Round 10**:

| Arm | $n_a$ | Successes | $\hat{\mu}_a$ | True $\mu_a$ |
|-----|-------|-----------|---------------|--------------|
| 1 | 3 | 1 | 0.33 | 0.3 |
| 2 | 5 | 3 | 0.60 | 0.5 |
| 3 | 2 | 0 | 0.0 | 0.2 |

### Regret Calculation

**Best arm**: Arm 2 with $\mu^* = 0.5$

**Optimal reward** (10 rounds): $10 \times 0.5 = 5.0$

**Our rewards**:
- Arm 1: 3 pulls × 0.3 expected = 0.9
- Arm 2: 5 pulls × 0.5 expected = 2.5
- Arm 3: 2 pulls × 0.2 expected = 0.4
- **Total expected**: 3.8

**Regret**: $5.0 - 3.8 = 1.2$

This is the price of exploration - but it's $O(\log T)$, not $O(T)$!

---

## Thompson Sampling: The Bayesian Approach

### The Intuition

**UCB**: Add uncertainty bonus deterministically.

**Thompson Sampling**: Model uncertainty **probabilistically** using Bayesian inference.

**For each arm**:
1. Maintain a **probability distribution** over the true reward rate
2. **Sample** from each distribution
3. Pull the arm with highest sample

### Why Beta Distribution?

For binary rewards (click/no-click), the Beta distribution is the **conjugate prior** for Bernoulli likelihood.

**Prior**: $\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$

**After observing** $s$ successes and $f$ failures:

**Posterior**: $\theta_a | \text{data} \sim \text{Beta}(\alpha_a + s, \beta_a + f)$

**Start with** $\text{Beta}(1, 1)$ = uniform prior (no information).

### Visual Understanding

**Beta distribution properties**:
- $\text{Beta}(1, 1)$: Uniform - no idea about $\theta$
- $\text{Beta}(2, 2)$: Slight peak at 0.5 - limited data suggests middle
- $\text{Beta}(10, 2)$: Strong peak near 0.83 - confident $\theta$ is high
- $\text{Beta}(2, 10)$: Strong peak near 0.17 - confident $\theta$ is low

### Numerical Example

**3 Arms**, after some observations:

| Arm | Successes | Failures | Posterior |
|-----|-----------|----------|-----------|
| 1 | 3 | 7 | Beta(4, 8) |
| 2 | 8 | 2 | Beta(9, 3) |
| 3 | 1 | 1 | Beta(2, 2) |

**Posterior means**:
- Arm 1: $\frac{4}{4+8} = 0.33$
- Arm 2: $\frac{9}{9+3} = 0.75$
- Arm 3: $\frac{2}{2+2} = 0.50$

**Pure exploitation**: Always pull Arm 2.

**Thompson Sampling**: Sample from each posterior.

Sample values (one draw):
- $\theta_1 \sim \text{Beta}(4, 8) = 0.28$
- $\theta_2 \sim \text{Beta}(9, 3) = 0.71$
- $\theta_3 \sim \text{Beta}(2, 2) = 0.61$

**Pull Arm 2** (0.71 highest).

Another draw:
- $\theta_1 \sim \text{Beta}(4, 8) = 0.41$
- $\theta_2 \sim \text{Beta}(9, 3) = 0.68$
- $\theta_3 \sim \text{Beta}(2, 2) = 0.82$

**Pull Arm 3** (0.82 highest)!

Because Arm 3 has high variance (low data), it occasionally samples high values, leading to exploration.

### Implementation

```python
import numpy as np

class ThompsonSampling:
    def __init__(self, n_arms):
        """
        Thompson Sampling for Bernoulli bandits.
        """
        self.n_arms = n_arms
        # Beta parameters: alpha = successes + 1, beta = failures + 1
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self):
        """
        Sample from each arm's posterior and pick the highest.
        """
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm, reward):
        """
        Update posterior after observing reward.

        reward: 1 for success, 0 for failure
        """
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


# Example
ts = ThompsonSampling(n_arms=3)

# Simulate 100 rounds
true_probs = [0.3, 0.5, 0.2]

for round in range(100):
    arm = ts.select_arm()
    reward = np.random.binomial(1, true_probs[arm])
    ts.update(arm, reward)

print(f"Final alpha (successes+1): {ts.alpha}")
print(f"Final beta (failures+1): {ts.beta}")
print(f"Estimated probabilities: {ts.alpha / (ts.alpha + ts.beta)}")
```

---

## Contextual Bandits: When Context Matters

### The Limitation of Basic Bandits

**Standard bandits**: Same reward distribution for all users.

**Reality**: Different users respond differently!

| User | Arm 1 (Trending) | Arm 2 (Personal) | Arm 3 (Diverse) |
|------|------------------|------------------|-----------------|
| Young user | 0.4 | 0.2 | 0.3 |
| Older user | 0.1 | 0.5 | 0.2 |

Best arm depends on user context!

### LinUCB: Linear Upper Confidence Bound

**Model**: Reward is linear in context features.

$$r_a = \mathbf{x}^T \boldsymbol{\theta}_a + \epsilon$$

where:
- $\mathbf{x}$ = context feature vector (user features)
- $\boldsymbol{\theta}_a$ = unknown parameter vector for arm $a$
- $\epsilon$ = noise

### LinUCB Derivation

**Step 1: Ridge Regression Estimate**

After observing $(\mathbf{x}_1, r_1), ..., (\mathbf{x}_t, r_t)$ for arm $a$:

$$\hat{\boldsymbol{\theta}}_a = (D_a^T D_a + \lambda I)^{-1} D_a^T \mathbf{r}_a$$

Let $A_a = D_a^T D_a + \lambda I$ and $\mathbf{b}_a = D_a^T \mathbf{r}_a$

Then $\hat{\boldsymbol{\theta}}_a = A_a^{-1} \mathbf{b}_a$

**Step 2: Confidence Bound**

The estimate has uncertainty. The confidence region:

$$|\mathbf{x}^T \boldsymbol{\theta}_a - \mathbf{x}^T \hat{\boldsymbol{\theta}}_a| \leq \alpha \sqrt{\mathbf{x}^T A_a^{-1} \mathbf{x}}$$

**Step 3: UCB for Linear Model**

$$\text{UCB}_a(\mathbf{x}) = \mathbf{x}^T \hat{\boldsymbol{\theta}}_a + \alpha \sqrt{\mathbf{x}^T A_a^{-1} \mathbf{x}}$$

The uncertainty term $\sqrt{\mathbf{x}^T A_a^{-1} \mathbf{x}}$ depends on:
1. How much data we have (through $A_a$)
2. How "unusual" the current context is (through $\mathbf{x}$)

### Implementation

```python
import numpy as np

class LinUCB:
    def __init__(self, n_arms, d, alpha=1.0, lambda_=1.0):
        """
        LinUCB for contextual bandits.

        n_arms: Number of arms
        d: Context feature dimension
        alpha: Exploration parameter
        lambda_: Ridge regression regularization
        """
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha

        # Initialize A_a = lambda * I and b_a = 0 for each arm
        self.A = [lambda_ * np.eye(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]

    def select_arm(self, x):
        """
        Select arm given context x.

        x: Context feature vector (d,)
        """
        ucb_values = []

        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]

            # UCB = x^T theta_hat + alpha * sqrt(x^T A^{-1} x)
            pred = x @ theta_hat
            uncertainty = self.alpha * np.sqrt(x @ A_inv @ x)

            ucb_values.append(pred + uncertainty)

        return np.argmax(ucb_values)

    def update(self, arm, x, reward):
        """
        Update model after observing reward.
        """
        # A_a += x x^T
        self.A[arm] += np.outer(x, x)

        # b_a += r * x
        self.b[arm] += reward * x


# Example
n_arms = 3
d = 5  # Context dimension

linucb = LinUCB(n_arms, d, alpha=0.5)

# Simulate
for round in range(100):
    # Random context
    x = np.random.randn(d)

    # Select arm
    arm = linucb.select_arm(x)

    # Simulate reward (true parameters unknown)
    true_theta = [np.array([1, 0, 0, 0, 0]),
                  np.array([0, 1, 0, 0, 0]),
                  np.array([0, 0, 1, 0, 0])]
    reward = x @ true_theta[arm] + np.random.randn() * 0.1

    # Update
    linucb.update(arm, x, reward)
```

---

## From Bandits to Full RL: MDP Formulation

### Why Bandits Aren't Enough

**Bandits assume**: Each action is independent. Today's choice doesn't affect tomorrow's state.

**But in recommendations**:
- Showing clickbait today makes user more likely to churn tomorrow
- Showing diverse content today changes user's learned preferences
- User's context (state) evolves based on our actions

### MDP: Markov Decision Process

**MDP**: Tuple $(S, A, P, R, \gamma)$

- **State** $s \in S$: User's current context (history, preferences, satisfaction)
- **Action** $a \in A$: Recommendation (which item to show)
- **Transition** $P(s' | s, a)$: How user state changes after action
- **Reward** $R(s, a)$: Immediate reward (click, watch time)
- **Discount** $\gamma \in [0, 1]$: How much to value future rewards

**Goal**: Find policy $\pi(a | s)$ maximizing:

$$J(\pi) = \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \right]$$

### Q-Learning and DQN

For larger state spaces, we use **Deep Q-Networks**:

$$Q(s, a; \theta) \approx Q^*(s, a)$$

**Loss**: Mean squared TD error

$$\mathcal{L}(\theta) = \mathbb{E}\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

### Implementation: DQN for Recommendations

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

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
        """Epsilon-greedy selection"""
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
```

---

## Reward Shaping for Recommendations

### The Challenge

**Sparse rewards**: User only provides feedback occasionally (rating after finishing movie).

**Delayed rewards**: Effect of recommendation seen days later (user retention).

**Solution**: Design intermediate rewards to guide learning.

### Reward Components

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

## What Can Go Wrong: RL Pitfalls in Production

### Pitfall 1: Exploration Hurts Short-Term Metrics

**The problem**: Exploration means showing suboptimal recommendations.

**Example**:
- Week 1: Deploy UCB algorithm
- Week 1 CTR: **-5%** (exploration costs)
- Management: "Turn it off!"
- Week 4 (if kept): **+10%** (learned optimal policy)

**Solutions**:
- Protected exploration budget (only explore on 5% of traffic)
- Offline evaluation before online deployment
- Set expectations with stakeholders

### Pitfall 2: Delayed Rewards Hard to Attribute

**The problem**: User churns 2 weeks after a bad recommendation. Which recommendation caused it?

**Credit assignment** is extremely difficult:
- 100 recommendations over 2 weeks
- Which one caused churn?
- Maybe it was a combination?
- Maybe external factors (competitor launched)?

**Solutions**:
- Shorter feedback loops (next-day return as proxy)
- Counterfactual analysis
- Careful experiment design (A/B test over long periods)

### Pitfall 3: Non-Stationarity

**The problem**: User preferences change over time. Optimal policy today may be suboptimal tomorrow.

**Examples**:
- Trending topics change
- User life events (new job, new city, new baby)
- Seasonal patterns

**Solutions**:
- Continuous learning with forgetting (recent data weighted higher)
- Monitor for distribution shift
- Periodic model refresh

### Pitfall 4: Safety and Constraints

**The problem**: RL explores freely, but some actions are dangerous.

**Dangerous exploration**:
- Recommending harmful content
- Showing inappropriate ads
- Violating user preferences (e.g., showing meat recipes to vegetarian)

**Solutions**:
- Constrained RL (never violate certain rules)
- Safe exploration (only explore among "safe" arms)
- Human-in-the-loop for edge cases

---

## Offline RL: Learning from Logged Data

### Why Offline?

**Online RL**: Interact with users in real-time. Risky!

**Offline RL**: Learn from historical data. Safe!

### The Challenge: Distribution Shift

**Problem**: Logged data was collected by old policy $\pi_b$.

New policy $\pi$ might want to take actions that $\pi_b$ never tried!

**Example**:
- $\pi_b$ always recommended popular items
- $\pi$ wants to recommend niche items
- No data on niche item performance!

### Conservative Q-Learning (CQL)

**Idea**: Penalize Q-values for unseen actions.

$$\min_Q \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp(Q(s, a)) - \mathbb{E}_{a \sim \pi_b} [Q(s, a)] \right] + \text{Bellman error}$$

**Effect**: Q-values for in-distribution actions are accurate, out-of-distribution actions are underestimated.

---

## Production Deployment

### The Path to Production

1. **Offline evaluation**: Replay logged data, measure offline metrics
2. **Simulation**: Build user simulator, train and test
3. **Shadow mode**: Run alongside production, compare predictions
4. **A/B test**: Small traffic (1%), measure real metrics
5. **Gradual rollout**: 1% to 10% to 50% to 100%

### Metrics to Track

**Immediate**:
- Click-through rate (CTR)
- Watch time / Dwell time

**Long-term**:
- 7-day retention
- 30-day retention
- Lifetime value

**Safety**:
- Policy violations
- User complaints
- Content quality scores

---

## Summary

**Key Takeaways**:

1. **Greedy fails long-term**: Maximizing immediate clicks leads to clickbait and user churn
2. **Explore-exploit tradeoff**: Must balance learning (exploration) with performance (exploitation)
3. **UCB**: Optimism in face of uncertainty - add confidence bonus to estimates
4. **Thompson Sampling**: Bayesian approach - sample from posterior distributions
5. **Contextual bandits**: When reward depends on user context, use LinUCB
6. **Full RL**: When actions affect future states, need MDP formulation

**The Hierarchy of Methods**:
```
Complexity    Method              When to Use
---------    ------              -----------
Low          Epsilon-greedy      Simple baseline
Medium       UCB                 Need theoretical guarantees
Medium       Thompson Sampling   Bayesian setting, natural exploration
High         LinUCB              Context-dependent rewards
Highest      Full RL (DQN)       Actions affect future states
```

**Critical Production Considerations**:
- Exploration hurts short-term metrics (set expectations)
- Delayed rewards need careful attribution
- Non-stationarity requires continuous adaptation
- Safety constraints must be built in
- Always start with offline evaluation

**Next**: Evaluation methodologies for RecSys - how to measure if any of this actually works!

---

## References

1. **Sutton, R. S., & Barto, A. G. (2018)**. *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
   - **RL fundamentals**

2. **Auer, P., et al. (2002)**. "Finite-time Analysis of the Multiarmed Bandit Problem". *Machine Learning*.
   - **UCB derivation**

3. **Chapelle, O., & Li, L. (2011)**. "An Empirical Evaluation of Thompson Sampling". *NeurIPS*.
   - **Thompson Sampling**

4. **Li, L., et al. (2010)**. "A Contextual-Bandit Approach to Personalized News Article Recommendation". *WWW*.
   - **LinUCB**

5. **Chen, M., et al. (2019)**. "Top-K Off-Policy Correction for a REINFORCE Recommender System". *WSDM*.
   - **YouTube RL system**

6. **Kumar, A., et al. (2020)**. "Conservative Q-Learning for Offline Reinforcement Learning". *NeurIPS*.
   - **CQL for offline RL**
