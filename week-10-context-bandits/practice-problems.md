# Week 10: Context-Aware and Bandit-Based Recommendations - Practice Problems

## Overview
Master contextual recommendations, multi-armed bandits, contextual bandits (LinUCB), and reinforcement learning for long-term optimization.

---

## Problem 1: Context-Aware Features
**Difficulty:** Easy

Design contextual features for restaurant recommendations:
- Time: breakfast (7-11am), lunch (11-2pm), dinner (6-10pm)
- Location: home, work, traveling
- Weather: sunny, rainy, cold
- Group size: solo, couple, family

**Tasks:**
1. How would you encode these features?
2. What interactions matter? (e.g., time × weather)
3. Design a tensor factorization model: User × Item × Context
4. Compare with standard MF (no context)

**Learning Outcomes:** Engineer contextual features, design tensor models, incorporate context

---

## Problem 2: Exploration vs. Exploitation
**Difficulty:** Medium

**Scenario:** E-commerce homepage with 10 slots. You can show popular items (safe) or new items (risky).

**Strategies:**
1. **Pure exploitation:** Always show top-10 popular items
2. **Pure exploration:** Show random items
3. **ε-greedy:** Show popular with prob 1-ε, random with prob ε
4. **UCB:** Show items with highest upper confidence bound

**Questions:**
1. Calculate expected revenue for each strategy (assume popularity correlates with CTR)
2. What ε would you choose? (0.05, 0.1, 0.2?)
3. How does UCB balance exploration/exploitation?
4. Which strategy is best long-term?

**Learning Outcomes:** Balance exploration/exploitation, choose strategies, analyze trade-offs

---

## Problem 3: LinUCB Algorithm
**Difficulty:** Hard

LinUCB maintains confidence intervals for each arm (item):

**Score:** $\theta_a^T x + \alpha \sqrt{x^T A_a^{-1} x}$

where:
- $\theta_a$ = estimated reward for arm a
- $x$ = context features
- $A_a$ = covariance matrix
- $\alpha$ = exploration parameter

**Tasks:**
1. What does the $\sqrt{x^T A_a^{-1} x}$ term represent?
2. Why does uncertainty decrease as arm is pulled more?
3. Implement one step of LinUCB
4. Compare with ε-greedy on simulated data

**Learning Outcomes:** Understand confidence bounds, implement LinUCB, quantify uncertainty

---

## Problem 4: Thompson Sampling
**Difficulty:** Hard

Thompson Sampling samples from posterior distributions:

**Algorithm:**
```
For each arm a:
  Sample reward_a ~ Beta(successes_a, failures_a)
Choose arm with highest sampled reward
```

**Questions:**
1. Why sample instead of choosing highest mean?
2. How does this implement exploration?
3. Compare with UCB theoretically
4. Implement for binary rewards (click/no-click)

**Learning Outcomes:** Implement Thompson Sampling, understand Bayesian bandits, compare strategies

---

## Problem 5: Reinforcement Learning for Recommendations
**Difficulty:** Very Hard

**MDP formulation:**
- **State:** User history, current context
- **Action:** Recommend item
- **Reward:** Immediate (click) + long-term (retention, satisfaction)
- **Policy:** Learned recommendation strategy

**Questions:**
1. Why RL instead of supervised learning?
2. What is the reward function? (How do you measure long-term value?)
3. How do you handle delayed rewards?
4. Design the state representation

**Learning Outcomes:** Formulate recommendations as RL, design reward functions, think long-term

---

## Programming Exercises

### Exercise 1: Implement ε-Greedy Bandit

```python
class EpsilonGreedyBandit:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)  # Explore
        else:
            return np.argmax(self.values)  # Exploit

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
```

**Evaluation:** Simulate with varying reward distributions, measure cumulative regret

---

### Exercise 2: Implement LinUCB

```python
class LinUCB:
    def __init__(self, n_arms, n_features, alpha=1.0):
        self.n_arms = n_arms
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, context):
        p = []
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv.dot(self.b[a])
            ucb = theta.dot(context) + self.alpha * np.sqrt(context.dot(A_inv).dot(context))
            p.append(ucb)
        return np.argmax(p)

    def update(self, arm, context, reward):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
```

---

### Exercise 3: Context-Aware Movie Recommendations

```python
def recommend_with_context(user, time_of_day, device):
    # Extract features
    user_features = get_user_features(user)
    context_features = encode_context(time_of_day, device)
    combined = np.concatenate([user_features, context_features])

    # Predict with context-aware model
    scores = model.predict(combined)
    return scores.argsort()[-10:][::-1]
```

**Evaluation:** A/B test context-aware vs. context-free recommendations

---

### Exercise 4: Simulate Reinforcement Learning

```python
class RecommenderEnv:
    def __init__(self, users, items):
        self.users = users
        self.items = items

    def step(self, user, item):
        # Simulate user interaction
        immediate_reward = self.click_probability(user, item)
        long_term_reward = self.satisfaction_change(user, item)

        reward = immediate_reward + 0.5 * long_term_reward
        next_state = self.update_user_state(user, item)

        return next_state, reward

# Train with Q-learning or policy gradient
```

---

## Discussion Questions

1. **Regret Minimization:** What is cumulative regret? How do different bandit algorithms compare?
2. **Cold Start:** How do bandits handle new items with no data?
3. **Ethical Exploration:** Is it ethical to explore (show suboptimal recommendations) on real users?
4. **Reward Shaping:** How do you balance clicks vs. long-term engagement?
5. **Off-Policy Evaluation:** How do you evaluate a new policy without deploying it?

---

## References
1. Li, L., et al. (2010). "A contextual-bandit approach to personalized news article recommendation". WWW. (LinUCB)
2. Chapelle, O., & Li, L. (2011). "An empirical evaluation of Thompson sampling". NIPS.
3. Chen, M., et al. (2019). "Top-K off-policy correction for a REINFORCE recommender system". WSDM. (YouTube RL)

---

*Return to [Week 10 Main Page](README.md)*
