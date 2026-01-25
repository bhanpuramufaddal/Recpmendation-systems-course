# Week 16: Emerging Applications and Specialized Domains - Practice Problems

## Overview
Explore Airbnb, Uber Eats, dating apps, news aggregation, and Cursor AI's online RL for code completion.

---

## Problem 1: Two-Sided Marketplace (Airbnb)
**Difficulty:** Hard

**Challenge:** Optimize for both guests (find good listings) and hosts (get bookings)

**Objectives:**
- Guest satisfaction
- Host earnings
- Platform revenue

**Design:**
1. Ranking function balancing objectives
2. Handle constraints (location, availability, price)
3. Prevent "rich get richer" (popular listings dominate)
4. Fair exposure for new listings

**Learning Outcomes:** Design two-sided systems, balance stakeholders, ensure fairness

---

## Problem 2: Context-Heavy Recommendations (Uber Eats)
**Difficulty:** Medium

**Context:**
- Time: breakfast, lunch, dinner, late-night
- Weather: hot, cold, rainy
- Location: home, work, outdoors
- Past orders

**Design:**
1. Feature engineering for context
2. Model architecture (include context features)
3. Handle sparse contexts (midnight + rainy + outdoors = rare)
4. Adapt to user feedback in real-time

**Learning Outcomes:** Incorporate rich context, handle sparsity, adapt dynamically

---

## Problem 3: Dating Apps - Mutual Interest
**Difficulty:** Hard

**Challenge:** Recommendations must be bidirectional (both must like each other)

**Approach:**
1. Predict P(A likes B) and P(B likes A)
2. Score = P(A likes B) × P(B likes A)
3. Optimize for matches (mutual likes)
4. Prevent exhaustion (running out of candidates)

**Questions:**
1. How do you handle gender imbalance?
2. How do you ensure diversity in recommendations?
3. Success = users leave platform. How to handle?
4. Ethical considerations?

**Learning Outcomes:** Design two-sided matching, handle constraints, consider ethics

---

## Problem 4: Cursor AI Online Reinforcement Learning
**Difficulty:** Very Hard

**Innovation:** Learn from production in real-time (1.5-2 hour loops)

**System:**
1. Deploy model
2. Collect user actions (accept/reject/ignore)
3. Compute rewards
4. Update policy with RL
5. Deploy updated model

**Challenges:**
1. Design reward function (accept=+1, reject=-1, silence=?)
2. On-policy vs. off-policy learning
3. Ensure stability (avoid catastrophic forgetting)
4. Monitor for degradation

**Learning Outcomes:** Implement online RL, design reward functions, ensure system stability

---

## Problem 5: News Aggregation - Freshness vs. Personalization
**Difficulty:** Medium

**Trade-off:**
- **Personalization:** Show topics user likes
- **Freshness:** Show breaking news
- **Diversity:** Prevent filter bubbles
- **Credibility:** Prioritize authoritative sources

**Design:**
1. Scoring function balancing factors
2. Decay function for article age
3. Source credibility scores
4. Filter bubble detection and mitigation

**Learning Outcomes:** Balance competing objectives, incorporate freshness, promote diversity

---

## Programming Exercises

### Exercise 1: Two-Sided Airbnb Ranker

```python
def rank_listings(guest, listings):
    scores = []
    for listing in listings:
        guest_score = predict_guest_satisfaction(guest, listing)
        host_score = predict_host_acceptance(guest, listing)
        platform_score = predict_booking_probability(guest, listing)

        # Multi-stakeholder score
        final_score = 0.5 * guest_score + 0.3 * host_score + 0.2 * platform_score
        scores.append((listing, final_score))

    return sorted(scores, key=lambda x: x[1], reverse=True)
```

---

### Exercise 2: Contextual Uber Eats Recommender

```python
def recommend_restaurants(user, time, weather, location):
    # Encode context
    context = encode_context(time, weather, location)

    # User preferences
    user_features = get_user_features(user)

    # Combine
    features = np.concatenate([user_features, context])

    # Predict
    scores = model.predict(features)

    # Boost contextually relevant (e.g., soup when cold)
    if weather == 'cold':
        scores[soup_restaurants] *= 1.2

    return scores.argsort()[-10:][::-1]
```

---

### Exercise 3: Online RL for Cursor AI

```python
class OnlineRLRecommender:
    def __init__(self, policy_model):
        self.policy = policy_model
        self.buffer = []

    def get_action(self, state):
        # Policy outputs: suggest or not
        return self.policy(state)

    def record_feedback(self, state, action, reward):
        self.buffer.append((state, action, reward))

        # Update every 1000 interactions
        if len(self.buffer) >= 1000:
            self.update_policy()
            self.buffer = []

    def update_policy(self):
        # Policy gradient update
        states, actions, rewards = zip(*self.buffer)
        loss = compute_policy_gradient_loss(states, actions, rewards)
        loss.backward()
        optimizer.step()
```

---

### Exercise 4: Dating App Match Predictor

```python
def predict_match_probability(userA, userB):
    # Predict both directions
    prob_A_likes_B = model.predict(userA, userB)
    prob_B_likes_A = model.predict(userB, userA)

    # Match probability
    match_prob = prob_A_likes_B * prob_B_likes_A

    return match_prob

def recommend_for_user(user):
    candidates = get_candidate_pool(user)
    match_probs = {c: predict_match_probability(user, c) for c in candidates}

    # Diversify (not all same age/location)
    diversified = diversify_matches(match_probs)

    return diversified[:20]
```

---

## Discussion Questions

1. **Two-Sided Fairness:** How to balance platform revenue vs. participant welfare?
2. **Context Overfitting:** Risk of overusing context (recommend soup every cold day)
3. **Dating Ethics:** Algorithmic matchmaking raises privacy, fairness, discrimination concerns
4. **Online RL Risks:** What if online learning makes model worse?
5. **News Diversity:** How much personalization before it becomes echo chamber?

---

## References
1. Haldar, M., et al. (2019). "Applying deep learning to Airbnb search". KDD.
2. Cui, Q., et al. (2020). "Deep learning for click-through rate estimation". IJCAI.
3. Cursor Team (2024). "Improving Cursor Tab with online RL".

---

*Return to [Week 16 Main Page](README.md)*
