# Week 14: Social Media Platforms - Practice Problems

## Overview
Study Facebook, Instagram, TikTok, and LinkedIn recommendation systems, multi-objective optimization, and feed ranking challenges.

---

## Problem 1: Multi-Objective Feed Ranking
**Difficulty:** Hard

**Objectives:**
- P(click): Maximize clicks
- P(like): Maximize engagement
- P(share): Maximize distribution
- E[time]: Maximize watch time

**Design scoring function:** $Score = w_1 P(click) + w_2 P(like) + w_3 P(share) + w_4 E[time]$

**Tasks:**
1. How do you learn weights w1, w2, w3, w4?
2. What if objectives conflict? (clickbait gets clicks but not shares)
3. How do you incorporate negative signals (hide, report)?
4. Design multi-task neural network

**Learning Outcomes:** Balance multiple objectives, design scoring functions, handle conflicts

---

## Problem 2: TikTok For You Page Algorithm
**Difficulty:** Medium

**Key signals:**
- Watch time (especially completion rate)
- Likes, shares, comments
- Video information (captions, sounds, hashtags)
- NOT: Follower count, past video performance

**Questions:**
1. Why doesn't follower count matter?
2. How does TikTok achieve such good personalization from day 1?
3. Design cold start strategy for new users
4. How to prevent filter bubbles while maintaining engagement?

**Learning Outcomes:** Understand viral algorithms, design rapid personalization, balance discovery and relevance

---

## Problem 3: Position Bias Correction
**Difficulty:** Hard

**Problem:** Items at top of feed get more clicks regardless of quality

**Data:**
- Position 1: 10% CTR
- Position 5: 3% CTR
- Position 10: 1% CTR

**Solutions:**
1. Train with position as feature, predict at position=1
2. Inverse propensity scoring (IPS)
3. Randomization for unbiased data collection

**Tasks:** Implement each method, compare effectiveness, design evaluation

**Learning Outcomes:** Debias training data, correct position effects, collect unbiased data

---

## Problem 4: Engagement vs. Well-being
**Difficulty:** Hard

**Trade-off:** Optimizing for time spent may harm user well-being (addiction, misinformation)

**Questions:**
1. How do you measure user well-being?
2. Design metrics beyond engagement (satisfaction surveys, retention)
3. Propose algorithm changes to balance engagement and well-being
4. How do you A/B test well-being?

**Learning Outcomes:** Balance business and ethics, measure well-being, design responsible algorithms

---

## Problem 5: Real-Time Feature Engineering
**Difficulty:** Hard

**Challenge:** Features must be computed in <100ms for feed ranking

**Features:**
- **Precomputed (offline):** User demographics, post content embeddings
- **Real-time (online):** Current session, trending score, recency

**Design:**
1. Feature store architecture
2. Online/offline feature separation
3. Caching strategy
4. Fallback for feature failures

**Learning Outcomes:** Design real-time systems, optimize feature computation, ensure reliability

---

## Programming Exercises

### Exercise 1: Multi-Task Feed Ranker

```python
class MultiTaskFeedRanker(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.click_head = nn.Linear(256, 1)
        self.like_head = nn.Linear(256, 1)
        self.share_head = nn.Linear(256, 1)
        self.time_head = nn.Linear(256, 1)

    def forward(self, x):
        shared_repr = self.shared(x)
        return {
            'click': torch.sigmoid(self.click_head(shared_repr)),
            'like': torch.sigmoid(self.like_head(shared_repr)),
            'share': torch.sigmoid(self.share_head(shared_repr)),
            'time': self.time_head(shared_repr)
        }

# Training
def multi_task_loss(outputs, targets, weights={'click': 0.2, 'like': 0.3, 'share': 0.4, 'time': 0.1}):
    losses = {}
    losses['click'] = F.binary_cross_entropy(outputs['click'], targets['click'])
    losses['like'] = F.binary_cross_entropy(outputs['like'], targets['like'])
    losses['share'] = F.binary_cross_entropy(outputs['share'], targets['share'])
    losses['time'] = F.mse_loss(outputs['time'], targets['time'])

    total_loss = sum(weights[k] * losses[k] for k in losses)
    return total_loss
```

---

### Exercise 2: Position Bias Debiasing

```python
def train_with_position_features(model, data):
    # Include position as feature during training
    for batch in data:
        features = torch.cat([batch['features'], batch['position'].unsqueeze(1)], dim=1)
        predictions = model(features)
        loss = criterion(predictions, batch['labels'])
        # ... training step ...

def predict_without_position(model, features):
    # At inference, set position = 1 (top position)
    position = torch.ones(len(features), 1)
    features_with_pos = torch.cat([features, position], dim=1)
    return model(features_with_pos)
```

---

## Discussion Questions

1. **Algorithm Transparency:** Should platforms reveal how feeds are ranked?
2. **Filter Bubbles:** How do you measure and prevent echo chambers?
3. **Misinformation:** What role do recommenders play in spreading misinformation?
4. **Addiction:** Are platforms responsible for addictive design?
5. **Content Moderation:** How to balance free speech and harmful content removal?
6. **Personalization Limits:** When is too much personalization harmful?

---

## References
1. Zhao, X., et al. (2019). "Recommending what video to watch next: A multitask ranking system". RecSys. (YouTube)
2. Beutel, A., et al. (2019). "Fairness in recommendation ranking through pairwise comparisons". KDD.

---

*Return to [Week 14 Main Page](README.md)*
