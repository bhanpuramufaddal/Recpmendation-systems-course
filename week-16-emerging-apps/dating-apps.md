# Week 16: Dating Apps (Tinder, Bumble, Hinge)

## Overview

**Dating apps**: Two-sided matching problem.

**Platforms**: Tinder (swipe-based), Bumble (women-first), Hinge (prompts-based).

**Unique challenges**:
1. **Mutual attraction**: Both must like each other
2. **Gender imbalance**: More men than women
3. **Ghosting**: Matches don't lead to conversations
4. **Ethical considerations**: Fairness, transparency

---

## Matching Algorithm (Tinder)

### Elo Rating System

**Idea**: Rank users by "desirability" (like chess ratings).

**Process**:
1. New user starts at base score (e.g., 1200)
2. **Swipe right** on high-Elo user → your Elo increases if they match
3. **Swipe left** by high-Elo user → your Elo decreases

**Formula** (simplified):
$$\text{New Elo} = \text{Old Elo} + K \cdot (\text{Actual} - \text{Expected})$$

where:
- $K$ = learning rate (e.g., 32)
- Actual = 1 if matched, 0 if not
- Expected = $\frac{1}{1 + 10^{(\text{Other Elo} - \text{Your Elo}) / 400}}$

**Matching**: Show users with similar Elo scores.

---

### Collaborative Filtering

**Beyond Elo**: Learn preferences from swipe patterns.

**Approach**: Matrix factorization on swipe data.

**Matrix**:
```
         User A  User B  User C
Profile 1   1      0      1     (1=swipe right, 0=left)
Profile 2   0      1      1
Profile 3   1      1      0
```

**Learn**: Latent factors for users and profiles → predict compatibility.

---

## Hinge's "Most Compatible"

### Machine Learning Approach

**Hinge dropped Elo** (2018) → ML-based.

**Goal**: Predict probability of conversation.

**Features**:
- Profile completeness
- Photo quality (face visibility, smile)
- Prompt responses (humor, thoughtfulness)
- Shared interests (music, hobbies)
- Geographic proximity

**Model**: Gradient boosted trees (XGBoost).

**Outcome**: "Most Compatible" shown daily (users 8x more likely to match).

---

## Bumble's Beeline

### Premium Feature

**Beeline**: See who already swiped right on you.

**Free users**: Blurred preview (incentive to upgrade).

**Recommendation**: Order beeline by compatibility.

**Metrics**:
- Mutual friends (Facebook integration)
- Shared interests
- Recent activity (active users ranked higher)

---

## Handling Gender Imbalance

### Problem

**Typical ratio**: 60-70% men, 30-40% women.

**Impact**:
- Women get overwhelmed (100+ matches)
- Men get few matches → frustration

---

### Solutions

**1. Bumble's approach**: Women must message first.
- Reduces spam for women
- Encourages meaningful conversations

**2. Quality over quantity**:
- Limit daily swipes (Tinder: 100/day for free users)
- Encourages selective swiping

**3. Elo/ranking**:
- Show high-quality profiles first
- Reduce low-effort swipes

---

## Conversation Prediction

### Goal

**Predict**: Will match lead to conversation?

**Features**:
- Profile completeness (both users)
- Message history (past conversations)
- Time to first message
- Response rate

**Use case**: Rank profiles by conversation likelihood (not just match likelihood).

**Outcome**: Higher-quality matches → better retention.

---

## Ethical Considerations

### Algorithmic Fairness

**Problem**: Algorithms can perpetuate biases.

**Examples**:
- Racial bias (profiles shown less based on race)
- Body type discrimination (certain body types ranked lower)

**Mitigation**:
- Audit for bias (demographic parity)
- User controls ("Show me everyone")
- Transparency ("Here's why we showed this profile")

---

### Addictive Design

**Problem**: Swipe mechanics can be addictive (slot machine effect).

**Variable reward**: Sometimes match, sometimes don't → dopamine loop.

**Ethical question**: Is maximizing engagement ethical in dating?

**Solutions**:
- Limit swipes per day
- Encourage meeting in person (Hinge: "Designed to be deleted")

---

## Location-Based Matching

### Proximity

**Default**: Show users within 5-50 miles.

**Ranking**: Closer users ranked higher.

**Use cases**:
- **Urban**: Tight radius (1-5 miles)
- **Rural**: Wider radius (50+ miles)

**Implementation**:
```python
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c

def proximity_score(user_location, candidate_location, max_distance=50):
    distance = haversine_distance(*user_location, *candidate_location)
    return max(0, 1 - distance / max_distance)
```

---

## Summary

**Key Takeaways**:
1. **Elo rating**: Rank users by desirability (Tinder)
2. **ML-based**: Predict conversation likelihood (Hinge)
3. **Gender imbalance**: Limit swipes, women-first messaging (Bumble)
4. **Conversation prediction**: Optimize for quality matches
5. **Ethics**: Fairness, transparency, avoid addictive design

**Metrics**: Match rate, conversation rate, date rate, retention.

---

## References

1. **Hinge Data Science Blog**: "The Dating Apocalypse" (2019).
2. **Austin, E. (2015)**. "A First-Timer's Guide to Tinder". *Vanity Fair*.
