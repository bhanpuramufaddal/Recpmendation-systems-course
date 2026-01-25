# Week 16: Emerging Applications and Specialized Domains

## Overview

Beyond social media and streaming, recommendation systems power diverse applications from travel to dating to code completion. This week explores specialized domains with unique constraints and opportunities.

## Learning Objectives

- Understand domain-specific recommendation challenges
- Learn about two-sided marketplace recommendations
- Master contextual and constrained recommendation problems
- Study cutting-edge applications (Cursor AI's online RL for code completion)

## Topics Covered

### [1. Airbnb Search and Recommendations](airbnb.md)
**Two-Sided Marketplace**

**Challenges**:
- Guest satisfaction AND host earnings
- Location constraints
- Real-time availability
- Price sensitivity

**Key Topics**:
- Listing quality scoring
- Guest-host compatibility
- Similar listings
- Supply-demand balancing

### [2. Uber Eats and Food Delivery](uber-eats.md)
**Context-Heavy Recommendations**

**Signals**:
- Time of day (breakfast vs. dinner)
- Weather (cold → soup, hot → salad)
- Location (delivery time)
- Dietary restrictions

**Features**:
- Restaurant ranking
- Reorder shortcuts
- New restaurant discovery

### [3. Dating Apps](dating-apps.md)
**Two-Sided Matching with Ethical Considerations**

**Platforms**: Tinder, Bumble, Hinge

**Challenges**:
- Mutual interest required (two-sided)
- Gender imbalance
- Success = users leave platform
- Ethical algorithm design

**Topics**:
- Collaborative filtering on swipe data
- Profile completeness signals
- Geographic constraints
- Preventing match exhaustion

### [4. News Aggregation](news-aggregation.md)
**Freshness vs. Personalization**

**Platforms**: Google News, Apple News

**Challenges**:
- Breaking news detection
- Misinformation handling
- Source credibility
- Filter bubble prevention

**Topics**:
- Topic diversification
- Freshness-relevance trade-off
- Cross-device behavior

### [5. GitHub and Developer Tools](github.md)
**Code and Repository Recommendations**

**Use Cases**:
- Repository discovery
- Trending topics
- Package suggestions
- Issue/PR recommendations

### [6. **Cursor AI: Code Completion with Online RL** ⭐](cursor-ai.md)
**Revolutionary Online Reinforcement Learning**

**Innovation**: First large-scale online RL in production for recommendations

**Scale**:
- 400M+ requests per day
- Multiple deployments per day
- 1.5-2 hour training loops

**Results**:
- 21% fewer suggestions (noise reduction)
- 28% higher accept rate
- Learning when NOT to suggest

**Technical Details**:
- On-policy data collection
- Policy gradient methods
- Reward shaping (acceptance, rejection, flow)
- Real-time learning from production

**Why Revolutionary**:
- First successful large-scale online RL
- vs. GitHub Copilot's offline logistic regression
- Continuous improvement from user feedback

**Paper**: Cursor Team (2024). "Improving Cursor Tab with online RL"

### [7. Education Platforms](education.md)
**Adaptive Learning**

**Platforms**: Coursera, Khan Academy, Duolingo

**Topics**:
- Learning path personalization
- Prerequisite modeling
- Difficulty adaptation
- Completion likelihood

## Comparison: Specialized Domain Challenges

| Domain | Primary Constraint | Success Metric | Unique Challenge |
|--------|-------------------|----------------|------------------|
| **Airbnb** | Availability, location | Bookings | Two-sided marketplace |
| **Uber Eats** | Delivery time, context | Orders | Real-time context (weather, time) |
| **Dating** | Mutual interest | Matches, dates | Users leave when successful |
| **News** | Freshness | Click-through, diversity | Misinformation, bias |
| **Cursor AI** | Code context, latency | Accept rate | Real-time RL from production |
| **Education** | Prerequisites | Learning outcomes | Knowledge dependencies |

## The Cursor AI Deep Dive

### Why This Matters

**First deployment** of online RL at massive scale for recommendations.

### Traditional Approach (GitHub Copilot)
1. Collect data offline
2. Train logistic regression filter
3. Deploy model
4. Repeat (slow iteration)

### Cursor's Online RL Approach
1. Deploy model
2. Collect accept/reject data in real-time
3. Train policy gradient model
4. Deploy updated model (1.5-2 hours)
5. Repeat continuously

### Reward Function

```python
reward = {
    'accept': +0.75,      # User accepted suggestion
    'reject': -0.25,      # User rejected suggestion
    'no_suggestion': 0     # Model chose not to suggest
}
```

**Key Insight**: Learning when NOT to suggest is as important as good suggestions.

### Architecture

```
User Context (code, cursor position, file) → Policy Network
    ↓
[Suggest | Don't Suggest] + Suggestion Content
    ↓
User Action (Accept/Reject/Ignore)
    ↓
Reward Signal
    ↓
Policy Gradient Update
```

### Results

| Metric | Before Online RL | After Online RL | Improvement |
|--------|------------------|-----------------|-------------|
| **Suggestions per session** | 100 | 79 | -21% (less noise) |
| **Accept rate** | 25% | 32% | +28% |
| **User satisfaction** | Baseline | Higher | Subjective |

### Lessons Learned

1. **On-policy critical**: Off-policy RL struggles in this domain
2. **Fast iteration**: 1.5-2 hour loops enable rapid improvement
3. **Reward shaping matters**: Balancing accept/reject/silence crucial
4. **Infrastructure investment**: Real-time data pipeline essential

## Required Reading

### Papers

1. **Cui, Q., et al. (2020)**. "Deep Learning for Click-Through Rate Estimation". *IJCAI*.
2. **Haldar, M., et al. (2019)**. "Applying deep learning to Airbnb search". *KDD*.
3. **Cursor Team (2024)**. "Improving Cursor Tab with online RL". ⭐

### Industry Blog Posts

1. **Airbnb Engineering**: "Machine Learning-Powered Search Ranking"
2. **Uber Engineering**: "Food Discovery with Uber Eats"
3. **Hinge**: "The Hinge Algorithm: Designed to be Deleted"

## Practice Exercises

### Exercise 1: Two-Sided Marketplace

Design a recommendation system for a freelance marketplace (like Upwork) that:
- Matches freelancers to jobs
- Balances freelancer earnings and client satisfaction
- Handles geographic and skill constraints

### Exercise 2: Online RL Simulation

Simulate Cursor AI's online RL approach:
1. Start with random policy
2. Collect user feedback (simulate with acceptance rates)
3. Update policy using policy gradient
4. Measure improvement over iterations

### Exercise 3: Contextual Recommendations

Design a food delivery recommendation system that uses:
- Time of day
- Weather
- Previous orders
- Dietary restrictions
- Delivery time estimates

## Assessment

**Project**: Build a recommendation system for one specialized domain
**Report**: Include domain analysis, algorithm choice, evaluation metrics
**Grading**: 10% of final grade

## Next Week

**Week 17**: Explainability and Interpretability
- Why recommendations are made
- SHAP, LIME for RecSys
- User trust and transparency

*Return to [Main Course Page](../README.md)*
