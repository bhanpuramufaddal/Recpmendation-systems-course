# Week 14: Social Media Platforms - Industry Case Studies

## Overview

This week explores how major social media platforms build and deploy recommendation systems at massive scale. You'll learn about real architectures, ranking signals, and the unique challenges of social feeds.

## Learning Objectives

- Understand production recommendation architectures at Facebook, Instagram, TikTok, and LinkedIn
- Learn multi-objective optimization for engagement, well-being, and revenue
- Master the challenges of feed ranking at billion-user scale
- Recognize the impact of recommendation algorithms on society

---

## Topics Covered

### [1. Facebook News Feed Ranking](facebook-newsfeed.md)
**Evolution from EdgeRank to Deep Learning**

**Key Topics**:
- EdgeRank (2010): Affinity × Weight × Time Decay
- Machine learning era (2013+): Thousands of signals
- Multi-task deep learning (2018+)
- Balancing engagement, meaningful interactions, well-being

**Architecture**:
```
User/Post Features → Neural Network → Multi-Task Heads
                                          ├─> P(like)
                                          ├─> P(comment)
                                          ├─> P(share)
                                          └─> E[time_spent]

Final Score = w1×P(like) + w2×P(comment) + w3×P(share) + w4×E[time]
```

**Challenges**:
- Misinformation and clickbait
- Filter bubbles and polarization
- Inventory management (limited feed slots)
- Real-time feature computation

**Latest Updates (2023-2024)**:
- Shift from engagement to "meaningful interactions"
- Reduced viral misinformation
- Transparent ranking criteria

---

### [2. Instagram Explore and Reels](instagram.md)
**Visual Discovery and Short-Form Video**

**Key Topics**:
- Explore page algorithm
- Reels ranking (competing with TikTok)
- Account recommendations
- Visual understanding with computer vision

**Top Ranking Signals (Confirmed Jan 2025)**:
1. **Watch time**: Especially first 3 seconds
2. **Likes per reach**: % of viewers who like
3. **Sends per reach**: DM sharing (most powerful)

**Reels Algorithm**:
- Watch-until-end prediction
- Trending audio/effects boost
- No watermarks from other platforms
- Original content prioritization

**Explore Page Flow**:
```
User's Likes/Saves → Seed Content
    ↓
Find Similar Content (Embeddings)
    ↓
Rank by Engagement Likelihood
    ↓
Diversify Topics → Display
```

**Latest Feature (Dec 2025)**:
- "Your Algorithm" tool: View and customize recommendation topics
- User control over interest steering

---

### [3. TikTok's For You Page](tiktok.md)
**The Ultimate Engagement Machine**

**Key Innovation**: Interest graph > Social graph

**Core Ranking Signals** (Official):
1. **User interactions**: Watch time, likes, shares, comments, skips
2. **Video information**: Captions, hashtags, sounds, effects
3. **User information**: Language, country, device type

**What Doesn't Matter** (Officially):
- Follower count
- Previous video performance
- Account verification status

**Algorithm Mechanics**:
1. New video shown to small test audience
2. High engagement → broader distribution
3. Personalization from day one
4. Rapid learning of preferences
5. Content freshness prioritized

**Technical Architecture**:
```
Video Upload → Content Understanding (CV, NLP)
    ↓
Test Distribution (Small Audience)
    ↓
Engagement Signals Collection
    ↓
If High Engagement → Broader Distribution
    ↓
Personalized For You Page Ranking
```

**Challenges**:
- Addictive design patterns
- Content moderation at scale
- Echo chambers and radicalization
- Mental health impact on teens

**2024 Transparency**:
- Public sharing of ranking factors
- User controls for sensitive content
- Break reminders after extended use

---

### [4. LinkedIn Feed and Job Recommendations](linkedin.md)
**Professional Network Dynamics**

**Unique Aspects**:
- Professional context (career advancement, not entertainment)
- B2B content (job posts, professional articles)
- Graph structure (colleagues, alumni, industry connections)

**Feed Ranking**:
- **Content types**: Posts, articles, job listings, ads, learning courses
- **Signals**: Industry relevance, seniority, skills, engagement history
- **Balance**: Organic content vs. sponsored content

**Job Recommendations**:
```
Job Posting Features:
  - Title, description, company, location, salary
  - Required skills, experience level

User Features:
  - Current role, skills (endorsed), experience
  - Career trajectory, job search activity

Matching Score = Skill Match × Experience Match × Location Match
                 × Company Preference × Application Likelihood
```

**People You May Know (PYMK)**:
- Mutual connections (2nd degree)
- Same company/school
- Similar industry/title
- Profile views (asymmetric)

**Challenges**:
- Spam and low-quality content
- Balancing ads with user experience
- International differences in professional norms

---

## Architectural Diagrams

### Two-Stage Recommendation Pipeline (Common Pattern)

```
┌──────────────────────────────────────────────────────────┐
│                    STAGE 1: CANDIDATE GENERATION          │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Social    │  │  Interest   │  │  Trending   │     │
│  │   Graph     │  │   Based     │  │   Content   │     │
│  │  (Friends)  │  │   (Topics)  │  │  (Viral)    │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                 │                 │             │
│         └─────────────────┴─────────────────┘             │
│                           │                               │
│                  500-1000 Candidates                      │
└───────────────────────────┬───────────────────────────────┘
                            │
┌───────────────────────────┴───────────────────────────────┐
│                    STAGE 2: RANKING                        │
│                                                            │
│   User Features + Post Features + Context Features        │
│                    ↓                                       │
│   ┌─────────────────────────────────────────────┐        │
│   │    Multi-Task Deep Neural Network           │        │
│   │   (Shared Layers + Task-Specific Towers)    │        │
│   └─────────────┬───────────────────────────────┘        │
│                 │                                          │
│   ┌─────────────┼─────────────┬────────────┐            │
│   │             │              │            │             │
│ P(Click)   P(Like)        P(Share)   E[Time_Spent]       │
│   │             │              │            │             │
│   └─────────────┴──────────────┴────────────┘             │
│                 │                                          │
│          Final Ranking Score                              │
└───────────────────────────┬────────────────────────────────┘
                            │
┌───────────────────────────┴────────────────────────────────┐
│                  STAGE 3: RE-RANKING                        │
│                                                             │
│  - Diversity (avoid similar posts in a row)                │
│  - Freshness (boost recent content)                        │
│  - Business rules (ad insertion, sponsored content)        │
│  - Fairness (creator exposure, prevent spam)               │
│                                                             │
│  → Final Feed Order (Top 20-50 items)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Comparison Table

| Platform | Primary Metric | Key Innovation | Main Challenge | Scale |
|----------|---------------|----------------|----------------|-------|
| **Facebook** | Meaningful interactions | Multi-objective optimization | Misinformation, polarization | 3B+ users |
| **Instagram** | Watch time (Reels) | Visual understanding (CV) | TikTok competition | 2B+ users |
| **TikTok** | Watch time, completion | Interest graph, rapid personalization | Addictive design, moderation | 1B+ users |
| **LinkedIn** | Professional engagement | Skills-based matching | Spam, low-quality content | 900M+ users |

---

## Multi-Objective Optimization

### The Challenge

**Cannot optimize for single metric** (e.g., time spent) because:
- Leads to clickbait, divisive content
- Harms user well-being
- Reduces long-term retention
- Societal concerns (polarization, mental health)

### Solution: Multi-Task Learning

**Model Architecture**:
```python
# Shared representation
shared = DenseNet(user_features + post_features + context)

# Task-specific towers
click_prob = DenseLayer(shared) → sigmoid → P(click)
like_prob = DenseLayer(shared) → sigmoid → P(like)
share_prob = DenseLayer(shared) → sigmoid → P(share)
time_spent = DenseLayer(shared) → linear → E[time]

# Combined objective
loss = w1×BCE(click_prob) + w2×BCE(like_prob)
       + w3×BCE(share_prob) + w4×MSE(time_spent)
```

**Weight Tuning**:
- Not just accuracy weights
- Encode business priorities
- Example: $w_{share} > w_{like} > w_{click}$ (shares are more meaningful)

### Real-World Objectives (Facebook)

| Objective | Weight | Rationale |
|-----------|--------|-----------|
| Clicks | Low | Easy to game with clickbait |
| Likes | Medium | Shows approval but passive |
| Comments | High | Active engagement |
| Shares | Very High | Strong signal of value |
| Time Spent | Medium | Can be manipulative |
| Hiding/Reporting | Negative | Clear negative signal |

**Final Score**:
$$\text{Score} = \sum_i w_i \cdot P(\text{action}_i) - \sum_j w_j \cdot P(\text{negative}_j)$$

---

## Position Bias Correction

### The Problem

Users more likely to interact with content at the top of the feed, **regardless of relevance**.

**Example**:
- Position 1: 10% click rate
- Position 10: 2% click rate (even for same quality content)

### Solutions

**1. Position-Aware Training**
```
Features = [content_features, user_features, position]
Model learns: P(click | content, user, position)
Inference: Set position = 1 for fair comparison
```

**2. Inverse Propensity Weighting**
```
Weight training examples by 1 / P(observe | position)
Downweight top positions, upweight lower positions
```

**3. Result Randomization**
- For small % of users, randomize order
- Collect unbiased data for training
- Use to calibrate main model

---

## Real-Time Feature Engineering

### Challenge

Features must be computed **in milliseconds** for real-time ranking.

### Approach: Feature Store

```
┌────────────────────────────────────────────┐
│           OFFLINE PROCESSING                │
│  (Batch jobs, daily/hourly updates)        │
│                                             │
│  - User historical features                │
│  - Post content features (CV, NLP)         │
│  - Graph features (friend connections)     │
│                                             │
│  → Feature Store (Redis, Cassandra)        │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────┴──────────────────────────┐
│         ONLINE SERVING (<100ms)             │
│                                              │
│  - Fetch precomputed features from store    │
│  - Compute real-time features (current session)  │
│  - Run model inference                      │
│  - Return ranked feed                       │
└──────────────────────────────────────────────┘
```

**Precomputed (Offline)**:
- User demographics
- Historical engagement rates
- Social graph embeddings
- Post content embeddings (text, image via CLIP)

**Real-Time (Online)**:
- Time of day, device
- Current session activity
- Post recency
- Trending score

---

## Ethical Considerations

### Filter Bubbles

**Problem**: Recommendation algorithms can create echo chambers

**Mitigation**:
- Diversity constraints in re-ranking
- Expose users to diverse viewpoints
- Transparent controls ("Why am I seeing this?")

### Mental Health

**Problem**: Infinite scroll + perfect personalization = addiction

**Mitigation**:
- Usage time reminders
- Optional daily limits
- "Take a break" prompts
- Reduced recommendation aggression for teens

### Misinformation

**Problem**: Viral false information spreads faster than truth

**Mitigation**:
- Fact-checking partnerships (3rd party verification)
- Reduce distribution of borderline content
- Authoritative sources prioritization
- Warning labels on disputed claims

### Content Moderation

**Scale**: Billions of posts per day

**Approach**:
- AI classifiers for policy violations
- Human review for edge cases
- User reporting mechanisms
- Contextual understanding (satire vs. hate speech)

---

## Required Reading

### Papers

1. **Beutel, A., et al. (2019)**. "Fairness in recommendation ranking through pairwise comparisons". *KDD*.

2. **Chen, J., et al. (2019)**. "Top-K off-policy correction for a REINFORCE recommender system". *WSDM*. (YouTube)

3. **Zhao, X., et al. (2019)**. "Recommending what video to watch next: A multitask ranking system". *RecSys*. (YouTube)

### Industry Blog Posts

1. **Facebook Engineering** (2021). "How Facebook's Feed Ranking System Works"
2. **Instagram Creators** (2024). "How Instagram's Ranking Algorithm Works"
3. **TikTok Newsroom** (2020). "How TikTok recommends videos #ForYou"
4. **LinkedIn Engineering** (2019). "AI at LinkedIn: Driving Members' Success"

---

## Practice Exercises

### Exercise 1: Multi-Objective Optimization

Design a scoring function for a social feed that balances:
- User engagement (clicks, likes, shares)
- Content diversity (avoid echo chambers)
- Creator fairness (small accounts get exposure)
- Platform revenue (ad clicks)

Propose weights and justify your choices.

### Exercise 2: A/B Test Analysis

Given:
- **Control**: Current ranking algorithm, 5% CTR, 10 min/session
- **Treatment**: New algorithm, 6% CTR, 8 min/session

Should you launch the new algorithm? Consider:
- Short-term metrics (CTR)
- Long-term metrics (retention)
- Ethical implications (is higher CTR always better?)

### Exercise 3: Position Bias

Compute the position bias from the following data:

| Position | Impressions | Clicks | CTR |
|----------|-------------|--------|-----|
| 1 | 10000 | 1000 | 10% |
| 2 | 10000 | 600 | 6% |
| 3 | 10000 | 400 | 4% |
| 10 | 10000 | 100 | 1% |

Design an inverse propensity weighting scheme to debias training data.

---

## Week Schedule

**Day 1**: Facebook News Feed
- Read: facebook-newsfeed.md
- Watch: Facebook Engineering talk on feed ranking

**Day 2**: Instagram Explore & Reels
- Read: instagram.md
- Analyze: Instagram's official transparency post (2025)

**Day 3**: TikTok For You Page
- Read: tiktok.md
- Discuss: Ethical implications of engagement optimization

**Day 4**: LinkedIn
- Read: linkedin.md
- Compare: Job recommendations vs. content recommendations

**Day 5**: Synthesis & Ethics
- Discussion: Balancing engagement, well-being, and revenue
- Debate: Should platforms optimize for time spent?

---

## Assessment

**Case Study Analysis**: Choose one platform and write a 5-page analysis covering:
- Architecture and algorithms
- Business objectives
- Ethical considerations
- Recommendations for improvement

**Grading**: 10% of final grade

---

## Next Week

**Week 15**: Streaming and E-Commerce Platforms
- Netflix, Spotify, YouTube, Amazon
- Different objectives (watch time, purchases)
- Long-form vs. short-form content

---

*Return to [Main Course Page](../README.md)*
