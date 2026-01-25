# Week 15: Streaming - Netflix Recommendation System

## Overview

Netflix is the **gold standard** for recommendation systems. With 80%+ of viewing driven by recommendations, their RecSys directly impacts billions of dollars in revenue and user retention.

This document covers Netflix's complete recommendation architecture, from the Netflix Prize era (2006-2009) to modern deep learning systems (2024).

**Business impact**: Estimated **\$1 billion per year** in value from recommendations (preventing churn).

---

## Learning Objectives

By the end of this section, you will:
- Understand Netflix's end-to-end recommendation architecture
- Master the concept of "row generation" and ranking
- Learn from Netflix's A/B testing culture
- Recognize why they didn't deploy the Netflix Prize winner
- Apply these lessons to build production RecSys

---

## Netflix at Scale (2024)

### The Numbers

- **Subscribers**: 260M+ globally (Q4 2024)
- **Content**: 15K+ titles
- **Hours watched**: 2B+ hours per day
- **Recommendations drive**: 80%+ of viewing
- **Regions**: 190+ countries
- **Languages**: 30+ UI languages

### The Challenge

**Problem**: Recommend personalized content to 260M users from 15K titles.

**Constraints**:
- Users expect fresh recommendations every time they open the app
- Content changes daily (new releases, expiring licenses)
- Preferences vary by region, culture, time of day
- Latency < 1 second for homepage load

---

## From Netflix Prize to Production

### The Netflix Prize (2006-2009)

**Goal**: Improve Cinematch (Netflix's algorithm) by 10% RMSE

**Winner**: BellKor's Pragmatic Chaos
- Ensemble of 100+ models (SVD++, RBMs, neighborhood methods)
- RMSE improvement: 10.06%
- Prize: \$1 million

**Why Netflix didn't deploy it**:

1. **Complexity**: 100+ models → expensive to train and serve
2. **Latency**: Ensemble inference too slow for real-time
3. **Rating prediction ≠ engagement**: Optimized RMSE, not watch time
4. **Shifting paradigm**: Moved from ratings to implicit feedback (plays, completion rate)
5. **Business priorities changed**: Homepage ranking > rating prediction

**Lesson**: Academic metrics (RMSE) ≠ business metrics (engagement, retention)

---

## Netflix Architecture (2024)

### The Homepage

**What users see**:

```
Continue Watching
───────────────────────
[Movie A] [Show B] [Movie C] ...

Top 10 in Your Country
───────────────────────
[#1] [#2] [#3] ...

Because You Watched "Stranger Things"
───────────────────────
[Similar 1] [Similar 2] ...

Trending Now
───────────────────────
[Movie X] [Show Y] ...

... (20-40 rows total)
```

**Two-level recommendation**:
1. **Row generation**: Which rows to show? (genres, themes)
2. **Ranking within rows**: Which titles in each row?

---

## Stage 1: Row Generation

### Personalized Rows

**Algorithm**: Select rows based on user profile

**Example rows**:
- **Continue Watching**: Resume in-progress titles
- **Top Picks for User**: Personalized top recommendations
- **Because You Watched X**: Content similar to recently watched
- **Trending in Your Country**: Popular in user's region
- **New Releases**: Recent additions
- **Award Winners**: Oscar/Emmy winners
- **Critically Acclaimed**: High-quality content

**Diversity**: Mix explicit genres (Action, Comedy) with themes (Feel-Good, Mind-Bending)

---

### Row Selection Algorithm

**Input**: User profile (watch history, ratings, demographics)

**Process**:
1. **Candidate rows**: Generate 100+ possible rows
2. **Relevance scoring**: Score each row for user
   $$\text{score}(\text{row}, \text{user}) = P(\text{user engages with row})$$
3. **Diversity filtering**: Remove similar rows
4. **Ranking**: Select top 20-40 rows

**Objective**: Maximize probability user finds something to watch.

---

## Stage 2: Ranking Within Rows

### Personalized Video Ranker (PVR)

**For each row, rank titles** based on predicted user engagement.

**Model**: Gradient Boosted Decision Trees (XGBoost) or Deep Neural Networks

**Features**:
- **User**: Watch history, ratings, demographics, device, time of day
- **Item**: Genre, actors, director, release year, popularity
- **Context**: Day of week, season, trending status
- **Engagement signals**: Click-through rate, completion rate, rewatch rate

**Prediction**:
$$P(\text{user watches item} | \text{features})$$

---

### Training Objective

**Positive signal**: User clicked and watched > 70% of title

**Negative signal**: User saw title but didn't click

**Loss**: Binary cross-entropy
$$\mathcal{L} = -\sum_{(u,i)} y_{ui} \log(\hat{y}_{ui}) + (1 - y_{ui}) \log(1 - \hat{y}_{ui})$$

**Challenge**: Position bias (titles at left are clicked more)

**Solution**: Inverse Propensity Scoring (IPS)
- Weight training examples by 1 / P(position)
- Debiases model

---

## Artwork Personalization

### The Problem

**Same title, different users** → show different artwork!

**Example: "Stranger Things"**
- User A (likes action): Show artwork with kids in danger
- User B (likes drama): Show artwork with emotional scene
- User C (likes sci-fi): Show artwork with Upside Down

**Impact**: 20-30% increase in engagement (A/B tested)

---

### Contextual Bandits

**Formulation**: Multi-Armed Bandit with context

**Context**: User features (watch history, demographics)

**Arms**: Different artworks for same title (3-10 variants)

**Reward**: Click or not (binary)

**Algorithm**: Thompson Sampling or Upper Confidence Bound (UCB)

**Process**:
1. For each user, select artwork with highest expected reward
2. Show artwork, observe click (reward)
3. Update posterior distribution for artwork
4. Repeat

**Result**: Personalized artwork improves CTR by 20%+

---

### Implementation

```python
# Simplified Thompson Sampling for artwork selection

class ArtworkBandit:
    def __init__(self, n_artworks):
        # Beta distribution parameters (prior: Beta(1,1))
        self.alpha = np.ones(n_artworks)  # Successes
        self.beta = np.ones(n_artworks)   # Failures

    def select_artwork(self):
        # Sample from posterior Beta distribution
        sampled_ctr = np.random.beta(self.alpha, self.beta)
        # Select artwork with highest sampled CTR
        return np.argmax(sampled_ctr)

    def update(self, artwork_id, reward):
        # Update posterior
        if reward == 1:
            self.alpha[artwork_id] += 1  # Success
        else:
            self.beta[artwork_id] += 1   # Failure

# Usage
bandit = ArtworkBandit(n_artworks=5)

for user in users:
    artwork = bandit.select_artwork()
    show_artwork(user, artwork)

    clicked = observe_click(user)
    bandit.update(artwork, clicked)
```

---

## Hydra: Multi-Task Learning (2024)

### The Problem

**Multiple objectives**:
- Maximize click-through rate (CTR)
- Maximize watch time
- Maximize completion rate
- Minimize churn risk

**Challenge**: Single model optimizing all objectives simultaneously.

---

### Hydra Architecture

**Multi-Task Learning**: Share lower layers, separate heads for each task.

```
Input Features
      ↓
Shared Layers (Dense NN)
      ↓
    Split
   /  |  \  \
CTR  Watch Completion Churn
Head  Time    Rate    Head
      Head    Head

```

**Loss**:
$$\mathcal{L}_{\text{total}} = \sum_{k=1}^K w_k \cdot \mathcal{L}_k$$

where $K$ = number of tasks, $w_k$ = task weight.

**Benefits**:
- Shared representations (transfer learning)
- Faster training
- Better generalization

---

## Session-Based Recommendations

### Capturing Short-Term Intent

**Observation**: User behavior within session differs from long-term profile.

**Example**:
- User typically watches dramas
- Today's session: Watched 2 comedies
- **Next rec**: More comedies (not dramas)

**Model**: Recurrent Neural Network (GRU) or Transformer

**Input**: Sequence of watched titles in current session

**Output**: Next title prediction

---

### Implementation

```python
import torch
import torch.nn as nn

class SessionRNN(nn.Module):
    def __init__(self, n_items, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_items)

    def forward(self, session_items):
        # session_items: (batch, seq_len)
        emb = self.item_embedding(session_items)  # (batch, seq_len, emb_dim)
        output, hidden = self.gru(emb)  # output: (batch, seq_len, hidden_dim)
        logits = self.fc(output[:, -1, :])  # Last timestep: (batch, n_items)
        return logits
```

---

## Cold Start Strategies

### New Content (Cold Items)

**Challenge**: No viewing history for new releases.

**Solutions**:

1. **Content-based features**:
   - Genre, actors, director, plot description
   - Extract features from metadata
   - Predict engagement based on similar titles

2. **Regional popularity**:
   - Use trending data from early-release regions
   - Example: Release in US first, use engagement to predict UK/EU interest

3. **"New Releases" row**:
   - Dedicate row to showcase new content
   - Collect data quickly

---

### New Users (Cold Users)

**Challenge**: No watch history.

**Solutions**:

1. **Onboarding quiz**:
   - "Select 3 titles you like"
   - Bootstrap user profile

2. **Demographic defaults**:
   - Use age, gender, location for initial recommendations

3. **Popular content**:
   - Show globally trending titles
   - Safe bet for new users

---

## Regional Personalization

### Cultural Differences

**Observation**: Content preferences vary significantly by region.

**Examples**:
- **India**: Bollywood, regional language films
- **Korea**: K-dramas, Korean variety shows
- **US**: Hollywood blockbusters, English-language content

**Solution**: Region-specific models or region as a feature.

---

### Language Preferences

**Challenge**: Multilingual content (dubbed/subtitled)

**Personalization**:
- Track user's language preference (UI language, subtitle language)
- Promote dubbed versions for users who prefer native language
- Promote subtitled versions for original audio fans

---

## Evaluation Metrics

### Offline Metrics

**Precision@K**:
$$\text{Precision@K} = \frac{|\text{relevant items in top K}|}{K}$$

**Recall@K**:
$$\text{Recall@K} = \frac{|\text{relevant items in top K}|}{|\text{all relevant items}|}$$

**NDCG@K**: Normalized Discounted Cumulative Gain

---

### Online Metrics (A/B Testing)

**Primary**:
- **Engagement rate**: % users who watch something
- **Watch time**: Average hours watched per user
- **Retention**: % users who return within 7/30 days

**Secondary**:
- **CTR**: Click-through rate on recommendations
- **Completion rate**: % of title watched
- **Replay rate**: % who rewatch content

**Business**:
- **Churn rate**: % users who cancel subscription
- **Revenue**: Impact on subscription revenue

**Goal**: Maximize engagement and retention, minimize churn.

---

## A/B Testing at Netflix

### Culture of Experimentation

**Scale**: 100+ A/B tests running concurrently

**Process**:
1. **Hypothesis**: New recommendation algorithm improves engagement
2. **Design test**: Treatment vs. Control (50/50 split)
3. **Run test**: 2-4 weeks
4. **Analyze**: Statistical significance (p < 0.05)
5. **Decision**: Ship, iterate, or kill

---

### Example: Personalized Row Order

**Hypothesis**: Reordering rows based on user preferences increases engagement.

**Test**:
- **Control**: Fixed row order (Continue Watching, Top Picks, New Releases, ...)
- **Treatment**: Personalized row order (e.g., move "Documentaries" higher for doc fans)

**Results** (hypothetical):
- **Engagement rate**: +2.5% (statistically significant)
- **Watch time**: +1.8%
- **Decision**: Ship to 100% of users

---

## Challenges and Solutions

### 1. The Filter Bubble

**Problem**: Users only see content similar to what they've watched.

**Solution**:
- **Exploration**: Inject 10-20% of diverse content
- **"Because You Watched" rows**: Limited to 30% of homepage
- **"Trending" rows**: Expose users to popular content outside their bubble

---

### 2. The Popularity Bias

**Problem**: Popular titles dominate recommendations.

**Solution**:
- **Regularization**: Penalize overly popular items in ranking
- **Long-tail promotion**: Dedicated rows for hidden gems
- **Fairness constraints**: Ensure niche content gets exposure

---

### 3. The Feedback Loop

**Problem**: Recommendations influence what users watch, biasing future recommendations.

**Solution**:
- **Counterfactual evaluation**: Estimate what user would have watched without recommendations
- **Logging policy**: Track recommendation source (algorithm vs. search vs. browsing)
- **Debiasing techniques**: Inverse propensity scoring, doubly robust estimators

---

## Technology Stack (Estimated)

**Data Storage**:
- **Hadoop/S3**: Petabytes of viewing data
- **Cassandra**: User profiles, real-time features

**Data Processing**:
- **Spark**: Batch processing (model training, feature engineering)
- **Flink**: Stream processing (real-time updates)

**Model Training**:
- **PyTorch/TensorFlow**: Deep learning models
- **XGBoost**: Gradient boosting
- **Custom frameworks**: Proprietary algorithms

**Model Serving**:
- **Microservices**: Personalized Video Ranker, Row Generation, Artwork Selection
- **Caching**: Redis for user features, model predictions
- **Load balancing**: 1000s of servers globally

**A/B Testing**:
- **Custom platform**: Experiment management, analysis
- **Metrics pipeline**: Real-time tracking of key metrics

---

## Lessons from Netflix

### 1. Offline Metrics ≠ Business Metrics

**Netflix Prize optimized RMSE**, but Netflix cares about **engagement and retention**.

**Takeaway**: Always validate offline improvements with online A/B tests.

---

### 2. Simplicity > Complexity

**Netflix Prize winner** was too complex to deploy. Simpler models (XGBoost, simple DNNs) work better in production.

**Takeaway**: Favor maintainable, interpretable models over complex ensembles.

---

### 3. Personalization at Multiple Levels

**Not just title recommendations**:
- Row generation (which rows to show)
- Ranking (which titles in rows)
- Artwork (which image to show)
- Messaging (which notifications to send)

**Takeaway**: Personalize every user touchpoint.

---

### 4. Continuous Experimentation

**100+ A/B tests** running at any time.

**Takeaway**: Build culture of experimentation. Ship fast, measure, iterate.

---

### 5. Context Matters

**Same user, different context** → different recommendations.
- Time of day: Lighter content at night
- Device: Movies on TV, shows on mobile
- Season: Holiday movies in December

**Takeaway**: Incorporate contextual features.

---

## Summary

**Key Takeaways**:
1. **80%+ of viewing from recommendations** → massive business impact
2. **Two-level system**: Row generation + ranking within rows
3. **Artwork personalization**: 20%+ CTR boost
4. **Multi-task learning**: Optimize multiple objectives (Hydra)
5. **Session-based**: Capture short-term intent with RNNs/Transformers
6. **A/B testing culture**: 100+ experiments, data-driven decisions
7. **Netflix Prize lesson**: RMSE ≠ engagement

**Architecture Overview**:
```
User → Row Generation → Ranking → Artwork Selection → Homepage
      (Which rows?)   (Which titles?) (Which image?)
```

**Technologies**:
- Models: XGBoost, PyTorch DNNs, GRUs, Contextual Bandits
- Scale: Petabytes of data, 260M users, 15K titles
- Latency: <1 second for homepage load

**For Builders**:
- Start simple (popularity, collaborative filtering)
- Add complexity incrementally (deep learning, multi-task)
- Always A/B test
- Personalize multiple touchpoints

---

## References

1. **Gomez-Uribe, C. A., & Hunt, N. (2016)**. "The Netflix Recommender System: Algorithms, Business Value, and Innovation". *ACM TMIS*.
   - **Comprehensive overview** of Netflix RecSys

2. **Amatriain, X., & Basilico, J. (2015)**. "Recommender Systems in Industry: A Netflix Case Study". *Recommender Systems Handbook*.
   - **Inside look** at Netflix's approach

3. **Chandrashekar, A., et al. (2017)**. "Artwork Personalization at Netflix". *Netflix Tech Blog*.
   - **Contextual bandits** for artwork

4. **Koren, Y. (2009)**. "The BellKor Solution to the Netflix Grand Prize". *Netflix Prize documentation*.
   - **Netflix Prize winner** technical details

5. **Basilico, J., & Raimond, Y. (2018)**. "Calibrated Recommendations". *RecSys*.
   - **Diversity and calibration** at Netflix
