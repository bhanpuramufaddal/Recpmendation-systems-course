# Week 8: Two-Tower Models - YouTube Case Study

## Overview

**YouTube's recommendation system** serves **1 billion+ hours** of video watched per day to **2 billion+ users**.

**Business impact**: Recommendations drive **70%+ of watch time** on YouTube.

**Challenge**: Recommend from **800 million+ videos** in **real-time** (<100ms).

**Solution**: Two-stage architecture with deep learning.

**Paper**: Covington et al., "Deep Neural Networks for YouTube Recommendations" (RecSys 2016)

This document covers YouTube's complete recommendation architecture.

---

## Learning Objectives

By the end of this section, you will:
- Understand YouTube's two-stage architecture
- Master candidate generation with two-tower models
- Learn ranking model design
- Apply lessons to build similar systems at scale
- Implement YouTube-style recommendations

---

## The YouTube Recommendation Challenge

### Scale (2016 Paper, ~10x larger today)

**Users**: 1B+ monthly active (now 2B+)
**Videos**: 500M+ (now 800M+)
**Uploads**: 300+ hours per minute (now 500+)
**Watch time**: 1B+ hours per day

**Constraints**:
- Fresh recommendations (new videos appear constantly)
- Low latency (<100ms for page load)
- Personalized for each user
- Handle extreme sparsity (billions of user-video pairs)

---

### Two-Stage Architecture

**Why two stages?**
- Can't score all 800M videos with complex model (too slow)
- Solution: **Retrieve** candidates fast (stage 1), **Rank** precisely (stage 2)

```
800M Videos
     ↓
Stage 1: Candidate Generation (fast, broad)
  → Retrieve ~1000 candidates
     ↓
1000 Videos
     ↓
Stage 2: Ranking (slow, precise)
  → Rank top-N for user
     ↓
~20 Videos (homepage)
```

**Latency**:
- Stage 1: <10ms
- Stage 2: <50ms
- Total: <100ms ✓

---

## Stage 1: Candidate Generation

### Architecture (Two-Tower Model)

**Goal**: Retrieve ~1000 relevant videos from 800M.

**Approach**: Encode user and videos into same embedding space.

```
User Features          Video Features
(watch history, search)   (title, tags, stats)
       ↓                        ↓
   User Tower              Video Tower
   (Deep NN)               (Deep NN)
       ↓                        ↓
User Embedding          Video Embedding
   (256-dim)                (256-dim)
       ↓                        ↓
     Dot Product → Similarity Score
```

---

### User Tower Features

**1. Watch History**:
- **Embedded watch history**: IDs of recently watched videos
- Embed each video ID, average embeddings
- Captures user's interests

**2. Search History**:
- **Embedded search tokens**: Words from recent searches
- Embed tokens, average
- Captures explicit intent

**3. Demographics**:
- Age, gender, geographic location
- Embed categorical features

**4. Context**:
- Device (mobile, desktop, TV)
- Time of day
- Day of week

---

### Video Tower Features

**1. Video ID**:
- Learned embedding for each video
- Most important feature

**2. Metadata**:
- Title (embedded with Word2Vec or BERT)
- Tags, category
- Upload timestamp

**3. Engagement Signals**:
- Total views, likes, shares
- Average watch time
- Click-through rate

---

### Training Objective

**Problem**: Classify which video user will watch next.

**Formulation**: Multi-class classification (each video = a class).

**Softmax**:
$$P(\text{video } i | \text{user } u) = \frac{\exp(\mathbf{u}^T \mathbf{v}_i)}{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}^T \mathbf{v}_j)}$$

**Challenge**: Denominator sums over all 800M videos → expensive!

**Solution**: **Sampled SoftMax** (negative sampling).

$$P(i | u) \approx \frac{\exp(\mathbf{u}^T \mathbf{v}_i)}{\sum_{j \in \text{sample}} \exp(\mathbf{u}^T \mathbf{v}_j)}$$

Sample ~1000 negatives instead of using all videos.

---

### Example Time as Feature

**Key insight**: Videos are time-sensitive (trending, news).

**Problem**: Model trained on old data recommends old videos.

**Solution**: **Example age as feature**.

**Feature**: "Time since video upload" at training time.

**At inference**: Set to 0 (pretend video just uploaded) → boosts new videos.

**Effect**: System recommends fresh content.

---

### Implementation (Simplified)

```python
import torch
import torch.nn as nn

class YouTubeCandidateModel(nn.Module):
    def __init__(self, n_videos, n_search_tokens, embedding_dim=256):
        super().__init__()

        # Video embeddings (for watch history)
        self.video_embedding = nn.Embedding(n_videos, embedding_dim)

        # Search token embeddings
        self.search_embedding = nn.Embedding(n_search_tokens, embedding_dim)

        # User tower (combines watch history + search + demographics)
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 10, 512),  # +10 for demographics/context
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

        # Video tower
        self.video_tower = nn.Sequential(
            nn.Linear(embedding_dim + 20, 512),  # +20 for metadata/engagement
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, watch_history, search_tokens, user_demographics, video_features):
        """
        watch_history: (batch, history_len) - video IDs
        search_tokens: (batch, search_len) - search token IDs
        user_demographics: (batch, 10) - age, gender, location, etc.
        video_features: (batch, 20) - video metadata + engagement
        """
        # Encode watch history (average)
        watch_embs = self.video_embedding(watch_history)  # (batch, history_len, emb_dim)
        watch_avg = watch_embs.mean(dim=1)  # (batch, emb_dim)

        # Encode search history (average)
        search_embs = self.search_embedding(search_tokens)  # (batch, search_len, emb_dim)
        search_avg = search_embs.mean(dim=1)  # (batch, emb_dim)

        # Combine user features
        user_features = torch.cat([watch_avg, search_avg, user_demographics], dim=-1)

        # User embedding
        user_emb = self.user_tower(user_features)  # (batch, emb_dim)

        # Video embedding
        video_emb = self.video_tower(video_features)  # (batch, emb_dim)

        # L2 normalize
        user_emb = nn.functional.normalize(user_emb, p=2, dim=1)
        video_emb = nn.functional.normalize(video_emb, p=2, dim=1)

        # Dot product
        scores = (user_emb * video_emb).sum(dim=1)  # (batch,)

        return scores, user_emb, video_emb
```

---

### Serving Candidate Generation

**Offline**:
1. Compute video embeddings for all 800M videos (daily batch)
2. Build ANN index (FAISS or custom)

**Online** (per request):
1. Compute user embedding from features
2. Query ANN index: top-1000 nearest videos
3. Return candidates to ranking stage

**Latency**: <10ms

---

## Stage 2: Ranking

### Architecture

**Goal**: Rank 1000 candidates to select top-20.

**Approach**: Richer model with more features + cross-features.

**Model**: Deep neural network with feature crosses.

```
User Features + Video Features + Context
              ↓
       Feature Crosses
  (user_age × video_category, etc.)
              ↓
       Deep Neural Network
         (3-4 layers)
              ↓
      Predicted Watch Time
```

---

### Ranking Features

**1. User Features**:
- Demographics (age, gender, location)
- Watch history (detailed: watch time per video, completion rate)
- Search history

**2. Video Features**:
- Metadata (title, tags, category, duration)
- Engagement (views, likes, shares, CTR)
- Freshness (time since upload)
- Channel (creator's past performance)

**3. User-Video Cross Features**:
- User's past engagement with this channel
- User's past engagement with this video category
- User's language vs. video language

**4. Contextual Features**:
- Time of day, day of week
- Device type
- User's location vs. video's popularity in that location

---

### Ranking Objective

**Goal**: Predict **expected watch time**.

$$\text{score}(u, v) = \mathbb{E}[\text{watch time} | u, v]$$

**Why watch time (not just click)?**
- Clickbait videos have high CTR but low watch time
- Watch time better aligns with user satisfaction

**Training**: Regression on actual watch time.

$$\mathcal{L} = \sum_{(u, v, t)} (\text{predicted}_t - \text{actual}_t)^2$$

where $t$ = watch time.

---

### Weighted Logistic Regression Trick

**Problem**: Most videos are not watched (label = 0) → class imbalance.

**YouTube's approach**: Treat as **weighted logistic regression**.

**Labels**:
- Positive: Watched (weight = watch time)
- Negative: Not watched (weight = 1)

**Benefit**: Positive samples weighted by engagement → longer watches matter more.

---

### Implementation (Simplified)

```python
class YouTubeRankingModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Predicted watch time
        )

    def forward(self, features):
        """
        features: (batch, n_features) - all user, video, cross features
        """
        watch_time_pred = self.layers(features)  # (batch, 1)
        return watch_time_pred.squeeze()


# Training
model_rank = YouTubeRankingModel(n_features=500)
optimizer = torch.optim.Adam(model_rank.parameters(), lr=0.001)
criterion = nn.MSELoss()

for batch in ranking_data_loader:
    features, watch_times = batch  # watch_times = actual watch time (seconds)

    # Predict
    pred_watch_time = model_rank(features)

    # Loss
    loss = criterion(pred_watch_time, watch_times)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

### Serving Ranking

**Input**: 1000 candidates from stage 1

**Process**:
1. Extract features for each (user, video) pair
2. Run through ranking model
3. Sort by predicted watch time
4. Return top-20

**Latency**: <50ms (1000 forward passes)

---

## Key Engineering Decisions

### 1. Negative Sampling

**Problem**: Training requires negative samples (videos user didn't watch).

**Naive**: Sample uniformly → biases toward obscure videos.

**YouTube's approach**: Sample proportional to video popularity.

$$P(\text{sample video } i) \propto (\text{popularity of } i)^{0.25}$$

**Effect**: Balances popular and niche videos in training.

---

### 2. Handling Fresh Content

**Challenge**: New videos have no watch history.

**Solutions**:

**a) Example Age Feature**:
- At training: Use actual video age
- At inference: Set to 0 → boosts new videos

**b) Exploration**:
- 10% of recommendations = fresh videos (random)
- Collect data → improve recommendations

---

### 3. Diversity

**Problem**: Model may recommend very similar videos (filter bubble).

**YouTube's approach**:
- Deduplicate: Don't show multiple videos from same creator in top-20
- Diversify topics: Mix different categories

---

## A/B Testing at YouTube

### Metrics

**Primary**:
- **Total watch time**: Sum of time users spend watching
- **Sessions per user**: How often users return

**Secondary**:
- **CTR** (click-through rate): Clicks / impressions
- **Completion rate**: % of video watched
- **Likes, shares, subscriptions**

**Guardrail**:
- **Satisfaction surveys**: Random polls

---

### Example A/B Test

**Hypothesis**: Adding "watch time" as objective improves engagement.

**Test**:
- **Control**: Optimize for CTR (clicks)
- **Treatment**: Optimize for watch time

**Results**:
- CTR: -5% (fewer clicks, less clickbait)
- Watch time: +10% (users watch more)
- Sessions: +3% (users return more often)

**Decision**: Ship treatment (watch time better aligns with goals).

---

## Evolution Over Time

### 2016 → 2024 Changes (Estimated)

**Scale**:
- Videos: 500M → 800M
- Users: 1B → 2B
- Embeddings: 256D → 512D (larger capacity)

**Models**:
- **Candidate generation**: Two-tower → Multi-tower (separate towers for shorts, live, etc.)
- **Ranking**: 3-layer DNN → Transformer-based models

**Features**:
- Added: Short-form content (Shorts), live streams, user comments
- Multimodal: Video frames (visual features), audio, transcripts

**Personalization**:
- More fine-grained (session-based, time-of-day, mood detection)

---

## Challenges & Solutions

### Challenge 1: Cold Start (New Users)

**Problem**: New users have no watch history.

**Solutions**:
- **Trending videos**: Show globally popular content
- **Onboarding**: Ask user to select interests
- **Demographic defaults**: Use age/gender/location to predict initial preferences

---

### Challenge 2: Cold Start (New Videos)

**Problem**: New videos have no engagement data.

**Solutions**:
- **Example age feature** (boost fresh content)
- **Creator history**: Use creator's past videos' performance
- **Content features**: Title, thumbnail, tags

---

### Challenge 3: Filter Bubble

**Problem**: Users only see similar content.

**Solutions**:
- **Exploration**: 10-20% of recommendations = diverse content
- **Topic mixing**: Show videos from multiple categories
- **"Break out of filter bubble"**: Periodic prompt to try new topics

---

### Challenge 4: Extreme Skew (Popularity Bias)

**Problem**: Popular videos dominate recommendations.

**Solutions**:
- **Calibration**: Adjust scores to promote diverse creators
- **Creator diversity**: Limit number of videos from same creator
- **Long-tail promotion**: Dedicated sections for niche content

---

## Lessons from YouTube

### 1. Two-Stage is Essential

**Candidate generation + Ranking** enables scale.

**Don't**: Try to rank all items with one model.

**Do**: Retrieve fast, rank precisely.

---

### 2. Watch Time > Clicks

**CTR optimizes** for clickbait.

**Watch time optimizes** for satisfaction.

**Lesson**: Choose objective that aligns with business goals.

---

### 3. Fresh Content Matters

**Users want** recent, trending content.

**Solution**: Boost fresh content (example age feature, exploration).

---

### 4. A/B Test Everything

**Even small changes** (e.g., feature engineering) → A/B test.

**Metrics matter**: Don't just look at CTR, measure watch time, sessions, satisfaction.

---

### 5. Simplicity First

**Start simple** (two-tower, basic features).

**Add complexity incrementally** (cross-features, transformers).

**YouTube's 2016 model**: Relatively simple (2-3 layer NNs) → huge impact.

---

## Summary

**Key Takeaways**:
1. **Two-stage**: Candidate generation (1000 from 800M) + Ranking (20 from 1000)
2. **Candidate generation**: Two-tower model, dot product similarity, ANN retrieval
3. **Ranking**: Deep NN with cross-features, predicts watch time
4. **Objective**: Optimize for watch time (not CTR)
5. **Fresh content**: Example age feature, exploration
6. **A/B testing**: Extensive testing, multiple metrics

**Architecture**:
```
800M Videos
    ↓
Stage 1: Candidate Generation (<10ms)
  Two-tower model + ANN → 1000 candidates
    ↓
Stage 2: Ranking (<50ms)
  Deep NN + cross-features → Top-20
    ↓
Homepage Recommendations
```

**Technologies**:
- Models: Two-tower (stage 1), Deep NN (stage 2)
- Retrieval: ANN (FAISS or custom)
- Scale: 800M videos, 2B users
- Latency: <100ms end-to-end

**For Builders**:
- Start with two-stage architecture
- Use two-tower for candidate generation
- Optimize for meaningful objective (not just clicks)
- A/B test extensively
- Handle fresh content explicitly

---

## References

1. **Covington, P., Adams, J., & Sargin, E. (2016)**. "Deep Neural Networks for YouTube Recommendations". *RecSys*.
   - **Original YouTube paper** (2016 system)

2. **Zhao, Z., et al. (2019)**. "Recommending What Video to Watch Next: A Multitask Ranking System". *RecSys*.
   - **Updated YouTube system** (multitask learning)

3. **Yi, X., et al. (2019)**. "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations". *RecSys*.
   - YouTube's approach to **negative sampling**

4. **Cheng, H.-T., et al. (2016)**. "Wide & Deep Learning for Recommender Systems". *RecSys*.
   - Google's Wide & Deep (used in Google Play, similar to YouTube ranking)

5. **Davidson, J., et al. (2010)**. "The YouTube Video Recommendation System". *RecSys*.
   - **Pre-deep learning** YouTube system (historical perspective)

---

## Practice Problems

### Problem 1: Two-Stage Speedup

**Given**:
```
Total videos: 800M
Candidate generation: Retrieves 1000 videos
Ranking model: 10ms per video

Without two-stage: 800M × 10ms = ?
With two-stage: Candidate (10ms) + Ranking (1000 × 10ms) = ?
```

**Solution**:
```
Without: 800M × 10ms = 8 billion ms = 92.6 days!
With: 10ms + 10s = ~10 seconds

Speedup: 800,000x faster!
```

---

### Problem 2: Watch Time Objective

**Given**:
```
Video A: CTR = 10%, Avg watch time = 30 seconds
Video B: CTR = 5%, Avg watch time = 5 minutes

Which to recommend if optimizing for watch time?
```

**Solution**:
```
Video A engagement: 0.10 × 30 = 3 seconds (expected)
Video B engagement: 0.05 × 300 = 15 seconds (expected)

Recommend Video B (higher expected watch time).
```

---

### Problem 3: Negative Sampling

**Given popularity distribution**:
```
Video A: 1M views
Video B: 100K views
Video C: 10K views
```

**Compute**: Sampling probabilities (proportional to popularity^0.25).

**Solution**:
```
A: 1,000,000^0.25 = 31.62
B: 100,000^0.25 = 17.78
C: 10,000^0.25 = 10.00

Sum = 59.4

P(A) = 31.62 / 59.4 = 0.53
P(B) = 17.78 / 59.4 = 0.30
P(C) = 10.00 / 59.4 = 0.17
```
