# System Design: Instagram Recommendation System

## Problem Statement & Requirements

### Interview Prompt

> "Design a recommendation system for Instagram with 2B+ users, powering Explore, Reels, and Feed personalization."

### Functional Requirements

1. **Explore page**: Discover new content and creators
2. **Reels feed**: Short-form video recommendations
3. **Feed ranking**: Posts from followed accounts + suggested
4. **Stories tray**: Order of story circles
5. **Suggested accounts**: "Accounts you might like"

### Non-Functional Requirements

1. **Latency**: Feed/Explore load < 100ms
2. **Scale**: 2B MAU, 500M DAU, billions of Reels plays
3. **Freshness**: New content eligible within minutes
4. **Availability**: 99.9% uptime

### Scope

**In scope**: Content ranking, personalization, computer vision integration
**Out of scope**: Content upload, messaging, advertising

---

## Scale Estimation (Back-of-Envelope)

### Users & Traffic

```
Users:
- Monthly Active Users (MAU): 2B+
- Daily Active Users (DAU): 500M+
- Peak concurrent users: 50M

Content:
- Photos/videos shared per day: 95M+
- Stories posted per day: 500M+
- Reels uploaded per day: 10M+
- Active posts (last 7 days): 700M

Traffic:
- Feed loads per user per day: 10
- Explore visits per user: 3
- Reels sessions per user: 5
- Total recommendation requests: 500M × 18 = 9B/day
- Average QPS: 100,000
- Peak QPS: 300,000
```

### Storage

```
User Embeddings:
- Users: 2B
- Embedding dimension: 256
- Storage: 2B × 256 × 4 bytes = 2TB

Content Embeddings:
- Active posts: 700M
- Embedding dimension: 512 (vision + text)
- Storage: 700M × 512 × 4 bytes = 1.4TB

Visual Features:
- Posts with images: 10B (all time)
- Feature vector: 2048-dim (ResNet)
- Compressed storage: ~50TB
```

### Latency Budget

```
Total budget: 100ms

User feature lookup: 10ms
Candidate retrieval (ANN): 20ms
Visual embedding lookup: 15ms
Ranking model inference: 30ms
Diversity re-ranking: 15ms
Network overhead: 10ms
```

---

## Overview

Instagram (owned by Meta) is a visual-first social platform with **2+ billion monthly active users** (2024). In January 2025, Instagram released unprecedented transparency about their ranking algorithms for **Explore, Reels, and Feed**.

This document covers Instagram's recommendation systems with **confirmed ranking signals** from official sources.

**Business impact**: 50%+ of time spent on Reels and Explore (recommended content).

---

## Learning Objectives

By the end of this section, you will:
- Understand Instagram's ranking systems (Explore, Reels, Feed)
- Master the top ranking signals (watch time, likes per reach, sends)
- Learn how computer vision integrates with collaborative filtering
- Recognize Instagram's transparency efforts (2025)
- Apply lessons to build similar systems

---

## Instagram at Scale (2024-2025)

### The Numbers

- **Users**: 2B+ monthly active
- **Daily active users**: 500M+
- **Reels plays per day**: Billions
- **Photos/videos shared per day**: 95M+
- **Stories posted per day**: 500M+
- **Countries**: 190+

### Content Types

1. **Feed**: Posts from accounts you follow
2. **Reels**: Short-form videos (TikTok competitor)
3. **Explore**: Personalized discovery page
4. **Stories**: 24-hour ephemeral content

---

## Instagram's Transparency (January 2025)

### Official Ranking Signals

**Instagram confirmed** (via blog post, January 2025) the top signals for each surface:

**Explore**:
1. Time spent viewing content
2. Likes per reach
3. Saves per reach
4. Shares/sends per reach

**Reels**:
1. Watch time (completion rate)
2. Likes per reach
3. Shares/sends per reach
4. Comments per reach

**Feed**:
1. Time spent viewing
2. Likes, comments, saves, shares
3. Profile visits after viewing post

**Key insight**: Engagement rate (per reach) matters more than absolute numbers.

---

## Explore Page

### What is Explore?

**Purpose**: Help users discover new content and creators.

**Content**: 90% from accounts user doesn't follow.

**Layout**: Grid of photos/videos personalized to user.

---

### Ranking Algorithm

**Two-stage system**:

**Stage 1: Candidate Generation**
- Identify ~500 posts user might be interested in
- Sources:
  1. **Accounts user has engaged with** (liked, commented, saved)
  2. **Similar users** (collaborative filtering)
  3. **Trending content** (popular in user's region/interests)

**Stage 2: Ranking**
- Score each candidate post
- Rank by predicted engagement

---

### Ranking Signals (Confirmed by Instagram, 2025)

**Primary signals** (in order of importance):

1. **Time spent viewing**:
   - How long user looks at content
   - Longer = higher interest

2. **Likes per reach**:
   - Not total likes, but likes / impressions
   - Filters out viral content with low engagement rate

3. **Saves per reach**:
   - Saving content = strong interest
   - Weighted heavily

4. **Sends per reach**:
   - Sharing with friends = very strong signal
   - Highest weight

**User-post relationship**:
- If user has engaged with creator before → boost
- If user follows similar accounts → boost

**Freshness**:
- Recent posts (last 48 hours) get boost
- Decays exponentially

---

### Scoring Function (Simplified)

$$\text{score}(u, p) = \sum_k w_k \cdot f_k(u, p)$$

where:
- $u$: User
- $p$: Post
- $f_k$: Feature $k$ (watch time, likes per reach, etc.)
- $w_k$: Learned weight for feature $k$

**Model**: Gradient Boosted Decision Trees or Deep Neural Network

---

## Reels

### The TikTok Competitor

**Launched**: August 2020

**Goal**: Compete with TikTok's short-form video dominance.

**Growth**: Fastest-growing product in Instagram history.

---

### Reels Ranking Signals (2025)

**Confirmed by Instagram**:

1. **Watch time / Completion rate**:
   - What % of reel did user watch?
   - Watched 100% → very strong positive
   - Dropped after 2 seconds → strong negative

2. **Likes per reach**:
   - Engagement rate, not absolute likes

3. **Shares per reach**:
   - Sending reel to friends = strong signal

4. **Comments per reach**:
   - Commenting = engagement

**Video attributes**:
- **Audio**: Trending audio gets boost
- **Effects**: Use of trending effects
- **Topics**: Category (comedy, dance, DIY, etc.)

---

### Video Understanding

**Challenge**: Understand what's in the video to recommend similar content.

**Solution**: Computer Vision models

**Signals extracted**:
- **Objects**: Detected objects (dog, car, food, etc.)
- **Actions**: Detected actions (dancing, cooking, working out)
- **Scene**: Indoor/outdoor, location type
- **Audio**: Music genre, speech transcription
- **Text**: On-screen text (OCR)

**Model**: Vision Transformer (ViT) or ResNet

---

### Implementation (Simplified)

```python
import torch
import torch.nn as nn

class ReelsRanker(nn.Module):
    def __init__(self, n_users, n_reels, video_emb_dim=512, user_emb_dim=128):
        super().__init__()

        # User embedding
        self.user_embedding = nn.Embedding(n_users, user_emb_dim)

        # Reel features (from CV model)
        self.video_projection = nn.Linear(video_emb_dim, 256)

        # Engagement history features
        self.engagement_mlp = nn.Sequential(
            nn.Linear(user_emb_dim + 256 + 10, 512),  # +10 for engagement features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Predicted engagement score
        )

    def forward(self, user_ids, video_embeddings, engagement_features):
        """
        user_ids: (batch,)
        video_embeddings: (batch, video_emb_dim) - from CV model
        engagement_features: (batch, 10) - likes_per_reach, saves_per_reach, etc.
        """
        user_emb = self.user_embedding(user_ids)  # (batch, user_emb_dim)
        video_proj = self.video_projection(video_embeddings)  # (batch, 256)

        # Concatenate all features
        combined = torch.cat([user_emb, video_proj, engagement_features], dim=1)

        # Predict engagement score
        score = self.engagement_mlp(combined)  # (batch, 1)

        return score.squeeze()
```

---

### Trending Audio

**Observation**: Reels using trending audio get more engagement.

**Signal**: Track audio usage across platform
- Audio used in 100K+ reels → trending
- Boost reels using trending audio

**Feedback loop**: Recommendations → more usage → more trending → more recommendations

**Risk**: Audio fatigue (same audio everywhere)

**Solution**: Diversify, rotate trending audios

---

## Feed Ranking

### Personalized Feed

**Content**: Posts from accounts user follows + suggested posts.

**Goal**: Show most interesting posts from friends and creators.

**Ranking signals** (confirmed, 2025):

1. **Time spent viewing**:
   - Dwell time on post

2. **Likes, comments, saves, shares**:
   - All engagement signals

3. **Profile visits after viewing**:
   - User clicked on creator's profile after seeing post → strong interest

**Recency**:
- Newest posts get priority
- But balanced with predicted engagement

---

### Suggested Posts

**Percentage**: 10-20% of feed is suggested (from accounts user doesn't follow)

**Goal**: Help users discover new creators.

**Ranking**: Similar to Explore (collaborative filtering + content-based)

**Controversy**: Users often don't want suggested content in feed.

**Instagram's response**: Added control to "Snooze suggested posts" for 30 days.

---

## Multi-Objective Optimization

### Balancing Multiple Goals

**Objectives**:
1. **Maximize engagement** (likes, comments, shares)
2. **Maximize time spent** (dwell time)
3. **Maximize discovery** (new creators found)
4. **Minimize negative actions** (hide, report, unfollow)

**Challenge**: Trade-offs between objectives.

**Example**:
- Viral meme → high engagement, low time spent
- Long documentary clip → low engagement, high time spent

---

### Multi-Task Learning

**Shared architecture** with task-specific heads:

```
User + Content Features
         ↓
    Shared Layers
         ↓
      Split
    /   |   \   \
Engagement Time Discovery Negative
  Head    Head   Head     Head

Objectives:
- Engagement: Maximize P(like | view)
- Time: Maximize E[watch time | view]
- Discovery: Maximize P(follow creator | view)
- Negative: Minimize P(hide/report | view)
```

**Combined loss**:
$$\mathcal{L} = w_1 \mathcal{L}_{\text{engagement}} + w_2 \mathcal{L}_{\text{time}} + w_3 \mathcal{L}_{\text{discovery}} - w_4 \mathcal{L}_{\text{negative}}$$

**Weights tuned** via A/B testing to optimize business metrics.

---

## Computer Vision Integration

### Visual Understanding

**Challenge**: Recommend visually similar content.

**Solution**: Extract visual embeddings from images/videos.

**Model**: Vision Transformer (ViT) or ResNet

**Process**:
1. **Extract features**: Pass image/video through CNN/ViT
2. **Embedding**: Get 512-D or 1024-D vector
3. **Similarity**: Cosine similarity between embeddings
4. **Recommendation**: "More like this" based on visual similarity

---

### Multi-Modal Embeddings

**Combine** visual + text + audio:

**Visual**: Image/video features (ViT)
**Text**: Caption, hashtags (BERT)
**Audio**: Audio features (for Reels)

**Fusion**:
$$\mathbf{e}_{\text{post}} = W_v \mathbf{e}_{\text{visual}} + W_t \mathbf{e}_{\text{text}} + W_a \mathbf{e}_{\text{audio}}$$

**Benefit**: Richer representation, better recommendations.

---

## Diversity and Exploration

### The Filter Bubble Problem

**Risk**: Users only see content similar to what they've engaged with.

**Solution**:

**1. Exploration**:
- Inject 10-20% of diverse content
- Sample from different categories/topics

**2. Interest expansion**:
- Gradually introduce adjacent interests
- Example: User likes cooking → show food photography → travel food

**3. Trending content**:
- Show popular content outside user's bubble
- Exposes to new trends

---

## Content Moderation and Safety

### Harmful Content Filtering

**Challenge**: Filter out harmful content before recommendation.

**Categories**:
- Violence, hate speech
- Misinformation
- Bullying, harassment
- Adult content

**Process**:
1. **Detection**: Computer vision + NLP models flag potential violations
2. **Review**: Human moderators review flagged content
3. **Action**: Remove or reduce distribution

**For recommendations**:
- Flagged content → Zero visibility in Explore/Reels
- Borderline content → Reduced distribution

---

## User Controls (2025 Transparency)

### What Users Can Control

**Instagram added controls** (2025):

1. **"Not Interested"**:
   - Mark content as not interesting
   - Instagram reduces similar content

2. **"See More Like This"**:
   - Explicitly request more similar content

3. **"Snooze Suggested Posts"**:
   - Pause suggested posts for 30 days

4. **"Reset Recommendations"**:
   - Clear recommendation history, start fresh

**Transparency**: Instagram shows why content was recommended.

---

## A/B Testing at Instagram

### Experimentation Culture

**Scale**: 1000s of A/B tests per year.

**Metrics**:
- **Primary**: Time spent, engagement rate
- **Secondary**: Creator diversity, negative actions
- **Business**: Monthly active users (MAU), daily active users (DAU)

**Example test**: New ranking signal (saves per reach)
- **Control**: Existing ranking
- **Treatment**: Add saves per reach signal
- **Results**: +5% engagement, +2% time spent → Ship

---

## Comparison: Instagram vs. TikTok

| Aspect | Instagram Reels | TikTok For You Page |
|--------|----------------|---------------------|
| **Primary signal** | Watch time, likes per reach | Watch time, completion rate |
| **Social graph** | Some weight on follows | Minimal weight (discovery-first) |
| **Diversity** | Moderate | High (very exploratory) |
| **Audio** | Trending audio boost | Critical (music-first) |
| **Engagement** | Likes, comments, shares | Replays, finishes, shares |
| **Cold start** | Good (uses Instagram history) | Excellent (personalizes day 1) |

**TikTok advantage**: Pure discovery, better cold start.

**Instagram advantage**: Social connections, cross-platform (Stories, Feed).

---

## Technical Architecture (Estimated)

**Data Pipeline**:
- **Kafka**: Real-time engagement events (likes, views)
- **Hive/Spark**: Batch processing for training data

**Model Training**:
- **PyTorch**: Deep learning models
- **XGBoost**: Ranking models
- **Distributed training**: Multi-GPU, multi-node

**Model Serving**:
- **Two-tower**: User encoder + Content encoder
- **ANN search**: FAISS for candidate retrieval (millions of posts)
- **Ranking**: Deployed as microservice (TensorFlow Serving)

**Latency**:
- Candidate retrieval: <50ms
- Ranking: <50ms
- Total: <100ms

---

## Lessons from Instagram

### 1. Engagement Rate > Absolute Engagement

**Likes per reach** matters more than total likes.

**Reason**: Filters out viral-but-low-quality content.

**Takeaway**: Normalize metrics by impressions/reach.

---

### 2. Visual Understanding is Critical

**Computer vision** extracts rich features from images/videos.

**Enables**: "More like this" recommendations without text.

**Takeaway**: Invest in CV models for visual platforms.

---

### 3. Multi-Objective Optimization

**Can't optimize just one metric** (engagement).

**Must balance**: Engagement, time spent, discovery, safety.

**Takeaway**: Use multi-task learning, tune weights via A/B tests.

---

### 4. Transparency Builds Trust

**Instagram's 2025 transparency**: Publicly shared ranking signals.

**Effect**: Users understand why they see content, more trust.

**Takeaway**: Consider transparency as a feature.

---

### 5. User Control Matters

**Letting users control recommendations** (Not Interested, Reset) increases satisfaction.

**Takeaway**: Give users agency over their experience.

---

## Summary

**Key Takeaways**:
1. **Ranking signals** (2025 confirmed): Watch time, likes per reach, sends per reach
2. **Two-stage**: Candidate retrieval (500 posts) → Ranking
3. **Computer vision**: Essential for visual content understanding
4. **Multi-objective**: Balance engagement, time, discovery, safety
5. **Transparency**: Instagram leading in algorithmic transparency (2025)
6. **User controls**: Not Interested, Reset, Snooze

**Architecture**:
```
User → Candidate Retrieval → Ranking → Multi-Objective → Recommendations
     (Collab Filtering)   (ML Model)  (Engagement/Time/Safety)
```

**Technologies**:
- Models: Two-tower, XGBoost, PyTorch DNNs, Vision Transformers
- Scale: 2B users, billions of posts/reels
- Latency: <100ms for recommendations

**Comparison with TikTok**:
- Instagram: Social graph + discovery
- TikTok: Pure discovery, better cold start

**For Builders**:
- Use engagement rate metrics (per reach)
- Invest in computer vision for visual content
- Multi-task learning for multiple objectives
- Give users control and transparency

---

---

## Course Concepts Applied

| Concept | Week | Application in Instagram |
|---------|------|--------------------------|
| **Collaborative Filtering** | 2-3 | Users who engage with similar content |
| **Matrix Factorization** | 3 | User-content embeddings |
| **Content-Based** | 4 | Visual embeddings (ViT/ResNet), caption NLP |
| **Neural CF** | 5 | Deep ranking models for engagement prediction |
| **Sequential Models** | 6 | Session-based Reels recommendations |
| **Graph-Based** | 7 | Social graph for suggested accounts |
| **Two-Tower** | 8 | User encoder + Content encoder for retrieval |
| **Multi-Task Learning** | 8 | Like, comment, share, save joint prediction |
| **Embeddings** | 9 | Multi-modal (vision + text + audio) embeddings |
| **Contextual Bandits** | 10 | Exploration in Explore page |
| **Evaluation** | 11 | Engagement rate metrics, watch time |
| **Bias/Fairness** | 12 | Creator exposure fairness, content diversity |
| **Production Systems** | 13 | Two-stage pipeline, visual feature serving |

---

## References

1. **Instagram Official Blog (January 2025)**. "How Instagram Ranks Reels, Feed, and Explore".
   - **Official ranking signals** confirmed

2. **Meta AI Research**. Various papers on recommendation systems.
   - Technical foundations

3. **Covington, P., Adams, J., & Sargin, E. (2016)**. "Deep Neural Networks for YouTube Recommendations". *RecSys*.
   - Similar two-tower architecture

4. **He, X., et al. (2017)**. "Neural Collaborative Filtering". *WWW*.
   - NCF applied at Instagram scale

5. **Instagram Engineering Blog**. "Powered by AI: Instagram's Explore recommender system".
   - Technical deep dive into Explore
