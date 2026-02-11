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

## Why Can't We Just Score All 800M Videos?

*Before we dive into YouTube's architecture, let's understand why we need it in the first place.*

### The Naive Approach: Score Everything

Imagine you're a YouTube engineer in 2016. Your CEO asks: "Why don't we just run our best model on all videos for each user?"

**Let's do the math together.**

**Given**:
- Total videos: 800,000,000
- Ranking model inference time: 0.01ms per video (very optimized!)
- User patience threshold: 100ms for page load

**Calculation**:
$$\text{Total time} = 800,000,000 \times 0.01\text{ms} = 8,000,000\text{ms} = 8,000\text{s} \approx 2.2 \text{ hours}$$

*Can you see the problem?* A user would wait **2.2 hours** to load their homepage!

Even if we parallelized across 1,000 GPUs:
$$\frac{8,000\text{s}}{1,000} = 8\text{s}$$

Still 80x too slow. And that's PER USER, with YouTube serving 2 billion users daily.

**The cost?** At $3/GPU-hour on cloud:
$$2\text{B users} \times 8\text{s} \times \frac{1,000 \text{ GPUs}}{3600} \times \$3 \approx \$13.3\text{M per day}$$

That's **$4.9 billion per year** just for inference!

*This is why the naive approach fails catastrophically.*

---

## The Two-Stage Solution

### The Key Insight

*Here's the brilliant insight that makes YouTube-scale recommendations possible:*

**Not all videos need careful consideration.**

Think about it: if you're a jazz enthusiast, do we need to carefully score videos about motorcycle repair? About Korean cooking tutorials? About competitive Fortnite gameplay?

**No!** We can quickly filter to videos that *might* be relevant (candidates), then carefully rank only those.

### Two-Stage Architecture

```
800M Videos
     |
     v
+----------------------------------+
| Stage 1: Candidate Generation    |
| - Fast, approximate              |
| - Embedding similarity           |
| - Time: <10ms                    |
+----------------------------------+
     |
     v  (~1000 videos)
+----------------------------------+
| Stage 2: Ranking                 |
| - Slow, precise                  |
| - Rich features + cross-features |
| - Time: <50ms                    |
+----------------------------------+
     |
     v  (~20 videos)
Homepage Recommendations
```

**Let's verify the latency:**

**Stage 1** (Candidate Generation):
- Compute user embedding: ~2ms
- ANN query for 1000 neighbors: ~8ms
- Total: **~10ms**

**Stage 2** (Ranking):
- Feature extraction for 1000 videos: ~10ms (batched)
- Model inference (1000 videos): ~30ms (batched)
- Sorting and business logic: ~10ms
- Total: **~50ms**

**End-to-end: 60ms < 100ms threshold**

*Notice how stage 1 reduced our problem from 800M to 1000 - that's an 800,000x reduction! This is the magic of the two-stage approach.*

---

## Stage 1: Candidate Generation

### The Two-Tower Architecture

**Goal**: Retrieve ~1000 relevant videos from 800M in <10ms.

*How can we possibly do this?*

**Key insight**: If we can embed users and videos into the same vector space, retrieval becomes a nearest-neighbor search.

```
User Features              Video Features
(watch history,            (title, tags,
 search history,            upload time,
 demographics)              engagement stats)
       |                         |
       v                         v
+-------------+           +-------------+
| User Tower  |           | Video Tower |
| (Deep NN)   |           | (Deep NN)   |
+-------------+           +-------------+
       |                         |
       v                         v
User Embedding            Video Embedding
    u (256-dim)              v (256-dim)
       |                         |
       +----------+  +-----------+
                  v  v
            Dot Product
           score = u^T * v
```

*Can you see why this is called "two-tower"?* Each side processes its inputs independently, like two separate towers, only meeting at the top for the similarity computation.

---

### User Tower: From Watch History to Embedding

**The Central Question**: How do we represent a user's preferences as a single 256-dimensional vector?

#### Step 1: Embed Individual Videos

Each video ID gets a learned embedding:
$$\mathbf{e}_i = \text{Embedding}(\text{video}_i) \in \mathbb{R}^{256}$$

*Why learn embeddings instead of using hand-crafted features?*

Because with 800M videos, we can't possibly engineer features for each one. The model discovers what matters.

#### Step 2: Aggregate Watch History

**Problem**: User watched 50 videos. How to combine into one vector?

**Simple approach**: Average pooling

$$\mathbf{h}_{\text{watch}} = \frac{1}{|\mathcal{W}|} \sum_{i \in \mathcal{W}} \mathbf{e}_i$$

where $\mathcal{W}$ = set of watched video IDs.

*What does this averaging capture?*

If a user watched 10 jazz videos and 2 cooking videos, the average will be "closer" to jazz in embedding space. The averaging naturally weights toward the user's dominant interests.

**Numerical Example**:

Let's trace through with concrete numbers. Suppose embeddings are 4-dimensional (simplified):

```
User watched 3 videos:
- Jazz tutorial:     e_1 = [0.8, 0.2, -0.1, 0.5]
- Jazz performance:  e_2 = [0.7, 0.3, -0.2, 0.4]
- Cooking show:      e_3 = [-0.3, 0.6, 0.8, 0.1]

Watch history embedding:
h_watch = (1/3) * ([0.8, 0.2, -0.1, 0.5] +
                   [0.7, 0.3, -0.2, 0.4] +
                   [-0.3, 0.6, 0.8, 0.1])

h_watch = (1/3) * [1.2, 1.1, 0.5, 1.0]
        = [0.4, 0.37, 0.17, 0.33]
```

*Notice that the result is closer to jazz (first two dimensions positive) than cooking (third dimension). The user's primary interest is preserved!*

#### Step 3: Combine All User Features

```python
# Concatenate all user signals
user_features = concat([
    h_watch,           # 256-dim: watch history embedding
    h_search,          # 256-dim: search history embedding
    demographics,      # 10-dim: age, gender, location encoded
    context            # 8-dim: device, time of day, day of week
])
# Total: 530 dimensions

# Feed through user tower (MLP)
user_embedding = user_tower(user_features)  # Output: 256-dim
```

#### The User Tower MLP

$$\mathbf{u} = \text{ReLU}(W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \cdot \mathbf{x} + b_1) + b_2) + b_3)$$

*Why multiple layers?*

**Layer 1**: Learn basic combinations (e.g., "watched jazz AND is 25-35 years old")

**Layer 2**: Learn higher-order patterns (e.g., "jazz enthusiast who searches late at night")

**Layer 3**: Compress into final user representation

---

### Video Tower Features

The video tower processes:

**1. Video ID Embedding** (256-dim):
- Most important feature
- Captures video's "identity" in recommendation space

**2. Metadata** (embedded):
- Title: Word2Vec average of title tokens
- Tags: Average of tag embeddings
- Category: One-hot encoded then embedded

**3. Engagement Signals**:
- Total views (log-transformed)
- Average watch time
- Like ratio
- Share count (log-transformed)

---

### Training Objective: Sampled Softmax

**Formulation**: Multi-class classification where each video is a class.

**Full Softmax**:
$$P(\text{video } i | \text{user } u) = \frac{\exp(\mathbf{u}^T \mathbf{v}_i)}{\sum_{j=1}^{800M} \exp(\mathbf{u}^T \mathbf{v}_j)}$$

*What's wrong with this?*

That denominator sums over ALL 800M videos! Computing this once would take hours.

**Solution**: Sampled Softmax

$$P(i | u) \approx \frac{\exp(\mathbf{u}^T \mathbf{v}_i)}{\exp(\mathbf{u}^T \mathbf{v}_i) + \sum_{j \in \text{negatives}} \exp(\mathbf{u}^T \mathbf{v}_j)}$$

Sample ~1000 "negative" videos (videos user didn't watch) instead of using all.

**Negative Sampling Strategy**:

*What happens if we sample uniformly?*

Uniform sampling over-represents obscure videos (there are millions of them!). The model wastes capacity learning "this jazz fan won't watch this random video with 12 views."

**YouTube's approach**: Sample proportional to popularity^0.75

$$P(\text{sample video } i) \propto (\text{popularity}_i)^{0.75}$$

*Why 0.75?*
- Exponent of 1.0 would sample purely by popularity (ignores niche content)
- Exponent of 0 would sample uniformly (wastes capacity on obscure negatives)
- 0.75 balances: popular enough to be meaningful, but includes some niche

---

### The "Example Age" Feature: A Brilliant Trick

*Here's one of the cleverest engineering tricks in the paper.*

**Problem**: Your model is trained on last month's data. But users want FRESH content - trending videos, breaking news, new releases.

**Observation**: During training, older videos are over-represented in the positive examples (they've had more time to accumulate watches).

**Naive fix**: Add a "freshness" feature. But how?

**The Trick**: Include "example age" as a feature during training.

$$\text{example\_age} = t_{\text{train}} - t_{\text{upload}}$$

where $t_{\text{train}}$ is when the training example was created.

**At inference time**: Set example_age = 0 for ALL videos.

*Why does this work?*

Let me walk through the intuition:

**During training**, the model sees:
- Old viral videos with high engagement AND high example_age
- New videos with moderate engagement AND low example_age

The model learns: "If example_age is low, this video is fresh and I should give it a chance even without huge engagement numbers."

**At inference**, when we set example_age = 0, we're telling the model: "Treat every video as if it were just uploaded. Judge it on its content, not its accumulated stats."

**Numerical demonstration**:

```
Without example_age feature:
- Old viral video: predicted score = 0.95
- New good video:  predicted score = 0.60  (less watch data)

With example_age = 0 at inference:
- Old viral video: predicted score = 0.75  (penalized for "being old")
- New good video:  predicted score = 0.70  (boosted for "being fresh")
```

The new video now has a fighting chance!

---

### Serving: Making It Fast with ANN

**Offline Pipeline** (runs daily):

1. **Compute all video embeddings**:
   ```
   For each video v in 800M:
       v_embedding = video_tower(video_features[v])
       store(v_embedding)
   ```
   Time: ~2 hours on 100 GPUs

2. **Build ANN index**:
   - Use FAISS or ScaNN
   - Hierarchical Navigable Small World (HNSW) graphs
   - Enables sub-linear retrieval

**Online Pipeline** (per user request):

```python
def get_candidates(user_features, k=1000):
    # Step 1: Compute user embedding (~2ms)
    user_embedding = user_tower(user_features)

    # Step 2: Query ANN index (~8ms)
    candidate_ids, scores = ann_index.search(user_embedding, k)

    return candidate_ids  # Top-1000 videos
```

**Total latency: <10ms**

---

## Stage 2: Ranking

### Why a Separate Ranking Stage?

*The candidate generation gave us 1000 videos. Why not just use those scores?*

**Two-tower limitations**:
1. **Only dot product similarity** - can't capture complex interactions
2. **No cross-features** - can't ask "how did THIS user interact with THIS creator before?"
3. **Embedding must be precomputed** - can't use real-time signals

**Ranking model advantages**:
1. Can use ANY features (including expensive ones)
2. Can compute cross-features at request time
3. More expressive model architecture

---

### Ranking Architecture

```
User Features    Video Features    Cross Features
     |                |                 |
     v                v                 v
+--------------------------------------------------+
|              Feature Processing                   |
|  (embeddings, normalization, crosses)            |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
|              Deep Neural Network                  |
|  Layer 1: 1024 units, ReLU, Dropout(0.3)        |
|  Layer 2: 512 units, ReLU, Dropout(0.3)         |
|  Layer 3: 256 units, ReLU                        |
+--------------------------------------------------+
                      |
                      v
             Predicted Watch Time
```

---

### Feature Engineering for Ranking

**1. User Features**:
- Demographics (age, gender, location)
- Detailed watch history (not just IDs - completion rates, rewatch counts)
- Search history with timestamps

**2. Video Features**:
- Metadata (title, duration, category)
- Engagement stats (views, likes, comments)
- **Freshness** (time since upload)
- Creator features (subscriber count, upload frequency, historical CTR)

**3. Cross Features** (the secret sauce):

$$\text{user\_creator\_affinity} = \frac{\text{videos watched from this creator}}{\text{total videos from this creator shown to user}}$$

$$\text{category\_preference} = \frac{\text{user's watch time in this category}}{\text{user's total watch time}}$$

*Can you see why cross features are so powerful?*

They let the model answer: "How does THIS specific user feel about THIS specific type of content?" - something impossible with just user or video features alone.

---

### Ranking Objective: Expected Watch Time

**Why not optimize for clicks?**

*Consider two videos:*

| Video | Thumbnail | Title | CTR | Avg Watch Time |
|-------|-----------|-------|-----|----------------|
| A | Sensational | "You WON'T BELIEVE..." | 15% | 30 seconds |
| B | Informative | "How Neural Networks Work" | 5% | 10 minutes |

If we optimize for CTR, Video A wins. But users feel **tricked** - they clicked expecting value and bounced quickly.

**Expected engagement calculation**:

$$E[\text{watch time} | \text{impression}] = P(\text{click}) \times E[\text{watch time} | \text{click}]$$

Video A: $0.15 \times 30 = 4.5$ seconds expected
Video B: $0.05 \times 600 = 30$ seconds expected

*Video B provides 6.7x more expected engagement!*

**Objective**: Predict watch time directly

$$\mathcal{L} = \frac{1}{N} \sum_{(u,v,t)} (\hat{t}_{uv} - t_{uv})^2$$

where $\hat{t}_{uv}$ is predicted watch time and $t_{uv}$ is actual watch time.

---

### Weighted Logistic Regression Trick

**Problem**: Most impressions result in NO click (negative examples vastly outnumber positives).

**YouTube's approach**: Use weighted logistic regression where:
- Positive examples (watched): weight = watch_time
- Negative examples (not clicked): weight = 1

**Effect**: A video watched for 10 minutes counts 10x more than one watched for 1 minute in the loss function.

**Mathematical formulation**:

$$\mathcal{L} = -\sum_{(u,v) \in \text{positives}} t_{uv} \log(\sigma(\hat{y}_{uv})) - \sum_{(u,v) \in \text{negatives}} \log(1 - \sigma(\hat{y}_{uv}))$$

*At inference*, we use the log-odds as our ranking score:
$$\text{score} = \log\left(\frac{P(\text{watch})}{1 - P(\text{watch})}\right) \approx E[\text{watch time}]$$

---

### Numerical Walkthrough: Ranking 5 Videos

Let's trace through the ranking model with actual numbers.

**Setup**: User wants to watch jazz content on mobile, evening time.

**User features** (simplified to 10 dimensions):
```
user_vec = [
    0.8,   # jazz_affinity (high)
    0.1,   # cooking_affinity (low)
    0.3,   # age_normalized (25-35)
    0.7,   # evening_activity (usually active)
    1.0,   # mobile_device (yes)
    0.6,   # avg_watch_completion
    0.4,   # days_since_last_visit (recent)
    0.9,   # subscription_count_norm
    0.2,   # search_recent (hasn't searched recently)
    0.5    # session_depth (mid-session)
]
```

**5 Candidate Videos**:

| Video | Category | Duration | Creator Affinity | Freshness |
|-------|----------|----------|------------------|-----------|
| V1 | Jazz tutorial | 15 min | 0.8 (watched before) | 2 days |
| V2 | Jazz concert | 45 min | 0.0 (new creator) | 30 days |
| V3 | Cooking show | 20 min | 0.9 (favorite creator) | 1 day |
| V4 | Jazz history | 25 min | 0.3 (watched once) | 7 days |
| V5 | Jazz + Cooking fusion | 12 min | 0.0 (new) | 3 hours |

**Video features** (10-dim each):
```
v1_vec = [0.9, 0.0, 0.3, 0.8, 0.9, 0.7, 0.8, 0.5, 0.6, 0.4]  # Jazz tutorial
v2_vec = [0.95, 0.0, 0.9, 0.0, 0.3, 0.6, 0.4, 0.7, 0.8, 0.2]  # Jazz concert
v3_vec = [0.0, 0.95, 0.4, 0.9, 0.95, 0.8, 0.9, 0.6, 0.7, 0.5]  # Cooking
v4_vec = [0.85, 0.0, 0.5, 0.3, 0.5, 0.7, 0.5, 0.6, 0.4, 0.3]  # Jazz history
v5_vec = [0.5, 0.5, 0.25, 0.0, 0.99, 0.4, 0.3, 0.3, 0.5, 0.6]  # Fusion (new!)
```

**Cross features** (per user-video pair):
```
cross_v1 = [0.8, 0.7, 0.6, 0.9, 0.5]  # High creator affinity, recent category engagement
cross_v2 = [0.0, 0.0, 0.6, 0.3, 0.5]  # New creator, but strong category
cross_v3 = [0.9, 0.8, 0.1, 0.2, 0.5]  # Favorite creator, but wrong category
cross_v4 = [0.3, 0.2, 0.6, 0.6, 0.5]  # Some history
cross_v5 = [0.0, 0.0, 0.4, 0.8, 0.9]  # New but fresh and bridging categories
```

**Model computation** (simplified):

```
Full feature vector for V1 = concat(user_vec, v1_vec, cross_v1) = 25-dim

Layer 1 (25 -> 16): h1 = ReLU(W1 @ features + b1)
Layer 2 (16 -> 8):  h2 = ReLU(W2 @ h1 + b2)
Layer 3 (8 -> 1):   score = W3 @ h2 + b3
```

**Predicted watch times** (in minutes):

| Video | Raw Score | Predicted Watch Time | Reasoning |
|-------|-----------|---------------------|-----------|
| V1 | 2.4 | 11.0 min | Strong match: jazz + known creator + fresh |
| V2 | 1.8 | 6.0 min | Good category but: unknown creator, old, very long |
| V3 | 1.2 | 3.3 min | Wrong category despite favorite creator |
| V4 | 1.9 | 6.7 min | Good match but: older, less creator history |
| V5 | 2.1 | 8.2 min | Fresh content, bridges interests, short duration |

**Final ranking**: V1 > V5 > V4 > V2 > V3

*Notice how the model balances multiple factors:*
- V1 wins because of creator affinity + category match + freshness
- V3 ranks last despite having the user's favorite creator - category mismatch dominates
- V5 does well because of extreme freshness (example_age trick!) and category bridging

---

### Implementation

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
        # Encode watch history (average pooling)
        watch_embs = self.video_embedding(watch_history)  # (batch, history_len, emb_dim)
        watch_avg = watch_embs.mean(dim=1)  # (batch, emb_dim)

        # Encode search history (average pooling)
        search_embs = self.search_embedding(search_tokens)  # (batch, search_len, emb_dim)
        search_avg = search_embs.mean(dim=1)  # (batch, emb_dim)

        # Combine user features
        user_features = torch.cat([watch_avg, search_avg, user_demographics], dim=-1)

        # User embedding through tower
        user_emb = self.user_tower(user_features)  # (batch, emb_dim)

        # Video embedding through tower
        video_emb = self.video_tower(video_features)  # (batch, emb_dim)

        # L2 normalize for cosine similarity
        user_emb = nn.functional.normalize(user_emb, p=2, dim=1)
        video_emb = nn.functional.normalize(video_emb, p=2, dim=1)

        # Dot product similarity
        scores = (user_emb * video_emb).sum(dim=1)  # (batch,)

        return scores, user_emb, video_emb


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
```

---

## What Can Go Wrong: Failure Modes and Solutions

*Now let's discuss the things that can silently break your recommendation system. These are the issues that won't show up in offline metrics but will hurt real users.*

### Failure Mode 1: Retrieval-Ranking Mismatch

**Symptom**: Offline ranking metrics look great, but A/B tests show no improvement or regression.

**What's happening**: The ranking model is trained on videos that appear in production, but the candidate generator retrieves a DIFFERENT set of videos.

**Example**:
```
Training data: User saw {V1, V2, V3, V4, V5} and clicked V2
  - Ranking model learns: Given these 5, predict V2

Production: Candidate gen retrieves {V6, V7, V8, V9, V10}
  - None of these were in training!
  - Ranking model has never seen these comparisons
```

**Root cause**: The candidate generation and ranking stages were trained independently on different data distributions.

**Solutions**:
1. **Train ranking on retrieved candidates**: Use actual retrieval output as training negatives
2. **Joint training**: End-to-end training of both stages
3. **Calibration**: Regularly audit overlap between training candidates and production candidates
4. **Monitoring**: Track "% of ranked videos that appeared in training"

---

### Failure Mode 2: Training-Serving Skew

**Symptom**: Model performance degrades over time even with retraining. Works on historical data, fails on fresh data.

**What's happening**: Features computed at training time differ from serving time.

**Common causes**:

**a) Feature computation differences**:
```
Training: video_popularity = count(watches_last_30_days)
Serving: video_popularity = count(watches_last_30_days)  # BUT computed at different time!
```
A video that had 1M views at training time might have 10M now.

**b) Look-ahead bias**:
```
Training: Used future information accidentally
  - "User subscription status" computed AFTER the interaction
  - Model learns to use information it won't have at serving time
```

**c) Stale features**:
```
Offline embeddings updated daily
User's taste changed THIS session
Model uses yesterday's representation
```

**Solutions**:
1. **Feature logging**: Log EXACT features used at serving time for training
2. **Time-travel testing**: Evaluate with features as they were at prediction time
3. **Feature freshness monitoring**: Alert when offline/online feature divergence exceeds threshold
4. **Timestamp discipline**: Never use features computed after the prediction timestamp

---

### Failure Mode 3: Feedback Loops and Popularity Bias

**Symptom**: Recommendations become increasingly homogeneous. New content never gets traction. A few videos dominate everything.

**What's happening**: Rich-get-richer dynamics

```
Day 1: Video A has 100 views, Video B has 10 views
  - Model recommends A more (more engagement data)

Day 2: Video A now has 1000 views, Video B still has 10
  - Gap widens

Day 30: Video A has 10M views, Video B was never shown
  - Model is CERTAIN A is better (but never tested B!)
```

**The feedback loop**:
```
Model recommends popular videos
     ↓
Popular videos get more watches
     ↓
Training data reinforces popularity
     ↓
Model recommends popular videos even more
     ↓
(repeat)
```

**Solutions**:
1. **Exploration**: Reserve 5-10% of recommendations for random/diverse content
2. **Popularity debiasing**: Train with inverse propensity weighting
   $$\text{weight}(v) = \frac{1}{\sqrt{\text{impressions}(v)}}$$
3. **Counterfactual evaluation**: Use logged propensities to estimate policy value
4. **Multi-armed bandits**: Balance exploitation (show best) vs exploration (try new)

---

### Failure Mode 4: Embedding Staleness

**Symptom**: New videos get very few recommendations. Trending content doesn't surface quickly.

**What's happening**: Video embeddings are computed in daily batch job, but video properties change hourly.

**Example**:
```
12:00 AM: New video uploaded, embedded with 0 views
11:00 PM: Video has gone viral (1M views)
           But embedding still reflects 0-view video
           Candidate generation doesn't retrieve it
```

**The math**:
```
Batch embedding update: Once per day (24h latency)
Viral video lifecycle: Peaks within 4-6 hours
Result: Miss the entire viral window!
```

**Solutions**:
1. **Example age feature**: At inference, set age=0 to boost fresh content
2. **Real-time embedding updates**: Stream processing for viral content
3. **Hybrid retrieval**:
   - 80% from ANN (stable embeddings)
   - 20% from recent uploads (freshness)
4. **Tiered refresh**: Update popular videos every hour, long-tail daily

---

### Failure Mode 5: Cold Start Cascades

**Symptom**: New users bounce immediately. New creators never get their first viewer.

**What's happening**: Two-stage system requires data at BOTH stages to work well.

**For new users**:
```
Stage 1: User embedding = average of 0 watched videos = ???
         Cannot compute meaningful similarity
         Retrieves random candidates

Stage 2: No watch history features
         No creator affinity features
         Random ranking

Result: Terrible recommendations → User leaves → Never get data → (stuck)
```

**For new creators**:
```
Stage 1: Video embedding exists, but no engagement signal
         Similarity to user embeddings is random

Stage 2: creator_historical_ctr = undefined
         creator_avg_watch_time = undefined
         Model uncertain, ranks low

Result: Never shown → Never watched → Never improve rankings → (stuck)
```

**Solutions**:

**For new users**:
1. **Onboarding flow**: Ask for 3-5 topic preferences
2. **Demographic priors**: Initialize embedding based on age/location
3. **Popular content**: Show globally popular until personalization kicks in
4. **Exploration boost**: More random recommendations for new users

**For new creators**:
1. **Content-based features**: Use video title/thumbnail/description when engagement missing
2. **Creator bootstrap**: Inherit stats from similar existing creators
3. **Explore slots**: Reserve homepage positions for new creator content
4. **Quality signals**: Prioritize production quality metrics initially

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

### 2016 to 2024 Changes (Estimated)

**Scale**:
- Videos: 500M to 800M
- Users: 1B to 2B
- Embeddings: 256D to 512D (larger capacity)

**Models**:
- **Candidate generation**: Two-tower to Multi-tower (separate towers for shorts, live, etc.)
- **Ranking**: 3-layer DNN to Transformer-based models

**Features**:
- Added: Short-form content (Shorts), live streams, user comments
- Multimodal: Video frames (visual features), audio, transcripts

**Personalization**:
- More fine-grained (session-based, time-of-day, mood detection)

---

## Key Lessons from YouTube

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

**Even small changes** (e.g., feature engineering) require A/B testing.

**Metrics matter**: Don't just look at CTR, measure watch time, sessions, satisfaction.

---

### 5. Simplicity First

**Start simple** (two-tower, basic features).

**Add complexity incrementally** (cross-features, transformers).

**YouTube's 2016 model**: Relatively simple (2-3 layer NNs) but massive impact.

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
    |
    v
Stage 1: Candidate Generation (<10ms)
  Two-tower model + ANN -> 1000 candidates
    |
    v
Stage 2: Ranking (<50ms)
  Deep NN + cross-features -> Top-20
    |
    v
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

Without two-stage: 800M x 10ms = ?
With two-stage: Candidate (10ms) + Ranking (1000 x 10ms) = ?
```

**Solution**:
```
Without: 800M x 10ms = 8 billion ms = 92.6 days!
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
Video A engagement: 0.10 x 30 = 3 seconds (expected)
Video B engagement: 0.05 x 300 = 15 seconds (expected)

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

**Compute**: Sampling probabilities (proportional to popularity^0.75).

**Solution**:
```
A: 1,000,000^0.75 = 31,623
B: 100,000^0.75 = 5,623
C: 10,000^0.75 = 1,000

Sum = 38,246

P(A) = 31,623 / 38,246 = 0.827
P(B) = 5,623 / 38,246 = 0.147
P(C) = 1,000 / 38,246 = 0.026
```

*Notice how the 0.75 exponent compresses the distribution - Video A is 100x more popular than C, but only ~32x more likely to be sampled.*
