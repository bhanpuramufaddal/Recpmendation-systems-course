# Week 15: YouTube Recommendations

## Overview

**YouTube**: 2+ billion users, 500+ hours uploaded per minute.

**Recommendation surfaces**:
1. **Home feed**: Personalized video recommendations
2. **Watch next**: Sidebar recommendations during video playback
3. **Search results**: Query-based recommendations
4. **Shorts feed**: Short-form vertical videos (TikTok competitor)
5. **Subscriptions**: Content from followed channels

**Scale challenges**:
- Billions of videos
- Millions of hours watched daily
- Fresh content uploaded constantly
- Diverse user preferences

---

## System Architecture

### Two-Stage Pipeline

**Stage 1: Candidate Generation**
- Input: User history (hundreds of videos)
- Output: Hundreds of candidate videos
- Goal: Fast retrieval from billions of videos

**Stage 2: Ranking**
- Input: Hundreds of candidates
- Output: Top ~20 ranked videos
- Goal: Precise ranking by watch time

---

## Candidate Generation

### Deep Neural Network Approach

**Architecture**: Two-tower model (user tower + video tower).

**User tower**:
- Watch history (video IDs)
- Search history (query tokens)
- Demographics (age, gender, geography)
- Context (time, device)

**Video tower**:
- Video ID
- Channel ID
- Upload time
- Video features (title, tags, duration)

**Output**: User embedding, video embedding.

**Similarity**: Dot product → Top-K candidates.

---

### Implementation

```python
import torch
import torch.nn as nn

class YouTubeCandidateGeneration(nn.Module):
    def __init__(self, n_videos, n_channels, embedding_dim=256):
        super().__init__()

        # Video embeddings
        self.video_embedding = nn.Embedding(n_videos, embedding_dim)

        # Channel embeddings
        self.channel_embedding = nn.Embedding(n_channels, embedding_dim)

        # User tower
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim * 3, 512),  # Watch history + search + demographics
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

        # Video tower
        self.video_tower = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),  # Video + channel
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def encode_user(self, watch_history, search_history, demographics):
        """
        Encode user to embedding.

        Args:
            watch_history: [batch_size, seq_len] video IDs
            search_history: [batch_size, search_len] query tokens
            demographics: [batch_size, demo_dim] user features

        Returns:
            user_emb: [batch_size, embedding_dim]
        """
        # Average watch history
        watch_emb = self.video_embedding(watch_history).mean(dim=1)

        # Average search history (simplified)
        search_emb = self.video_embedding(search_history).mean(dim=1)

        # Concatenate features
        user_features = torch.cat([watch_emb, search_emb, demographics], dim=1)

        # User embedding
        user_emb = self.user_tower(user_features)

        return user_emb

    def encode_video(self, video_ids, channel_ids):
        """
        Encode videos to embeddings.

        Args:
            video_ids: [batch_size] video IDs
            channel_ids: [batch_size] channel IDs

        Returns:
            video_emb: [batch_size, embedding_dim]
        """
        video_emb = self.video_embedding(video_ids)
        channel_emb = self.channel_embedding(channel_ids)

        # Concatenate
        video_features = torch.cat([video_emb, channel_emb], dim=1)

        # Video embedding
        video_emb = self.video_tower(video_features)

        return video_emb

    def forward(self, user_data, video_ids, channel_ids):
        """
        Compute similarity scores.

        Returns:
            scores: [batch_size] dot product scores
        """
        watch_history, search_history, demographics = user_data

        user_emb = self.encode_user(watch_history, search_history, demographics)
        video_emb = self.encode_video(video_ids, channel_ids)

        # Dot product
        scores = (user_emb * video_emb).sum(dim=1)

        return scores


# Training with softmax cross-entropy
model = YouTubeCandidateGeneration(n_videos=1000000, n_channels=100000, embedding_dim=256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for user_data, video_ids_pos, channel_ids_pos, video_ids_neg, channel_ids_neg in train_loader:
        # Positive scores
        pos_scores = model(user_data, video_ids_pos, channel_ids_pos)

        # Negative scores (sampled)
        neg_scores = model(user_data, video_ids_neg, channel_ids_neg)

        # Softmax loss
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long)  # Positive is index 0

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

---

### Approximate Nearest Neighbor (ANN)

**Problem**: Billions of videos, can't compute all dot products.

**Solution**: ANN search (FAISS, ScaNN).

**Process**:
1. Pre-compute all video embeddings
2. Build ANN index (HNSW, IVF)
3. Query: User embedding → Top-K similar videos (sub-second)

---

## Ranking Model

### Deep Neural Network

**Goal**: Predict watch time (or watch time / duration).

**Features**:
- **User features**: Watch history, demographics, context
- **Video features**: Title, duration, upload time, channel
- **Contextual**: Time of day, device, user location
- **Historical**: Video's click-through rate, watch time

**Architecture**: Deep feed-forward network.

---

### Implementation

```python
class YouTubeRanking(nn.Module):
    def __init__(self, n_features, hidden_dims=[512, 256, 128]):
        super().__init__()

        layers = []
        prev_dim = n_features

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        # Output: watch time prediction
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.model = nn.Sequential(*layers)

    def forward(self, features):
        """
        Predict expected watch time.

        Args:
            features: [batch_size, n_features]

        Returns:
            watch_time: [batch_size] predicted watch time (seconds)
        """
        watch_time = self.model(features).squeeze()
        return watch_time


# Training
model = YouTubeRanking(n_features=200, hidden_dims=[512, 256, 128])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for features, watch_times in train_loader:
        # Predict watch time
        pred_watch_time = model(features)

        # Loss: MSE or weighted cross-entropy
        loss = F.mse_loss(pred_watch_time, watch_times)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

---

## Watch Time Optimization

### Primary Metric

**Objective**: Maximize total watch time on platform.

**Why watch time?**
- Aligns with user engagement
- Proxy for user satisfaction
- Drives ad revenue

**Ranking score**: Expected watch time.

---

### Addressing Biases

**Problem**: Videos already popular get more recommendations → rich get richer.

**Mitigation**:
1. **Freshness**: Boost recently uploaded videos
2. **Diversity**: Ensure variety in recommendations
3. **Exploration**: Occasionally recommend new channels

---

### Watch Time Prediction

**Challenge**: Predict watch time before user watches.

**Approach**: Regression on historical watch times.

**Features**:
- Video duration
- User's average watch time for similar videos
- Video's historical completion rate
- User-video interaction features

```python
def predict_watch_time(user, video, model):
    """
    Predict expected watch time.

    Returns:
        expected_watch_time: Float (seconds)
    """
    features = extract_features(user, video)

    with torch.no_grad():
        watch_time = model(features).item()

    return watch_time


# Rank by predicted watch time
def rank_candidates(user, candidates, model):
    """
    Rank candidate videos by expected watch time.

    Returns:
        ranked_videos: List of videos sorted by watch time
    """
    scores = []

    for video in candidates:
        watch_time = predict_watch_time(user, video, model)
        scores.append((video, watch_time))

    # Sort descending
    scores.sort(key=lambda x: x[1], reverse=True)

    return [video for video, _ in scores]
```

---

## Addressing Clickbait and Misinformation

### Content Quality Signals

**Problem**: Clickbait maximizes clicks but hurts user satisfaction.

**Solution**: Multi-objective optimization.

**Objectives**:
1. **Watch time**: User engagement
2. **User satisfaction**: Surveys, likes, dislikes
3. **Content quality**: Authoritative sources, fact-checking

**Ranking formula**:
$$\text{Score} = w_1 \cdot \text{Watch Time} + w_2 \cdot \text{Satisfaction} - w_3 \cdot \text{Clickbait Score}$$

---

### Reducing Borderline Content

**Borderline content**: Violates community guidelines but doesn't warrant removal.

**Examples**: Conspiracy theories, sensationalism, misleading health claims.

**Strategy**: De-rank borderline content (reduce visibility, not remove).

**Implementation**:
- Classify videos as borderline using ML
- Apply ranking penalty
- Limit recommendations in Home feed

---

## Cold Start for New Channels

### New Creator Problem

**Challenge**: New channels have no watch history → hard to recommend.

**Solutions**:
1. **Content-based**: Use video title, description, tags
2. **Channel bootstrapping**: Recommend to subscribers of similar channels
3. **Exploration**: Show to random users, measure engagement

---

### Implementation

```python
def cold_start_score(video, similar_channels):
    """
    Score new video for cold start.

    Args:
        video: New video with no watch history
        similar_channels: Channels with similar content

    Returns:
        score: Float score
    """
    score = 0

    # Content-based features
    if video.title_quality > 0.7:  # High-quality title
        score += 0.3

    # Similar channel performance
    for channel in similar_channels:
        avg_watch_time = channel.avg_watch_time
        score += avg_watch_time / len(similar_channels)

    # Upload recency bonus
    days_since_upload = (today - video.upload_date).days
    if days_since_upload < 7:
        score *= 1.2  # Boost recent uploads

    return score
```

---

## Shorts vs. Long-Form Recommendations

### YouTube Shorts

**Format**: Vertical, 15-60 second videos (TikTok competitor).

**Different ranking criteria**:
- **Completion rate**: Did user watch entire short?
- **Swipe behavior**: Did user swipe away quickly?
- **Engagement**: Likes, comments, shares

**Challenge**: Separate feed from long-form (different user intent).

---

### Dual Feed Architecture

**Long-form feed** (Home):
- Optimizes for watch time
- Horizontal videos
- Longer engagement sessions

**Shorts feed**:
- Optimizes for completion rate + engagement
- Vertical videos
- Rapid consumption (minutes, not hours)

**Recommendation models**: Separate for each feed.

---

### Implementation

```python
class ShortsRanking(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Predict: completion, like, share
        )

    def forward(self, features):
        """
        Predict short engagement metrics.

        Returns:
            completion_prob, like_prob, share_prob: [batch_size] probabilities
        """
        logits = self.model(features)

        # Separate predictions
        completion_logit = logits[:, 0]
        like_logit = logits[:, 1]
        share_logit = logits[:, 2]

        return completion_logit, like_logit, share_logit


def rank_shorts(user, candidates, model):
    """
    Rank shorts by engagement score.

    Returns:
        ranked_shorts: List of shorts sorted by score
    """
    scores = []

    for short in candidates:
        features = extract_features(user, short)

        completion_logit, like_logit, share_logit = model(features)

        # Weighted combination
        score = (
            0.5 * torch.sigmoid(completion_logit) +
            0.3 * torch.sigmoid(like_logit) +
            0.2 * torch.sigmoid(share_logit)
        )

        scores.append((short, score.item()))

    # Sort descending
    scores.sort(key=lambda x: x[1], reverse=True)

    return [short for short, _ in scores]
```

---

## Subscription Feed vs. Recommendations

### Balancing Subscriptions and Discovery

**Subscriptions**: Users explicitly follow channels (high intent).

**Recommendations**: Algorithmic discovery (explore new content).

**Challenge**: Balance familiarity vs. serendipity.

**Strategy**:
- Subscription feed: Show content from followed channels
- Home feed: Mix of subscriptions (40%) + recommendations (60%)

---

### Implementation

```python
def generate_home_feed(user, subscription_videos, recommended_videos, k=20):
    """
    Generate home feed mixing subscriptions and recommendations.

    Args:
        subscription_videos: Videos from channels user follows
        recommended_videos: Algorithmically recommended videos
        k: Number of videos in feed

    Returns:
        feed: List of k videos
    """
    feed = []

    # Subscription ratio
    n_subscriptions = int(k * 0.4)
    n_recommendations = k - n_subscriptions

    # Select top subscriptions
    feed.extend(subscription_videos[:n_subscriptions])

    # Select top recommendations (excluding already included)
    for video in recommended_videos:
        if video not in feed:
            feed.append(video)

        if len(feed) >= k:
            break

    return feed
```

---

## Evaluation Metrics

### Online Metrics

**A/B testing metrics**:
1. **Watch time**: Total minutes watched per user
2. **CTR (Click-Through Rate)**: Clicks / Impressions
3. **User retention**: % users returning next day/week
4. **Session length**: Time spent per session
5. **User satisfaction**: Surveys, like/dislike ratio

**Trade-offs**:
- CTR vs. Watch time (clickbait has high CTR, low watch time)
- Short-term engagement vs. long-term retention

---

### Offline Metrics

**Prediction accuracy**:
- **Watch time prediction**: RMSE, MAE
- **Click prediction**: AUC, log-loss
- **Ranking quality**: NDCG, MRR

**Challenge**: Offline metrics don't always correlate with online metrics.

---

## Summary

**Key Takeaways**:
1. **Two-stage**: Candidate generation (billions → hundreds) + ranking (hundreds → 20)
2. **Watch time**: Primary optimization metric
3. **Multi-objective**: Balance watch time, satisfaction, content quality
4. **Cold start**: Content-based + exploration for new channels
5. **Shorts**: Separate feed optimizing for completion rate
6. **Subscriptions**: Mix explicit follows (40%) with recommendations (60%)

**Scale**: Billions of videos, millions of hours watched daily.

**Challenges**: Clickbait, misinformation, filter bubbles, creator fairness.

---

## Practice Problems

**Problem 1**: Implement two-stage pipeline (candidate generation + ranking) on MovieLens. Measure latency and accuracy.

**Problem 2**: Train ranking model with multi-objective loss (click, watch time, like). Compare with single-objective (watch time only).

**Problem 3**: Implement cold start strategy for new videos. Measure watch time for cold-start videos vs. warm-start videos.

**Problem 4**: Design shorts ranking model. What features are most predictive of completion rate?

---

## References

1. **Covington, P., et al. (2016)**. "Deep Neural Networks for YouTube Recommendations". *RecSys*.
   - Seminal paper on YouTube's deep learning approach

2. **Davidson, J., et al. (2010)**. "The YouTube Video Recommendation System". *RecSys*.
   - Early YouTube recommendation system

3. **Zhao, X., et al. (2019)**. "Recommending What Video to Watch Next: A Multitask Ranking System". *RecSys*.
   - Multi-task learning for watch next recommendations

4. **Chen, M., et al. (2019)**. "Top-K Off-Policy Correction for a REINFORCE Recommender System". *WSDM*.
   - Reinforcement learning for YouTube recommendations

5. **YouTube Creator Academy**. "How the YouTube Algorithm Works" (2021).
   - Official YouTube documentation
