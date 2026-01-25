# Week 1: The Recommendation Pipeline

## Learning Objectives

- Understand the end-to-end recommendation pipeline
- Recognize the purpose of each pipeline stage
- Learn why multi-stage architectures are necessary at scale

---

## Overview: The Complete Pipeline

```
┌─────────────────┐
│ User Interaction│
│  (clicks, views)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Collection │
│   & Logging     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Candidate    │
│   Generation    │  Reduce millions → hundreds
│  (Fast & Broad) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Ranking     │
│ (Rich Features) │  Rank hundreds → dozens
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Re-ranking    │
│  (Diversity,    │  Optimize final list
│   Business)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Display      │
│   (UI/UX)       │  Present to user
└─────────────────┘
         │
         ▼
    [Feedback Loop]
```

---

## Stage 1: Data Collection and User Interaction Logging

### Purpose
Capture all user interactions to build training data and personalization signals.

### Data Types

#### Explicit Feedback
- **Ratings**: Stars, thumbs up/down
- **Reviews**: Text feedback
- **Preferences**: Likes, favorites

#### Implicit Feedback
- **Clicks**: Which items clicked
- **Views**: What was displayed and for how long
- **Purchases**: Transaction history
- **Watch time**: Video streaming duration
- **Scrolling**: How far user scrolled
- **Hovers**: Mouse movements
- **Searches**: Query history

### Implementation Considerations

**Logging Schema**:
```json
{
  "user_id": "u12345",
  "item_id": "i98765",
  "action": "click",
  "timestamp": "2024-01-15T10:30:00Z",
  "context": {
    "device": "mobile",
    "location": "homepage",
    "position": 3,
    "session_id": "sess_abc"
  }
}
```

**Storage**:
- **Real-time**: Kafka, Kinesis for streaming
- **Batch**: S3, HDFS for historical data
- **Processed**: Feature stores (Feast, Tecton)

**Privacy**:
- **GDPR compliance**: User consent, right to deletion
- **Anonymization**: Hash user IDs
- **Retention policies**: Delete old data

---

## Stage 2: Candidate Generation

### Purpose
**Rapidly filter** millions/billions of items down to hundreds of candidates.

**Goal**: High recall (don't miss relevant items) over precision

### Why Necessary?
- **Scale**: Can't score all items (too slow, too expensive)
- **Latency**: Need sub-100ms response times
- **Example**: YouTube has 800M videos but shows ~20 recommendations

### Common Approaches

#### 1. **Collaborative Filtering**
- Matrix factorization: $\text{score}(u, i) = \mathbf{u}^T \mathbf{v_i}$
- Precompute user and item embeddings
- Top-K nearest neighbors via ANN

#### 2. **Content-Based**
- Item similarity: "Users who liked X also liked..."
- Precompute item-item similarity matrix
- Fast lookup at serving time

#### 3. **Two-Tower Neural Networks**
```
User Features → User Tower → User Embedding ─┐
                                              ├─> Dot Product → Score
Item Features → Item Tower → Item Embedding ─┘
```
- Separate encoders for users and items
- Dot product for similarity
- ANN search over item embeddings

#### 4. **Multiple Candidate Sources**
Combine multiple retrieval strategies:
- **CF**: Similar users' items
- **Content**: Similar items to user's history
- **Popularity**: Trending items
- **Exploration**: Random sampling for diversity
- **Fresh**: New items (recency bias)

**Example (YouTube)**:
- Source 1: Videos similar to watch history
- Source 2: Videos from subscribed channels
- Source 3: Trending videos in user's region
- Source 4: Videos from similar users

**Union** of all sources → hundreds of candidates

### Implementation: ANN Search

**Problem**: Finding nearest neighbors in high-dimensional space

**Libraries**:
- **FAISS** (Facebook): GPU acceleration, billions of vectors
- **ScaNN** (Google): Anisotropic vector quantization
- **HNSW**: Graph-based, high accuracy
- **Annoy** (Spotify): Tree-based

**Trade-off**: Speed vs. accuracy
- Exact search: Too slow for millions of items
- Approximate: 10-100x faster, 95%+ recall

**Example Code** (FAISS):
```python
import faiss
import numpy as np

# Item embeddings: 1M items, 128 dimensions
item_embeddings = np.random.randn(1000000, 128).astype('float32')

# Build index
index = faiss.IndexFlatIP(128)  # Inner product
index.add(item_embeddings)

# User embedding
user_embedding = np.random.randn(1, 128).astype('float32')

# Find top 100 candidates
k = 100
scores, indices = index.search(user_embedding, k)
```

**Latency**: ~1-10ms for 1M items

---

## Stage 3: Ranking

### Purpose
**Precisely score** hundreds of candidates using rich features.

**Goal**: High precision (top items are truly relevant)

### Why Separate from Candidate Generation?
- **Rich features**: Can afford expensive feature computation for fewer items
- **Complex models**: Deep neural networks with billions of parameters
- **Latency budget**: 10-50ms acceptable (vs. 1ms for candidate generation)

### Feature Types

#### User Features
- Demographics: age, gender, location
- Historical interactions: past clicks, purchases, ratings
- Session context: current browsing, time of day, device
- Derived: user embeddings from CF, engagement patterns

#### Item Features
- Metadata: category, price, duration, genre
- Content: text, images, audio (via embeddings)
- Popularity: view count, average rating, recency
- Derived: item embeddings from CF

#### User-Item Interaction Features
- Historical interaction: has user engaged with item before?
- Similarity: cosine similarity between user and item embeddings
- Cross features: user_age × item_category

#### Contextual Features
- Time: hour of day, day of week, season
- Location: country, city, timezone
- Device: mobile, desktop, tablet
- Session: position in session, session length

### Ranking Models

#### Traditional: Logistic Regression
```
P(click | user, item, context) = σ(w^T x)
```

**Pros**: Fast, interpretable, scalable
**Cons**: Limited expressiveness, manual feature engineering

#### Modern: Deep Neural Networks
```
User Features ──┐
                ├─> Concatenate → MLP → Softmax → P(engagement)
Item Features ──┘
```

**Architectures**:
- **Wide & Deep** (Google, 2016): Linear + DNN
- **DeepFM** (2017): Factorization machines + DNN
- **DCN** (2017): Deep & Cross Network for feature interactions
- **DLRM** (Facebook, 2019): Embeddings + MLP for ads

### Multi-Task Learning
Predict multiple objectives simultaneously:
- Click probability
- Watch time (for videos)
- Like probability
- Purchase probability

**Architecture (YouTube)**:
```
Shared Layers
     ├─> Task 1: Click prediction
     ├─> Task 2: Watch time prediction
     └─> Task 3: Like prediction

Final Score = w1 × P(click) + w2 × E[watch_time] + w3 × P(like)
```

### Calibration
Ensure predicted probabilities match actual frequencies.

**Example**: If model predicts 30% click rate, 30% of those items should be clicked.

**Techniques**:
- Platt scaling
- Isotonic regression
- Temperature scaling

---

## Stage 4: Re-ranking

### Purpose
Optimize the **final list** for business goals beyond relevance.

### Why Necessary?
Pure relevance ranking can lead to:
- **Filter bubbles**: Showing same type of content
- **Popularity bias**: Always recommending popular items
- **Poor diversity**: Similar items clustered together
- **Suboptimal business metrics**: Revenue, long-term engagement

### Re-ranking Objectives

#### 1. **Diversity**
Ensure variety in recommendations.

**Metrics**:
- **Intra-list diversity**: How different are items from each other?

$$\text{Diversity} = \frac{1}{|L|(|L|-1)} \sum_{i \in L} \sum_{j \in L, j \neq i} \text{distance}(i, j)$$

**Algorithms**:
- **MMR** (Maximal Marginal Relevance): Balance relevance and diversity
- **DPP** (Determinantal Point Processes): Probabilistic diverse subsets

**Example**: Don't show 10 action movies; mix genres

#### 2. **Fairness**
- **Provider fairness**: Exposure for small creators vs. large creators
- **Consumer fairness**: Equal quality recommendations for all user demographics

#### 3. **Business Rules**
- **Inventory management**: Promote items needing sales
- **Contractual obligations**: Featured content, sponsored items
- **Freshness**: Boost recent items

#### 4. **Exploration**
Explore new items to gather data.

**ε-greedy**:
- With probability ε: show random item (exploration)
- With probability 1-ε: show top-ranked item (exploitation)

**Thompson Sampling**: Bayesian approach balancing exploration and exploitation

### Position Bias Correction
Users more likely to click top items regardless of relevance.

**Solution**: Train model aware of position bias, remove at inference.

### Algorithms

#### Maximal Marginal Relevance (MMR)
```
Start with empty list L
While |L| < K:
  For each candidate i:
    score[i] = λ × relevance(i) - (1-λ) × max similarity(i, j) for j in L
  Add argmax(score) to L
```

**λ**: Tunable parameter (relevance vs. diversity)

#### Determinantal Point Processes (DPP)
Probabilistic model for diverse subset selection.

**Probability of subset S**:

$$P(S) \propto \det(L_S)$$

where $L$ is a kernel matrix encoding similarity.

**Property**: Naturally assigns lower probability to similar item sets.

---

## Stage 5: Evaluation and Online Testing

### Offline Evaluation
- **Metrics**: Precision@K, NDCG, MAP
- **Data**: Historical interactions
- **Limitation**: Doesn't capture real user behavior

### Online Evaluation (A/B Testing)
- **Treatment**: New model/algorithm
- **Control**: Current production model
- **Metrics**: CTR, conversion, watch time, retention
- **Duration**: 1-4 weeks
- **Statistical significance**: t-test, bootstrap

### Guardrail Metrics
Ensure no negative side effects:
- User retention
- Revenue
- Latency
- System load

---

## End-to-End Example: YouTube

### Input
- User: u123456
- Context: Homepage on mobile at 8 PM

### Candidate Generation (Stage 1)
- **Input**: 800M videos
- **Sources**:
  - Similar to watch history (CF)
  - From subscriptions (content-based)
  - Trending in user's country
  - Explore (random)
- **Output**: 500 candidates
- **Latency**: <5ms

### Ranking (Stage 2)
- **Input**: 500 candidates
- **Model**: Deep neural network
- **Features**:
  - User: watch history, demographics, device
  - Video: title, channel, views, upload date
  - Context: time, location, session
- **Output**: Scored list of 500 videos
- **Latency**: 20ms

### Re-ranking (Stage 3)
- **Input**: Top 100 from ranking
- **Objectives**:
  - Diversity: Mix topics and channels
  - Freshness: Boost recent uploads
  - Exploration: Include 10% random
- **Output**: Final 20 videos
- **Latency**: 5ms

### Display (Stage 4)
- **Thumbnails**: Personalized via contextual bandits
- **Order**: Top 20 in optimized order
- **Total latency**: <50ms

### Feedback (Loop)
- User clicks video #3, watches for 5 minutes
- Logged for future training

---

## Implementation Challenges

### Latency Constraints
- **Real-time**: <100ms total
- **Solution**: Precomputation, caching, ANN search

### Scalability
- **Billions of users/items**
- **Solution**: Distributed systems, sharding, model compression

### Freshness
- **New content/users** need immediate inclusion
- **Solution**: Online learning, incremental updates

### Cold Start
- **New users**: No history → use demographics, popularity
- **New items**: No interactions → use content features

### Data Quality
- **Noisy labels**: Clicks ≠ satisfaction
- **Biased data**: Popular items over-represented
- **Solution**: Debiasing techniques, inverse propensity weighting

---

## Summary

The recommendation pipeline has 5 key stages:

1. **Data Collection**: Log all user interactions
2. **Candidate Generation**: Millions → hundreds (fast, broad)
3. **Ranking**: Hundreds → dozens (precise, rich features)
4. **Re-ranking**: Optimize diversity, fairness, business goals
5. **Evaluation**: A/B testing for online metrics

**Key Insight**: Multi-stage architecture necessary for scale and latency.

**Next**: See **challenges.md** for common problems (cold start, sparsity, etc.)

---

## References

1. Covington, P., et al. (2016). "Deep neural networks for YouTube recommendations". *RecSys*.
2. Cheng, H. T., et al. (2016). "Wide & deep learning for recommender systems". *DLRS*.
3. Chen, M., et al. (2019). "Sampling-bias-corrected neural modeling for large corpus item recommendations". *RecSys*.
