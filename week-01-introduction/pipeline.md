# Week 1: The Recommendation Pipeline

## Learning Objectives

- Understand the end-to-end recommendation pipeline
- Recognize the purpose of each pipeline stage
- Learn why multi-stage architectures are necessary at scale
- **Derive the latency math** that forces a multi-stage design
- **Map each stage** to its optimization target: recall, precision, business rules

---

## The Opening Problem: The 10-Millisecond Challenge

*"You have 1 million items and 10 milliseconds to respond. How do you do it?"*

Let me give you the numbers that keep recommendation engineers up at night.

**The Reality of Scale**:
- Netflix: 15,000+ titles
- Amazon: 350+ million products
- YouTube: 800+ million videos
- Spotify: 100+ million tracks

**The User Expectation**: Response in under 100 milliseconds. Any slower, and users notice. Any much slower, and they leave.

**The Naive Approach**: Score every item with a sophisticated model.

Let us do the math together:
- 1 million items
- A decent deep learning model: 0.1ms per item
- Total time: **100,000 milliseconds = 100 seconds**

*That is a 1000x overshoot of our latency budget.*

**The Question**: How do we bridge this 1000x gap?

*Pause and think: What strategies come to mind?*

---

## The Fundamental Insight: Funnel Architecture

The solution emerges from a simple observation: **not all items need the same level of scrutiny**.

Think of it like hiring for a company:
1. **Resume screening** (fast, approximate): 10,000 applicants down to 100
2. **Phone interviews** (moderate effort): 100 down to 10
3. **On-site interviews** (expensive, thorough): 10 down to 1

The recommendation pipeline follows the same principle:

```
Stage           | Items In  | Items Out | Time/Item | Total Time | Optimizes For
----------------|-----------|-----------|-----------|------------|---------------
Candidate Gen   | 1,000,000 | 1,000     | 0.001ms   | 1ms        | RECALL
Ranking         | 1,000     | 100       | 0.01ms    | 10ms       | PRECISION
Re-ranking      | 100       | 10        | 0.1ms     | 10ms       | BUSINESS RULES
----------------|-----------|-----------|-----------|------------|---------------
TOTAL           | 1,000,000 | 10        | -         | ~21ms      | -
```

*This is how you turn an impossible problem into a tractable one.*

---

## Deriving the Pipeline: Why This Architecture?

### Why Not Use One Perfect Model?

*"Professor, why not just build one really good model that does everything?"*

Excellent question. Let us derive why this is impossible.

**Attempt 1: Score Everything with a Rich Model**

A production ranking model uses:
- 100+ dense features
- 1000+ sparse features (categorical embeddings)
- Multiple neural network layers

**Cost per item**: ~0.1ms (on optimized hardware)

For 1M items: $1{,}000{,}000 \times 0.1\text{ms} = 100{,}000\text{ms}$

**Verdict**: 1000x too slow.

**Attempt 2: Score Everything with a Simple Model**

Use just a dot product: $\text{score}(u, i) = \mathbf{u}^T \mathbf{v}_i$

**Cost per item**: ~0.0001ms

For 1M items: $1{,}000{,}000 \times 0.0001\text{ms} = 100\text{ms}$

**Verdict**: Barely acceptable latency, but terrible accuracy. We cannot use rich features.

**The Insight**: We need both:
- Simple models (fast) to reduce the candidate space
- Rich models (accurate) on a reduced candidate set

*This tension between speed and accuracy forces the multi-stage design.*

---

## The Complete Pipeline Diagram

```
+-------------------+
| User Interaction  |
|  (clicks, views)  |
+---------+---------+
          |
          v
+-------------------+
| Data Collection   |
|   & Logging       |
+---------+---------+
          |
          v
+-------------------+
|    Candidate      |
|   Generation      |  1M -> 1K items (1-5ms)
|  (Fast & Broad)   |  OPTIMIZE FOR: RECALL
+---------+---------+
          |
          v
+-------------------+
|     Ranking       |
| (Rich Features)   |  1K -> 100 items (10-20ms)
|                   |  OPTIMIZE FOR: PRECISION
+---------+---------+
          |
          v
+-------------------+
|   Re-ranking      |
|  (Diversity,      |  100 -> 10 items (5-10ms)
|   Business)       |  OPTIMIZE FOR: BUSINESS RULES
+---------+---------+
          |
          v
+-------------------+
|    Display        |
|   (UI/UX)         |  Present to user
+---------+---------+
          |
          v
    [Feedback Loop]
```

---

## Numerical Walkthrough: The YouTube Example

Let me walk you through exactly how YouTube serves recommendations.

### The Setup
- **Catalog**: 800 million videos
- **User**: Logged in, mobile, 8 PM
- **Latency budget**: 100ms total

### Stage 1: Candidate Generation

**Input**: 800,000,000 videos
**Output**: 1,000 videos
**Latency**: 5ms

**How it works**:
```
User embedding (128-dim) -> ANN index -> Top 1000 similar items

Computation:
- ANN search with HNSW/FAISS: ~5ms for 800M vectors
- Why so fast? Pre-computed embeddings + approximate search
```

**What we sacrifice**: We might miss some relevant videos (recall < 100%)
**What we gain**: 800,000x reduction in candidates

### Stage 2: Ranking

**Input**: 1,000 videos
**Output**: 100 videos (ordered)
**Latency**: 20ms

**How it works**:
```
For each of 1,000 videos:
  - Compute 500+ features
  - Pass through deep neural network
  - Predict: P(click), E[watch_time], P(like)

Final score = w1*P(click) + w2*E[watch_time] + w3*P(like)
```

**Features used**:
- User: watch history (last 50 videos), age, country, device
- Video: title embedding, channel stats, upload date, duration
- Cross: user_watched_channel_before, video_in_user_language

**Cost**: $1000 \times 0.02\text{ms} = 20\text{ms}$

### Stage 3: Re-ranking

**Input**: 100 videos
**Output**: 20 videos (final list)
**Latency**: 5ms

**Business rules applied**:
- No more than 3 videos from same channel
- At least 2 videos from subscribed channels
- 10% exploration (random videos for learning)
- Boost videos uploaded in last 24 hours

**Diversity enforcement**:
- MMR algorithm ensures variety in topics
- No consecutive videos on same topic

### The Math Summary

| Stage | Items | ms/Item | Total ms | Cumulative |
|-------|-------|---------|----------|------------|
| Candidate Gen | 800M -> 1K | 0.000006 | 5 | 5ms |
| Ranking | 1K -> 100 | 0.02 | 20 | 25ms |
| Re-ranking | 100 -> 20 | 0.05 | 5 | 30ms |
| Network/Display | - | - | 20 | **50ms** |

**Total latency: 50ms** - well under the 100ms budget.

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

## Stage 2: Candidate Generation - Optimizing for Recall

### Purpose
**Rapidly filter** millions/billions of items down to hundreds of candidates.

**Optimization Target**: **RECALL** - Do not miss relevant items

*"It is better to let some bad items through than to miss good ones."*

### The Recall vs. Latency Trade-off

| Approach | Latency | Recall | Use Case |
|----------|---------|--------|----------|
| Exact KNN | O(n) slow | 100% | Small catalogs |
| Approximate NN | O(log n) fast | 95-99% | Production |
| Inverted Index | O(k) fast | Varies | Text/sparse |

**Key Insight**: We accept 1-5% recall loss to get 100x speedup.

### Why Necessary?
- **Scale**: Cannot score all items (too slow, too expensive)
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

This is the modern workhorse of candidate generation:

```
User Features -> User Tower -> User Embedding -+
                                               |-> Dot Product -> Score
Item Features -> Item Tower -> Item Embedding -+
```

**Why Two Towers?**
- User tower: computed once per request
- Item tower: pre-computed offline for all items
- Serving: just dot products (blazingly fast)

**Feature Engineering for Two-Tower**:

| User Tower Features | Item Tower Features |
|--------------------|---------------------|
| User ID embedding | Item ID embedding |
| Age bucket | Category |
| Country | Upload date |
| Watch history (aggregated) | View count bucket |
| Device type | Duration bucket |

*Critical*: Features must be **independent** of each other (no cross-features).

#### 4. **Multiple Candidate Sources**

Production systems use multiple retrieval strategies:

| Source | What it retrieves | Why |
|--------|-------------------|-----|
| CF | Items similar users liked | Personalization |
| Content | Items similar to user history | Relevance |
| Popularity | Trending items | Social proof |
| Exploration | Random sampling | Learning |
| Fresh | New items | Recency |

**Example (YouTube)**:
- Source 1: Videos similar to watch history (500 candidates)
- Source 2: Videos from subscribed channels (200 candidates)
- Source 3: Trending videos in user's region (200 candidates)
- Source 4: Videos from similar users (100 candidates)

**Union** of all sources -> ~1000 candidates (after dedup)

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

## Stage 3: Ranking - Optimizing for Precision

### Purpose
**Precisely score** hundreds of candidates using rich features.

**Optimization Target**: **PRECISION** - Top items must be truly relevant

*"Now that we have 1000 candidates, we can afford to be picky."*

### Why Separate from Candidate Generation?
- **Rich features**: Can afford expensive feature computation for fewer items
- **Complex models**: Deep neural networks with billions of parameters
- **Cross-features**: Can compute user-item interactions
- **Latency budget**: 10-50ms acceptable (vs. 1ms for candidate generation)

### Feature Engineering by Stage

This is where production systems differ dramatically from academic papers:

| Feature Type | Candidate Gen | Ranking |
|--------------|---------------|---------|
| User ID embedding | Yes | Yes |
| Item ID embedding | Yes | Yes |
| User history (aggregated) | Yes | Yes |
| User history (detailed) | No | Yes |
| Cross-features (user x item) | **No** | **Yes** |
| Real-time features | No | Yes |
| Position bias features | No | Yes |

**Why no cross-features in candidate generation?**

Cross-features like `user_clicked_this_author_before` require:
- Looking up each user-item pair
- Cannot be pre-computed for the item tower

This would defeat the purpose of two-tower independence.

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
- Cross features: user_age x item_category

#### Contextual Features
- Time: hour of day, day of week, season
- Location: country, city, timezone
- Device: mobile, desktop, tablet
- Session: position in session, session length

### Ranking Models

#### Traditional: Logistic Regression
```
P(click | user, item, context) = sigmoid(w^T x)
```

**Pros**: Fast, interpretable, scalable
**Cons**: Limited expressiveness, manual feature engineering

#### Modern: Deep Neural Networks
```
User Features --+
               |-> Concatenate -> MLP -> Softmax -> P(engagement)
Item Features --+
```

**Architectures**:
- **Wide and Deep** (Google, 2016): Linear + DNN
- **DeepFM** (2017): Factorization machines + DNN
- **DCN** (2017): Deep and Cross Network for feature interactions
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
     +-> Task 1: Click prediction
     +-> Task 2: Watch time prediction
     +-> Task 3: Like prediction

Final Score = w1 * P(click) + w2 * E[watch_time] + w3 * P(like)
```

### Calibration
Ensure predicted probabilities match actual frequencies.

**Example**: If model predicts 30% click rate, 30% of those items should be clicked.

**Techniques**:
- Platt scaling
- Isotonic regression
- Temperature scaling

---

## Stage 4: Re-ranking - Optimizing for Business Rules

### Purpose
Optimize the **final list** for business goals beyond relevance.

**Optimization Target**: **BUSINESS OBJECTIVES** - Diversity, fairness, revenue

*"The ranker found the most clickable items. Now we make sure the list is actually good for users AND the business."*

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

**Example**: Do not show 10 action movies; mix genres

#### 2. **Fairness**
- **Provider fairness**: Exposure for small creators vs. large creators
- **Consumer fairness**: Equal quality recommendations for all user demographics

#### 3. **Business Rules**
- **Inventory management**: Promote items needing sales
- **Contractual obligations**: Featured content, sponsored items
- **Freshness**: Boost recent items

#### 4. **Exploration**
Explore new items to gather data.

**epsilon-greedy**:
- With probability epsilon: show random item (exploration)
- With probability 1-epsilon: show top-ranked item (exploitation)

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
    score[i] = lambda * relevance(i) - (1-lambda) * max similarity(i, j) for j in L
  Add argmax(score) to L
```

**lambda**: Tunable parameter (relevance vs. diversity)

#### Determinantal Point Processes (DPP)
Probabilistic model for diverse subset selection.

**Probability of subset S**:

$$P(S) \propto \det(L_S)$$

where $L$ is a kernel matrix encoding similarity.

**Property**: Naturally assigns lower probability to similar item sets.

---

## Real System Examples: How the Giants Do It

### YouTube (Covington et al., 2016)

| Stage | Implementation | Scale |
|-------|----------------|-------|
| Candidate Gen | Two-tower DNN + multiple sources | 800M -> 500 |
| Ranking | Wide-and-Deep multi-task model | 500 -> 20 |
| Re-ranking | Diversity + freshness rules | Final ordering |

**Key Innovation**: Watch time prediction, not just click prediction.

### Netflix (Gomez-Uribe and Hunt, 2015)

| Stage | Implementation | Scale |
|-------|----------------|-------|
| Candidate Gen | Multiple algorithms per "row" | 15K -> 100s per row |
| Ranking | Personalized ranking per row | Row-level ordering |
| Re-ranking | Row selection + ordering | Page composition |

**Key Innovation**: Row-based organization (trending, because you watched X, etc.)

### Amazon (Smith and Linden, 2017)

| Stage | Implementation | Scale |
|-------|----------------|-------|
| Candidate Gen | Item-to-item collaborative filtering | 350M -> 100s |
| Ranking | Purchase probability model | Per-widget ranking |
| Re-ranking | Business rules + inventory | Final selection |

**Key Innovation**: "Customers who bought X also bought Y" - item-based CF.

### Feature Comparison Across Systems

| Feature | YouTube | Netflix | Amazon |
|---------|---------|---------|--------|
| Primary signal | Watch time | Rating + completion | Purchase |
| User embedding dim | 256 | 100-200 | Varies |
| Candidate sources | 4-5 | 10+ (per row) | 3-4 |
| Ranking model | Multi-task DNN | Ensemble | Gradient boosted |
| Latency target | 100ms | 250ms | 100ms |

---

## What Can Go Wrong: Failure Modes

### 1. Stage Mismatch

**Problem**: Candidate generation misses items that ranking would have scored highly.

**Example**:
```
User loves obscure indie films.
Candidate gen uses popularity-weighted embeddings.
Result: No indie films in candidate set.
Ranker never gets a chance to score them.
```

**Solution**: Ensure candidate generation recall is measured against ranking preferences, not just clicks.

### 2. Feature Inconsistency (Training-Serving Skew)

**Problem**: Features computed differently at training vs. serving time.

**Example**:
```
Training: user_avg_watch_time computed over all history
Serving: user_avg_watch_time computed over last 30 days (for speed)

Result: Model learns patterns that do not exist at serving time.
```

**Solution**: Use the same feature computation code for training and serving (feature stores).

### 3. Cascade Failures

**Problem**: An error in an early stage propagates and amplifies.

**Example**:
```
Candidate generation returns 0 items (bug in ANN index).
Ranking has nothing to rank.
User sees empty recommendations.
```

**Solution**:
- Fallbacks at each stage (popularity-based backup)
- Monitoring for empty/small candidate sets
- Circuit breakers

### 4. Feedback Loop Amplification

**Problem**: Recommendations create the data that trains future recommendations.

**Example**:
```
Day 1: System recommends action movies (they have high CTR)
Day 2: User watches action movies (only option)
Day 3: System sees "user likes action" -> more action movies
Result: User trapped in filter bubble
```

**Solution**:
- Exploration (epsilon-greedy, Thompson sampling)
- Diversity constraints in re-ranking
- Counterfactual evaluation

### 5. Latency Spikes

**Problem**: A stage occasionally takes too long, breaking latency SLA.

**Example**:
```
Ranking model: p50 = 10ms, p99 = 100ms
At p99, total latency = 150ms (unacceptable)
```

**Solution**:
- Timeout and fallback to simpler model
- Tail latency optimization
- Request hedging (send duplicate requests)

---

## Stage 5: Evaluation and Online Testing

### Offline Evaluation
- **Metrics**: Precision@K, NDCG, MAP
- **Data**: Historical interactions
- **Limitation**: Does not capture real user behavior

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
- **New users**: No history -> use demographics, popularity
- **New items**: No interactions -> use content features

### Data Quality
- **Noisy labels**: Clicks != satisfaction
- **Biased data**: Popular items over-represented
- **Solution**: Debiasing techniques, inverse propensity weighting

---

## Socratic Questions to Test Understanding

1. **"Why not use one perfect model for everything?"**
   - Hint: What is the time complexity of scoring N items with a complex model?

2. **"What happens if candidate generation has 80% recall instead of 99%?"**
   - Hint: Can the ranker recover missed items?

3. **"Why does YouTube optimize for watch time instead of clicks?"**
   - Hint: What user behavior does each metric incentivize?

4. **"Why can't we use cross-features in two-tower models?"**
   - Hint: What must be pre-computed for ANN search?

5. **"If re-ranking adds diversity, why not add diversity to ranking directly?"**
   - Hint: What is the difference between pointwise and listwise optimization?

---

## Summary

The recommendation pipeline has 5 key stages:

1. **Data Collection**: Log all user interactions
2. **Candidate Generation**: Millions -> thousands (fast, optimize for **RECALL**)
3. **Ranking**: Thousands -> hundreds (precise, optimize for **PRECISION**)
4. **Re-ranking**: Hundreds -> final list (optimize for **BUSINESS RULES**)
5. **Evaluation**: A/B testing for online metrics

**Key Insight**: The multi-stage architecture is not a choice - it is a mathematical necessity forced by the latency constraints.

**The Numbers to Remember**:
- 1M items, 10ms budget -> impossible with one stage
- 1M -> 1K -> 100 -> 10 with increasing model complexity -> achievable

**Next**: See **challenges.md** for common problems (cold start, sparsity, etc.)

---

## References

1. Covington, P., et al. (2016). "Deep neural networks for YouTube recommendations". *RecSys*.
2. Cheng, H. T., et al. (2016). "Wide & deep learning for recommender systems". *DLRS*.
3. Chen, M., et al. (2019). "Sampling-bias-corrected neural modeling for large corpus item recommendations". *RecSys*.
4. Gomez-Uribe, C. A., & Hunt, N. (2015). "The Netflix recommender system: Algorithms, business value, and innovation". *ACM TMIS*.
5. Smith, B., & Linden, G. (2017). "Two decades of recommender systems at Amazon.com". *IEEE Internet Computing*.
