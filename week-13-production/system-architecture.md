# Week 13: Production System Architecture

## Overview

**Production RecSys** serve millions of users in real-time with strict latency requirements (<100ms).

**Architecture patterns**:
1. **Lambda**: Batch + real-time (Twitter, LinkedIn)
2. **Kappa**: Streaming-only (Netflix, Spotify)
3. **Multi-stage**: Retrieval → Ranking → Re-ranking (YouTube, Pinterest)

---

## Multi-Stage Architecture

### Why Multi-Stage?

**Challenge**: Millions of items, can't score all in real-time.

**Solution**: Funnel from millions → thousands → tens.

**Stages**:
1. **Candidate Generation** (Retrieval): 1M items → 1000 candidates (fast, simple model)
2. **Ranking**: 1000 → 100 (complex model, more features)
3. **Re-ranking**: 100 → 10 (business logic, diversity, fairness)

---

### Stage 1: Retrieval

**Goal**: Quickly find ~1000 candidates from millions.

**Methods**:
- **Collaborative filtering**: Matrix factorization, ALS
- **Two-tower models**: User/item embeddings + ANN search
- **Multiple retrievers**: Combine CF + content-based + trending

**Latency**: <10ms

```python
def retrieve_candidates(user_id, n_candidates=1000):
    candidates = []

    # CF retrieval
    cf_items = collaborative_filter.top_k(user_id, k=500)
    candidates.extend(cf_items)

    # Content-based retrieval
    user_profile = get_user_profile(user_id)
    content_items = content_recommender.top_k(user_profile, k=300)
    candidates.extend(content_items)

    # Trending items
    trending = get_trending_items(k=200)
    candidates.extend(trending)

    return list(set(candidates))[:n_candidates]
```

---

### Stage 2: Ranking

**Goal**: Rank 1000 candidates by predicted engagement.

**Model**: Deep neural network (Wide & Deep, DeepFM, DIN).

**Features**:
- User: demographics, history (embeddings)
- Item: content features, popularity
- Context: time, device, location
- Interactions: user-item cross-features

**Latency**: 10-50ms

```python
def rank_candidates(user_id, candidates):
    user_features = get_user_features(user_id)
    context = get_context()  # time, device, etc.

    scores = []
    for item in candidates:
        item_features = get_item_features(item)
        features = combine_features(user_features, item_features, context)

        score = ranking_model.predict(features)
        scores.append((item, score))

    # Sort by score
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return [item for item, score in ranked[:100]]
```

---

### Stage 3: Re-ranking

**Goal**: Apply business logic, diversity, fairness.

**Rules**:
- **Diversity**: No more than 2 items from same category
- **Freshness**: Include recent uploads
- **Exploration**: 10% random items
- **Personalization**: Adjust for user preferences

**Latency**: 10-30ms

```python
def rerank(ranked_items, user_id, k=10):
    result = []
    categories_seen = set()

    for item in ranked_items:
        # Diversity: limit per category
        category = get_category(item)
        if categories_seen.count(category) >= 2:
            continue

        result.append(item)
        categories_seen.add(category)

        if len(result) >= k * 0.9:  # 90% ranked
            break

    # Add exploration items (10%)
    exploration = sample_random_items(k - len(result))
    result.extend(exploration)

    return result[:k]
```

---

## Lambda Architecture

### Components

**1. Batch Layer**:
- Train models on historical data (daily/weekly)
- Generate item embeddings offline
- High-quality, slow updates

**2. Speed Layer**:
- Real-time updates from streaming data
- Approximate, fast updates
- Handle recent user interactions

**3. Serving Layer**:
- Merge batch + speed layer outputs
- Serve predictions

---

### Example

**Netflix**:
- **Batch**: Retrain models nightly on previous day's data
- **Speed**: Update user embeddings as they watch (session-based)
- **Serving**: Combine for real-time recommendations

---

## Kappa Architecture

### Concept

**Stream-only**: No batch layer, everything streamed.

**Benefits**:
- Simpler (one code path)
- Fresher (continuous updates)

**Challenges**:
- Harder to debug
- Need robust streaming infra (Kafka, Flink)

---

## Feature Store

### Purpose

**Centralize features** for training and serving.

**Benefits**:
- **Consistency**: Same features in training/serving
- **Reusability**: Share features across models
- **Monitoring**: Track feature drift

**Examples**: Feast, Tecton, AWS SageMaker Feature Store

---

### Architecture

```
Data Sources (logs, DBs) → Feature Engineering Pipeline
                                    ↓
                              Feature Store
                                ↓       ↓
                          Training  Serving
                          (batch)  (real-time)
```

---

## Caching

### Why Cache?

**Problem**: Feature computation expensive (embedding lookups, DB queries).

**Solution**: Cache frequently accessed features.

**Layers**:
1. **L1 (in-memory)**: User embeddings (Redis, Memcached)
2. **L2 (disk)**: Item embeddings (SSD)
3. **L3 (database)**: Full features (PostgreSQL, Cassandra)

---

### Cache Invalidation

**Strategies**:
1. **TTL** (Time to Live): Cache for 1 hour, then refresh
2. **Event-driven**: Invalidate when user interacts
3. **Lazy**: Refresh on next request

```python
import redis

cache = redis.Redis()

def get_user_embedding(user_id):
    key = f"user_emb:{user_id}"
    cached = cache.get(key)

    if cached:
        return pickle.loads(cached)

    # Compute embedding
    embedding = compute_user_embedding(user_id)

    # Cache for 1 hour
    cache.setex(key, 3600, pickle.dumps(embedding))

    return embedding
```

---

## Summary

**Key Takeaways**:
1. **Multi-stage**: Retrieval (1M→1K) → Ranking (1K→100) → Re-rank (100→10)
2. **Lambda**: Batch (quality) + Speed (freshness)
3. **Kappa**: Stream-only (simplicity)
4. **Feature store**: Centralize features
5. **Caching**: Reduce latency

**Latency Budget** (typical):
- Retrieval: 10ms
- Ranking: 30ms
- Re-ranking: 20ms
- Feature lookup: 20ms
- **Total**: <100ms

**Next**: Scalability challenges.

---

## References

1. **Covington, P., et al. (2016)**. "Deep Neural Networks for YouTube Recommendations". *RecSys*.
2. **Grbovic, M., &Cheng, H. (2018)**. "Real-time Personalization using Embeddings for Search Ranking at Airbnb". *KDD*.
3. **Amatriain, X. (2013)**. "Building Industrial-Scale Real-World Recommender Systems". *RecSys Tutorial*.
