# Week 13: Production System Architecture

## Opening Problem: The YouTube Challenge

*"Before we dive into architectures, let me pose a question that will guide our entire discussion today."*

**The Problem**: YouTube serves over 1 billion users, with a catalog of hundreds of millions of videos. When you open the app, recommendations appear in under 100 milliseconds.

**Let's do the math together**:

Suppose we have:
- 500 million videos in the catalog
- A ranking model that takes 0.1ms per video (extremely optimistic!)
- Target latency: 100ms

**Naive approach - score everything**:

$$\text{Time to score all videos} = 500{,}000{,}000 \times 0.1\text{ms} = 50{,}000{,}000\text{ms} = 13.9 \text{ hours}$$

*"So if we try to score every video, we'd need nearly 14 hours. Our user wants results in 100 milliseconds. That's off by a factor of 500,000!"*

**What about parallelization?**

Even with 10,000 GPUs scoring in parallel:

$$\text{Parallel time} = \frac{13.9 \text{ hours}}{10{,}000} = 5 \text{ seconds}$$

*"Still 50x too slow, and we'd need a small data center just for one user request. This is economically and technically infeasible."*

**The insight**: We cannot score all items. We must be clever about which items we even consider.

---

## The Birth of Multi-Stage Architecture

### Socratic Derivation: Why Three Stages?

*"Let's derive the solution together. What if we could somehow reduce the number of items we need to score?"*

**Key observation**: Most items are completely irrelevant to any given user.

A user who watches cooking videos probably doesn't need us to score:
- Gaming walkthroughs in languages they don't speak
- Academic physics lectures
- Children's nursery rhymes (unless they have kids)

**Design principle**: Use cheap, fast methods to eliminate obviously irrelevant items first.

**Stage 1 - Candidate Generation (Retrieval)**:
- **Goal**: Quickly find ~1000 "plausibly relevant" items from millions
- **Method**: Simple models, approximate nearest neighbors
- **Latency budget**: ~10ms
- **Accuracy**: Can afford some mistakes (we'll refine later)

**Stage 2 - Ranking**:
- **Goal**: Accurately order the 1000 candidates
- **Method**: Complex neural networks with rich features
- **Latency budget**: ~30-50ms
- **Accuracy**: Must be precise - these determine user experience

**Stage 3 - Re-ranking**:
- **Goal**: Apply business constraints (diversity, freshness, fairness)
- **Method**: Rule-based and lightweight optimization
- **Latency budget**: ~10-20ms
- **Accuracy**: Business logic, not ML accuracy

*"Notice how each stage has a different job. Retrieval optimizes for recall ('don't miss good items'). Ranking optimizes for precision ('get the order right'). Re-ranking optimizes for business objectives ('make money while keeping users happy')."*

---

## Latency Budget Breakdown: A Numerical Example

*"Let's allocate our 100ms budget. This is like planning a construction project - every team needs their time, and delays cascade."*

### The Budget Allocation

| Component | Time Budget | What's Happening |
|-----------|-------------|------------------|
| Network overhead | 10ms | Request routing, load balancing |
| Feature lookup | 20ms | Fetch user features from cache/DB |
| Candidate generation | 10ms | ANN search across retrievers |
| Ranking model inference | 40ms | GPU inference on 1000 items |
| Re-ranking | 10ms | Diversity, business rules |
| Response serialization | 10ms | Format and send response |
| **Total** | **100ms** | |

### Why This Allocation?

*"Notice ranking gets the lion's share (40ms). Why?"*

**Answer**: Ranking directly determines what users see. A 1% improvement in ranking quality can mean millions in revenue. We invest latency where it matters most.

**Feature lookup gets 20ms** because:
- User features may require database queries
- Real-time features (session behavior) need computation
- Caching reduces this, but cache misses are expensive

*"What happens if candidate generation takes 15ms instead of 10ms?"*

**Cascade effect**: Everything downstream is compressed. Ranking might get only 35ms, reducing model complexity or batch size. This is why SLAs (Service Level Agreements) exist for each stage.

---

## Stage 1: Candidate Generation Deep Dive

### The Recall vs. Latency Tradeoff

*"Let's derive why candidate generation is fundamentally about trading recall for speed."*

**Definition - Recall@K**: Of all relevant items, what fraction did we retrieve in our K candidates?

$$\text{Recall@K} = \frac{|\text{Retrieved} \cap \text{Relevant}|}{|\text{Relevant}|}$$

**The tradeoff**:

| Retrieve More | Retrieve Fewer |
|---------------|----------------|
| Higher recall | Lower recall |
| More items for ranker | Fewer items for ranker |
| Ranker overloaded | Ranker underloaded |
| Latency blows up | Miss good items |

*"If recall is 80% at K=1000, we're missing 20% of relevant items before ranking even begins. Those items have zero chance of being recommended, no matter how good our ranker is."*

### Approximate Nearest Neighbor (ANN) Search

**Why ANN?** Two-tower models produce embeddings. Finding similar items means finding nearest neighbors in embedding space.

**Exact nearest neighbor** complexity:

$$O(n \cdot d)$$

where $n$ is the number of items (millions) and $d$ is embedding dimension (hundreds).

**For 500M items, d=256**:

$$500{,}000{,}000 \times 256 = 128 \text{ billion operations per query}$$

*"That's way too slow. Enter approximate methods."*

### ANN Algorithms Comparison

| Algorithm | Time Complexity | Space | Recall | Use Case |
|-----------|-----------------|-------|--------|----------|
| **HNSW** | $O(\log n)$ | $O(n \cdot M)$ | 95%+ | General purpose |
| **IVF** | $O(\sqrt{n})$ | $O(n)$ | 90%+ | Large scale |
| **LSH** | $O(1)$ | $O(n \cdot L)$ | 80%+ | Streaming |
| **PQ** | $O(n/k)$ | $O(n \cdot m)$ | 85%+ | Memory constrained |

**HNSW (Hierarchical Navigable Small World)**:
- Build a graph where similar items are connected
- Search by "hopping" between nodes
- Achieves $O(\log n)$ search time with 95%+ recall

*"With HNSW, our 500M item search goes from 128 billion operations to about $\log_2(500M) \approx 29$ graph hops. That's a 4-billion-fold speedup!"*

### Multiple Retrievers Strategy

*"Here's a key insight: different retrieval methods have different blind spots."*

**Collaborative filtering** misses:
- New items (cold start)
- Items from underrepresented categories

**Content-based** misses:
- Items the user would like but with different features
- Serendipitous discoveries

**Solution**: Run multiple retrievers in parallel, merge results.

```python
def retrieve_candidates(user_id, n_candidates=1000):
    """
    Multi-retriever candidate generation.

    Why multiple retrievers?
    - CF captures behavioral patterns
    - Content captures feature similarity
    - Trending ensures freshness
    - Diversity across retriever outputs
    """
    candidates = []

    # Collaborative filtering retrieval (behavioral patterns)
    # Uses user-item interaction history
    cf_items = collaborative_filter.top_k(user_id, k=500)
    candidates.extend(cf_items)

    # Content-based retrieval (feature similarity)
    # Uses item features user has engaged with
    user_profile = get_user_profile(user_id)
    content_items = content_recommender.top_k(user_profile, k=300)
    candidates.extend(content_items)

    # Trending items (freshness, popularity)
    # Ensures we don't miss viral content
    trending = get_trending_items(k=200)
    candidates.extend(trending)

    # Deduplicate and limit
    return list(set(candidates))[:n_candidates]
```

---

## Stage 2: Ranking Model Design

### Socratic Question: "Why not use the ranking model for all items?"

*"This is the question students always ask. Let's work through it."*

**The math of ranking inference**:

Suppose our ranking model:
- Takes 0.5ms per item (GPU, batched)
- Needs 50 features per item
- Uses a transformer with 10M parameters

**Scoring all 500M items**:

$$\text{Time} = 500{,}000{,}000 \times 0.5\text{ms} = 250{,}000 \text{ seconds} \approx 3 \text{ days}$$

**Scoring 1000 candidates**:

$$\text{Time} = 1{,}000 \times 0.5\text{ms} = 500\text{ms}$$

*"Still too slow! But here's where batching saves us."*

### Batch Size Math

**GPU inference is parallelized**. With proper batching:

| Batch Size | Per-Item Time | Total for 1000 items |
|------------|---------------|----------------------|
| 1 | 5ms | 5000ms |
| 10 | 0.8ms | 80ms |
| 100 | 0.15ms | 15ms |
| 1000 | 0.04ms | 40ms |

*"By batching all 1000 candidates into a single GPU call, we achieve near-constant time regardless of candidate count (up to memory limits)."*

**Why heavier models are affordable for ranking**:

1. **Fixed input size**: Always ~1000 items (not millions)
2. **Batching efficiency**: GPU utilization is high
3. **Feature caching**: User features computed once, reused for all items
4. **Model parallelism**: Split large models across GPUs

### Ranking Model Architecture

```python
def rank_candidates(user_id, candidates):
    """
    Deep ranking model inference.

    Key insight: We can afford complex features because
    we only score ~1000 items, and we batch everything.
    """
    # Fetch user features ONCE (amortized across all candidates)
    user_features = get_user_features(user_id)  # ~5ms
    context = get_context()  # time, device, location

    # Batch all candidates for efficient GPU inference
    batch_features = []
    for item in candidates:
        item_features = get_item_features(item)  # Cached
        combined = combine_features(user_features, item_features, context)
        batch_features.append(combined)

    # Single batched inference call
    # 1000 items in one GPU batch = ~40ms total
    scores = ranking_model.predict_batch(batch_features)

    # Sort by predicted engagement probability
    ranked = sorted(zip(candidates, scores),
                   key=lambda x: x[1], reverse=True)

    return [item for item, score in ranked[:100]]
```

---

## Stage 3: Re-ranking

### Beyond Pure Relevance

*"The ranking model optimizes for predicted engagement. But engagement isn't everything."*

**Business constraints**:
- **Diversity**: Users get bored seeing similar items
- **Freshness**: New content needs exposure to gather signals
- **Fairness**: Don't always favor the same creators
- **Monetization**: Promoted content needs slots

```python
def rerank(ranked_items, user_id, k=10):
    """
    Apply business logic and diversity constraints.

    This stage exists because ML models optimize for
    a single objective, but businesses have multiple goals.
    """
    result = []
    categories_seen = {}

    for item in ranked_items:
        # Diversity: limit items per category
        category = get_category(item)
        if categories_seen.get(category, 0) >= 2:
            continue

        result.append(item)
        categories_seen[category] = categories_seen.get(category, 0) + 1

        if len(result) >= int(k * 0.9):  # 90% from ranking
            break

    # Add exploration items (10% for learning)
    # These help us discover new user preferences
    exploration = sample_exploration_items(k - len(result))
    result.extend(exploration)

    return result[:k]
```

---

## Numerical Walkthrough: Tracing the Full Pipeline

*"Let's trace a single recommendation request through the entire system."*

### Scenario

- **User**: Alice, ID #12345
- **Catalog**: 1,000,000 items
- **Target**: Show 10 recommendations
- **Latency budget**: 100ms

### Step-by-Step Trace

**T=0ms: Request arrives**
```
GET /recommendations?user_id=12345
```

**T=10ms: Feature lookup complete**
- User embedding: [0.23, -0.45, 0.12, ...] (256 dims)
- Recent history: [item_892, item_1204, item_445]
- User segment: "tech enthusiast, 25-34, high engagement"

**T=20ms: Candidate generation complete**
- CF retriever: 500 candidates (based on similar users)
- Content retriever: 300 candidates (based on past views)
- Trending retriever: 200 candidates (popular this hour)
- After deduplication: **1,000 unique candidates**

*"Notice: We just reduced our search space from 1,000,000 to 1,000 - a 1000x reduction!"*

**T=60ms: Ranking complete**
- All 1000 candidates scored by deep model
- Top 100 selected for re-ranking

**Ranking scores sample**:
| Item | Score | Why High? |
|------|-------|-----------|
| item_7234 | 0.92 | Similar to recent views |
| item_892 | 0.89 | From favorite creator |
| item_1156 | 0.87 | Trending + relevant category |
| ... | ... | ... |
| item_9901 | 0.12 | Tangentially related |

**T=70ms: Re-ranking complete**
- Applied diversity: max 2 per category
- Added 1 exploration item (new creator)
- Final 10 items selected

**T=80ms: Response sent**
```json
{
  "recommendations": [
    {"id": "item_7234", "score": 0.92},
    {"id": "item_892", "score": 0.89},
    // ... 8 more items
  ],
  "latency_ms": 80,
  "stage_times": {
    "feature_lookup": 10,
    "retrieval": 10,
    "ranking": 40,
    "reranking": 10
  }
}
```

### The Funnel Visualization

```
1,000,000 items (full catalog)
     |
     | Candidate Generation (10ms)
     | Method: ANN search, multiple retrievers
     | Filter: Embedding similarity > threshold
     v
   1,000 candidates
     |
     | Ranking (40ms)
     | Method: Deep neural network
     | Filter: Keep top 100 by predicted engagement
     v
    100 ranked items
     |
     | Re-ranking (10ms)
     | Method: Business rules, diversity
     | Filter: Apply constraints, select final 10
     v
    10 recommendations shown to user
```

**Reduction at each stage**:
- Retrieval: 1000x reduction (1M to 1K)
- Ranking: 10x reduction (1K to 100)
- Re-ranking: 10x reduction (100 to 10)
- **Total**: 100,000x reduction

---

## What Can Go Wrong: Failure Modes

*"Now let's talk about what happens when things break. Because in production, things always break."*

### 1. Cascade Failures

**Scenario**: Retrieval service is slow (20ms instead of 10ms)

**Impact**:
```
Retrieval:  20ms (+10ms over budget)
Ranking:    40ms (no buffer left)
Re-ranking: 10ms
Feature:    20ms
Network:    10ms
------------------------
Total:      100ms (exactly at limit)
```

*"We're now at 100% of budget with no room for variance. The next slow request will timeout."*

**Defense strategies**:
- **Circuit breakers**: If retrieval is slow, use cached candidates
- **Graceful degradation**: Reduce candidate count under load
- **Timeouts per stage**: Hard limits that trigger fallbacks

### 2. Feature Freshness Issues

**Scenario**: User watched 5 videos, but features only update hourly.

**Problem**: Ranking model sees stale user embedding, recommends based on old preferences.

```
User state (reality):     Just watched 5 cooking videos
User state (features):    Last updated 45 min ago, no cooking signal
Recommendations:          Tech videos (old preference)
User experience:          "Why isn't YouTube learning?"
```

**Defense strategies**:
- **Real-time feature updates**: Stream user events to feature store
- **Feature freshness monitoring**: Alert if features are stale
- **Session-aware ranking**: Inject recent behavior as explicit features

### 3. Model-Serving Mismatch (Training-Serving Skew)

*"This is the silent killer of ML systems."*

**Scenario**: Training and serving compute features differently.

```python
# Training code (offline, batched)
def compute_feature_training(user_history):
    return np.mean(user_history[-100:])  # Last 100 items

# Serving code (online, real-time)
def compute_feature_serving(user_history):
    return np.mean(user_history[-50:])   # Bug: only last 50!
```

**Impact**: Model trained on one feature distribution, served on another. Performance degrades mysteriously.

**Defense strategies**:
- **Feature store**: Single source of truth for feature computation
- **Feature logging**: Log serving features, compare to training
- **Integration tests**: Verify feature parity before deployment

### 4. Traffic Spikes

**Scenario**: Breaking news causes 10x traffic surge.

**Impact**:
- Candidate generation: Queue backs up, latencies spike
- Ranking GPU: Memory exhaustion, OOM errors
- Feature store: Cache hit rate drops (new users)

**Defense strategies**:
- **Auto-scaling**: Add capacity based on request rate
- **Load shedding**: Reject low-priority requests under load
- **Cached responses**: Serve stale recommendations temporarily
- **Reduced quality mode**: Simpler models, fewer candidates

### 5. Cold Start Cascade

**Scenario**: New item goes viral, but has no features.

```
Item state:     New upload, no engagement data
CF retrieval:   Can't find similar items (no interactions)
Content:        Only has basic metadata
Ranking:        No historical CTR, defaults to prior
Result:         Systematically underranked despite being relevant
```

**Defense strategies**:
- **Content-based fallback**: Use item features when behavior is sparse
- **Exploration budget**: Reserve slots for new items
- **Creator reputation**: Transfer trust from creator's other content

---

## Architecture Patterns

### Lambda Architecture

*"The Lambda architecture is like having two chefs in the kitchen - one prepares meals in advance, the other handles last-minute requests."*

**Components**:

**1. Batch Layer** (the prep chef):
- Trains models on historical data (daily/weekly)
- Generates item embeddings offline
- High-quality, slow updates
- Handles the "heavy lifting"

**2. Speed Layer** (the line cook):
- Real-time updates from streaming data
- Approximate, fast updates
- Handles recent user interactions
- Bridges the gap until next batch run

**3. Serving Layer** (the waiter):
- Merges batch + speed layer outputs
- Serves final predictions to users
- Handles request routing

**Example - Netflix**:
- **Batch**: Retrain models nightly on previous day's data
- **Speed**: Update user embeddings as they watch (session-based)
- **Serving**: Combine for real-time recommendations

### Kappa Architecture

*"Kappa says: why have two code paths when one will do?"*

**Concept**: Stream-only architecture, no separate batch layer.

**Benefits**:
- Simpler (one code path to maintain)
- Fresher (continuous updates)
- Easier to reason about (single source of truth)

**Challenges**:
- Harder to debug (no batch "checkpoint")
- Requires robust streaming infrastructure (Kafka, Flink)
- Reprocessing historical data is complex

---

## Feature Store Architecture

### Why Feature Stores Exist

*"Let me tell you a horror story. Team A trains a model using feature X computed one way. Team B serves the model using feature X computed differently. The model fails mysteriously in production. Nobody knows why for weeks."*

**Feature stores solve this by**:
- **Consistency**: Same features in training and serving
- **Reusability**: Share features across models and teams
- **Monitoring**: Track feature drift and freshness

**Popular implementations**: Feast, Tecton, AWS SageMaker Feature Store

### Architecture Diagram

```
Data Sources (logs, DBs, streams)
              |
              v
    Feature Engineering Pipeline
    (Spark, Flink, Python)
              |
              v
        Feature Store
        /           \
       v             v
   Training       Serving
   (batch)      (real-time)

   - Same feature definitions
   - Same computation logic
   - Versioned and monitored
```

---

## Caching Strategies

### The Caching Hierarchy

*"Think of caching like a library. The books you use daily are on your desk (L1). Popular books are on nearby shelves (L2). Everything else is in the stacks (L3)."*

**Cache Layers**:

| Layer | Storage | Latency | Contents |
|-------|---------|---------|----------|
| L1 | In-memory (Redis) | <1ms | User embeddings, hot features |
| L2 | SSD | 1-5ms | Item embeddings, recent items |
| L3 | Database | 5-20ms | Full features, cold items |

### Cache Invalidation Strategies

*"There are only two hard things in Computer Science: cache invalidation and naming things. -- Phil Karlton"*

**Strategies**:

1. **TTL (Time to Live)**: Cache for fixed duration, then refresh
   - Simple, but may serve stale data

2. **Event-driven**: Invalidate when underlying data changes
   - Fresh, but complex to implement

3. **Lazy refresh**: Refresh on next request after staleness threshold
   - Balances freshness and simplicity

```python
import redis
import pickle

cache = redis.Redis()

def get_user_embedding(user_id):
    """
    Multi-tier caching for user embeddings.

    Cache hierarchy:
    1. Redis (L1): 1ms latency, 1 hour TTL
    2. Compute (fallback): 50ms latency
    """
    key = f"user_emb:{user_id}"

    # Try L1 cache first
    cached = cache.get(key)
    if cached:
        return pickle.loads(cached)

    # Cache miss: compute embedding
    embedding = compute_user_embedding(user_id)

    # Store in cache with 1-hour TTL
    cache.setex(key, 3600, pickle.dumps(embedding))

    return embedding
```

---

## Summary: Key Architectural Principles

*"Let's consolidate what we've learned today."*

### The Core Insights

1. **Multi-stage is mandatory**: You cannot score millions of items in real-time. The three-stage funnel (Retrieval -> Ranking -> Re-ranking) is not optional.

2. **Each stage has a different objective**:
   - Retrieval: Maximize recall (don't miss good items)
   - Ranking: Maximize precision (get the order right)
   - Re-ranking: Maximize business value (constraints, diversity)

3. **Latency budgets are contracts**: Every stage has a time budget. Violations cascade through the system.

4. **Feature consistency is critical**: Training-serving skew is a silent killer. Use feature stores.

5. **Plan for failure**: Cascade failures, stale features, traffic spikes. Build graceful degradation.

### The Numbers to Remember

| Metric | Typical Value |
|--------|---------------|
| Total latency budget | <100ms |
| Retrieval time | 10ms |
| Ranking time | 30-50ms |
| Re-ranking time | 10-20ms |
| Catalog size | Millions |
| Candidates from retrieval | ~1000 |
| Items from ranking | ~100 |
| Final recommendations | ~10-50 |

---

## Exercises for Understanding

### Exercise 1: Latency Analysis

*"YouTube's latency budget is 100ms. If retrieval takes 15ms, ranking takes 50ms, and feature lookup takes 25ms, what happens?"*

**Work through**:
- Total: 15 + 50 + 25 = 90ms
- Remaining for re-ranking and network: 10ms
- Is this feasible? What are the risks?

### Exercise 2: Retrieval Tradeoff

*"If you increase retrieval K from 1000 to 2000 candidates, what happens to recall? What happens to ranking latency?"*

### Exercise 3: Failure Scenario

*"Design a degradation strategy for when the ranking service is down. What do you serve users?"*

---

## References

1. **Covington, P., et al. (2016)**. "Deep Neural Networks for YouTube Recommendations". *RecSys*. [The foundational paper on multi-stage architecture]

2. **Grbovic, M., & Cheng, H. (2018)**. "Real-time Personalization using Embeddings for Search Ranking at Airbnb". *KDD*. [Practical embedding-based retrieval]

3. **Amatriain, X. (2013)**. "Building Industrial-Scale Real-World Recommender Systems". *RecSys Tutorial*. [System architecture patterns]

4. **Malkov, Y., & Yashunin, D. (2018)**. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs". *IEEE TPAMI*. [HNSW algorithm]

5. **Zhao, Z., et al. (2019)**. "Recommending What Video to Watch Next: A Multitask Ranking System". *RecSys*. [YouTube's multi-objective ranking]

---

## Next Steps

**Coming up in Week 14**: Scalability Challenges - How do we scale these systems to billions of users and items? Distributed training, model serving at scale, and A/B testing infrastructure.

*"Remember: a recommendation system that doesn't meet latency requirements is a recommendation system that doesn't get used. Architecture matters as much as algorithms."*
