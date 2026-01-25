# Week 13: Production Systems and MLOps - Practice Problems

## Overview
Design production architectures, handle scalability, solve cold start, and implement ML Ops practices for recommendation systems.

---

## Problem 1: Batch vs. Streaming Architecture
**Difficulty:** Medium

**Batch (Lambda):** Precompute recommendations daily
**Streaming (Kappa):** Update in real-time as events arrive

**Scenarios:**
1. Movie recommendations (catalog changes slowly)
2. News recommendations (content changes hourly)
3. E-commerce (inventory changes constantly)

For each: Choose batch or streaming, justify, design architecture

**Learning Outcomes:** Choose architectures, understand trade-offs, design systems

---

## Problem 2: Feature Store Design
**Difficulty:** Hard

**Requirements:**
- **Offline:** Batch features for training (user demographics, item stats)
- **Online:** Real-time features for serving (current session, trending)
- **Consistency:** Same features in training and serving

**Design:**
1. Schema for user/item features
2. Offline compute pipeline (Spark)
3. Online serving (Redis/Cassandra)
4. Feature versioning
5. Monitoring for data drift

**Learning Outcomes:** Design feature stores, ensure train-serve consistency, monitor data quality

---

## Problem 3: Cold Start Solutions
**Difficulty:** Medium

**New user (no history):**
1. Popularity baseline
2. Demographic matching
3. Onboarding questionnaire
4. Explore-exploit (bandits)

**New item (no interactions):**
1. Content-based features
2. Similar items
3. Exploration boost

**Tasks:** Design cold start strategy, measure effectiveness, transition to personalized

**Learning Outcomes:** Handle cold start, design onboarding, measure success

---

## Problem 4: Model Serving Optimization
**Difficulty:** Hard

**Latency budget:** 100ms total
- Model inference: 50ms
- Feature fetching: 30ms
- Candidate generation: 10ms
- Re-ranking: 10ms

**Optimizations:**
1. Model quantization (reduce size/latency)
2. Batching requests
3. Caching user embeddings
4. Pre-filtering candidates

**Tasks:** Measure impact of each, design serving architecture, meet SLA

**Learning Outcomes:** Optimize inference, design low-latency systems, meet SLAs

---

## Problem 5: A/B Testing Infrastructure
**Difficulty:** Hard

**Requirements:**
- Run 100+ experiments simultaneously
- Random traffic splitting
- Metric computation
- Statistical analysis
- Guardrail monitoring

**Design:** Experimentation platform with proper randomization, metric tracking, analysis

**Learning Outcomes:** Design A/B testing platforms, ensure statistical rigor, scale experiments

---

## Programming Exercises

### Exercise 1: Feature Store Implementation

```python
# Offline feature computation (Spark/Pandas)
def compute_user_features(interactions):
    user_features = interactions.groupby('user_id').agg({
        'item_id': 'count',  # Total interactions
        'rating': 'mean',    # Average rating
        'timestamp': 'max'   # Last interaction time
    })
    return user_features

# Store in Redis
import redis
r = redis.Redis()

def store_features(user_id, features):
    r.hmset(f"user:{user_id}", features)

def get_features(user_id):
    return r.hgetall(f"user:{user_id}")
```

---

### Exercise 2: Model Serving with Caching

```python
from functools import lru_cache

class RecommendationService:
    def __init__(self, model, cache_size=10000):
        self.model = model
        self.item_embeddings = self._precompute_items()

    def _precompute_items(self):
        # Pre-compute and cache all item embeddings
        return {item_id: self.model.get_item_embedding(item_id) for item_id in all_items}

    @lru_cache(maxsize=10000)
    def get_user_embedding(self, user_id):
        # Cache user embeddings (TTL: 1 hour in production)
        return self.model.get_user_embedding(user_id)

    def recommend(self, user_id, k=10):
        user_emb = self.get_user_embedding(user_id)
        scores = {item: np.dot(user_emb, item_emb) for item, item_emb in self.item_embeddings.items()}
        return sorted(scores, key=scores.get, reverse=True)[:k]
```

---

### Exercise 3: Online Learning System

```python
class OnlineRecommender:
    def __init__(self, model):
        self.model = model
        self.buffer = []

    def recommend(self, user_id):
        return self.model.recommend(user_id)

    def record_interaction(self, user_id, item_id, reward):
        self.buffer.append((user_id, item_id, reward))

        # Update model incrementally
        if len(self.buffer) >= 1000:
            self.update_model()
            self.buffer = []

    def update_model(self):
        # Incremental update (gradient descent step)
        for user_id, item_id, reward in self.buffer:
            self.model.update(user_id, item_id, reward)
```

---

## Discussion Questions

1. **Build vs. Buy:** When to build custom vs. use SaaS (e.g., AWS Personalize)?
2. **Serving Costs:** Model inference is expensive. How to reduce costs?
3. **Data Pipeline:** How to ensure data quality in production?
4. **Model Versioning:** How to manage multiple model versions?
5. **Rollback:** Model causes drop in metrics. How to rollback quickly?
6. **Monitoring:** What alerts/dashboards do you set up?

---

## References
1. Amatriain, X., & Basilico, J. (2015). "Recommender systems in industry: A Netflix case study".
2. Sculley, D., et al. (2015). "Hidden technical debt in machine learning systems". NIPS.

---

*Return to [Week 13 Main Page](README.md)*
