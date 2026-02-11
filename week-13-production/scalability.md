# Week 13: Scalability Challenges

## The Moment of Truth: Why Your Laptop Model Fails at Production Scale

*"You've built a beautiful recommendation model. It works great on your laptop with 10,000 users and 1,000 items. Your manager says 'Ship it!' You deploy to production and... everything crashes. What happened?"*

Let me show you the brutal arithmetic of scale.

**The Production Reality**:
- Users: 100 million
- Items: 10 million
- Potential user-item pairs: 100M x 10M = **10^15 pairs** (one quadrillion)

**Memory Requirements**:
$$\text{Memory} = 10^{15} \text{ pairs} \times 4 \text{ bytes/score} = 4 \times 10^{15} \text{ bytes} = \textbf{4 Petabytes}$$

*That's 4,000 terabytes. Your laptop has maybe 16GB of RAM. You'd need 250,000 laptops just to store the scores.*

**Latency Budget**:
- Users expect recommendations in **100 milliseconds**
- Scoring all 10M items at 1 microsecond each = **10 seconds**
- You're 100x over budget before you even start

*This is why we need scalability. Not because it's fancy, but because the naive approach is mathematically impossible.*

---

## The Two-Stage Retrieval Architecture

*"If we can't score everything, we score... less. But how do we pick what to score?"*

### The Mathematical Insight

**Problem**: Score 10M items in 100ms budget.

**Solution**: Two-stage funnel.

**Stage 1 - Retrieval** (10ms budget):
- Use cheap, approximate method
- Retrieve top 1,000 candidates from 10M items
- **Reduction factor**: 10,000x

**Stage 2 - Ranking** (90ms budget):
- Use expensive, precise model
- Score and rank top 100 from 1,000 candidates
- **Reduction factor**: 10x

*"Why 1,000 then 100? Let's derive it."*

**Derivation of Optimal Retrieval Size**:

Let's say:
- $N$ = total items (10M)
- $k_1$ = retrieval candidates
- $k_2$ = final recommendations (10)
- $t_{retrieve}$ = time per item in retrieval (1 microsecond)
- $t_{rank}$ = time per item in ranking (100 microseconds)

Total latency:
$$T = N \cdot t_{retrieve}^{ANN} + k_1 \cdot t_{rank}$$

With ANN giving O(log N) complexity:
$$T \approx C \cdot \log(N) + k_1 \cdot t_{rank}$$

For N = 10M, log(N) ~ 23, with C = 10 microseconds:
$$T = 10 \times 23 + k_1 \times 100 = 230\mu s + k_1 \times 100\mu s$$

With 90ms budget for ranking:
$$k_1 = \frac{90,000\mu s}{100\mu s} = 900 \approx 1000$$

*That's where the "retrieve 1000, rank 100" comes from - it's not arbitrary, it's math.*

---

## Approximate Nearest Neighbors: The Heart of Scalable Retrieval

*"We need to find 1000 similar items from 10 million in 10 milliseconds. Brute force is O(N). We need O(log N). How?"*

### Numerical Walkthrough: The Speed Imperative

**Scenario**: 1M items, need top-10 recommendations

| Method | Comparisons | Time | Speedup |
|--------|-------------|------|---------|
| Brute Force | 1,000,000 | 100ms | 1x |
| IVF (100 clusters) | 10,000 | 1ms | 100x |
| HNSW | 1,000 | 0.1ms | 1000x |

*"HNSW is 1000x faster. How? Let me explain each algorithm."*

---

### Algorithm 1: Locality Sensitive Hashing (LSH)

*"Imagine you want to find people who look similar to you in a crowd of millions. What if you had special glasses that made similar people glow the same color?"*

**Core Insight**: Design hash functions that give similar items the same hash with high probability.

**The Math**:

For cosine similarity, we use random hyperplanes:
$$h_{\vec{r}}(\vec{x}) = \text{sign}(\vec{r} \cdot \vec{x})$$

where $\vec{r}$ is a random unit vector.

**Probability of Same Hash**:
$$P[h_{\vec{r}}(\vec{x}) = h_{\vec{r}}(\vec{y})] = 1 - \frac{\theta}{\pi}$$

where $\theta$ is the angle between $\vec{x}$ and $\vec{y}$.

*If two vectors are similar (small angle), they're likely to get the same hash.*

**Multiple Hash Tables**:
- Use $L$ different hash tables with $k$ hash functions each
- Query all tables, union results
- Probability of finding true neighbor: $1 - (1 - p^k)^L$

**Example**:
```
Item A embedding: [0.8, 0.6, 0.2]
Item B embedding: [0.7, 0.7, 0.1]  (similar to A)
Item C embedding: [-0.3, 0.9, 0.4] (different from A)

Random hyperplane 1: [1, 0, 0]
  A: sign(0.8) = +1
  B: sign(0.7) = +1  <- Same as A!
  C: sign(-0.3) = -1 <- Different from A

Random hyperplane 2: [0, 1, 0]
  A: sign(0.6) = +1
  B: sign(0.7) = +1  <- Same as A!
  C: sign(0.9) = +1

Hash(A) = [+1, +1], Hash(B) = [+1, +1], Hash(C) = [-1, +1]
A and B collide -> They're candidates for neighbors!
```

---

### Algorithm 2: HNSW (Hierarchical Navigable Small World)

*"Imagine navigating from New York to a specific coffee shop in San Francisco. You don't check every building in America. You fly to SF (long-range), then take a cab to the neighborhood (medium-range), then walk to the shop (short-range)."*

**Core Insight**: Build a multi-layer graph. Top layers have long-range connections (few nodes), bottom layers have short-range connections (all nodes).

**Why O(log N)?**

Each layer has exponentially fewer nodes:
$$\text{Layer } l \text{ has } N \cdot e^{-l} \text{ nodes}$$

Number of layers:
$$L = \log(N)$$

At each layer, we do constant work (traverse a few edges). Total work:
$$O(L \cdot c) = O(\log N)$$

**The Algorithm**:

```
SEARCH(query, top_k):
    entry_point = top_layer_entry

    For layer L down to 0:
        # Greedy search on this layer
        While can find closer neighbor:
            Move to closer neighbor

        If not at bottom layer:
            Drop to layer below (same node)

    Return top_k nearest from bottom layer
```

**Visual Intuition**:
```
Layer 2 (sparse):    A -------- B
                      \        /
                       \      /
Layer 1 (medium):   A -- C -- B -- D
                     |   |   |   |
Layer 0 (dense):    A-C-E-F-B-G-D-H-I-J...

Query: Find nearest to X
1. Start at A (layer 2)
2. A or B closer? B is closer, move to B
3. Drop to layer 1, still at B
4. B-C-D, C is closest, move to C
5. Drop to layer 0
6. Fine-grained search around C
```

**Code Example**:

```python
import hnswlib
import numpy as np
import time

# Index 1M item embeddings
num_items = 1_000_000
dim = 128

# Create index
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(
    max_elements=num_items,
    ef_construction=200,  # Quality during build
    M=16                  # Connections per node
)

# Add embeddings
item_embeddings = np.random.randn(num_items, dim).astype('float32')
index.add_items(item_embeddings)

# Query benchmark
user_emb = np.random.randn(1, dim).astype('float32')

# Set search quality
index.set_ef(100)  # Higher = better recall, slower

start = time.time()
labels, distances = index.knn_query(user_emb, k=100)
elapsed = (time.time() - start) * 1000

print(f"Retrieved {len(labels[0])} candidates in {elapsed:.2f}ms")
# Output: Retrieved 100 candidates in 0.15ms
```

---

### Algorithm 3: IVF (Inverted File Index)

*"Divide and conquer. Instead of searching all items, first find the relevant cluster, then search within it."*

**Core Insight**: Cluster items. At query time, find nearest cluster centroid, search only that cluster.

**The Math**:

With $\sqrt{N}$ clusters, each cluster has $\sqrt{N}$ items on average.

**Query Complexity**:
$$O(\sqrt{N}) \text{ for centroid search} + O(\sqrt{N}) \text{ for in-cluster search} = O(\sqrt{N})$$

For 1M items: $\sqrt{1M} = 1000$ comparisons instead of 1M.

**Multi-probe IVF**:
Search `nprobe` nearest clusters to improve recall:
$$\text{Complexity} = O(\sqrt{N}) + O(nprobe \cdot \sqrt{N})$$

```python
import faiss

# Training data for clustering
dim = 128
n_items = 1_000_000
n_clusters = 1024  # sqrt(1M) ~ 1000

# Create IVF index
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, n_clusters)

# Train on sample data
training_data = np.random.randn(100_000, dim).astype('float32')
index.train(training_data)

# Add all items
all_items = np.random.randn(n_items, dim).astype('float32')
index.add(all_items)

# Search
index.nprobe = 10  # Search 10 nearest clusters
query = np.random.randn(1, dim).astype('float32')
D, I = index.search(query, k=100)
```

---

### The Recall vs Latency Tradeoff

*"There's no free lunch. Faster search means lower recall."*

**Recall Definition**:
$$\text{Recall@k} = \frac{|\text{ANN top-k} \cap \text{True top-k}|}{k}$$

**Empirical Tradeoff** (typical values for 1M items, 128 dimensions):

| Method | Latency | Recall@10 | Memory |
|--------|---------|-----------|--------|
| Brute Force | 100ms | 100% | 512MB |
| HNSW (ef=10) | 0.05ms | 85% | 1GB |
| HNSW (ef=50) | 0.15ms | 95% | 1GB |
| HNSW (ef=200) | 0.5ms | 99% | 1GB |
| IVF (nprobe=1) | 0.1ms | 70% | 512MB |
| IVF (nprobe=10) | 0.8ms | 92% | 512MB |
| IVF (nprobe=100) | 5ms | 99% | 512MB |

*The tradeoff curve is not linear. You can get 95% recall for 500x speedup, but getting to 99% recall only gives you 200x speedup. That last 4% is expensive.*

---

## Sharding Strategies: When One Machine Isn't Enough

*"What if your embedding table is 512GB but your server has 64GB RAM? You shard."*

### Strategy 1: Shard by User

**How it works**:
- User ID 0-999,999 -> Shard 0
- User ID 1,000,000-1,999,999 -> Shard 1
- ...

**Routing**:
```python
def get_shard(user_id, num_shards=10):
    return user_id % num_shards
```

**Pros**: Each request hits exactly one shard.

**Cons**: Popular users create hot shards. Can't scale item catalog independently.

---

### Strategy 2: Shard by Item

**How it works**:
- Each shard holds a subset of items
- Query fans out to all shards, aggregates results

**Routing**:
```python
def query_all_shards(user_embedding, num_shards=10):
    results = []
    for shard_id in range(num_shards):
        shard_results = query_shard(shard_id, user_embedding)
        results.extend(shard_results)
    return top_k(results, k=100)
```

**Pros**: New items easy to add. Natural load balancing.

**Cons**: Every query hits every shard (fan-out overhead).

---

### Strategy 3: Shard by Consistent Hash

**How it works**:
- Hash items to a ring
- Each shard owns a range on the ring

**Benefit**: Adding/removing shards only moves ~1/N items.

```python
import hashlib

class ConsistentHash:
    def __init__(self, nodes, virtual_nodes=100):
        self.ring = {}
        self.sorted_keys = []

        for node in nodes:
            for i in range(virtual_nodes):
                key = self._hash(f"{node}:{i}")
                self.ring[key] = node
                self.sorted_keys.append(key)

        self.sorted_keys.sort()

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def get_node(self, item_id):
        if not self.ring:
            return None

        key = self._hash(str(item_id))

        # Find first node with key >= item's key
        for node_key in self.sorted_keys:
            if node_key >= key:
                return self.ring[node_key]

        return self.ring[self.sorted_keys[0]]

# Usage
shards = ConsistentHash(["shard_0", "shard_1", "shard_2"])
print(shards.get_node(item_id=12345))  # -> "shard_1"
```

---

## Distributed Training: Making GPUs Work Together

*"Training on 1TB of data with a single GPU would take a year. With 100 GPUs, we do it in a day. But getting 100 GPUs to cooperate is hard."*

### Data Parallelism

*"Everyone has the same model. Everyone sees different data. Everyone shares what they learned."*

**The Process**:

1. **Replicate**: Copy model to all N GPUs
2. **Partition**: Split batch across GPUs (each gets batch_size/N)
3. **Forward**: Each GPU computes loss on its data
4. **Backward**: Each GPU computes gradients
5. **Aggregate**: Average gradients across all GPUs
6. **Update**: Apply same update to all models

**The Math**:

If GPU $i$ computes gradient $g_i$, the aggregated gradient is:
$$g = \frac{1}{N} \sum_{i=1}^{N} g_i$$

This is mathematically equivalent to a larger batch on a single GPU:
$$g = \frac{1}{N \cdot B} \sum_{i=1}^{N} \sum_{j=1}^{B} \nabla L(x_{i,j})$$

**Implementation**:

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Create model and wrap with DDP
model = RecommendationModel().cuda()
model = DDP(model, device_ids=[local_rank])

# Training loop - each GPU sees different data
for batch in train_loader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()  # Gradients automatically averaged!
    optimizer.step()
```

---

### Model Parallelism

*"When your embedding table is 512GB and your GPU has 40GB, you can't fit it. Split the model itself across GPUs."*

**Example**: 1B users x 128 dims x 4 bytes = 512GB embedding table

**Sharded Embeddings**:
- GPU 0: Users 0 - 249M
- GPU 1: Users 250M - 499M
- GPU 2: Users 500M - 749M
- GPU 3: Users 750M - 999M

```python
class ShardedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_shards):
        super().__init__()
        self.num_shards = num_shards
        self.shard_size = num_embeddings // num_shards

        # Each GPU holds one shard
        self.local_embedding = nn.Embedding(
            self.shard_size, embedding_dim
        )
        self.rank = dist.get_rank()

    def forward(self, indices):
        # Determine which indices belong to this shard
        local_mask = (indices // self.shard_size) == self.rank
        local_indices = indices[local_mask] % self.shard_size

        # Look up local embeddings
        local_embeds = self.local_embedding(local_indices)

        # All-gather to collect embeddings from all shards
        all_embeds = [torch.zeros_like(local_embeds) for _ in range(self.num_shards)]
        dist.all_gather(all_embeds, local_embeds)

        return torch.cat(all_embeds, dim=0)
```

---

### Gradient Aggregation: AllReduce vs Parameter Server

*"After each GPU computes its gradient, how do they share information?"*

**AllReduce** (Decentralized):
- Every GPU sends to every other GPU
- Symmetric, no bottleneck
- Best for homogeneous setups

```
GPU 0 <--> GPU 1
  ^          ^
  |          |
  v          v
GPU 2 <--> GPU 3

Each GPU ends up with the average of all gradients
```

**Parameter Server** (Centralized):
- Workers send gradients to server
- Server aggregates and sends back
- Good for heterogeneous/async setups

```
       Parameter Server
       /    |    |    \
    GPU0  GPU1  GPU2  GPU3

Workers push gradients, pull updated params
```

**Comparison**:

| Aspect | AllReduce | Parameter Server |
|--------|-----------|------------------|
| Synchronization | Synchronous | Can be async |
| Bottleneck | Network bandwidth | Server capacity |
| Best for | Dense gradients | Sparse gradients |
| Frameworks | PyTorch DDP | TensorFlow PS |

---

## Feature Store Design: The Bridge Between Training and Serving

*"Features computed during training must be available during serving. Sounds simple? It's the source of countless production bugs."*

### Why Features Need to be Precomputed

**Problem**: At serving time, you have 100ms to:
1. Look up user features
2. Look up item features
3. Compute score
4. Return result

**Real-time feature computation is expensive**:
- User's average rating: Aggregate over 10,000 ratings -> 50ms
- Item's click-through rate: Aggregate over 1M impressions -> 200ms
- User-item feature cross: Compute similarity -> 10ms

*That's already 260ms. We're 2.6x over budget, and we haven't even scored anything.*

**Solution**: Precompute features, store in a fast key-value store.

---

### Online vs Offline Features

**Offline Features** (batch computed):
- User's lifetime stats (average rating, genre preferences)
- Item's aggregate stats (popularity, average rating)
- Updated hourly/daily
- Stored in feature store (Redis, DynamoDB)

**Online Features** (real-time):
- User's last 5 actions (session context)
- Time since last visit
- Current device/location
- Computed at request time

**Feature Store Architecture**:

```
+----------------+     +-----------------+     +----------------+
|  Batch Jobs    | --> | Feature Store   | --> | Serving Layer  |
| (Spark, Flink) |     | (Redis, Feast)  |     | (Model Server) |
+----------------+     +-----------------+     +----------------+
                              ^
                              |
                       +----------------+
                       | Streaming Jobs |
                       | (Kafka, Kinesis)|
                       +----------------+
```

```python
class FeatureStore:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.feature_ttl = 3600 * 24  # 24 hours

    def set_user_features(self, user_id, features):
        """Called by batch job"""
        key = f"user:{user_id}:features"
        self.redis.hset(key, mapping=features)
        self.redis.expire(key, self.feature_ttl)

    def get_user_features(self, user_id):
        """Called at serving time"""
        key = f"user:{user_id}:features"
        features = self.redis.hgetall(key)
        return features or self.get_default_features()

    def get_default_features(self):
        """Fallback for missing users"""
        return {
            "avg_rating": 3.5,
            "num_ratings": 0,
            "days_since_signup": 0
        }
```

---

### Training-Serving Skew: The Silent Killer

*"Your model trained on feature X computed one way, but at serving time, X is computed differently. Your model silently degrades."*

**Common Sources of Skew**:

1. **Aggregation windows**: Training uses 30-day average, serving uses 7-day average
2. **Missing value handling**: Training fills with mean, serving fills with 0
3. **Feature version**: Training used feature V1, serving has feature V2
4. **Timestamp bugs**: Training uses event time, serving uses processing time

**Prevention**:

```python
class FeatureDefinition:
    """Single source of truth for feature computation"""

    def __init__(self, name, aggregation_window_days=30,
                 missing_value_strategy="mean"):
        self.name = name
        self.window = aggregation_window_days
        self.missing_strategy = missing_value_strategy
        self.version = "v1"

    def compute(self, raw_data):
        """Used by BOTH training and serving pipelines"""
        # Same logic everywhere
        if raw_data is None:
            return self._handle_missing()

        filtered = self._filter_window(raw_data)
        return self._aggregate(filtered)

    def _handle_missing(self):
        if self.missing_strategy == "mean":
            return 3.5  # Global mean
        return 0

# Same definition used in training and serving
user_avg_rating = FeatureDefinition(
    name="user_avg_rating",
    aggregation_window_days=30,
    missing_value_strategy="mean"
)
```

---

## What Can Go Wrong: Production Failure Modes

*"I've seen every one of these in production. Learn from my pain."*

### Failure Mode 1: ANN Gives Wrong Results (Recall < 100%)

**The Problem**: Your ANN index returns top-100 candidates, but the true best item wasn't in that set.

**Real Example**:
- User loves obscure 1970s Italian horror films
- True best item: "Suspiria (1977)" with similarity 0.95
- ANN returns: Popular horror films with similarity 0.70-0.85
- Suspiria is in a sparse region of embedding space, HNSW navigated away from it

**Detection**:
```python
def measure_recall(index, ground_truth_index, queries, k=100):
    """Compare ANN results to brute force"""
    recalls = []

    for query in queries:
        # ANN result
        ann_ids, _ = index.knn_query(query, k=k)
        ann_set = set(ann_ids[0])

        # Ground truth (brute force)
        true_ids, _ = ground_truth_index.knn_query(query, k=k)
        true_set = set(true_ids[0])

        recall = len(ann_set & true_set) / k
        recalls.append(recall)

    return np.mean(recalls)

# Monitor in production
recall = measure_recall(prod_index, exact_index, sample_queries)
if recall < 0.95:
    alert("ANN recall degraded!")
```

**Mitigation**:
- Increase `ef` parameter (search beam width)
- Use hybrid: ANN for most users, exact for high-value users
- Periodic recall audits with random samples

---

### Failure Mode 2: Embedding Staleness

**The Problem**: Index built on Tuesday's embeddings. Model updated on Wednesday. Index still uses old embeddings.

**Timeline of Disaster**:
```
Monday:    Train model V1, build index V1    [In sync]
Tuesday:   Serving with model V1, index V1   [In sync]
Wednesday: Train model V2, still index V1    [OUT OF SYNC]
Thursday:  User searches, gets V1 neighbors  [Wrong results!]
           for V2 query embeddings
```

**Why It's Bad**:
- User embedding (from V2): [0.8, 0.6]
- Item embeddings in index (from V1): Completely different meaning for same dimensions
- Similar items in V1 space are random in V2 space

**Detection**:
```python
def check_embedding_freshness(index_metadata, current_model_version):
    index_version = index_metadata["model_version"]
    index_build_time = index_metadata["build_timestamp"]

    if index_version != current_model_version:
        raise EmbeddingMismatchError(
            f"Index uses model {index_version}, "
            f"but serving model is {current_model_version}"
        )

    hours_stale = (datetime.now() - index_build_time).hours
    if hours_stale > 24:
        alert(f"Index is {hours_stale} hours old")
```

**Mitigation**:
- Atomic deployment: Model + Index updated together
- Version tagging: Index knows which model version it was built from
- Continuous index rebuild pipeline

---

### Failure Mode 3: Hot Shards (Popularity Imbalance)

**The Problem**: Popular items concentrate on one shard, overloading it.

**Scenario**:
- 10 shards, items distributed by item_id % 10
- Taylor Swift's new album: item_id = 12345, goes to shard 5
- 50% of traffic wants Taylor Swift
- Shard 5 gets 50% of queries, other shards get 5.5% each

**Traffic Distribution**:
```
Shard 0: ########## (10%)
Shard 1: ########## (10%)
Shard 2: ########## (10%)
Shard 3: ########## (10%)
Shard 4: ########## (10%)
Shard 5: ################################################## (50%)  <- OVERLOADED
Shard 6: ##### (5%)
Shard 7: ##### (5%)
Shard 8: ##### (5%)
Shard 9: ##### (5%)
```

**Detection**:
```python
def monitor_shard_balance(shard_metrics):
    qps_values = [m["qps"] for m in shard_metrics]
    mean_qps = np.mean(qps_values)
    max_qps = np.max(qps_values)

    imbalance_ratio = max_qps / mean_qps

    if imbalance_ratio > 2.0:
        hottest_shard = np.argmax(qps_values)
        alert(f"Shard {hottest_shard} is hot: {imbalance_ratio:.1f}x average")
```

**Mitigation**:
- Replicate hot items across multiple shards
- Popularity-aware sharding (spread popular items)
- Caching layer for hot items

```python
class PopularityAwareSharding:
    def __init__(self, num_shards, hot_item_threshold=1000):
        self.num_shards = num_shards
        self.hot_items = set()  # Replicated everywhere
        self.threshold = hot_item_threshold

    def get_shards(self, item_id, item_request_count):
        if item_request_count > self.threshold:
            self.hot_items.add(item_id)
            return list(range(self.num_shards))  # All shards
        else:
            return [item_id % self.num_shards]  # Single shard
```

---

### Failure Mode 4: Feature Drift (Training vs Serving Mismatch)

**The Problem**: Features in production don't match training data distribution.

**Scenario**:
- Training data: user_age mean=35, std=10
- Production (after mobile launch): user_age mean=22, std=8
- Model trained on 35-year-olds, applied to 22-year-olds

**Impact**:
- Features are out of distribution
- Model's learned patterns don't apply
- Predictions are unreliable

**Detection**:
```python
class FeatureDriftMonitor:
    def __init__(self, training_stats):
        self.training_stats = training_stats  # {feature: {mean, std}}

    def check_drift(self, serving_features, threshold=2.0):
        """Alert if serving features are > threshold std from training"""
        alerts = []

        for feature, value in serving_features.items():
            if feature not in self.training_stats:
                alerts.append(f"Unknown feature: {feature}")
                continue

            train_mean = self.training_stats[feature]["mean"]
            train_std = self.training_stats[feature]["std"]

            z_score = abs(value - train_mean) / train_std

            if z_score > threshold:
                alerts.append(
                    f"{feature}: z-score={z_score:.1f} "
                    f"(value={value}, expected={train_mean}+/-{train_std})"
                )

        return alerts

# Usage
monitor = FeatureDriftMonitor({
    "user_age": {"mean": 35, "std": 10},
    "days_active": {"mean": 180, "std": 90}
})

# In serving pipeline
alerts = monitor.check_drift({"user_age": 18, "days_active": 5})
# -> ["user_age: z-score=1.7 (value=18, expected=35+/-10)"]
```

**Mitigation**:
- Feature normalization relative to training distribution
- Monitoring dashboards for feature distributions
- Regular model retraining on recent data
- Fallback to simpler model for out-of-distribution users

---

### Failure Mode 5: Cascading Failures (Dependency Hell)

**The Problem**: One service fails, causing others to fail, causing everything to fail.

**Chain Reaction**:
```
1. Feature store Redis runs out of memory
2. Feature lookups timeout
3. Model server waits for features, request queue grows
4. Model server runs out of memory
5. Load balancer sees unhealthy servers, routes all traffic to remaining servers
6. Remaining servers overload
7. Full system outage
```

**Mitigation - Circuit Breakers**:
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self.failures = 0
        self.threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = "CLOSED"  # CLOSED = normal, OPEN = failing
        self.last_failure_time = None

    def call(self, func, fallback):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF-OPEN"
            else:
                return fallback()

        try:
            result = func()
            if self.state == "HALF-OPEN":
                self.state = "CLOSED"
                self.failures = 0
            return result
        except Exception:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.threshold:
                self.state = "OPEN"

            return fallback()

# Usage
feature_store_breaker = CircuitBreaker()

def get_features(user_id):
    return feature_store_breaker.call(
        func=lambda: redis.get(f"user:{user_id}"),
        fallback=lambda: default_features()
    )
```

---

## Online Learning: Adapting in Real-Time

*"The world changes. Yesterday's hit movie is today's old news. Your model must keep up."*

### Incremental Updates

**Goal**: Update model as new data arrives (no full retrain).

**Methods**:
1. **SGD update**: New interaction -> single gradient step
2. **Mini-batch**: Accumulate 1000 interactions -> update
3. **Windowed**: Retrain on last 7 days of data

```python
class OnlineRecommender:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate
        )
        self.update_count = 0

    def update(self, user, item, feedback):
        """Update model with single interaction"""
        self.model.train()

        prediction = self.model(user, item)
        loss = (prediction - feedback) ** 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        if self.update_count % 1000 == 0:
            print(f"Applied {self.update_count} updates")

# Continuous learning loop
async def learning_loop(recommender, interaction_stream):
    async for user, item, feedback in interaction_stream:
        recommender.update(user, item, feedback)
```

---

### Concept Drift

**Problem**: User preferences change over time.

**Example**:
- March 2020: Users want home workout videos
- March 2021: Users want travel videos
- Model trained on 2020 data fails in 2021

**Solution**: Decay old data, emphasize recent.

**Exponential decay weighting**:
$$w(t) = e^{-\lambda (t_{\text{now}} - t)}$$

**Interpretation**:
- $\lambda$ = decay rate (higher = faster forgetting)
- $\lambda = 0.1$ with $t$ in days: 10-day-old data has weight $e^{-1} \approx 0.37$
- 30-day-old data has weight $e^{-3} \approx 0.05$

```python
def weighted_loss(predictions, targets, timestamps, lambda_decay=0.1):
    """Apply time decay to loss"""
    now = datetime.now()
    ages = [(now - t).days for t in timestamps]
    weights = np.exp(-lambda_decay * np.array(ages))

    losses = (predictions - targets) ** 2
    weighted_losses = losses * torch.tensor(weights)

    return weighted_losses.mean()
```

---

## Compression: Making Billion-Scale Fit

*"1 billion items x 128 dimensions x 4 bytes = 512 GB. That's too much. Let's compress."*

### Int8 Quantization

**Idea**: Map float32 (4 bytes) to int8 (1 byte). 4x compression.

**The Math**:

Given embedding values in range $[min, max]$:
$$\text{scale} = \frac{max - min}{255}$$
$$\text{quantized} = \text{round}\left(\frac{value - min}{\text{scale}}\right)$$

**Dequantization**:
$$\text{value} = \text{quantized} \times \text{scale} + min$$

**Implementation**:

```python
def quantize_embedding(emb, num_bits=8):
    """Quantize float32 to int8"""
    min_val, max_val = emb.min(), emb.max()
    scale = (max_val - min_val) / (2 ** num_bits - 1)

    quantized = ((emb - min_val) / scale).round().astype(np.uint8)

    return quantized, min_val, scale

def dequantize(quantized, min_val, scale):
    return quantized.astype(np.float32) * scale + min_val

# Accuracy impact
original = np.random.randn(128).astype(np.float32)
quantized, min_val, scale = quantize_embedding(original)
reconstructed = dequantize(quantized, min_val, scale)

error = np.mean((original - reconstructed) ** 2)
print(f"MSE: {error:.6f}")  # Typically < 0.001
```

**Memory Savings**:
- Original: 1B items x 128 dims x 4 bytes = 512 GB
- Quantized: 1B items x 128 dims x 1 byte = 128 GB
- Overhead: 1B items x 2 floats (min, scale) x 4 bytes = 8 GB
- **Total: 136 GB (3.8x reduction)**

---

## Summary

**The Scalability Journey**:

1. **The Wall**: Production scale (100M users x 10M items) is mathematically impossible to brute-force
2. **Two-Stage Retrieval**: Cheap retrieval (1000 candidates) + expensive ranking (100 final)
3. **ANN Algorithms**: LSH (hash-based), HNSW (graph-based), IVF (cluster-based)
4. **Distributed Training**: Data parallelism (split data) + Model parallelism (split model)
5. **Feature Stores**: Precompute offline features, compute online features in real-time
6. **Failure Modes**: ANN recall, embedding staleness, hot shards, feature drift

**Scale Benchmarks**:
- Training: 1TB data in <1 day (100 GPUs)
- Serving: 1M QPS, <100ms p99 latency
- Storage: 1B embeddings in <100GB (compressed)

*"Scalability isn't about making your code run faster. It's about designing systems that work when your scale increases 1000x."*

**Next**: Cold start problem.

---

## References

1. **Malkov, Y., & Yashunin, D. (2018)**. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs". *TPAMI*.
2. **Johnson, J., et al. (2019)**. "Billion-scale Similarity Search with GPUs". *IEEE Transactions on Big Data*.
3. **Dean, J., & Ghemawat, S. (2008)**. "MapReduce: Simplified Data Processing on Large Clusters". *CACM*.
4. **Jouppi, N., et al. (2017)**. "In-Datacenter Performance Analysis of a Tensor Processing Unit". *ISCA*.
5. **Naumov, M., et al. (2019)**. "Deep Learning Recommendation Model for Personalization and Recommendation Systems". *arXiv*.
