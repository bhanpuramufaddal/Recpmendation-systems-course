# Week 13: Scalability Challenges

## Overview

**Scale**: Billion users, million items, trillion interactions.

**Challenges**:
1. Training: Distributed training on TB of data
2. Serving: Sub-100ms latency for millions QPS
3. Storage: Petabytes of embeddings, features

---

## Distributed Training

### Data Parallelism

**Split data** across machines, same model on each.

**Process**:
1. Each worker trains on data shard
2. Aggregate gradients
3. Update shared model

**Frameworks**: PyTorch DDP, TensorFlow distributed

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize
dist.init_process_group(backend='nccl')

model = MyModel()
model = DistributedDataParallel(model)

# Training loop (each GPU processes different batch)
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # Gradients averaged across GPUs
    optimizer.step()
```

---

### Model Parallelism

**Split model** across machines (for huge models).

**Use case**: Model doesn't fit on single GPU.

**Example**: Embedding table with 1B users × 128 dims = 512 GB.

---

### Parameter Server

**Architecture**:
- **Workers**: Compute gradients
- **Param servers**: Store and update parameters

**Benefits**: Asynchronous updates (faster than sync).

---

## Online Learning

### Incremental Updates

**Goal**: Update model as new data arrives (no full retrain).

**Methods**:
1. **SGD update**: New interaction → single gradient step
2. **Mini-batch**: Accumulate 1000 interactions → update
3. **Windowed**: Retrain on last 7 days of data

```python
class OnlineRecommender:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def update(self, user, item, feedback):
        """Update model with single interaction"""
        prediction = self.model(user, item)
        loss = (prediction - feedback) ** 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Continuous learning loop
for user, item, feedback in interaction_stream:
    recommender.update(user, item, feedback)
```

---

### Concept Drift

**Problem**: User preferences change over time.

**Solution**: Decay old data, emphasize recent.

**Exponential decay**:
$$w(t) = e^{-\lambda (t_{\text{now}} - t)}$$

---

## Approximate Nearest Neighbors (ANN)

### Problem

**Exact NN**: $O(N)$ per query (N = millions).

**Need**: <5ms retrieval.

**Solution**: ANN - trade accuracy for speed.

---

### HNSW (Hierarchical Navigable Small World)

**Idea**: Multi-layer graph, navigate from top to bottom.

**Complexity**: $O(\log N)$ per query.

**Accuracy**: >95% recall@10.

```python
import hnswlib

# Index 1M item embeddings
num_items = 1_000_000
dim = 128

index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=num_items, ef_construction=200, M=16)

item_embeddings = np.random.randn(num_items, dim)
index.add_items(item_embeddings)

# Query
user_emb = np.random.randn(dim)
labels, distances = index.knn_query(user_emb, k=100)

print(f"Retrieved {len(labels[0])} candidates in <5ms")
```

---

### FAISS (Facebook AI Similarity Search)

**Features**:
- GPU support (10x faster)
- Compression (PQ, OPQ)
- Billion-scale indices

**Example**:
```python
import faiss

# Create index
index = faiss.IndexFlatL2(dim)
index.add(item_embeddings)

# Search
D, I = index.search(user_emb.reshape(1, -1), k=100)
```

---

## Load Balancing

### Problem

**Hot users**: Popular users get many requests → overload servers.

**Solution**: Distribute load evenly.

---

### Consistent Hashing

**Assign** users/items to servers consistently.

**Benefit**: Minimal reshuffling when servers added/removed.

```python
import hashlib

def get_server(user_id, num_servers=100):
    hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
    return hash_val % num_servers

# Route request
server_id = get_server(user_id=12345)
```

---

## Compression

### Embedding Compression

**Problem**: 1B items × 128 dims × 4 bytes = 512 GB.

**Solution**: Quantization.

**Int8 quantization**: 4 bytes → 1 byte (4x compression).

```python
def quantize_embedding(emb, num_bits=8):
    """Quantize float32 to int8"""
    min_val, max_val = emb.min(), emb.max()
    scale = (max_val - min_val) / (2 ** num_bits - 1)

    quantized = ((emb - min_val) / scale).astype(np.uint8)

    return quantized, min_val, scale

def dequantize(quantized, min_val, scale):
    return quantized.astype(np.float32) * scale + min_val
```

---

## Summary

**Key Takeaways**:
1. **Distributed training**: Data parallelism (PyTorch DDP)
2. **Online learning**: Incremental updates, concept drift
3. **ANN**: HNSW, FAISS for fast retrieval
4. **Load balancing**: Consistent hashing
5. **Compression**: Int8 quantization (4x reduction)

**Scale Benchmarks**:
- Training: 1TB data in <1 day (100 GPUs)
- Serving: 1M QPS, <100ms p99 latency
- Storage: 1B embeddings in <100GB (compressed)

**Next**: Cold start problem.

---

## References

1. **Malkov, Y., & Yashunin, D. (2018)**. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs". *TPAMI*.
2. **Johnson, J., et al. (2019)**. "Billion-scale Similarity Search with GPUs". *IEEE Transactions on Big Data*.
3. **Dean, J., & Ghemawat, S. (2008)**. "MapReduce: Simplified Data Processing on Large Clusters". *CACM*.
