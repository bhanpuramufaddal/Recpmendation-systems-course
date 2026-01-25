# Week 8: Two-Tower Models - Retrieval at Scale

## Overview

**The scalability challenge**: How do you recommend from **millions of items** to **billions of users** in **milliseconds**?

**The solution**: **Approximate Nearest Neighbor** (ANN) search.

This document covers the infrastructure that powers recommendation at companies like Google, Facebook, Spotify, and Netflix:
1. **FAISS** (Facebook AI Similarity Search)
2. **ScaNN** (Google's Scalable Nearest Neighbors)
3. **HNSW** (Hierarchical Navigable Small World graphs)

---

## Learning Objectives

By the end of this section, you will:
- Understand the retrieval bottleneck at scale
- Master ANN algorithms (FAISS, ScaNN, HNSW)
- Implement fast candidate retrieval
- Optimize latency vs. accuracy trade-offs
- Deploy production-ready retrieval systems

---

## The Retrieval Problem

### Two-Stage Architecture

Modern recommendation systems use **two stages**:

```
Stage 1: Candidate Retrieval
    Input: User
    Process: Find ~100-1000 relevant items from millions
    Latency: <50ms
    Method: ANN search

Stage 2: Ranking
    Input: Candidates
    Process: Rank candidates with complex model
    Latency: <50ms
    Method: Deep neural network

Total Latency: <100ms
```

**Why two stages?**
- Can't run expensive neural network on millions of items
- ANN retrieves good candidates fast
- Ranking refines with complex models

---

### The Scale Challenge

**Example: YouTube (2016)**
- **Users**: 2 billion
- **Videos**: 800 million
- **Latency requirement**: <100ms
- **Challenge**: Score 800M videos per user in <50ms → **Impossible with brute force!**

**Brute force complexity**:
- Compute scores: $O(|I| \cdot d)$ where $d$ is embedding dimension
- Example: $800M \times 128 = 102$ billion operations
- Even on GPU: **seconds** per user

**Solution**: Approximate Nearest Neighbor (ANN)
- Trade perfect accuracy for speed
- Find ~99% of true nearest neighbors
- **1000x faster** than brute force

---

## Embedding-Based Retrieval

### Two-Tower Architecture

**User Tower**:
$$\mathbf{u} = f_{\text{user}}(\text{user features})$$

**Item Tower**:
$$\mathbf{v} = f_{\text{item}}(\text{item features})$$

**Similarity**:
$$\text{score}(u, i) = \mathbf{u}^T \mathbf{v}$$

**Key property**: User and item embeddings are computed **independently**!

---

### Pre-Computation Strategy

**Offline** (once per day):
1. Compute item embeddings for all items
   $$\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_{|I|}$$
2. Build ANN index on item embeddings

**Online** (per request):
1. Compute user embedding $\mathbf{u}$
2. Query ANN index: find nearest items to $\mathbf{u}$
3. Return top-K items

**Latency**: $O(\log |I|)$ instead of $O(|I|)$!

---

## ANN Algorithms

### 1. FAISS (Facebook AI Similarity Search)

**Paper**: Johnson et al., "Billion-scale similarity search with GPUs" (IEEE, 2019)

**Developed by**: Facebook AI Research (Meta)

**Open source**: https://github.com/facebookresearch/faiss

---

#### FAISS Algorithms

**a) Flat (Exact Search)**
- Brute force: compute all distances
- **O(N)** time
- Perfect recall
- Use for: Small datasets (<10K items)

**b) IVF (Inverted File Index)**
- Cluster items into $K$ clusters (k-means)
- Search only nearby clusters
- **O(N/K + K)** time
- Recall ~95-99%

**c) IVFPQ (IVF + Product Quantization)**
- Compress vectors (128D → 8 bytes)
- Store compressed vectors
- **10-100x less memory**
- Recall ~90-95%

**d) HNSW (Hierarchical Navigable Small World)**
- Graph-based index
- **O(log N)** time
- Recall ~99%+
- Best accuracy-speed trade-off

---

#### FAISS Example

```python
import faiss
import numpy as np

# Toy dataset
n_items = 1000000
d = 128  # Embedding dimension

# Random item embeddings (in practice, from model)
item_embeddings = np.random.randn(n_items, d).astype('float32')

# Normalize (for cosine similarity)
faiss.normalize_L2(item_embeddings)

# Build index
index = faiss.IndexFlatIP(d)  # Flat index (exact search)
index.add(item_embeddings)

# Query
user_embedding = np.random.randn(1, d).astype('float32')
faiss.normalize_L2(user_embedding)

k = 10  # Top-10
distances, indices = index.search(user_embedding, k)

print(f"Top-{k} items: {indices[0]}")
print(f"Scores: {distances[0]}")
```

**Output**:
```
Top-10 items: [492815 832901 123456 ...]
Scores: [0.98 0.97 0.96 ...]
```

---

#### FAISS with IVF (Faster)

```python
import faiss

# Build IVF index
n_clusters = 1000  # Number of clusters
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, n_clusters)

# Train (cluster items)
index.train(item_embeddings)
index.add(item_embeddings)

# Search (probe 10 nearest clusters)
index.nprobe = 10
distances, indices = index.search(user_embedding, k)

print(f"Top-{k} items: {indices[0]}")
```

**Speed**: 10-100x faster than flat
**Recall**: ~95% (adjustable with `nprobe`)

---

#### FAISS with Product Quantization (Memory-Efficient)

```python
# IVFPQ: IVF + Product Quantization
n_clusters = 1000
m = 8  # Number of subquantizers
n_bits = 8  # Bits per subquantizer

index = faiss.IndexIVFPQ(quantizer, d, n_clusters, m, n_bits)
index.train(item_embeddings)
index.add(item_embeddings)

# Search
index.nprobe = 10
distances, indices = index.search(user_embedding, k)

print(f"Memory: {index.ntotal * m / 1e6:.2f} MB")  # Compressed!
```

**Memory**: 128D vectors → 8 bytes (16x compression!)

---

### 2. ScaNN (Google's Scalable Nearest Neighbors)

**Paper**: Guo et al., "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" (ICML, 2020)

**Developed by**: Google Research

**Open source**: https://github.com/google-research/google-research/tree/master/scann

---

#### Key Innovation: Anisotropic Quantization

**Standard quantization** (FAISS PQ):
- Split vector into equal chunks
- Quantize each chunk independently

**Anisotropic quantization** (ScaNN):
- Weighted quantization based on data distribution
- More bits for important dimensions
- **Higher accuracy** at same compression

---

#### ScaNN Example

```python
import scann

# Build ScaNN index
searcher = scann.scann_ops_pybind.builder(item_embeddings, k, "dot_product") \
    .tree(num_leaves=1000, num_leaves_to_search=10, training_sample_size=100000) \
    .score_ah(2, anisotropic_quantization_threshold=0.2) \
    .reorder(100) \
    .build()

# Search
neighbors, distances = searcher.search(user_embedding, final_num_neighbors=k)

print(f"Top-{k} items: {neighbors}")
print(f"Scores: {distances}")
```

**Performance**:
- **2-3x faster** than FAISS IVFPQ
- **Higher recall** at same speed

---

### 3. HNSW (Hierarchical Navigable Small World)

**Paper**: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (IEEE, 2018)

**Key idea**: Build hierarchical graph, navigate from coarse to fine.

**Library**: `hnswlib` (C++ with Python bindings)

---

#### HNSW Algorithm

**Structure**: Multi-layer graph
- Layer 0 (bottom): All items, dense connections
- Layer 1: Subset of items (1/2), sparser connections
- Layer 2: Smaller subset (1/4)
- ...
- Layer L (top): Few items, very sparse

**Search**:
1. Start at top layer
2. Greedy search to nearest neighbor in current layer
3. Move down to next layer
4. Repeat until layer 0
5. Return K nearest neighbors

**Complexity**: $O(\log N)$

---

#### HNSW Example

```python
import hnswlib

# Initialize index
index = hnswlib.Index(space='cosine', dim=d)

# Build index
index.init_index(max_elements=n_items, ef_construction=200, M=16)
index.add_items(item_embeddings, ids=np.arange(n_items))

# Search
index.set_ef(50)  # Exploration factor (higher = more accurate)
labels, distances = index.knn_query(user_embedding, k=k)

print(f"Top-{k} items: {labels[0]}")
print(f"Scores: {distances[0]}")
```

**Parameters**:
- `M`: Number of connections per node (typical: 16-64)
- `ef_construction`: Build-time exploration (typical: 100-200)
- `ef`: Query-time exploration (typical: 50-500)

**Trade-off**: Higher `ef` → better recall, slower search

---

## Comparison of ANN Libraries

| Library | Algorithm | Speed | Recall | Memory | GPU Support |
|---------|-----------|-------|--------|--------|-------------|
| **FAISS Flat** | Brute force | Baseline | 100% | High | Yes |
| **FAISS IVF** | Clustering | 10x | 95% | Medium | Yes |
| **FAISS IVFPQ** | IVF + Compression | 100x | 90% | Low | Yes |
| **FAISS HNSW** | Graph | 50x | 99% | Medium | No |
| **ScaNN** | Anisotropic Quant | 150x | 95% | Low | Yes (TPU) |
| **hnswlib** | Graph | 50x | 99% | Medium | No |

**Recommendation**:
- **FAISS HNSW**: Best accuracy-speed trade-off
- **ScaNN**: Best for very large scale (Google-level)
- **FAISS IVFPQ**: Best for memory-constrained environments
- **hnswlib**: Lightweight, easy to use

---

## Latency vs. Accuracy Trade-offs

### Metrics

**Recall@K**:
$$\text{Recall@K} = \frac{|\text{ANN top-K} \cap \text{True top-K}|}{K}$$

**Typical targets**:
- Recall@10 = 0.95 (find 9.5 out of 10 true nearest neighbors)
- Latency < 10ms

---

### Benchmark Results

**Dataset**: 1M items, 128D embeddings

| Method | Latency (ms) | Recall@10 | Memory (GB) |
|--------|--------------|-----------|-------------|
| Flat | 50 | 1.00 | 0.5 |
| IVF (nprobe=10) | 5 | 0.95 | 0.5 |
| IVFPQ | 2 | 0.90 | 0.03 |
| HNSW (ef=50) | 1 | 0.98 | 1.0 |
| ScaNN | 0.8 | 0.95 | 0.05 |

**Conclusion**: HNSW or ScaNN for production systems.

---

## Production Deployment

### Architecture

```
User Request
    ↓
User Embedding Service (GPU)
    ↓
ANN Index (HNSW/FAISS)
    ↓
Top-1000 Candidates
    ↓
Ranking Service (GPU)
    ↓
Top-10 Recommendations
    ↓
Response
```

---

### Handling Updates

**Challenge**: Items change (new videos uploaded, products added)

**Solutions**:

**1. Periodic Rebuild**
- Rebuild index every 1-24 hours
- Simple, consistent
- Lag: New items not searchable immediately

**2. Incremental Updates**
- Add new items to index dynamically
- HNSW supports incremental adds
- More complex, eventual consistency issues

**3. Hybrid**
- Main index (rebuilt daily)
- Delta index (recent items, rebuilt hourly)
- Merge results from both

**Typical**: Hybrid approach for large systems

---

### Sharding

**Problem**: Index too large for one machine (billions of items)

**Solution**: Shard by item category or hash

```
Shards:
    Movies → FAISS Index 1
    TV Shows → FAISS Index 2
    Documentaries → FAISS Index 3

Query:
    Search all relevant shards in parallel
    Merge results
```

**Latency**: Same (parallel queries)
**Throughput**: 3x (with 3 shards)

---

## Optimization Tips

### 1. Embedding Dimension

**Trade-off**: Higher dimension → better accuracy, slower search

**Benchmark** (1M items):

| Dimension | Latency (ms) | Recall@10 |
|-----------|--------------|-----------|
| 32 | 0.5 | 0.85 |
| 64 | 1.0 | 0.92 |
| 128 | 2.0 | 0.97 |
| 256 | 4.0 | 0.98 |

**Recommendation**: 64-128D for most applications

---

### 2. Quantization

**Reduce precision**: float32 → int8

```python
import faiss

# Scalar quantization
index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)
index.train(item_embeddings)
index.add(item_embeddings)

# 4x memory reduction (32 bits → 8 bits)
```

**Accuracy loss**: ~1-2% recall
**Speed gain**: 2-4x (fewer bytes to load)

---

### 3. GPU Acceleration

**FAISS on GPU**:

```python
import faiss

# Build index on CPU
index_cpu = faiss.IndexIVFFlat(quantizer, d, n_clusters)
index_cpu.train(item_embeddings)
index_cpu.add(item_embeddings)

# Move to GPU
res = faiss.StandardGpuResources()
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)

# Search on GPU (10-100x faster)
distances, indices = index_gpu.search(user_embedding, k)
```

**Speedup**: 10-100x for large queries

---

## Case Studies

### YouTube (2016)

**Scale**:
- 2B users
- 800M videos
- <100ms latency

**Architecture**:
- Two-tower: User encoder + Video encoder
- Embeddings: 256D
- ANN: Custom distributed index
- Retrieval: ~1000 candidates in <50ms
- Ranking: Deep neural network

**Result**: 80%+ of watch time from recommendations

---

### Spotify (2023)

**Scale**:
- 500M users
- 100M tracks
- <100ms latency

**Architecture**:
- Two-tower: User + Track encoders
- Embeddings: 128D
- ANN: FAISS HNSW
- Retrieval: 500 candidates
- Ranking: Multi-objective model

**Result**: 30%+ of streams from recommendations

---

### Pinterest (2020)

**Scale**:
- 450M users
- 200B pins
- <100ms latency

**Architecture**:
- PinSage (GNN) for embeddings
- Embeddings: 256D
- ANN: Custom graph-based index
- Retrieval: ~2000 candidates
- Multi-stage ranking

**Result**: 40%+ of engagement from recommendations

---

## Summary

**Key Takeaways**:
1. **ANN is essential** for billion-scale retrieval
2. **FAISS/ScaNN/HNSW** are production-ready libraries
3. **Trade-off**: Latency vs. Recall (aim for 95%+ recall in <10ms)
4. **Two-stage**: Retrieval (fast, approximate) + Ranking (slow, accurate)
5. **Embedding dimension matters**: 64-128D is sweet spot

**Algorithm Choice**:
- **FAISS HNSW**: Best default choice (99% recall, <5ms)
- **ScaNN**: Google-scale systems (billions of items)
- **FAISS IVFPQ**: Memory-constrained (10x compression)

**Production Checklist**:
- ✅ Choose ANN library (FAISS HNSW recommended)
- ✅ Tune parameters (recall target: 95%+)
- ✅ Handle index updates (periodic or incremental)
- ✅ Shard for scale (if >100M items)
- ✅ Monitor latency (p50, p95, p99)

---

## References

1. **Johnson, J., Douze, M., & Jégou, H. (2019)**. "Billion-scale similarity search with GPUs". *IEEE Transactions on Big Data*.
   - **FAISS paper**, Facebook's library

2. **Guo, R., et al. (2020)**. "Accelerating Large-Scale Inference with Anisotropic Vector Quantization". *ICML*.
   - **ScaNN paper**, Google's approach

3. **Malkov, Y. A., & Yashunin, D. A. (2018)**. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs". *IEEE TPAMI*.
   - **HNSW algorithm**

4. **Covington, P., Adams, J., & Sargin, E. (2016)**. "Deep Neural Networks for YouTube Recommendations". *RecSys*.
   - YouTube's two-tower architecture

5. **Ying, R., et al. (2018)**. "Graph Convolutional Neural Networks for Web-Scale Recommender Systems". *KDD*.
   - **PinSage** at Pinterest, billion-scale GNN
