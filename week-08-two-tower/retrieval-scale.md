# Week 8: Two-Tower Models - Retrieval at Scale

## The Opening Problem

*Professor walks in, writes on board:*

**"You need to search 100 million items in 10 milliseconds. Brute force takes 10 seconds. Now what?"**

*Pauses, looks at class*

Let me make this concrete. You're building recommendations for a streaming service. You have:
- 100 million items (songs, videos, products)
- Each item represented as a 256-dimensional embedding
- A user walks in, and you need to find their top 10 recommendations
- You have 10 milliseconds. Go.

**The brutal math**:
- Brute force: compute dot product with all 100M items
- Each dot product: 256 multiplications + 255 additions = 511 operations
- Total: 100,000,000 * 511 = 51.1 billion operations
- Modern CPU: ~10 billion operations/second
- Time: **~5 seconds**

Your latency budget? **10 milliseconds**. You're off by a factor of **500x**.

*So what do we do? We approximate.*

---

## Why Approximate is Good Enough: The Core Insight

Before we dive into algorithms, let me ask you a question that will save your career:

> **"Why is 95% recall often good enough in recommendations?"**

*Waits for answers*

Think about what we're actually doing. We're retrieving candidates for a **ranking model**. The ranking model will:
1. Re-score all candidates with a complex neural network
2. Consider context the retrieval couldn't see
3. Apply business rules
4. Make the final decision

If our retrieval finds 950 of the "true" top 1000 candidates, and misses 50... what happens to those 50? They were ranked #950-#1000 by embedding similarity. Were they really going to beat the top 10 after full ranking? **Almost never.**

**The approximation error bound** (informal):

Let $R_{true}$ be the true top-K set, and $R_{ann}$ be the ANN top-K set.

If $\text{Recall@K} = 0.95$, then:
- We find $0.95K$ of the true top-K
- We miss $0.05K$ items, but these are the *worst* of the top-K
- The "intruders" (false positives) are items ranked K+1 to K+something
- By embedding similarity, these are *close* to the boundary

**The downstream impact**: If your ranking model is even slightly better than random at distinguishing true top-K from intruders, your final recommendations barely change.

*This is why companies ship 95% recall systems and sleep soundly.*

---

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
- **Challenge**: Score 800M videos per user in <50ms. **Impossible with brute force!**

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

## ANN Intuition: Why Approximate Works

*Let's build intuition before algorithms.*

### The Geometry of High-Dimensional Space

In high dimensions, something strange happens. Consider 256-dimensional embeddings:

**Observation 1**: Most items are "far" from any given query
- In high-D space, volume concentrates in a thin shell
- Most items have similar distance to query (curse of dimensionality)
- Only a tiny fraction are "close"

**Observation 2**: The "close" items cluster together
- Similar items have similar embeddings (by design!)
- If item A is close to query Q, item B (similar to A) is likely close to Q too

**Observation 3**: We only need the TOP items, not ALL close items
- We want top-K, not all items within distance r
- Missing the 1001st closest item when retrieving top-1000? **Who cares.**

### The Error Bound Derivation

Let's be precise about what "approximate" means.

**Definition**: An ANN algorithm has $(c, r)$-approximate guarantee if:
- For any query $q$ with true nearest neighbor at distance $r$
- The algorithm returns a point at distance at most $c \cdot r$

**For recommendations**, we care about **recall**, not distance guarantees:

$$\text{Recall@K} = \frac{|\text{ANN-TopK} \cap \text{True-TopK}|}{K}$$

**Empirical finding** (across many datasets):
- Recall@K = 0.95-0.99 is achievable with 10-100x speedup
- The "missed" items are boundary cases (ranked ~K in true ordering)
- These rarely affect downstream ranking quality

**Why this works mathematically**:

Let $s_i = \mathbf{q}^T \mathbf{v}_i$ be the true score of item $i$.

The items we "miss" have scores:
$$s_{\text{missed}} \in [s_K - \epsilon, s_K]$$

where $\epsilon$ is small (the items are near the decision boundary).

The items we "incorrectly include" have scores:
$$s_{\text{intruder}} \in [s_K - \delta, s_K]$$

where $\delta$ is also small.

**Key insight**: Both missed items and intruders are **near the boundary**. The ranking model will sort them correctly because it has more features than just embedding similarity.

---

## HNSW: Building the Hierarchical Graph (Step by Step)

*Now let's derive HNSW from first principles. This is the algorithm that powers most production systems.*

### Starting Point: Why Graphs?

**Observation**: In a nearest neighbor graph, similar items are connected.

If we're at item A, and A is connected to its 10 nearest neighbors, one of those neighbors is likely close to our query (if A is close to the query).

**Greedy search on a graph**:
1. Start at a random node
2. Move to the neighbor closest to query
3. Repeat until no neighbor is closer
4. Return current node

**Problem**: Gets stuck in local minima!

### The NSW (Navigable Small World) Insight

**Solution**: Add "long-range" connections!

In a regular k-NN graph:
- All edges are short (connect nearby items)
- Greedy search explores slowly

In a Small World graph:
- Most edges are short (local structure)
- Some edges are long (shortcuts)
- Greedy search can "jump" across the space

**Construction**:
1. Insert items one by one
2. For each new item, connect to M nearest neighbors in current graph
3. Key: Early items have few choices, so they get "long" connections
4. Late items have many choices, so they get "local" connections

**Result**: A graph where you can reach any node in $O(\log N)$ hops.

### The Hierarchical Extension

*But NSW still has a problem: how do we find a good starting point?*

**HNSW Solution**: Build multiple layers!

**Layer Structure**:
- Layer 0: All N items, dense connections
- Layer 1: N/2 items (random subset), sparser connections
- Layer 2: N/4 items
- ...
- Layer L: ~log(N) items, very sparse

**Insertion Algorithm** (for new item $v$):

```
1. Randomly choose max_layer for v (geometric distribution)
   - P(layer >= l) = (1/M)^l
   - Expected max_layer = log_M(N)

2. Start at top layer, entry point

3. For each layer from top to max_layer:
   - Greedy search: find closest node to v
   - This becomes entry point for next layer

4. For each layer from max_layer to 0:
   - Find M closest nodes to v
   - Add bidirectional edges
   - Keep only top M connections per node (prune)
```

**Search Algorithm**:

```
1. Start at top layer, entry point

2. For each layer from top to 1:
   - Greedy search: find closest node to query
   - This becomes entry point for next layer

3. At layer 0:
   - Beam search with width ef
   - Return top K from beam
```

### Complexity Analysis

**Search complexity**: $O(\log N)$

*Let's derive this:*

1. **Number of layers**: $L = O(\log N)$
   - Each layer has half the items of layer below
   - $N \cdot (1/2)^L = 1 \Rightarrow L = \log_2 N$

2. **Work per layer**: $O(M \cdot \log N)$
   - Greedy search visits $O(\log N)$ nodes per layer (small world property)
   - Each node has $M$ neighbors to check

3. **Total**: $O(M \cdot \log^2 N)$

**In practice**: $M = 16-64$, so this is effectively $O(\log N)$.

**Memory**: $O(N \cdot M \cdot L) = O(N \cdot M \cdot \log N)$

For N = 10M, M = 32, L = 24:
- Edges: 10M * 32 * 24 = 7.68 billion
- Memory: ~60 GB (just for graph structure, plus embeddings)

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
- Compress vectors (128D to 8 bytes)
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

**Memory**: 128D vectors stored in 8 bytes (16x compression!)

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

**Trade-off**: Higher `ef` leads to better recall but slower search.

---

## FAISS vs ScaNN: When to Use Which

*Students often ask: "Which library should I use?" Let's be systematic.*

### Decision Framework

| Factor | Choose FAISS | Choose ScaNN |
|--------|-------------|--------------|
| **Scale** | < 100M items | > 100M items |
| **Hardware** | GPU available | TPU available |
| **Memory** | Flexible | Very constrained |
| **Update frequency** | Need incremental adds | Batch rebuild OK |
| **Ecosystem** | Need HNSW option | Fine with tree-based |
| **Maturity** | Want battle-tested | OK with newer |

### Detailed Comparison

**FAISS Strengths**:
1. **Algorithm variety**: Flat, IVF, IVFPQ, HNSW, and combinations
2. **GPU support**: Excellent CUDA implementation
3. **Incremental updates**: HNSW supports adding items
4. **Community**: Huge, well-documented, many tutorials
5. **Production proven**: Used at Meta, Spotify, many others

**ScaNN Strengths**:
1. **Raw speed**: 2-3x faster than FAISS IVFPQ at same recall
2. **Anisotropic quantization**: Better compression accuracy
3. **TPU support**: Designed for Google infrastructure
4. **Memory efficiency**: Excellent for memory-constrained
5. **Production proven**: Powers Google Search, YouTube, etc.

### My Recommendation

*Here's what I tell my students who are building production systems:*

**Start with FAISS HNSW** because:
- Highest recall (99%+) with reasonable speed
- Simple API, easy to tune
- Supports incremental updates
- Well-documented failure modes

**Graduate to ScaNN when**:
- You have > 500M items
- Memory is critical constraint
- You can tolerate batch rebuilds
- You have engineering resources for tuning

**Use FAISS IVFPQ when**:
- Memory is extremely constrained
- You can accept 90-95% recall
- You need GPU acceleration

---

## Numerical Walkthrough: 10M Items, 256-Dim Embeddings

*Let's make this concrete with actual numbers you'll see in production.*

### Setup

- **Items**: N = 10,000,000 (10 million)
- **Embedding dimension**: d = 256
- **Data type**: float32 (4 bytes per value)
- **Top-K**: K = 100 candidates

### Memory Analysis by Index Type

**1. Flat Index (Brute Force)**

Raw embedding storage:
$$\text{Memory} = N \times d \times 4 \text{ bytes} = 10M \times 256 \times 4 = \textbf{10.24 GB}$$

No additional index structure needed.

**2. IVF Index (Clustering)**

Same embedding storage: 10.24 GB

Plus cluster centroids (1000 clusters):
$$\text{Centroids} = 1000 \times 256 \times 4 = 1.02 \text{ MB}$$

Plus inverted lists (item-to-cluster mapping):
$$\text{Lists} = 10M \times 4 = 40 \text{ MB}$$

**Total: ~10.3 GB**

**3. IVFPQ Index (Compressed)**

Product quantization with m=32 subquantizers, 8 bits each:
$$\text{Codes} = N \times m = 10M \times 32 = 320 \text{ MB}$$

Plus codebooks (256 centroids per subquantizer):
$$\text{Codebooks} = m \times 256 \times (d/m) \times 4 = 32 \times 256 \times 8 \times 4 = 262 \text{ KB}$$

Plus cluster centroids and lists: ~41 MB

**Total: ~361 MB** (28x compression!)

**4. HNSW Index (Graph)**

Raw embeddings: 10.24 GB

Graph structure (M=32 connections, average 24 layers):
$$\text{Edges} \approx N \times M \times 2 = 10M \times 32 \times 2 = 640M \text{ edges}$$
$$\text{Edge storage} = 640M \times 4 = 2.56 \text{ GB}$$

**Total: ~12.8 GB**

### Query Time Analysis

**Test conditions**: Single thread, Intel Xeon, no GPU

| Index Type | Build Time | Query Time (p50) | Query Time (p99) | Recall@100 |
|------------|------------|------------------|------------------|------------|
| Flat | - | 850 ms | 920 ms | 100% |
| IVF (nprobe=10) | 45 min | 8.5 ms | 12 ms | 92% |
| IVF (nprobe=50) | 45 min | 42 ms | 55 ms | 98% |
| IVFPQ (nprobe=10) | 2 hr | 3.2 ms | 5 ms | 85% |
| IVFPQ (nprobe=50) | 2 hr | 15 ms | 22 ms | 93% |
| HNSW (ef=50) | 3 hr | 1.8 ms | 3.5 ms | 97% |
| HNSW (ef=200) | 3 hr | 6.5 ms | 11 ms | 99.5% |
| ScaNN | 1.5 hr | 1.2 ms | 2.1 ms | 95% |

### The Memory-Latency-Recall Triangle

*You can optimize for two, but not all three.*

```
                    RECALL
                      /\
                     /  \
                    /    \
                   / HNSW \
                  /________\
                 /          \
                / ScaNN/IVF  \
               /______________\
              /                \
             /     IVFPQ        \
            /____________________\
         MEMORY ←------------→ LATENCY
```

**HNSW**: Best recall, moderate memory, good latency
**ScaNN/IVF**: Balanced all three
**IVFPQ**: Best memory, acceptable latency, lower recall

---

## Product Quantization: Why Compression Works

*This is one of the most elegant ideas in ANN. Let's derive it.*

### The Problem

We have 10M items, 256 dimensions, float32:
$$\text{Memory} = 10M \times 256 \times 4 = 10.24 \text{ GB}$$

This doesn't fit in L3 cache (~30MB). Every distance computation is a memory access. **Memory bandwidth is the bottleneck, not compute.**

### The Key Insight

Instead of storing the full vector, store a **compressed code** that lets us **approximate** the distance.

**Observation**: We don't need exact distances!
- We only need to rank items by distance
- Getting the order roughly right is sufficient
- Even 90% recall is often acceptable

### Product Quantization Derivation

**Step 1**: Split the vector into subvectors

For a 256-dim vector $\mathbf{v}$, split into $m=32$ subvectors of length 8:
$$\mathbf{v} = [\mathbf{v}^1 | \mathbf{v}^2 | \ldots | \mathbf{v}^{32}]$$

where $\mathbf{v}^j \in \mathbb{R}^8$.

**Step 2**: Learn codebooks for each subspace

For subspace $j$, run k-means with 256 centroids:
$$\mathcal{C}^j = \{c^j_1, c^j_2, \ldots, c^j_{256}\}$$

Each centroid is 8-dimensional.

**Step 3**: Encode each vector

For item $i$, subvector $j$, find nearest centroid:
$$q^j_i = \arg\min_k \|\mathbf{v}^j_i - c^j_k\|^2$$

Store only the index $q^j_i \in \{1, \ldots, 256\}$ (8 bits).

**Compression ratio**:
- Original: $256 \times 4 = 1024$ bytes
- Compressed: $32 \times 1 = 32$ bytes
- **32x compression!**

### Distance Approximation

**Original distance**:
$$\|\mathbf{q} - \mathbf{v}\|^2 = \sum_{j=1}^{32} \|\mathbf{q}^j - \mathbf{v}^j\|^2$$

**Approximated distance**:
$$\|\mathbf{q} - \mathbf{v}\|^2 \approx \sum_{j=1}^{32} \|\mathbf{q}^j - c^j_{q^j_i}\|^2$$

**Key optimization**: Pre-compute a distance table!

For a query $\mathbf{q}$:
1. Compute all subvector-to-centroid distances: $d^j_k = \|\mathbf{q}^j - c^j_k\|^2$
2. This is $32 \times 256 = 8192$ distances (once per query)
3. For each item, look up 32 values and sum: $\sum_j d^j_{q^j_i}$

**Query complexity**: $O(32)$ lookups per item, not $O(256)$ multiplications!

### Why This Works: The Approximation Bound

**Lemma**: Let $\mathbf{v}$ be a vector and $\hat{\mathbf{v}}$ its PQ approximation. Then:
$$\mathbb{E}[\|\mathbf{v} - \hat{\mathbf{v}}\|^2] \leq \text{quantization error}$$

The quantization error depends on:
1. Number of centroids (more = lower error)
2. Data distribution (clustered = lower error)
3. Subvector dimension (smaller = lower error, but more subvectors)

**Empirical finding**: With 256 centroids per subspace, the ranking order is preserved for ~90% of true top-K items.

### Anisotropic Quantization (ScaNN's Improvement)

**Problem with standard PQ**: All subspaces treated equally.

**But**: Some dimensions matter more for the dot product!

If $\mathbf{q}^j$ has large magnitude, errors in subspace $j$ matter more.

**ScaNN's solution**: Weight the quantization by expected query magnitude:
$$\text{minimize } \mathbb{E}_\mathbf{q}[\|\mathbf{q}^T\mathbf{v} - \mathbf{q}^T\hat{\mathbf{v}}\|^2]$$

This allocates more bits to dimensions that queries care about.

**Result**: 10-20% better recall at same compression.

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
    |
User Embedding Service (GPU)
    |
ANN Index (HNSW/FAISS)
    |
Top-1000 Candidates
    |
Ranking Service (GPU)
    |
Top-10 Recommendations
    |
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
    Movies -> FAISS Index 1
    TV Shows -> FAISS Index 2
    Documentaries -> FAISS Index 3

Query:
    Search all relevant shards in parallel
    Merge results
```

**Latency**: Same (parallel queries)
**Throughput**: 3x (with 3 shards)

---

## What Can Go Wrong: Production Pitfalls

*Every production system I've seen has hit at least one of these. Learn from others' mistakes.*

### 1. Index Staleness

**The problem**: Your embeddings update daily, but user behavior changes hourly.

**Symptoms**:
- New items never get recommended
- Trending items underperform
- Recommendations feel "stale"

**Real example**: A streaming service updated their index every 24 hours. A viral video uploaded at 9 AM wouldn't appear in recommendations until the next day's index build at 6 AM. By then, the viral moment had passed.

**Solutions**:
1. **Hybrid index**: Main index (daily) + delta index (hourly) for recent items
2. **Streaming updates**: HNSW supports incremental adds
3. **Freshness boost**: Manually boost recent items in ranking

### 2. Embedding Drift

**The problem**: The retrieval index uses old embeddings, but your model has been retrained.

**Symptoms**:
- Recall drops without code changes
- A/B tests show regression
- Recommendations become less relevant over time

**Why it happens**:
- You retrain the two-tower model weekly
- New embeddings are in a "different space" than old ones
- User embedding (from new model) searches index of old item embeddings
- Dot products become meaningless

**Mathematical illustration**:

Old model: $\mathbf{u}_{old}, \mathbf{v}_{old}$ trained together
New model: $\mathbf{u}_{new}, \mathbf{v}_{new}$ trained together

If you use $\mathbf{u}_{new}$ to query $\mathbf{v}_{old}$:
$$\mathbf{u}_{new}^T \mathbf{v}_{old} \neq \text{similarity}(user, item)$$

The spaces don't align!

**Solutions**:
1. **Atomic updates**: Update user model and item index together
2. **Version pinning**: User service knows which index version to query
3. **Alignment loss**: Train new model to stay close to old embedding space

### 3. Recall vs Latency: The Monitoring Gap

**The problem**: You optimize for latency, recall silently degrades.

**How it happens**:
1. Traffic spikes, p99 latency increases
2. On-call engineer reduces `ef` or `nprobe` to improve latency
3. Latency improves! Ship it.
4. Nobody notices recall dropped from 97% to 85%
5. Weeks later: "Why are recommendations worse?"

**The insidious part**: Recall degradation is invisible in standard monitoring. Users don't complain "I'm missing 15% of relevant items" - they just engage less.

**Solutions**:
1. **Monitor recall**: Sample queries, compute exact top-K, compare to ANN top-K
2. **Alert on recall**: Set threshold (e.g., <95% triggers alert)
3. **A/B test parameter changes**: Don't ship latency optimizations without testing impact

### 4. Cold Start Meets ANN

**The problem**: New items have random/zero embeddings, ANN behavior is unpredictable.

**Scenarios**:
- New item with no interactions has random embedding (from random init)
- Item embedding is mean of empty set (zero vector)
- Fallback embedding is trained on different data

**What happens**:
- Random embeddings: Item appears in random users' retrievals
- Zero vector: Item never appears (dot product with anything is 0)
- Fallback: Item appears in wrong contexts

**Solutions**:
1. **Content-based fallback**: Use content features for new items until behavioral data exists
2. **Explore-exploit**: Randomly inject new items into retrieval for exploration
3. **Dedicated cold-start index**: Separate index with content-only embeddings

### 5. The Curse of High Recall

**The problem**: 99% recall sounds great, but it might be hurting you.

**Why higher recall can be bad**:
1. **Latency cost**: Going from 95% to 99% recall might 3x your latency
2. **Diminishing returns**: The extra 4% are boundary items anyway
3. **Ranking load**: More candidates = more ranking computation

**A Socratic question for you**:

> "If 99% recall takes 15ms and 95% recall takes 5ms, which should you choose?"

*Pauses*

**The answer depends on what you do with the saved 10ms!**

Option A: Serve at 15ms with 99% recall
Option B: Serve at 5ms with 95% recall
Option C: Use the saved 10ms for better ranking

Option C often wins. A sophisticated ranking model running on more candidates (in the same total time) beats a simple ranker running on fewer perfect candidates.

---

## Socratic Dialogue: Testing Your Understanding

*Let's make sure these concepts are solid.*

### Question 1: Why Two Stages?

**Student**: "Why not just use a more accurate ANN algorithm and skip ranking entirely?"

**Professor**: "Think about what ANN can and cannot see. What information does the retrieval stage have access to?"

**The answer**: Retrieval only sees the user embedding and item embeddings. It cannot see:
- Real-time context (time of day, device, session history)
- Cross-item features (diversity, already-watched filtering)
- Business rules (promoted content, content warnings)
- User's explicit intent (search query if hybrid system)

Even a perfect ANN (100% recall, 0ms latency) cannot replace ranking because ranking uses **more information**.

### Question 2: The 95% Recall Threshold

**Student**: "Why do you keep saying 95% recall is good enough? Shouldn't we maximize accuracy?"

**Professor**: "Consider the full pipeline. What happens to the 5% we miss?"

**The answer**: The 5% we miss are items ranked around position K in the true ordering. After ranking:
- Some of the 95% we found will rank high
- Some will rank low
- The "intruders" (items we incorrectly included) might rank high if the ranker likes them
- The missed items would have ranked low anyway (they were borderline)

**The math**: If missed items would have ranked in bottom 20% after full ranking, missing them has near-zero impact on final recommendations.

**When 95% is NOT enough**:
- Legal/compliance requirements (must show certain content)
- Fairness constraints (must consider all creator groups)
- Exact matching systems (search, not recommendations)

### Question 3: Memory vs. Latency

**Student**: "IVFPQ has 30x less memory but only slightly higher latency than HNSW. Why not always use IVFPQ?"

**Professor**: "Look at the recall numbers again. What's the business cost of lower recall?"

**The answer**: IVFPQ typically achieves 85-93% recall vs. HNSW's 97-99%. That 5-10% recall gap means:
- 5-10% of "true" recommendations are replaced by worse ones
- Cumulative over millions of users: significant engagement loss
- Harder to A/B test (signal is diluted)

**When IVFPQ wins**:
- Memory is the binding constraint (can't fit HNSW in RAM)
- Recall gap is acceptable for your use case
- You have a strong ranking model that compensates

### Question 4: Index Staleness

**Student**: "Our index rebuilds every 24 hours. How stale is too stale?"

**Professor**: "It depends on your item corpus. What's the half-life of relevance?"

**Framework for thinking about staleness**:

| Domain | Item half-life | Acceptable staleness |
|--------|---------------|---------------------|
| News | Hours | Minutes to hours |
| Social media | Hours | Minutes to hours |
| E-commerce (fashion) | Days | Hours |
| E-commerce (electronics) | Weeks | Days |
| Music/movies | Months | Days to weeks |
| Academic papers | Years | Weeks |

**The question to ask**: "If an item becomes relevant NOW, how much does it hurt to not recommend it for X hours?"

---

## Optimization Tips

### 1. Embedding Dimension

**Trade-off**: Higher dimension leads to better accuracy but slower search.

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

**Reduce precision**: float32 to int8

```python
import faiss

# Scalar quantization
index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)
index.train(item_embeddings)
index.add(item_embeddings)

# 4x memory reduction (32 bits -> 8 bits)
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
- Choose ANN library (FAISS HNSW recommended)
- Tune parameters (recall target: 95%+)
- Handle index updates (periodic or incremental)
- Shard for scale (if >100M items)
- Monitor latency (p50, p95, p99)
- **Monitor recall** (sample-based estimation)
- Plan for embedding drift (versioned updates)
- Handle cold start items (content fallback)

---

## The Closing Thought

*Professor wraps up*

We started with an impossible problem: search 100M items in 10ms when brute force takes 10 seconds.

We solved it by being willing to be **wrong** - but wrong in the right way.

The items we miss? Borderline cases that wouldn't have won anyway.
The items we incorrectly include? Close enough that the ranker can fix it.
The latency we save? Used for better ranking, fresher results, more features.

This is the engineering mindset for scale: **perfect is the enemy of good, and good is the enemy of shipped.**

Now go build something that serves a billion users.

---

## References

1. **Johnson, J., Douze, M., & Jegou, H. (2019)**. "Billion-scale similarity search with GPUs". *IEEE Transactions on Big Data*.
   - **FAISS paper**, Facebook's library

2. **Guo, R., et al. (2020)**. "Accelerating Large-Scale Inference with Anisotropic Vector Quantization". *ICML*.
   - **ScaNN paper**, Google's approach

3. **Malkov, Y. A., & Yashunin, D. A. (2018)**. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs". *IEEE TPAMI*.
   - **HNSW algorithm**

4. **Covington, P., Adams, J., & Sargin, E. (2016)**. "Deep Neural Networks for YouTube Recommendations". *RecSys*.
   - YouTube's two-tower architecture

5. **Ying, R., et al. (2018)**. "Graph Convolutional Neural Networks for Web-Scale Recommender Systems". *KDD*.
   - **PinSage** at Pinterest, billion-scale GNN
