# Week 7: Graph Neural Networks - LightGCN

## Overview

**LightGCN** (He et al., 2020) is a **simplified** Graph Convolutional Network that achieves **state-of-the-art** performance on collaborative filtering by removing unnecessary complexity.

**Counterintuitive result**: Simpler is better!
- Remove feature transformations → **+16% improvement**
- Remove nonlinear activations → Faster training
- Linear propagation only → Better performance

This document covers LightGCN's design, implementation, and why simplicity wins for recommendation.

---

## Learning Objectives

By the end of this section, you will:
- Understand user-item bipartite graphs
- Master LightGCN's simplified propagation
- Implement LightGCN from scratch (PyTorch)
- Recognize why removing complexity helps
- Apply graph-based CF to real datasets

---

## The User-Item Bipartite Graph

### Graph Representation

**Nodes**: Users $\mathcal{U}$ and Items $\mathcal{I}$

**Edges**: Interactions (ratings, clicks, purchases)
- Edge $(u, i)$ exists if user $u$ interacted with item $i$

**Bipartite**: No user-user or item-item edges (only user-item)

**Example**:
```
Users:      U1    U2    U3
             |    / \    |
             |   /   \   |
             |  /     \  |
Items:      I1 I2    I3 I4

Edges: (U1,I1), (U2,I1), (U2,I3), (U3,I4)
```

---

### Adjacency Matrix

**Sparse matrix** $A \in \mathbb{R}^{(|U|+|I|) \times (|U|+|I|)}$:

$$A = \begin{bmatrix} 0 & R \\ R^T & 0 \end{bmatrix}$$

where:
- $R \in \mathbb{R}^{|U| \times |I|}$: User-item interaction matrix
- $R^T$: Transpose (item-user)
- Block structure: Users don't connect to users, items don't connect to items

---

## The Evolution: From GCN to LightGCN

### Standard GCN (Kipf & Welling, 2017)

**Layer-wise propagation**:
$$\mathbf{h}_i^{(l+1)} = \sigma\left( \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(j)|}} \mathbf{h}_j^{(l)} W^{(l)} \right)$$

**Components**:
1. **Neighbor aggregation**: Sum over neighbors
2. **Feature transformation**: Multiply by weight matrix $W^{(l)}$
3. **Normalization**: Symmetric normalization
4. **Nonlinearity**: $\sigma$ (ReLU, tanh)

---

### NGCF (Wang et al., 2019)

**Neural Graph Collaborative Filtering** for recommendation:

$$\mathbf{h}_i^{(l+1)} = \sigma\left( W_1^{(l)} \mathbf{h}_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(j)|}} (W_2^{(l)} \mathbf{h}_j^{(l)} + W_3^{(l)} (\mathbf{h}_i^{(l)} \odot \mathbf{h}_j^{(l)})) \right)$$

where:
- $\mathbf{h}_i^{(l)}$ = embedding of node $i$ at layer $l$
- $\mathcal{N}(i)$ = neighbors of node $i$ in the user-item bipartite graph
- $\sigma$ = nonlinear activation function (e.g., ReLU, LeakyReLU)
- $\odot$ = element-wise (Hadamard) product
- $W_1^{(l)} \in \mathbb{R}^{d \times d}$ = weight matrix for **self-connection** (transforms node's own embedding)
- $W_2^{(l)} \in \mathbb{R}^{d \times d}$ = weight matrix for **neighbor message** (transforms neighbor embeddings)
- $W_3^{(l)} \in \mathbb{R}^{d \times d}$ = weight matrix for **interaction term** (transforms element-wise product of node and neighbor)
- $\frac{1}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(j)|}}$ = symmetric normalization coefficient

**Added complexity compared to basic GCN**:
- Self-connection: $W_1^{(l)} \mathbf{h}_i^{(l)}$ - preserves node's own information
- Neighbor aggregation: $W_2^{(l)} \mathbf{h}_j^{(l)}$ - captures neighbor features
- Element-wise product: $W_3^{(l)} (\mathbf{h}_i^{(l)} \odot \mathbf{h}_j^{(l)})$ - models explicit interaction between node and neighbors
- Three separate weight matrices per layer

**Problem**: More parameters = overfitting on sparse data!

---

### LightGCN: Simplification

**Key insight**: For collaborative filtering, feature transformation and nonlinearity **hurt** performance!

**Simplified propagation**:
$$\mathbf{h}_i^{(l+1)} = \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(j)|}} \mathbf{h}_j^{(l)}$$

**What's removed**:
1. ❌ Feature transformation ($W$)
2. ❌ Nonlinear activation ($\sigma$)
3. ❌ Self-connections
4. ❌ Element-wise products

**What's kept**:
✅ Neighborhood aggregation
✅ Normalization

**Result**: Fewer parameters, better generalization, faster training!

---

## LightGCN Architecture

### Layer-wise Propagation

**Initialization** (layer 0):
$$\mathbf{h}_u^{(0)} = \mathbf{e}_u, \quad \mathbf{h}_i^{(0)} = \mathbf{e}_i$$

where $\mathbf{e}_u, \mathbf{e}_i \in \mathbb{R}^d$ are learnable embeddings.

**Propagation** (layer $l = 1, 2, \ldots, L$):
$$\mathbf{h}_u^{(l)} = \sum_{i \in \mathcal{N}(u)} \frac{1}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(i)|}} \mathbf{h}_i^{(l-1)}$$

$$\mathbf{h}_i^{(l)} = \sum_{u \in \mathcal{N}(i)} \frac{1}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(u)|}} \mathbf{h}_u^{(l-1)}$$

**Normalization**: $\frac{1}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(i)|}}$ prevents exploding/vanishing values.

---

### Layer Combination

**Final representation**: Weighted sum of all layers

$$\mathbf{h}_u = \sum_{l=0}^L \alpha_l \mathbf{h}_u^{(l)}, \quad \mathbf{h}_i = \sum_{l=0}^L \alpha_l \mathbf{h}_i^{(l)}$$

where $\alpha_l$ is the importance of layer $l$.

**Typical**: Uniform weighting $\alpha_l = \frac{1}{L+1}$

**Intuition**:
- Layer 0: Direct user/item embeddings
- Layer 1: 1-hop neighbors (items user liked, users who liked item)
- Layer 2: 2-hop neighbors (items similar users liked)
- Layer 3+: Higher-order collaborative signals

---

### Prediction

**Score** for user $u$ and item $i$:
$$\hat{y}_{ui} = \mathbf{h}_u^T \mathbf{h}_i$$

Simple dot product!

---

## Loss Function

### BPR Loss (Bayesian Personalized Ranking)

$$\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j) \in \mathcal{D}} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda \|\mathbf{E}\|^2$$

where:
- $(u, i, j)$: User $u$ interacted with item $i$ (positive), not with item $j$ (negative)
- $\sigma(x) = \frac{1}{1 + e^{-x}}$: Sigmoid
- $\lambda$: L2 regularization on embeddings $\mathbf{E}$

**Pairwise ranking**: Positive item should rank higher than negative item.

---

## Matrix Form (Efficient Implementation)

### Propagation as Matrix Multiplication

**Graph convolution**:
$$\mathbf{H}^{(l+1)} = \tilde{A} \mathbf{H}^{(l)}$$

where:
- $\mathbf{H}^{(l)} \in \mathbb{R}^{(|U|+|I|) \times d}$: Embeddings at layer $l$ (stacked users and items)
- $\tilde{A}$: Normalized adjacency matrix

**Normalized adjacency**:
$$\tilde{A} = D^{-1/2} A D^{-1/2}$$

where $D$ is the degree matrix:
$$D_{ii} = \sum_j A_{ij}$$

**Efficient**: Use sparse matrix multiplication!

---

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        # Learnable embeddings (only at layer 0)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Initialize
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, adj_matrix):
        """
        adj_matrix: Normalized adjacency matrix (sparse)
        """
        # Initial embeddings (layer 0)
        user_emb = self.user_embedding.weight  # (n_users, d)
        item_emb = self.item_embedding.weight  # (n_items, d)

        # Stack: [users; items]
        all_emb = torch.cat([user_emb, item_emb], dim=0)  # (n_users+n_items, d)

        # Store embeddings at each layer
        emb_list = [all_emb]

        # Layer-wise propagation
        for layer in range(self.n_layers):
            all_emb = matmul(adj_matrix, all_emb)  # Sparse matrix multiplication
            emb_list.append(all_emb)

        # Layer combination (average)
        final_emb = torch.stack(emb_list, dim=0).mean(dim=0)  # (n_users+n_items, d)

        # Split back to users and items
        users_final = final_emb[:self.n_users]
        items_final = final_emb[self.n_users:]

        return users_final, items_final

    def bpr_loss(self, users, pos_items, neg_items, adj_matrix):
        """
        Compute BPR loss for a batch.

        users: (batch,) - user IDs
        pos_items: (batch,) - positive item IDs
        neg_items: (batch,) - negative item IDs
        """
        # Get final embeddings
        users_emb, items_emb = self.forward(adj_matrix)

        # Lookup
        user_emb = users_emb[users]  # (batch, d)
        pos_item_emb = items_emb[pos_items]
        neg_item_emb = items_emb[neg_items]

        # Scores
        pos_scores = (user_emb * pos_item_emb).sum(dim=1)  # (batch,)
        neg_scores = (user_emb * neg_item_emb).sum(dim=1)

        # BPR loss
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        # L2 regularization on initial embeddings
        reg_loss = (self.user_embedding.weight[users].norm(2).pow(2) +
                    self.item_embedding.weight[pos_items].norm(2).pow(2) +
                    self.item_embedding.weight[neg_items].norm(2).pow(2)) / users.size(0)

        return bpr_loss, reg_loss

    def predict(self, users, items, adj_matrix):
        """
        Predict scores for (user, item) pairs.
        """
        users_emb, items_emb = self.forward(adj_matrix)
        user_emb = users_emb[users]
        item_emb = items_emb[items]
        scores = (user_emb * item_emb).sum(dim=1)
        return scores

# Prepare normalized adjacency matrix
def build_adj_matrix(user_item_edges, n_users, n_items):
    """
    Build normalized adjacency matrix from edges.

    user_item_edges: (2, n_edges) - [user_ids; item_ids]
    Returns: SparseTensor (n_users+n_items, n_users+n_items)
    """
    n_nodes = n_users + n_items

    # Shift item IDs
    user_ids = user_item_edges[0]
    item_ids = user_item_edges[1] + n_users  # Offset by n_users

    # Bidirectional edges
    edge_index = torch.cat([
        torch.stack([user_ids, item_ids]),
        torch.stack([item_ids, user_ids])
    ], dim=1)  # (2, 2*n_edges)

    # Create sparse adjacency
    adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                       sparse_sizes=(n_nodes, n_nodes))

    # Compute degree
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # Normalize: D^{-0.5} A D^{-0.5}
    adj_norm = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    return adj_norm

# Training loop
def train(model, train_edges, n_users, n_items, epochs=100, lr=0.001, reg=1e-4, batch_size=1024):
    adj_matrix = build_adj_matrix(train_edges, n_users, n_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Sample batches
        n_edges = train_edges.size(1)
        perm = torch.randperm(n_edges)

        for start in range(0, n_edges, batch_size):
            batch_idx = perm[start:start+batch_size]
            users = train_edges[0, batch_idx]
            pos_items = train_edges[1, batch_idx]

            # Sample negative items
            neg_items = torch.randint(0, n_items, (batch_idx.size(0),))

            # Compute loss
            bpr_loss, reg_loss = model.bpr_loss(users, pos_items, neg_items, adj_matrix)
            loss = bpr_loss + reg * reg_loss

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}: Loss = {total_loss/n_edges:.4f}")

# Recommendation
@torch.no_grad()
def recommend(model, user_id, adj_matrix, top_k=10, exclude_items=None):
    """
    Recommend top-K items for a user.
    """
    model.eval()
    users_emb, items_emb = model.forward(adj_matrix)

    user_emb = users_emb[user_id]  # (d,)
    scores = user_emb @ items_emb.T  # (n_items,)

    # Mask already interacted
    if exclude_items is not None:
        scores[exclude_items] = -float('inf')

    top_items = torch.topk(scores, top_k).indices.cpu().numpy()
    return top_items
```

---

## Why LightGCN Works: Ablation Study

### Experimental Results (MovieLens-1M, He et al., 2020)

| Model | Recall@20 | NDCG@20 |
|-------|-----------|---------|
| MF (baseline) | 0.1779 | 0.1457 |
| NGCF (complex GCN) | 0.2090 | 0.1728 |
| **LightGCN** | **0.2424** | **0.2000** |

**Improvement**: LightGCN beats NGCF by **+16% Recall@20**!

---

### Ablation: What Matters?

**Remove components one by one**:

| Variant | Recall@20 | What's Removed |
|---------|-----------|----------------|
| NGCF (full) | 0.2090 | Baseline |
| -NonLinear | 0.2210 | Remove ReLU → **+5.7%** |
| -FeatureTransform | 0.2350 | Remove $W$ → **+12.4%** |
| **LightGCN** | **0.2424** | Remove all → **+16.0%** |

**Insight**: Each removed component **improves** performance!

**Why?**
1. **Overfitting**: NGCF has too many parameters for sparse data
2. **Smoothing**: Linear propagation acts as regularization
3. **Collaborative signal**: Neighborhood aggregation is all you need

---

## Comparison with Other Models

### Collaborative Filtering Methods

| Model | Type | Recall@20 | Parameters |
|-------|------|-----------|------------|
| MF | Matrix Factorization | 0.1779 | $O(d \cdot (|U|+|I|))$ |
| NCF | Neural CF | 0.1920 | $O(d \cdot (|U|+|I|) + L \cdot d^2)$ |
| NGCF | GNN (complex) | 0.2090 | $O(d \cdot (|U|+|I|) + 3Ld^2)$ |
| **LightGCN** | GNN (simple) | **0.2424** | $O(d \cdot (|U|+|I|))$ |

**LightGCN**: Best performance with fewest parameters!

---

## Hyperparameters

| Parameter | Symbol | Typical Range | Recommendation |
|-----------|--------|---------------|----------------|
| Embedding dim | $d$ | 32-256 | 64 for small, 128-256 for large |
| Layers | $L$ | 2-4 | 3 (diminishing returns after) |
| Learning rate | - | 0.0001-0.01 | 0.001 (Adam) |
| L2 reg | $\lambda$ | 1e-5 to 1e-3 | 1e-4 |
| Batch size | - | 1024-4096 | 2048 |

**Layer depth**:
- $L=1$: Only 1-hop neighbors
- $L=2$: 2-hop (friends of friends)
- $L=3$: 3-hop (**best** in most cases)
- $L>3$: Over-smoothing (all embeddings become similar)

---

## Scalability

### Time Complexity

**Training** (per epoch):
- Propagation: $O(L \cdot |E| \cdot d)$ (sparse matrix-vector products)
- BPR loss: $O(|E| \cdot d)$

where $|E|$ is number of edges (interactions).

**Inference** (top-K for one user):
- Propagation (cached): $O(1)$
- Score computation: $O(|I| \cdot d)$ (dot product with all items)
- Sorting: $O(|I| \log K)$

**Typical**: 100M edges, $d=128$, $L=3$ → **< 1 minute per epoch** on GPU.

---

### Sparse Matrix Optimization

**Key**: Use sparse representations!

```python
from torch_sparse import SparseTensor

# Sparse adjacency
adj = SparseTensor(row=row_indices, col=col_indices, sparse_sizes=(n, n))

# Sparse matmul (much faster than dense)
output = torch_sparse.matmul(adj, embeddings)  # O(|E| * d)
```

**Speedup**: 10-100x faster than dense operations.

---

## Practical Tips

### 1. Negative Sampling

**Uniform sampling**:
- Sample random items as negatives
- Simple, unbiased

**Popularity-based**:
- Sample proportional to $\text{popularity}^{0.75}$
- Harder negatives, better training

---

### 2. Early Stopping

**Monitor** validation Recall@20 every epoch:
- Stop if no improvement for 10 epochs
- Prevents overfitting

---

### 3. Cold Start

**New users** (no interactions):
- Cannot use LightGCN directly (no graph connections)
- Fallback: Popularity-based or content-based

**New items** (no interactions):
- Use item features (if available)
- Or wait for first few interactions

---

## Summary

**Key Takeaways**:
1. **LightGCN simplifies GCNs** → removes transformations, nonlinearities
2. **Simplicity wins** → +16% improvement over complex NGCF
3. **Linear propagation** is sufficient for collaborative filtering
4. **Fewer parameters** → better generalization on sparse data
5. **State-of-the-art** results (2020-2024)

**Why it works**:
- **Over-smoothing as regularization**: Linear propagation smooths embeddings
- **Collaborative signal**: Neighbor aggregation captures user-item patterns
- **Less overfitting**: Fewer parameters for sparse interaction data

**Limitations**:
- Requires graph structure (edges)
- Cold start problem (new users/items)
- Not easily extended to features (side information)

**When to use LightGCN**:
- Implicit feedback data (clicks, views)
- Bipartite user-item graphs
- Want state-of-the-art CF performance
- Have sufficient interactions (not extreme cold start)

---

## References

1. **He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020)**. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation". *SIGIR*.
   - **Original LightGCN paper**

2. **Wang, X., He, X., Wang, M., Feng, F., & Chua, T. S. (2019)**. "Neural Graph Collaborative Filtering". *SIGIR*.
   - NGCF (complex baseline)

3. **Kipf, T. N., & Welling, M. (2017)**. "Semi-Supervised Classification with Graph Convolutional Networks". *ICLR*.
   - Original GCN paper

4. **Berg, R. van den, Kipf, T. N., & Welling, M. (2017)**. "Graph Convolutional Matrix Completion". *arXiv*.
   - GCN for collaborative filtering

5. **Ying, R., He, R., Chen, K., Eksombatchai, P., Hamilton, W. L., & Leskovec, J. (2018)**. "Graph Convolutional Neural Networks for Web-Scale Recommender Systems". *KDD*.
   - PinSage (Pinterest's GNN), industrial scale
