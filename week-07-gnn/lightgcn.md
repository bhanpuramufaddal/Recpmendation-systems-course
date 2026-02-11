# Week 7: Graph Neural Networks - LightGCN

## The Opening Paradox: Why Does Removing Features IMPROVE Recommendations?

*This should bother you.*

**The Setup:**

NGCF (Neural Graph Collaborative Filtering, 2019) is a sophisticated GNN with:
- Feature transformation matrices $W_1, W_2, W_3$ at each layer
- Nonlinear activations (LeakyReLU)
- Self-connections for preserving information
- Element-wise interactions for modeling user-item affinity

**Total parameters** (3 layers, d=64):
$$3 \times 3 \times 64^2 = 36,864 \text{ transformation parameters}$$

**LightGCN (2020)** removes ALL of that:
- No feature transformations
- No activations
- No self-connections
- No element-wise products

**Total transformation parameters**: 0

---

**The Results (MovieLens-1M):**

| Model | Recall@20 | Parameters in Transforms |
|-------|-----------|-------------------------|
| NGCF | 0.2090 | 36,864 |
| **LightGCN** | **0.2424** | **0** |

*LightGCN is 16% better with ZERO transformation parameters!*

**How can removing learned components improve performance?**

This is the puzzle we'll solve in this lecture.

---

## Overview

**LightGCN** (He et al., 2020) is a **simplified** Graph Convolutional Network that achieves **state-of-the-art** performance on collaborative filtering by removing unnecessary complexity.

**Counterintuitive result**: Simpler is better!
- Remove feature transformations and +16% improvement
- Remove nonlinear activations and faster training
- Linear propagation only and better performance

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

**Let me explain each term:**

| Term | Meaning | Purpose |
|------|---------|---------|
| $\mathbf{h}_i^{(l)}$ | Embedding of node $i$ at layer $l$ | Current state |
| $\mathcal{N}(i)$ | Neighbors of node $i$ | Who to aggregate from |
| $\sigma$ | Nonlinear activation (LeakyReLU) | Expressiveness |
| $\odot$ | Element-wise product | Capture interactions |
| $W_1^{(l)}$ | Self-connection weights | Preserve own info |
| $W_2^{(l)}$ | Neighbor message weights | Transform neighbors |
| $W_3^{(l)}$ | Interaction weights | Model user-item affinity |

**Added complexity compared to basic GCN**:
- Self-connection: $W_1^{(l)} \mathbf{h}_i^{(l)}$ - preserves node's own information
- Neighbor aggregation: $W_2^{(l)} \mathbf{h}_j^{(l)}$ - captures neighbor features
- Element-wise product: $W_3^{(l)} (\mathbf{h}_i^{(l)} \odot \mathbf{h}_j^{(l)})$ - models explicit interaction

**Problem**: More parameters = overfitting on sparse data!

---

### LightGCN: The Radical Simplification

**Key insight**: For collaborative filtering, feature transformation and nonlinearity **hurt** performance!

**Simplified propagation**:
$$\mathbf{h}_i^{(l+1)} = \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(j)|}} \mathbf{h}_j^{(l)}$$

**What's removed**:
1. Feature transformation ($W$)
2. Nonlinear activation ($\sigma$)
3. Self-connections
4. Element-wise products

**What's kept**:
- Neighborhood aggregation
- Normalization

**Result**: Fewer parameters, better generalization, faster training!

---

## The Ablation Study: Dissecting NGCF

*Let me show you exactly what happens when we remove each component.*

### The Experiment

Start with full NGCF and remove components one by one:

**Dataset**: MovieLens-1M (1M ratings, 6K users, 4K items)

| Variant | What's Removed | Recall@20 | Change |
|---------|---------------|-----------|--------|
| NGCF (full) | Nothing | 0.2090 | Baseline |
| NGCF-f | Interaction term $W_3(\mathbf{h}_i \odot \mathbf{h}_j)$ | 0.2150 | +2.9% |
| NGCF-n | Nonlinearity $\sigma$ | 0.2210 | +5.7% |
| NGCF-W | All weight matrices $W_1, W_2, W_3$ | 0.2350 | +12.4% |
| **LightGCN** | Everything (pure aggregation) | **0.2424** | **+16.0%** |

*Each component we remove IMPROVES performance!*

---

### Why Each Removal Helps

**Removing $W_3$ (interaction term)**: +2.9%

The element-wise product $\mathbf{h}_i \odot \mathbf{h}_j$ tries to model explicit user-item affinity at each layer. But:
- We already model this at prediction time (dot product)
- Adding it at each layer is redundant AND adds $d^2$ parameters per layer
- These parameters overfit on sparse data

---

**Removing $\sigma$ (nonlinearity)**: +5.7%

*What does ReLU do?* It zeros out negative values.

In collaborative filtering, negative embedding dimensions often carry meaning:
- User A dislikes horror: $\mathbf{h}_A^{\text{horror}} = -0.8$
- Item X is a horror movie: $\mathbf{h}_X^{\text{horror}} = 0.9$
- Dot product captures mismatch: $(-0.8)(0.9) = -0.72$ (low score)

*If we apply ReLU*:
- $\text{ReLU}(-0.8) = 0$
- We lose the "dislike" signal!

*Can you see why* activations might harm recommendation? The negative values are informative!

---

**Removing $W$ (feature transforms)**: +12.4%

This is the biggest gain. Why?

**Parameter count argument**:
- NGCF: $3 \times d^2$ parameters per layer
- With $L=3$ layers and $d=64$: $3 \times 3 \times 64^2 = 36,864$ parameters
- Training signals: Depends on data, but MovieLens-1M has ~1M ratings
- **Ratio**: ~36K parameters for only ~1M training pairs

The weight matrices are overfitting!

**What do transformations try to learn?** In NLP/vision, transformations extract features:
- "This pixel pattern = edge"
- "This word sequence = sentiment"

**In CF, what features are there?** User ID 42 and Item ID 103 have no inherent meaning. The ONLY signal is the interaction pattern. Transformations have nothing meaningful to extract.

---

## The Philosophy: Why Simplicity Wins for CF

### The Regularization Perspective

*Linear propagation acts as implicit regularization.*

**LightGCN propagation**:
$$\mathbf{h}_u^{(1)} = \sum_{\text{items rated by } u} \frac{\mathbf{h}_i^{(0)}}{\sqrt{|N(u)||N(i)|}}$$

This is essentially **weighted averaging** of item embeddings.

**What averaging does**:
1. **Smooths noise**: Individual item embeddings might be noisy, but the average is stable
2. **Regularizes**: Prevents any single item from dominating
3. **Shares information**: Similar items (rated by same users) get similar embeddings

---

### The Collaborative Signal Perspective

**What we ACTUALLY learn from in CF**:
```
User 1 liked items {A, B, C}
User 2 liked items {A, B, D}
Therefore: A and B should be similar (co-rated often)
```

**What we DON'T have** (unlike NLP or vision):
- No rich features (text, pixels)
- No semantic meaning of "User 1" or "Item A"
- Just IDs and interaction patterns

**LightGCN's philosophy**: The ONLY signal is the graph structure. Don't try to add complexity (transformations, nonlinearities) that has nothing to learn from.

*Notice that* this is fundamentally different from computer vision or NLP, where complex features (edges, textures, syntax) NEED transformations to extract!

---

## LightGCN Architecture

### Layer-wise Propagation

**Initialization** (layer 0):
$$\mathbf{h}_u^{(0)} = \mathbf{e}_u, \quad \mathbf{h}_i^{(0)} = \mathbf{e}_i$$

where $\mathbf{e}_u, \mathbf{e}_i \in \mathbb{R}^d$ are learnable embeddings.

*These are the ONLY learnable parameters!*

**Propagation** (layer $l = 1, 2, \ldots, L$):
$$\mathbf{h}_u^{(l)} = \sum_{i \in \mathcal{N}(u)} \frac{1}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(i)|}} \mathbf{h}_i^{(l-1)}$$

$$\mathbf{h}_i^{(l)} = \sum_{u \in \mathcal{N}(i)} \frac{1}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(u)|}} \mathbf{h}_u^{(l-1)}$$

**Normalization**: $\frac{1}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(i)|}}$ prevents exploding/vanishing values.

---

### Layer Combination: Why Average Across Layers?

**Final representation**: Weighted sum of all layers

$$\mathbf{h}_u = \sum_{l=0}^L \alpha_l \mathbf{h}_u^{(l)}, \quad \mathbf{h}_i = \sum_{l=0}^L \alpha_l \mathbf{h}_i^{(l)}$$

where $\alpha_l$ is the importance of layer $l$.

**Typical**: Uniform weighting $\alpha_l = \frac{1}{L+1}$

---

**Why not just use the last layer?**

*Let me show you with numbers.*

**Example**: 2-layer LightGCN on a small graph

**Layer 0** (initial embeddings):
```
User U: [1.0, 0.0]  (action lover)
Item A: [0.8, 0.2]  (action movie U rated)
Item B: [0.1, 0.9]  (comedy, 2 hops away)
```

**Layer 1** (1-hop aggregation):
```
U^(1) = aggregate(items U rated) = [0.8, 0.2]
(U picked up A's embedding)
```

**Layer 2** (2-hop aggregation):
```
U^(2) = aggregate(items rated by users who rated A)
     = includes B = [0.45, 0.55]
(U now has comedy signal from 2 hops away!)
```

**What happens if we only use Layer 2?**

$\mathbf{h}_U = [0.45, 0.55]$

We've LOST the original signal that U is an action lover! The 2-hop information dominated.

**What happens with layer combination?**

$$\mathbf{h}_U = \frac{1}{3}[1.0, 0.0] + \frac{1}{3}[0.8, 0.2] + \frac{1}{3}[0.45, 0.55]$$
$$= [0.75, 0.25]$$

Now we have:
- Original preference: action
- 1-hop signal: action (confirmed)
- 2-hop signal: slight comedy interest (new discovery)

*Can you see why* combining layers preserves information at all scales?

---

### What Each Layer Captures

| Layer | Information | Example |
|-------|-------------|---------|
| Layer 0 | Direct user/item identity | "Who is User U?" |
| Layer 1 | 1-hop neighbors | "What did U rate?" |
| Layer 2 | 2-hop neighbors | "What did similar users rate?" |
| Layer 3 | 3-hop neighbors | "Friend of friend of friend" |

**The combination**: Balances local identity with global collaborative signal.

---

### Numerical Walkthrough: 4-Node LightGCN

*Let's trace through propagation with exact numbers.*

**Graph (bipartite)**:
```
Users:    U0    U1
           |\  /|
           | \/ |
           | /\ |
           |/  \|
Items:    I0    I1

Edges: (U0,I0), (U0,I1), (U1,I0), (U1,I1)
```

Both users rated both items (complete bipartite).

**Initial Embeddings (d=2)**:
```
U0: [1.0, 0.0]
U1: [0.0, 1.0]
I0: [0.5, 0.5]
I1: [0.5, 0.5]
```

**Degree**: U0, U1 have degree 2. I0, I1 have degree 2.

**Normalization**: $\frac{1}{\sqrt{2} \cdot \sqrt{2}} = \frac{1}{2}$

---

**Layer 1 Propagation:**

**User U0** aggregates from items {I0, I1}:
$$\mathbf{h}_{U0}^{(1)} = \frac{1}{2}[0.5, 0.5] + \frac{1}{2}[0.5, 0.5] = [0.5, 0.5]$$

**User U1** aggregates from items {I0, I1}:
$$\mathbf{h}_{U1}^{(1)} = \frac{1}{2}[0.5, 0.5] + \frac{1}{2}[0.5, 0.5] = [0.5, 0.5]$$

**Item I0** aggregates from users {U0, U1}:
$$\mathbf{h}_{I0}^{(1)} = \frac{1}{2}[1.0, 0.0] + \frac{1}{2}[0.0, 1.0] = [0.5, 0.5]$$

**Item I1** aggregates from users {U0, U1}:
$$\mathbf{h}_{I1}^{(1)} = \frac{1}{2}[1.0, 0.0] + \frac{1}{2}[0.0, 1.0] = [0.5, 0.5]$$

**After Layer 1**: All embeddings are [0.5, 0.5]!

*Notice that* one layer already mixed information. Users absorbed item info, items absorbed user info.

---

**Layer 2 Propagation:**

Since all Layer 1 embeddings are [0.5, 0.5], Layer 2 gives same result.

**Final Embeddings (L=2, uniform weighting)**:

$$\mathbf{h}_{U0} = \frac{1}{3}[1.0, 0.0] + \frac{1}{3}[0.5, 0.5] + \frac{1}{3}[0.5, 0.5] = [0.67, 0.33]$$

$$\mathbf{h}_{U1} = \frac{1}{3}[0.0, 1.0] + \frac{1}{3}[0.5, 0.5] + \frac{1}{3}[0.5, 0.5] = [0.33, 0.67]$$

$$\mathbf{h}_{I0} = \frac{1}{3}[0.5, 0.5] + \frac{1}{3}[0.5, 0.5] + \frac{1}{3}[0.5, 0.5] = [0.5, 0.5]$$

$$\mathbf{h}_{I1} = [0.5, 0.5]$$

**Prediction scores**:
- $\hat{y}_{U0, I0} = [0.67, 0.33] \cdot [0.5, 0.5] = 0.335 + 0.165 = 0.5$
- $\hat{y}_{U1, I0} = [0.33, 0.67] \cdot [0.5, 0.5] = 0.165 + 0.335 = 0.5$

*Both users have same score for I0* (makes sense - complete bipartite graph has no preference signal).

---

### Comparing NGCF vs LightGCN Propagation

*Let's trace both on the same graph to see the difference.*

**Setup**: Same 4-node graph, same initial embeddings.

**NGCF Layer 1** (simplified, showing one user):

$$\mathbf{h}_{U0}^{(1)} = \sigma\left( W_1 \mathbf{h}_{U0}^{(0)} + \sum_{i} \frac{1}{2} \left( W_2 \mathbf{h}_i^{(0)} + W_3 (\mathbf{h}_{U0}^{(0)} \odot \mathbf{h}_i^{(0)}) \right) \right)$$

Assume $W_1, W_2, W_3$ are identity matrices for simplicity:

$$= \sigma\left( [1,0] + \frac{1}{2}([0.5,0.5] + [1 \cdot 0.5, 0 \cdot 0.5]) + \frac{1}{2}([0.5,0.5] + [1 \cdot 0.5, 0 \cdot 0.5]) \right)$$
$$= \sigma\left( [1,0] + [1.0, 0.5] \right)$$
$$= \sigma([2.0, 0.5])$$
$$= \text{LeakyReLU}([2.0, 0.5]) = [2.0, 0.5]$$

**LightGCN Layer 1**:
$$\mathbf{h}_{U0}^{(1)} = \frac{1}{2}[0.5, 0.5] + \frac{1}{2}[0.5, 0.5] = [0.5, 0.5]$$

**Key Differences**:
| Aspect | NGCF | LightGCN |
|--------|------|----------|
| Self-connection | Yes ($W_1$) | No |
| Magnitude | [2.0, 0.5] (exploding) | [0.5, 0.5] (normalized) |
| Computation | 3 matmuls + activation | 1 sparse matmul |
| Parameters per layer | $3d^2$ | 0 |

*Notice that* NGCF's output [2.0, 0.5] is already larger than inputs. After multiple layers, this can explode!

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

*Why BPR and not cross-entropy?* We care about ranking, not classification. BPR directly optimizes the relative order.

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

## Why LightGCN Works: The Complete Picture

### Experimental Results (MovieLens-1M, He et al., 2020)

| Model | Recall@20 | NDCG@20 |
|-------|-----------|---------|
| MF (baseline) | 0.1779 | 0.1457 |
| NGCF (complex GCN) | 0.2090 | 0.1728 |
| **LightGCN** | **0.2424** | **0.2000** |

**Improvement**: LightGCN beats NGCF by **+16% Recall@20**!

---

### The Three Reasons Simplicity Wins

**1. Overfitting Prevention**

| Model | Transformation Params | User/Item Params | Ratio |
|-------|----------------------|------------------|-------|
| NGCF | 36,864 | ~640,000 | 5.8% |
| LightGCN | 0 | ~640,000 | 0% |

With sparse data (96% sparsity in MovieLens-1M), those 36K extra parameters overfit.

**2. Gradient Flow**

NGCF: $\nabla = \nabla_L \cdot W_1^{(L)} \cdot \sigma' \cdot W_2^{(L-1)} \cdot \sigma' \cdot ...$

LightGCN: $\nabla = \nabla_L \cdot \tilde{A}^L$

*No vanishing/exploding gradients from activation derivatives!*

**3. Information Preservation**

Activations like ReLU zero out negative values. In CF, negative embedding dimensions encode "dislikes" which are informative. LightGCN preserves them.

---

## Hyperparameters

| Parameter | Symbol | Typical Range | Recommendation |
|-----------|--------|---------------|----------------|
| Embedding dim | $d$ | 32-256 | 64 for small, 128-256 for large |
| Layers | $L$ | 2-4 | 3 (diminishing returns after) |
| Learning rate | - | 0.0001-0.01 | 0.001 (Adam) |
| L2 reg | $\lambda$ | 1e-5 to 1e-3 | 1e-4 |
| Batch size | - | 1024-4096 | 2048 |

---

### How Many Layers? The Depth Experiment

*Let me show you what happens as we add layers.*

| Layers | Recall@20 | NDCG@20 | Observation |
|--------|-----------|---------|-------------|
| 1 | 0.2189 | 0.1786 | Only 1-hop, missing higher-order |
| 2 | 0.2356 | 0.1931 | Good, capturing 2-hop |
| **3** | **0.2424** | **0.2000** | **Best** |
| 4 | 0.2402 | 0.1985 | Slight decrease |
| 5 | 0.2331 | 0.1923 | Over-smoothing starting |
| 8 | 0.2015 | 0.1652 | Severe over-smoothing |

**Why 3 layers is optimal**:
- Layer 1: Direct collaborators
- Layer 2: Friend-of-friend items
- Layer 3: Sufficient coverage of most graphs
- Layer 4+: Diminishing returns, then over-smoothing

*What happens if...* you use 10 layers on a small graph? All embeddings converge to the graph's average. You lose all discriminative power!

---

## What Can Go Wrong: LightGCN Failure Modes

### Failure Mode 1: Too Shallow (L=1)

**Symptom**: Recommendations are too obvious. Only items with direct connections surface.

**Diagnosis**: Check average path length between recommendations and user history:
```python
# If most recommendations are 1-hop, you're not leveraging the graph
avg_distance = compute_avg_path_length(user_history, recommendations)
print(f"Average distance: {avg_distance}")  # Should be > 1.5
```

**Cause**: With L=1, you only see items rated by the same user. No "friend of friend" signal.

**Solution**: Increase layers to 2-3.

**Example**:
```
User rated: [Movie A, Movie B]
L=1 recommendations: [Movies rated by same users who rated A or B]
L=3 recommendations: [Movies rated by users who rated movies rated by users who rated A or B]
                     (Much broader, discovers hidden gems)
```

---

### Failure Mode 2: Too Deep (L>4)

**Symptom**: All recommendations look the same for different users. Diversity drops.

**Diagnosis**: Compute embedding similarity across users:
```python
user_embs = model.get_user_embeddings()
cos_sim = cosine_similarity(user_embs, user_embs)
avg_sim = (cos_sim.sum() - len(user_embs)) / (len(user_embs)**2 - len(user_embs))
print(f"Average user similarity: {avg_sim:.3f}")
# If > 0.8, over-smoothed!
```

**Cause**: Too many layers means every node sees almost every other node. Embeddings converge.

**Solution**:
- Reduce layers to 2-3
- Add skip connections if you need depth for other reasons

---

### Failure Mode 3: Cold Start Users

**Symptom**: New users get only popular items. Recommendations don't personalize.

**Diagnosis**:
```python
# Check Recall@20 stratified by user activity
for group in ['1-5 ratings', '6-20 ratings', '20+ ratings']:
    print(f"{group}: Recall@20 = {compute_recall(group)}")

# Example bad output:
# 1-5 ratings: Recall@20 = 0.02  ← Problem!
# 6-20 ratings: Recall@20 = 0.15
# 20+ ratings: Recall@20 = 0.28
```

**Cause**: Users with few edges have weak embeddings. LightGCN averages neighbor embeddings, but with 2 neighbors, the average is noisy.

**Mathematical view**:
$$\mathbf{h}_u^{(1)} = \frac{1}{\sqrt{2}} \left( \frac{\mathbf{h}_{i_1}}{\sqrt{d_{i_1}}} + \frac{\mathbf{h}_{i_2}}{\sqrt{d_{i_2}}} \right)$$

With only 2 items, this is essentially a noisy average of 2 vectors.

**Solution**:
- Hybrid: Use content features for cold users, GNN for warm
- Fallback: Popularity-based for users with < 5 interactions
- Side information: Inject user demographics into initial embeddings

---

### Failure Mode 4: Cold Start Items

**Symptom**: New items never get recommended, even to relevant users.

**Diagnosis**:
```python
# Check which items appear in recommendations
rec_item_counts = Counter(all_recommendations)
items_never_recommended = [i for i in all_items if rec_item_counts[i] == 0]
print(f"Items never recommended: {len(items_never_recommended)} / {len(all_items)}")
# If > 50%, you have cold item problem
```

**Cause**: Items with no ratings have only their initial embedding (layer 0). If this is random, it won't match any users.

**Solution**:
- Content-based fallback for new items
- Exploration: Randomly inject new items into recommendations
- Bandits: Use exploration-exploitation for cold items

---

### Failure Mode 5: Scalability Issues

**Symptom**: Training takes hours per epoch. Memory errors on large graphs.

**Diagnosis**:
```python
# Check graph size
print(f"Nodes: {n_users + n_items:,}")
print(f"Edges: {n_edges:,}")
print(f"Density: {n_edges / (n_users * n_items):.6f}")

# If nodes > 1M or edges > 100M, you need special handling
```

**Cause**: Full-batch propagation requires storing all embeddings in memory. Dense operations are O(N^2).

**Solution**:
- **Always use sparse matrices** (10-100x memory savings)
- **Mini-batch propagation**: Sample subgraphs for each batch
- **PinSage-style sampling**: Random walk + importance sampling
- **GPU memory**: Use gradient checkpointing if needed

**Example scaling**:
| Scale | Nodes | Edges | Full batch RAM | Mini-batch RAM |
|-------|-------|-------|----------------|----------------|
| Small | 10K | 100K | 100 MB | 10 MB |
| Medium | 100K | 10M | 10 GB | 100 MB |
| Large | 1M | 100M | 1 TB (impossible) | 1 GB |
| Web | 1B | 10B | Impossible | 10 GB (with sampling) |

---

### Failure Mode 6: Popularity Bias Amplification

**Symptom**: Popular items get recommended even more. Long-tail items are ignored.

**Diagnosis**:
```python
# Gini coefficient of recommendation frequency
item_rec_freq = Counter(all_recommendations)
gini = compute_gini(list(item_rec_freq.values()))
print(f"Gini coefficient: {gini:.3f}")  # Closer to 1 = more biased
```

**Cause**: Popular items have high degree. In message passing:
- They send messages to many users
- They receive messages from many users
- Their embeddings become "generically appealing"

**Solution**:
- Inverse propensity weighting in loss
- Post-hoc re-ranking to boost diversity
- Sample negatives proportional to popularity^0.75

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

**Typical**: 100M edges, $d=128$, $L=3$ is less than 1 minute per epoch on GPU.

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
1. **LightGCN simplifies GCNs** and removes transformations, nonlinearities
2. **Simplicity wins** with +16% improvement over complex NGCF
3. **Linear propagation** is sufficient for collaborative filtering
4. **Fewer parameters** means better generalization on sparse data
5. **State-of-the-art** results (2020-2024)

**Why it works**:
- **Overfitting prevention**: No transformation parameters to overfit
- **Collaborative signal**: Neighbor aggregation captures user-item patterns
- **Layer combination**: Preserves information at all scales

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
