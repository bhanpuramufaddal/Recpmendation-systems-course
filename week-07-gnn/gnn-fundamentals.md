# Week 7: Graph Neural Networks - GNN Fundamentals

## The Opening Question: Why Can't Matrix Factorization See the Whole Picture?

*Let's start with a puzzle that should bother you.*

**The Setup:**
```
Alice rates: Inception (5), Interstellar (5), The Matrix (4)
Bob rates:   Inception (5), Interstellar (5), Fight Club (5)
Carol rates: Fight Club (5), Pulp Fiction (5), Se7en (5)
```

**Matrix Factorization's view:**
- Alice and Bob are similar (both love Inception, Interstellar)
- Bob and Carol are similar (both love Fight Club)
- Alice and Carol? MF sees **NO direct connection**

**But wait!** Through Bob, Alice is connected to Carol. And Carol loves dark psychological thrillers. Maybe Alice would like Se7en too?

**MF's prediction for Alice → Se7en:** Low confidence (no direct signal)

**The Multi-Hop Connection:**
```
Alice ←→ Bob ←→ Carol → Se7en
        (2 hops away)
```

*This is the "friend of a friend" pattern. MF completely misses it.*

**Let's put numbers to this failure:**

| Approach | How it sees Alice-Carol | Alice→Se7en prediction |
|----------|-------------------------|------------------------|
| **MF** | No connection | 2.1 (low - random guess) |
| **2-hop GNN** | Connected via Bob | 4.2 (high - propagated signal) |

**The gap is 2.1 vs 4.2!** That's the difference between "not recommended" and "top-10 recommendation."

*Can you see why exploiting graph structure matters?*

---

## Overview

**Graph Neural Networks (GNNs)** extend deep learning to graph-structured data by performing **neighborhood aggregation**: each node learns from its neighbors.

**Key idea**: Node's representation = function of its features + neighbors' features.

**Breakthrough for RecSys**: GNNs achieve state-of-the-art results by exploiting user-item graph structure.

**Notable models**:
- **GCN** (Kipf & Welling, 2017): Graph Convolutional Networks
- **GraphSAGE** (Hamilton et al., 2017): Inductive learning
- **GAT** (Velickovic et al., 2018): Attention-based aggregation
- **PinSage** (Ying et al., 2018): Pinterest's billion-scale GNN
- **LightGCN** (He et al., 2020): Simplified GNN for RecSys

This document covers GNN fundamentals for recommendation systems.

---

## Learning Objectives

By the end of this section, you will:
- Understand GNN architecture and neighborhood aggregation
- Master GCN, GraphSAGE, and GAT
- Implement LightGCN for collaborative filtering
- Apply GNNs to billion-scale recommendation (PinSage)
- Compare GNNs with traditional methods

---

## From Graphs to GNNs

### Limitation of Traditional Graph Methods

**DeepWalk, Node2Vec**:
- Random walks then embeddings
- **Problem**: Fixed embeddings (can't generalize to new nodes)

**Matrix Factorization**:
- User/item embeddings
- **Problem**: Only captures direct connections, misses multi-hop relationships

*Notice that* MF learns $\mathbf{p}_u$ and $\mathbf{q}_i$ independently. The embedding of Alice doesn't "know" about Carol's preferences, even though Bob connects them!

**GNN Solution**: Learn node embeddings by aggregating neighbor information **recursively**.

---

### GNN Paradigm: Nodes Talk to Their Neighbors

**The Core Intuition:**

Imagine a room full of people at a party. Each person only knows their immediate friends. Now we play a game:

*Round 1:* Everyone tells their friends what movies they like.
*Round 2:* Everyone summarizes what they heard and tells their friends again.
*Round 3:* Same thing.

After 3 rounds, what does Alice know?
- Round 1: Her own preferences + what Bob told her
- Round 2: Bob's summary (which includes Carol's info!)
- Round 3: Information from people 3 connections away

**This is message passing.** Each round is a GNN layer.

---

### The GNN Update Rule

**Goal**: Learn node embedding $\mathbf{h}_v$ for each node $v$.

**Process** (layer-wise):
$$\mathbf{h}_v^{(k)} = \sigma\left(\mathbf{W}^{(k)} \cdot \text{AGG}^{(k)}\left(\{\mathbf{h}_u^{(k-1)} : u \in N(v)\}\right)\right)$$

**Let me break down each term:**

| Term | What it does | Intuition |
|------|--------------|-----------|
| $\mathbf{h}_v^{(k)}$ | Embedding of node $v$ at layer $k$ | "What Alice knows after $k$ rounds" |
| $N(v)$ | Neighbors of node $v$ | "Alice's direct friends" |
| $\text{AGG}$ | Aggregation function (mean, sum, max, attention) | "How to combine friends' messages" |
| $\mathbf{W}^{(k)}$ | Learnable weight matrix | "What information to extract" |
| $\sigma$ | Activation (ReLU, etc.) | "Nonlinearity for expressiveness" |

**Why each component?**
- **AGG**: We need to combine multiple neighbors into one vector (order-invariant)
- **W**: Different information matters at different depths
- **$\sigma$**: Without it, $k$ layers collapse to 1 linear layer

*Can you see why* the choice of AGG matters? Mean aggregation treats all neighbors equally. Max picks the strongest signal. Attention learns which neighbors matter most.

---

## Graph Convolutional Network (GCN)

### Architecture

**Paper**: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017)

**Key idea**: Convolve over graph structure (like CNN over images).

**Layer update**:
$$\mathbf{H}^{(k+1)} = \sigma(\tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \mathbf{H}^{(k)} \mathbf{W}^{(k)})$$

where:
- $\mathbf{H}^{(k)}$ = node features at layer $k$ (matrix, all nodes)
- $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ = adjacency matrix + self-loops
- $\tilde{\mathbf{D}}$ = degree matrix of $\tilde{\mathbf{A}}$
- $\mathbf{W}^{(k)}$ = learnable weights

---

### Deriving the Normalization: Why $\tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2}$?

*Let me show you step-by-step why this normalization is necessary. We'll derive it by fixing problems one at a time.*

**Step 1: Naive aggregation (just sum neighbors)**

$$\mathbf{h}_v^{(k+1)} = \sum_{u \in N(v)} \mathbf{h}_u^{(k)}$$

*What happens if...* we have nodes with very different degrees?

```
Node A: 2 neighbors  → aggregated sum = 2 × (average embedding)
Node B: 100 neighbors → aggregated sum = 100 × (average embedding)
```

Node B's representation is **50x larger** just because it has more neighbors!

After a few layers, high-degree nodes explode in magnitude. Training becomes unstable.

**Symptom**: Loss goes to NaN. Embeddings overflow.

---

**Step 2: Normalize by degree (divide by neighbor count)**

$$\mathbf{h}_v^{(k+1)} = \frac{1}{|N(v)|} \sum_{u \in N(v)} \mathbf{h}_u^{(k)}$$

This is **row normalization**: $\tilde{\mathbf{D}}^{-1} \tilde{\mathbf{A}}$

*Better, but what goes wrong now?*

Consider:
- Node A has 2 neighbors, each with degree 100 (connected to hubs)
- Node B has 2 neighbors, each with degree 2 (connected to specialists)

Both A and B average over 2 neighbors, but:
- A's neighbors are "hubs" — their information is spread across 100 connections, diluted
- B's neighbors are "specialists" — their information is focused, meaningful

*We're treating a message from a hub the same as a message from a specialist. That's wrong!*

---

**Step 3: Symmetric normalization (the GCN solution)**

$$\mathbf{h}_v^{(k+1)} = \sum_{u \in N(v)} \frac{1}{\sqrt{|N(v)|} \cdot \sqrt{|N(u)|}} \mathbf{h}_u^{(k)}$$

**The factor $\frac{1}{\sqrt{|N(v)|} \cdot \sqrt{|N(u)|}}$ means:**

- Divide by $\sqrt{|N(v)|}$: "I have many neighbors, so each contributes less to MY representation"
- Divide by $\sqrt{|N(u)|}$: "This neighbor is very popular, so THEIR signal is diluted"

**In matrix form**: $\tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2}$

*Why square root?* It preserves the **spectral properties** of the graph Laplacian, making training more stable. (Deep math reason: eigenvalues stay in [-1, 1].)

*Notice that* this is the geometric mean of the two normalizations!

---

### Numerical Walkthrough: 4-Node GCN Propagation

*Let's trace through exactly what happens with actual numbers.*

**Our Graph:**
```
    [0]----[1]
     |      |
     |      |
    [2]----[3]
```

**Adjacency Matrix (with self-loops):**
```
      0   1   2   3
  0 [ 1   1   1   0 ]   degree = 3
  1 [ 1   1   0   1 ]   degree = 3
  2 [ 1   0   1   1 ]   degree = 3
  3 [ 0   1   1   1 ]   degree = 3
```

All nodes have degree 3 (including self-loop).

**Initial Embeddings (d=2):**
```
h_0^(0) = [1.0, 0.0]   (Node 0: "action lover")
h_1^(0) = [0.0, 1.0]   (Node 1: "comedy lover")
h_2^(0) = [1.0, 1.0]   (Node 2: "likes both")
h_3^(0) = [0.0, 0.0]   (Node 3: "cold start user")
```

**Normalization Factor:**
Since all degrees are 3: $\frac{1}{\sqrt{3} \cdot \sqrt{3}} = \frac{1}{3}$

---

**Layer 1 Propagation (no weight matrix for clarity):**

**Node 0** aggregates from neighbors {0, 1, 2}:
$$\mathbf{h}_0^{(1)} = \frac{1}{3}[1.0, 0.0] + \frac{1}{3}[0.0, 1.0] + \frac{1}{3}[1.0, 1.0] = [\frac{2}{3}, \frac{2}{3}] = [0.67, 0.67]$$

**Node 1** aggregates from neighbors {0, 1, 3}:
$$\mathbf{h}_1^{(1)} = \frac{1}{3}[1.0, 0.0] + \frac{1}{3}[0.0, 1.0] + \frac{1}{3}[0.0, 0.0] = [\frac{1}{3}, \frac{1}{3}] = [0.33, 0.33]$$

**Node 2** aggregates from neighbors {0, 2, 3}:
$$\mathbf{h}_2^{(1)} = \frac{1}{3}[1.0, 0.0] + \frac{1}{3}[1.0, 1.0] + \frac{1}{3}[0.0, 0.0] = [\frac{2}{3}, \frac{1}{3}] = [0.67, 0.33]$$

**Node 3** aggregates from neighbors {1, 2, 3}:
$$\mathbf{h}_3^{(1)} = \frac{1}{3}[0.0, 1.0] + \frac{1}{3}[1.0, 1.0] + \frac{1}{3}[0.0, 0.0] = [\frac{1}{3}, \frac{2}{3}] = [0.33, 0.67]$$

**After Layer 1:**
```
h_0^(1) = [0.67, 0.67]  ← Picked up comedy from Node 1
h_1^(1) = [0.33, 0.33]  ← More balanced now
h_2^(1) = [0.67, 0.33]  ← Still action-leaning
h_3^(1) = [0.33, 0.67]  ← COLD START NODE GOT PREFERENCES!
```

*Can you see what happened?* Node 3 started with [0, 0] (no preferences), but after one layer, it has [0.33, 0.67] — it learned that its neighbors (1 and 2) lean toward comedy!

---

**Layer 2 Propagation:**

**Node 0** aggregates from new embeddings:
$$\mathbf{h}_0^{(2)} = \frac{1}{3}[0.67, 0.67] + \frac{1}{3}[0.33, 0.33] + \frac{1}{3}[0.67, 0.33]$$
$$= [\frac{0.67 + 0.33 + 0.67}{3}, \frac{0.67 + 0.33 + 0.33}{3}] = [0.56, 0.44]$$

*Notice that* Node 0 now has information from Node 3 (via Node 1 and Node 2). The 2-hop connection is captured!

**After Layer 2:**
```
h_0^(2) = [0.56, 0.44]
h_1^(2) = [0.44, 0.56]
h_2^(2) = [0.56, 0.56]
h_3^(2) = [0.44, 0.44]
```

*What happens if...* we keep going? All embeddings will converge to the same value! This is **over-smoothing**, which we'll discuss later.

---

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, A_norm):
        """
        X: (N, in_features) - node features
        A_norm: (N, N) - normalized adjacency matrix
        """
        # Aggregate from neighbors
        aggregated = torch.mm(A_norm, X)  # (N, in_features)

        # Linear transformation
        output = self.linear(aggregated)  # (N, out_features)

        return output


class GCN(nn.Module):
    def __init__(self, n_features, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.gc1 = GCNLayer(n_features, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, X, A_norm):
        # Layer 1
        h = self.gc1(X, A_norm)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)

        # Layer 2
        output = self.gc2(h, A_norm)

        return output


# Example
N = 100  # Number of nodes
n_features = 50  # Initial features per node
hidden_dim = 128
output_dim = 64

# Random node features
X = torch.randn(N, n_features)

# Random adjacency matrix (sparse, symmetric)
A = torch.randint(0, 2, (N, N), dtype=torch.float32)
A = (A + A.T) / 2  # Make symmetric

# Add self-loops
A_tilde = A + torch.eye(N)

# Compute degree matrix
D_tilde = torch.diag(A_tilde.sum(dim=1))

# Normalized adjacency
D_tilde_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(D_tilde)))
A_norm = torch.mm(torch.mm(D_tilde_inv_sqrt, A_tilde), D_tilde_inv_sqrt)

# GCN model
model = GCN(n_features, hidden_dim, output_dim)
embeddings = model(X, A_norm)

print(f"Node embeddings shape: {embeddings.shape}")  # (100, 64)
```

---

## GraphSAGE

### Architecture

**Paper**: Hamilton et al., "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)

**Key innovation**: **Sampling** neighbors (don't use all neighbors).

**Motivation**: Large graphs have nodes with 1000s of neighbors. Aggregating all is expensive.

**Process**:
1. **Sample**: Sample $K$ neighbors per node
2. **Aggregate**: Aggregate sampled neighbors' features
3. **Update**: Combine with own features

**Aggregators**:
- **Mean**: $\text{AGG} = \frac{1}{|N(v)|} \sum_{u \in N(v)} \mathbf{h}_u$
- **Max**: $\text{AGG} = \max_{u \in N(v)} \mathbf{h}_u$
- **LSTM**: Apply LSTM to neighbor sequence

*Why sample?* Consider a celebrity with 1M followers. Aggregating all 1M is O(1M). Sampling 10 is O(10). Similar information, 100,000x faster!

---

### Implementation

```python
class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features, aggregator='mean'):
        super().__init__()
        self.aggregator = aggregator

        # Transform aggregated neighbors
        self.linear_neigh = nn.Linear(in_features, out_features)

        # Transform self
        self.linear_self = nn.Linear(in_features, out_features)

    def forward(self, X, sampled_neighbors):
        """
        X: (N, in_features)
        sampled_neighbors: list of lists, sampled_neighbors[i] = neighbor indices for node i
        """
        N = X.size(0)
        aggregated = torch.zeros(N, X.size(1))

        # Aggregate from sampled neighbors
        for i in range(N):
            neighbors = sampled_neighbors[i]
            if len(neighbors) > 0:
                if self.aggregator == 'mean':
                    aggregated[i] = X[neighbors].mean(dim=0)
                elif self.aggregator == 'max':
                    aggregated[i] = X[neighbors].max(dim=0)[0]

        # Combine neighbor aggregation + self
        h_neigh = self.linear_neigh(aggregated)
        h_self = self.linear_self(X)

        # Concatenate and normalize
        output = h_neigh + h_self
        output = F.normalize(output, p=2, dim=1)  # L2 normalization

        return output


# Example: Sample K neighbors per node
def sample_neighbors(adjacency_list, K=10):
    """Sample K neighbors for each node."""
    sampled = []
    for node, neighbors in enumerate(adjacency_list):
        if len(neighbors) > K:
            sampled.append(random.sample(neighbors, K))
        else:
            sampled.append(neighbors)
    return sampled


# adjacency_list = {0: [1, 2, 3], 1: [0, 2], ...}
# sampled_neighbors = sample_neighbors(adjacency_list, K=10)
# model = GraphSAGELayer(in_features=50, out_features=128, aggregator='mean')
# embeddings = model(X, sampled_neighbors)
```

**Advantage**: Inductive (can generalize to unseen nodes).

---

## Graph Attention Network (GAT)

### Architecture

**Paper**: Velickovic et al., "Graph Attention Networks" (ICLR 2018)

**Key idea**: **Attention** mechanism to weight neighbors.

*Not all neighbors are equally important!* Your best friend's movie opinion matters more than a random acquaintance's.

**Attention weight**:
$$\alpha_{uv} = \frac{\exp(e_{uv})}{\sum_{k \in N(v)} \exp(e_{vk})}$$

where $e_{uv} = \text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_u || \mathbf{W}\mathbf{h}_v])$.

**Aggregation**:
$$\mathbf{h}_v' = \sigma\left(\sum_{u \in N(v)} \alpha_{uv} \mathbf{W} \mathbf{h}_u\right)$$

**Benefit**: Automatically learns which neighbors are important.

---

### Step-by-Step GAT Calculation with 3 Nodes

*Let me walk through the attention mechanism with concrete numbers.*

**Graph**:
```
    Node A --- Node B
       \        /
        \      /
         Node C
```

Node B has neighbors {A, C}. Let's compute B's new representation.

**Initial embeddings** (d=2 for simplicity):
```
h_A = [0.5, 0.3]
h_B = [0.8, 0.2]
h_C = [0.1, 0.9]
```

**Step 1: Transform embeddings** (W is 2x2 identity for simplicity)
```
Wh_A = [0.5, 0.3]
Wh_B = [0.8, 0.2]
Wh_C = [0.1, 0.9]
```

**Step 2: Compute attention scores for B's neighbors**

For edge A to B: Concatenate [Wh_A || Wh_B] = [0.5, 0.3, 0.8, 0.2]
For edge C to B: Concatenate [Wh_C || Wh_B] = [0.1, 0.9, 0.8, 0.2]

**Attention vector** $\mathbf{a}$ (learned, assume $\mathbf{a} = [0.2, 0.1, 0.3, 0.4]$):

$$e_{AB} = \text{LeakyReLU}(\mathbf{a}^T [0.5, 0.3, 0.8, 0.2])$$
$$= \text{LeakyReLU}(0.2 \cdot 0.5 + 0.1 \cdot 0.3 + 0.3 \cdot 0.8 + 0.4 \cdot 0.2)$$
$$= \text{LeakyReLU}(0.1 + 0.03 + 0.24 + 0.08) = \text{LeakyReLU}(0.45) = 0.45$$

$$e_{CB} = \text{LeakyReLU}(\mathbf{a}^T [0.1, 0.9, 0.8, 0.2])$$
$$= \text{LeakyReLU}(0.02 + 0.09 + 0.24 + 0.08) = \text{LeakyReLU}(0.43) = 0.43$$

**Step 3: Softmax to get attention weights**

$$\alpha_{AB} = \frac{e^{0.45}}{e^{0.45} + e^{0.43}} = \frac{1.568}{1.568 + 1.537} = \frac{1.568}{3.105} \approx 0.505$$

$$\alpha_{CB} = \frac{e^{0.43}}{e^{0.45} + e^{0.43}} \approx 0.495$$

*Both neighbors get roughly equal attention (they're similarly relevant).*

**Step 4: Weighted aggregation**

$$\mathbf{h}_B' = \sigma(0.505 \cdot [0.5, 0.3] + 0.495 \cdot [0.1, 0.9])$$
$$= \sigma([0.253 + 0.050, 0.152 + 0.446])$$
$$= \sigma([0.303, 0.598])$$

*B's new representation is a weighted mix of A and C's features.*

---

### Implementation

```python
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.dropout = dropout
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, X, A):
        """
        X: (N, in_features)
        A: (N, N) - adjacency matrix
        """
        # Transform features
        Wh = self.W(X)  # (N, out_features)

        # Compute attention scores
        N = Wh.size(0)
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1)  # (N*N, 2*out_features)
        e = self.leakyrelu(torch.mm(a_input, self.a).squeeze(1))  # (N*N,)
        e = e.view(N, N)  # (N, N)

        # Mask non-neighbors
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(A > 0, e, zero_vec)

        # Softmax
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Aggregate
        h_prime = torch.mm(attention, Wh)  # (N, out_features)

        return F.elu(h_prime)


# Example
model_gat = GATLayer(in_features=50, out_features=64)
embeddings_gat = model_gat(X, A)
print(f"GAT embeddings shape: {embeddings_gat.shape}")
```

**Advantage**: Learns importance of each neighbor (interpretable).

---

## The Over-Smoothing Problem: When GNNs Go Wrong

*What happens if we stack too many GNN layers?*

### The Problem

**The issue**: After many layers, ALL nodes have seen information from ALL other nodes.

**Layer-by-layer view**:
```
Layer 0: Node knows itself
Layer 1: Node knows 1-hop neighbors
Layer 2: Node knows 2-hop neighbors (neighbors of neighbors)
...
Layer 5: Node knows 5-hop neighbors (almost everyone in a small graph!)
```

**The math**: After enough layers, every node's representation becomes:
$$\mathbf{h}_v \approx \text{weighted average of all nodes in the graph}$$

*All node embeddings converge to the same vector!*

---

### Numerical Demonstration of Over-Smoothing

*Let me show you exactly how embeddings converge.*

**Starting embeddings (4 nodes, d=2):**
```
Layer 0:
  Node 0: [1.0, 0.0]  ■□
  Node 1: [0.0, 1.0]  □■
  Node 2: [0.5, 0.5]  ◧
  Node 3: [0.8, 0.2]  ■□
```

**After 2 layers** (connected graph, symmetric normalization):
```
Layer 2:
  Node 0: [0.58, 0.42]  ◧
  Node 1: [0.52, 0.48]  ◧
  Node 2: [0.55, 0.45]  ◧
  Node 3: [0.57, 0.43]  ◧
```

*Notice that* nodes are becoming more similar!

**After 6 layers:**
```
Layer 6:
  Node 0: [0.575, 0.425]  ○
  Node 1: [0.575, 0.425]  ○
  Node 2: [0.575, 0.425]  ○
  Node 3: [0.575, 0.425]  ○
```

**All nodes are now IDENTICAL!** We've lost all discriminative power.

**The intuition**: It's like playing "telephone" across the graph. After enough rounds, everyone has heard a garbled mix of everyone's original message.

---

### Measuring Over-Smoothing

**Mean Average Distance (MAD):**
$$\text{MAD} = \frac{1}{N^2} \sum_{i,j} \|\mathbf{h}_i - \mathbf{h}_j\|_2$$

| Layers | MAD | Interpretation |
|--------|-----|----------------|
| 1 | 0.82 | Nodes are distinct |
| 2 | 0.61 | Some smoothing, still good |
| 3 | 0.43 | Starting to converge |
| 4 | 0.25 | Losing information |
| 6 | 0.08 | Severely over-smoothed |
| 10 | 0.01 | Essentially identical |

*Can you see why* 2-3 layers is the sweet spot?

---

### Solutions to Over-Smoothing

**1. Use fewer layers** (2-3 is usually optimal)

**2. Skip connections** (preserve original information):
$$\mathbf{h}_v^{(k+1)} = \mathbf{h}_v^{(k+1)} + \mathbf{h}_v^{(k)}$$

**3. JKNet** (Jumping Knowledge): Concatenate ALL layer outputs, let the model choose:
$$\mathbf{h}_v^{\text{final}} = [\mathbf{h}_v^{(0)} || \mathbf{h}_v^{(1)} || ... || \mathbf{h}_v^{(K)}]$$

**4. DropEdge**: Randomly drop edges during training to slow down message diffusion

---

## PinSage (Pinterest)

### Architecture

**Paper**: Ying et al., "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (KDD 2018)

**Use case**: Recommend pins (images) on Pinterest.

**Scale**: 3 billion nodes (pins), 18 billion edges.

**Key innovations**:
1. **Importance pooling**: Weighted aggregation (not uniform)
2. **Efficient sampling**: Sample neighbors by random walk + importance
3. **MapReduce training**: Distributed training on graph

**Performance**: 150%+ improvement over prior methods.

---

### Importance Pooling

**Problem**: Not all neighbors are equally important.

**Solution**: Weight neighbors by importance.

**Importance**:
$$w_{uv} = \frac{|N(u) \cap N(v)|}{|N(u)|}$$

(Jaccard similarity between neighborhoods)

**Aggregation**:
$$\mathbf{h}_v = \text{ReLU}\left(\mathbf{W} \cdot \sum_{u \in N(v)} \frac{w_{uv}}{\sum_{k} w_{kv}} \mathbf{h}_u\right)$$

---

## LightGCN (State-of-the-Art for RecSys)

### Architecture

**Paper**: He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" (SIGIR 2020)

**Key insight**: For RecSys, **simpler is better**. Remove feature transformation and activation.

**Layer update** (simplified):
$$\mathbf{h}_v^{(k)} = \sum_{u \in N(v)} \frac{1}{\sqrt{|N(v)|} \sqrt{|N(u)|}} \mathbf{h}_u^{(k-1)}$$

**Final embedding**: Weighted sum of all layers.

$$\mathbf{h}_v = \sum_{k=0}^K \alpha_k \mathbf{h}_v^{(k)}$$

where $\alpha_k = \frac{1}{K+1}$ (uniform weighting).

**Benefits**:
- Fewer parameters (faster training)
- Better performance (state-of-the-art on multiple benchmarks)

---

### Implementation

```python
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Initialize
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, user_ids, item_ids, graph):
        """
        user_ids: (batch,)
        item_ids: (batch,)
        graph: adjacency matrix (normalized)
        """
        # Initial embeddings
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)  # (N_total, dim)

        embeddings_layers = [all_embeddings]

        # Propagate through layers
        for k in range(self.n_layers):
            all_embeddings = torch.sparse.mm(graph, all_embeddings)
            embeddings_layers.append(all_embeddings)

        # Aggregate (mean)
        final_embeddings = torch.stack(embeddings_layers, dim=0).mean(dim=0)

        # Split back into users and items
        user_embs, item_embs = torch.split(final_embeddings, [self.n_users, self.n_items])

        # Get embeddings for batch
        u_emb = user_embs[user_ids]  # (batch, dim)
        i_emb = item_embs[item_ids]  # (batch, dim)

        # Dot product
        scores = (u_emb * i_emb).sum(dim=1)

        return scores


# Example
n_users = 1000
n_items = 500
model_lgcn = LightGCN(n_users, n_items, embedding_dim=64, n_layers=3)

# Sample batch
user_ids = torch.randint(0, n_users, (32,))
item_ids = torch.randint(0, n_items, (32,))

# Need to construct normalized graph (adjacency matrix)
# ... (graph construction code)

# scores = model_lgcn(user_ids, item_ids, graph)
```

---

## Training GNNs for RecSys

### Loss Function

**BPR Loss** (Bayesian Personalized Ranking):
$$\mathcal{L}_{\text{BPR}} = -\sum_{(u, i, j)} \log \sigma(\hat{y}_{ui} - \hat{y}_{uj})$$

where:
- $i$ = positive item (user interacted)
- $j$ = negative item (user didn't interact)

---

### Training Loop

```python
import torch.optim as optim

def train_lightgcn(model, graph, interactions, n_epochs=100, lr=0.001, batch_size=1024):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        # Mini-batch training
        for batch in create_batches(interactions, batch_size):
            user_ids, pos_items, neg_items = batch

            # Positive scores
            pos_scores = model(user_ids, pos_items, graph)

            # Negative scores
            neg_scores = model(user_ids, neg_items, graph)

            # BPR loss
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model
```

---

## What Can Go Wrong: GNN Failure Modes

### Failure Mode 1: Over-Smoothing

**Symptom**: All embeddings look the same after training. Recommendations are nearly random.

**Diagnosis**: Check embedding diversity:
```python
# Compute pairwise cosine similarity
similarities = cosine_similarity(embeddings)
avg_sim = (similarities.sum() - len(embeddings)) / (len(embeddings)**2 - len(embeddings))
print(f"Average pairwise similarity: {avg_sim:.3f}")
# If > 0.9, you have over-smoothing
```

**Cause**: Too many layers (>3-4 on small graphs).

**Solution**:
- Reduce layers to 2-3
- Add skip connections
- Use DropEdge during training

---

### Failure Mode 2: Overfitting on Small Graphs

**Symptom**: Training loss drops to near-zero, but validation metrics get worse.

**Diagnosis**:
```
Epoch 10:  Train Loss: 0.42, Val Recall@20: 0.18
Epoch 50:  Train Loss: 0.05, Val Recall@20: 0.21
Epoch 100: Train Loss: 0.01, Val Recall@20: 0.15  ← Getting worse!
```

**Cause**: Too many parameters for sparse data. GNNs with transformations overfit.

**Solution**:
- Use LightGCN (no feature transformations)
- Stronger L2 regularization ($\lambda = 10^{-3}$ instead of $10^{-5}$)
- Early stopping based on validation metrics
- Dropout on embeddings

---

### Failure Mode 3: Degree Bias

**Symptom**: Popular items always recommended. Long-tail items never surface.

**Diagnosis**: Check recommendation frequency distribution:
```python
# Count how often each item is recommended
rec_counts = Counter(all_recommendations)
top_10_items = rec_counts.most_common(10)
# If top 10 items account for 80%+ of recommendations, you have degree bias
```

**Cause**: High-degree nodes (popular items) dominate message passing. Their embeddings are "averaged" across many users, making them generically appealing.

**Solution**:
- Use symmetric normalization (already in GCN/LightGCN)
- Re-weight loss to penalize easy positive examples
- Post-hoc re-ranking to boost diversity

---

### Failure Mode 4: Cold Start Collapse

**Symptom**: New users/items get random or popularity-based recommendations only.

**Diagnosis**: Filter metrics by user activity level:
```
Users with 50+ interactions: Recall@20 = 0.28
Users with 10-50 interactions: Recall@20 = 0.19
Users with 1-10 interactions: Recall@20 = 0.04  ← Problem!
```

**Cause**: GNN embeddings are defined by neighbors. Nodes with few/no neighbors have poor embeddings.

**Solution**:
- Hybrid approach: Use content features for cold users, GNN for warm
- Fallback to popularity for cold start
- Meta-learning for few-shot users
- Side information (user demographics, item features) in initial embeddings

---

## Comparison of GNN Models

| Model | Aggregation | Advantage | Disadvantage | Use Case |
|-------|-------------|-----------|--------------|----------|
| **GCN** | Mean (all neighbors) | Simple, effective | Not scalable (all neighbors) | Small/medium graphs |
| **GraphSAGE** | Sampling + Mean/Max/LSTM | Inductive, scalable | Sampling may miss info | Large graphs, new nodes |
| **GAT** | Attention-weighted | Learns importance | More expensive | Interpretability needed |
| **PinSage** | Importance pooling | Billion-scale | Complex implementation | Web-scale systems |
| **LightGCN** | Simplified (no transform) | State-of-the-art for RecSys | Only for RecSys | Collaborative filtering |

**Recommendation**: Start with **LightGCN** for RecSys (best accuracy, simplest).

---

## Real-World Applications

### 1. Pinterest (PinSage)

**Scale**: 3B pins, 18B edges
**Task**: Recommend related pins
**Result**: 150%+ improvement, powers 40% of engagement

---

### 2. Alibaba

**Use case**: Product recommendations
**Approach**: GNN on user-item-product knowledge graph
**Result**: 10%+ CTR improvement

---

### 3. Twitter

**Use case**: Who to follow
**Approach**: GraphSAGE on user-user graph
**Result**: 15%+ increase in follows

---

## Summary

**Key Takeaways**:
1. **GNNs**: Learn node embeddings via neighborhood aggregation
2. **GCN**: Convolve over graph (all neighbors)
3. **GraphSAGE**: Sample neighbors (scalable, inductive)
4. **GAT**: Attention-based aggregation (interpretable)
5. **PinSage**: Billion-scale GNN (importance pooling)
6. **LightGCN**: State-of-the-art for RecSys (simplified)

**Best Practices**:
- Use **LightGCN** for collaborative filtering
- Use **GraphSAGE** for large, dynamic graphs
- Use **GAT** when interpretability matters
- 2-3 layers sufficient (over-smoothing beyond that)

**When to use GNNs**:
- Rich graph structure (social, knowledge graphs)
- Need to leverage neighbors' information
- State-of-the-art accuracy required

**Next**: Week 8: Two-tower models and retrieval at scale.

---

## References

1. **Kipf, T. N., & Welling, M. (2017)**. "Semi-Supervised Classification with Graph Convolutional Networks". *ICLR*.
   - **GCN** paper

2. **Hamilton, W. L., Ying, R., & Leskovec, J. (2017)**. "Inductive Representation Learning on Large Graphs". *NeurIPS*.
   - **GraphSAGE**

3. **Velickovic, P., et al. (2018)**. "Graph Attention Networks". *ICLR*.
   - **GAT**

4. **Ying, R., et al. (2018)**. "Graph Convolutional Neural Networks for Web-Scale Recommender Systems". *KDD*.
   - **PinSage** (Pinterest)

5. **He, X., et al. (2020)**. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation". *SIGIR*.
   - **LightGCN** (state-of-the-art)

---

## Practice Problems

### Problem 1: GCN Layer

**Given**:
```
Node features X: (4, 3) - 4 nodes, 3 features each
Adjacency A: 4x4 (with self-loops added)

X = [[1, 0, 1],
     [0, 1, 0],
     [1, 1, 0],
     [0, 0, 1]]

A_norm (normalized):
     [[0.5, 0.5, 0.0, 0.0],
      [0.5, 0.5, 0.0, 0.0],
      [0.0, 0.0, 0.7, 0.3],
      [0.0, 0.0, 0.3, 0.7]]
```

**Compute**: Aggregated features (before weight matrix).

**Solution**:
```
Aggregated = A_norm @ X

Node 0: 0.5*[1,0,1] + 0.5*[0,1,0] = [0.5, 0.5, 0.5]
Node 1: 0.5*[1,0,1] + 0.5*[0,1,0] = [0.5, 0.5, 0.5]
Node 2: 0.7*[1,1,0] + 0.3*[0,0,1] = [0.7, 0.7, 0.3]
Node 3: 0.3*[1,1,0] + 0.7*[0,0,1] = [0.3, 0.3, 0.7]
```
