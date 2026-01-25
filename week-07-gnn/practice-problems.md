# Week 7: Graph Neural Networks for Recommendations - Practice Problems

## Overview
These problems test your understanding of graph-based recommendations, message passing, GCN/GraphSAGE/LightGCN architectures, and knowledge graph integration. Focus on graph structure, propagation mechanisms, and scalability.

---

## Problem 1: User-Item Bipartite Graph
**Difficulty:** Easy
**Topics:** Graph construction, bipartite graphs

Given interactions:
- User 1 rated: Item A (5★), Item B (4★)
- User 2 rated: Item A (3★), Item C (5★)
- User 3 rated: Item B (4★), Item C (2★)

**Tasks:**
1. Draw the user-item bipartite graph
2. Compute node degrees for each user and item
3. What does high item degree indicate?
4. How would you incorporate ratings as edge weights?

**Learning Outcomes:**
- Construct bipartite graphs from interactions
- Understand graph topology
- Interpret node degrees

---

## Problem 2: Graph Collaborative Signal
**Difficulty:** Medium
**Topics:** High-order connectivity, collaborative filtering on graphs

In the graph from Problem 1:
1. Find all paths of length 2 from User 1 to User 2
2. What does a path User1→ItemA→User2 represent?
3. How is this similar to item-based collaborative filtering?
4. What additional information does the graph structure provide?

**Hints:**
- Paths connect users through shared items
- Length-2 paths capture item co-interactions
- Higher-order paths capture indirect signals

**Learning Outcomes:**
- Understand collaborative signal in graphs
- Connect graph-based and traditional CF
- Recognize multi-hop information

---

## Problem 3: Message Passing Mechanism
**Difficulty:** Medium
**Topics:** GCN, message passing, aggregation

Graph Convolution layer:
$$h_v^{(l+1)} = \sigma\left(\sum_{u \in N(v)} \frac{1}{\sqrt{deg(u) \cdot deg(v)}} W^{(l)} h_u^{(l)}\right)$$

**Given:**
- Node A neighbors: [B, C, D]
- deg(A)=3, deg(B)=2, deg(C)=4, deg(D)=2
- h_B = [1, 0], h_C = [0, 1], h_D = [1, 1]
- W = identity matrix, σ = ReLU

**Tasks:**
1. Compute the aggregated message to node A
2. Why normalize by $\frac{1}{\sqrt{deg(u) \cdot deg(v)}}$?
3. What happens without normalization?

**Learning Outcomes:**
- Compute message passing manually
- Understand normalization importance
- Implement GCN layers

---

## Problem 4: LightGCN Simplification
**Difficulty:** Hard
**Topics:** LightGCN, feature transformation, non-linearity

**LightGCN removes:**
1. Feature transformation (no weight matrix W)
2. Non-linear activation (no σ)

**Propagation:**
$$h_u^{(l+1)} = \sum_{i \in N(u)} \frac{1}{\sqrt{|N(u)| \cdot |N(i)|}} h_i^{(l)}$$

**Questions:**
1. Why does removing transformations improve performance?
2. What does LightGCN rely on instead?
3. Calculate computational savings (FLOPs) vs. standard GCN
4. When might standard GCN be better?

**Learning Outcomes:**
- Understand simplification benefits
- Recognize when less is more
- Analyze computational complexity

---

## Problem 5: Graph Embedding Visualization
**Difficulty:** Medium
**Topics:** Embeddings, visualization, interpretation

After training LightGCN on MovieLens:
- Action movies cluster together
- Users who like action cluster near action movies
- Romcom movies are far from action movies

**Questions:**
1. Why do semantically similar items cluster in embedding space?
2. What does the distance between user and item embeddings represent?
3. How would you visualize high-dimensional embeddings (100D → 2D)?
4. Can you interpret individual embedding dimensions?

**Learning Outcomes:**
- Interpret learned embeddings
- Visualize graph representations
- Understand latent space structure

---

## Problem 6: Scalability of GNNs
**Difficulty:** Hard
**Topics:** Scalability, sampling, mini-batch training

**Challenge:** Full-batch GCN on large graphs (1M users, 1M items) is infeasible.

**Solutions:**
1. **Node sampling:** Sample subset of neighbors
2. **Layer sampling:** Sample different neighbors per layer
3. **Subgraph sampling:** Sample connected subgraphs

**Questions:**
1. Compare memory usage: full-batch vs. mini-batch with neighbor sampling
2. How many neighbors should you sample per node?
3. What is the trade-off between sampling and accuracy?
4. How does PinSage (Pinterest) handle billion-scale graphs?

**Learning Outcomes:**
- Understand GNN scalability challenges
- Implement sampling strategies
- Make engineering trade-offs

---

## Problem 7: Knowledge Graph Integration
**Difficulty:** Hard
**Topics:** Knowledge graphs, KGAT, heterogeneous graphs

**Movie knowledge graph:**
- Movie --genre→ Action
- Movie --director→ Nolan
- Movie --actor→ DiCaprio
- User --rated→ Movie

**Tasks:**
1. Design a heterogeneous graph schema
2. How would you propagate information across different edge types?
3. What additional information does KG provide vs. collaborative graph?
4. Design an attention mechanism for different relation types

**Learning Outcomes:**
- Work with heterogeneous graphs
- Integrate knowledge graphs
- Design multi-relational GNNs

---

## Problem 8: GNN vs. Matrix Factorization
**Difficulty:** Medium
**Topics:** Comparison, expressiveness, complexity

**Compare:**
| Aspect | Matrix Factorization | LightGCN |
|--------|---------------------|----------|
| Input | User-item matrix | User-item graph |
| Parameters | User/item embeddings | User/item embeddings |
| Computation | Dot product | Multi-hop aggregation |
| Complexity | O(K) per prediction | O(L × K) per prediction |

**Questions:**
1. LightGCN uses same embeddings as MF but propagates them. Why is this better?
2. What information does GNN capture that MF misses?
3. When would MF be preferable (simpler, faster)?
4. Can you prove LightGCN with 0 layers = MF?

**Learning Outcomes:**
- Connect GNNs to traditional methods
- Understand expressiveness gains
- Recognize computational costs

---

## Programming Exercises

### Exercise 1: Build User-Item Bipartite Graph
**Dataset:** MovieLens 100K
**Task:** Construct and analyze the graph

```python
import networkx as nx
import numpy as np

def build_bipartite_graph(ratings):
    G = nx.Graph()

    for _, row in ratings.iterrows():
        user = f"u{row['user_id']}"
        item = f"i{row['item_id']}"
        rating = row['rating']

        G.add_edge(user, item, weight=rating)

    return G

# Analysis
G = build_bipartite_graph(ratings)
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Avg degree: {np.mean([d for n, d in G.degree()])}")

# Visualize degree distribution
user_degrees = [d for n, d in G.degree() if n.startswith('u')]
item_degrees = [d for n, d in G.degree() if n.startswith('i')]
```

**Analysis:** Plot degree distributions, identify hubs

---

### Exercise 2: Implement LightGCN from Scratch (PyTorch)
**Dataset:** MovieLens 100K
**Task:** Build LightGCN with layer aggregation

```python
import torch
import torch.nn as nn
import scipy.sparse as sp

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        # Initialize embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, adj_matrix):
        # adj_matrix: normalized adjacency matrix (sparse)

        ego_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)

        all_embeddings = [ego_embeddings]

        # Multi-layer propagation
        for layer in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj_matrix, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        # Layer aggregation (mean)
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)

        user_embeddings = final_embeddings[:self.n_users]
        item_embeddings = final_embeddings[self.n_users:]

        return user_embeddings, item_embeddings

    def predict(self, user_ids, item_ids):
        user_emb, item_emb = self.forward(self.adj_matrix)
        user_vec = user_emb[user_ids]
        item_vec = item_emb[item_ids]
        scores = torch.sum(user_vec * item_vec, dim=-1)
        return scores
```

**Normalization:**
```python
def create_adj_matrix(user_item_edges, n_users, n_items):
    # Build adjacency matrix
    n_nodes = n_users + n_items
    rows = []
    cols = []

    for u, i in user_item_edges:
        rows.extend([u, n_users + i])
        cols.extend([n_users + i, u])

    data = np.ones(len(rows))
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # Normalize: D^{-1/2} A D^{-1/2}
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return norm_adj.tocoo()
```

**Training:**
- BPR loss or BCE loss
- Negative sampling
- Optimizer: Adam

**Evaluation:** Recall@20, NDCG@20

---

### Exercise 3: Compare GCN vs. LightGCN
**Dataset:** MovieLens 1M
**Task:** Implement both and compare

**Standard GCN:**
- Include weight matrices
- Include non-linearity (ReLU)

**LightGCN:**
- No weight matrices
- No non-linearity
- Just aggregation

**Comparison Metrics:**
1. Recall@20, NDCG@20
2. Training time per epoch
3. Memory usage
4. Number of parameters

**Expected Result:** LightGCN slightly better, much faster

---

### Exercise 4: Neighbor Sampling for Scalability
**Dataset:** Amazon product graph (large)
**Task:** Implement GraphSAGE with neighbor sampling

```python
def sample_neighbors(node, graph, num_samples=10):
    neighbors = list(graph.neighbors(node))
    if len(neighbors) > num_samples:
        return np.random.choice(neighbors, num_samples, replace=False)
    return neighbors

class GraphSAGE(nn.Module):
    def aggregate(self, node_embedding, neighbor_embeddings):
        # Mean aggregation
        return torch.mean(torch.cat([node_embedding.unsqueeze(0), neighbor_embeddings]), dim=0)

    def forward(self, nodes, graph, num_samples=10):
        # Sample neighbors
        sampled_neighbors = [sample_neighbors(n, graph, num_samples) for n in nodes]

        # Aggregate
        # ... implementation ...
```

**Experiment:** Compare full-batch vs. sampling on runtime and accuracy

---

### Exercise 5: Knowledge Graph Attention (KGAT)
**Dataset:** MovieLens + IMDB metadata (genre, director, actors)
**Task:** Build heterogeneous graph with different edge types

**Graph Schema:**
- User --rates→ Movie
- Movie --hasGenre→ Genre
- Movie --directedBy→ Director
- Movie --starring→ Actor

**Attention Mechanism:**
```python
def relation_attention(h_head, h_tail, relation_type):
    # Learn attention weights for different relations
    W_r = self.relation_transform[relation_type]
    e = torch.tanh(torch.mm(torch.mm(h_head, W_r), h_tail.t()))
    attention = torch.softmax(e, dim=-1)
    return attention
```

**Evaluation:** Does KG improve cold-start items?

---

### Exercise 6: Graph Visualization with t-SNE
**Dataset:** MovieLens
**Task:** Visualize learned embeddings

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Train LightGCN
user_emb, item_emb = model.get_embeddings()

# Reduce dimensionality
tsne = TSNE(n_components=2, random_state=42)
item_emb_2d = tsne.fit_transform(item_emb.detach().cpu().numpy())

# Plot with genre colors
plt.figure(figsize=(12, 8))
for genre in genres:
    genre_items = items[items['genre'] == genre].index
    plt.scatter(item_emb_2d[genre_items, 0], item_emb_2d[genre_items, 1], label=genre, alpha=0.6)
plt.legend()
plt.title("Item Embeddings Colored by Genre")
plt.show()
```

**Analysis:** Do items cluster by genre? Director? Year?

---

## Discussion Questions

1. **Over-smoothing:** Deep GNNs (many layers) can over-smooth embeddings (all nodes become similar). How do you prevent this?

2. **Cold Start:** How do GNNs handle new users/items with no edges? Compare with MF and content-based methods.

3. **Explanation:** How would you explain why an item was recommended using GNN? (e.g., "because users similar to you liked it")

4. **Dynamic Graphs:** User-item graphs change over time. How do you update GNN incrementally?

5. **Negative Edges:** Should you add negative edges (dislikes) to the graph? How would this affect propagation?

6. **Homophily:** GNNs assume connected nodes are similar. Is this true for user-item graphs?

7. **Directed vs. Undirected:** User-item graphs are bipartite and undirected. What about follow graphs (Twitter)?

8. **Graph Sparsity:** Most users interact with few items. How does sparsity affect GNN performance?

---

## Challenge Problem: Temporal Graph Neural Networks

**Difficulty:** Very Hard
**Topics:** Dynamic graphs, temporal modeling, TGNs

**Task:** Extend LightGCN to handle temporal dynamics

**Key Idea:** Edges have timestamps. Recent interactions matter more.

**Approach:**
1. Add time-decay to edge weights: $w_{ui}(t) = e^{-\lambda(t_{now} - t_{interaction})}$
2. Temporal aggregation: Weight neighbors by recency
3. Dynamic embeddings: Embeddings evolve over time

**Architecture:**
```python
class TemporalLightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64):
        super(TemporalLightGCN, self).__init__()
        # Static embeddings
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)

        # Temporal attention
        self.time_attention = nn.Linear(1, 1)

    def temporal_aggregate(self, node, neighbors, timestamps, current_time):
        # Compute time decay
        time_diffs = current_time - timestamps
        decay_weights = torch.exp(-self.decay * time_diffs)

        # Aggregate with temporal weights
        neighbor_embs = self.item_emb(neighbors)
        weighted_embs = neighbor_embs * decay_weights.unsqueeze(-1)
        aggregated = torch.sum(weighted_embs, dim=0) / torch.sum(decay_weights)

        return aggregated
```

**Evaluation:**
- Predict next interaction given temporal history
- Compare with static LightGCN

---

## References

### Papers
1. He, X., et al. (2020). "LightGCN: Simplifying and powering graph convolution network". SIGIR.
2. Ying, R., et al. (2018). "Graph convolutional neural networks for web-scale recommender systems". KDD. (PinSage)
3. Wang, X., et al. (2019). "KGAT: Knowledge graph attention network for recommendation". KDD.
4. Wu, S., et al. (2019). "Session-based recommendation with graph neural networks". AAAI.

### Libraries
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- DGL (Deep Graph Library): https://www.dgl.ai/
- NetworkX: https://networkx.org/

### Datasets
- MovieLens: https://grouplens.org/datasets/movielens/
- Amazon: http://jmcauley.ucsd.edu/data/amazon/
- Yelp: https://www.yelp.com/dataset

---

*Return to [Week 7 Main Page](README.md)*
