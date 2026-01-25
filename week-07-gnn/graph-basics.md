# Week 7: Graph Neural Networks - Graph Basics

## Overview

**Graph-based recommendation** models users and items as nodes in a graph, with edges representing interactions (clicks, purchases, ratings).

**Why graphs?**
- Capture **relational structure** (user-user, item-item, user-item)
- Leverage **graph topology** for better recommendations
- **Neighborhood aggregation**: Learn from connected nodes

**Evolution**:
- **Matrix Factorization** (2006): Ignores graph structure
- **Graph-based methods** (2020+): Exploit connections

This document covers graph fundamentals for recommendation systems.

---

## Learning Objectives

By the end of this section, you will:
- Understand graphs for recommendation (nodes, edges, types)
- Master graph representation and storage
- Learn random walk-based methods (DeepWalk, Node2Vec)
- Implement graph-based collaborative filtering
- Apply graph algorithms to real-world RecSys

---

## Graphs for Recommendation

### Graph Definition

**Graph**: $G = (V, E)$
- $V$: Set of nodes (vertices)
- $E$: Set of edges (connections)

**For RecSys**:
- **Nodes**: Users, items
- **Edges**: Interactions (rated, clicked, purchased)

---

### Types of Graphs

**1. Bipartite Graph (User-Item)**

```
Users        Items
  U1 -------- I1
  U2 -------- I2
   |   \       |
  U3    ----  I3
```

**Edges**: Only between users and items (no user-user or item-item edges).

**Example**: Netflix (users rate movies).

---

**2. Knowledge Graph**

```
Nodes: Users, Items, Attributes
Edges: Relationships

User1 --clicked--> Movie1
Movie1 --genre--> Action
Movie1 --director--> Nolan
Nolan --directed--> Movie2
```

**Richer structure**: Multi-relational.

---

**3. Social Graph**

```
Users connected by friendship:
  U1 ---- U2
   |   ×   |
  U3 ---- U4
```

**Edges**: Social connections (friends, followers).

**Use**: Social recommendation (recommend what friends like).

---

### Graph Representation

**Adjacency Matrix** (for small graphs):

$$A_{ij} = \begin{cases}
1 & \text{if edge from } i \text{ to } j \\
0 & \text{otherwise}
\end{cases}$$

**Example**:
```
Graph:
  1 → 2
  1 → 3
  2 → 3

Adjacency Matrix:
     1  2  3
1 [ [0, 1, 1],
2   [0, 0, 1],
3   [0, 0, 0] ]
```

**Problem**: Sparse, memory-intensive for large graphs (millions of nodes).

---

**Adjacency List** (for large graphs):

```python
graph = {
    1: [2, 3],  # Node 1 connects to 2, 3
    2: [3],     # Node 2 connects to 3
    3: []       # Node 3 connects to nothing
}
```

**Advantage**: Memory-efficient for sparse graphs.

---

### Implementation: User-Item Bipartite Graph

```python
from collections import defaultdict

class BipartiteGraph:
    def __init__(self):
        self.user_to_items = defaultdict(set)  # User → items they interacted with
        self.item_to_users = defaultdict(set)  # Item → users who interacted

    def add_interaction(self, user, item):
        """Add edge between user and item."""
        self.user_to_items[user].add(item)
        self.item_to_users[item].add(user)

    def get_user_items(self, user):
        """Get items user has interacted with."""
        return self.user_to_items[user]

    def get_item_users(self, item):
        """Get users who interacted with item."""
        return self.item_to_users[item]

    def user_similarity(self, user1, user2):
        """Compute Jaccard similarity between two users."""
        items1 = self.user_to_items[user1]
        items2 = self.user_to_items[user2]

        intersection = len(items1 & items2)
        union = len(items1 | items2)

        return intersection / union if union > 0 else 0


# Example
graph = BipartiteGraph()
graph.add_interaction(user=1, item=101)
graph.add_interaction(user=1, item=102)
graph.add_interaction(user=2, item=102)
graph.add_interaction(user=2, item=103)

print(f"User 1 items: {graph.get_user_items(1)}")  # {101, 102}
print(f"Item 102 users: {graph.get_item_users(102)}")  # {1, 2}
print(f"Similarity(U1, U2): {graph.user_similarity(1, 2):.2f}")  # 0.33 (1 common out of 3 total)
```

---

## Graph-Based Collaborative Filtering

### ItemKNN on Graph

**Idea**: Items are similar if they share many common users.

**Similarity**:
$$\text{sim}(i, j) = \frac{|U_i \cap U_j|}{\sqrt{|U_i| \cdot |U_j|}}$$

where $U_i$ = set of users who interacted with item $i$.

**Recommendation**: Recommend items similar to user's past items.

---

### Implementation

```python
import numpy as np

class ItemKNNGraph:
    def __init__(self, graph):
        self.graph = graph
        self.item_similarity = {}

    def compute_similarities(self):
        """Precompute item-item similarities."""
        items = list(self.graph.item_to_users.keys())

        for i in range(len(items)):
            for j in range(i+1, len(items)):
                item1, item2 = items[i], items[j]

                users1 = self.graph.item_to_users[item1]
                users2 = self.graph.item_to_users[item2]

                intersection = len(users1 & users2)
                denominator = np.sqrt(len(users1) * len(users2))

                similarity = intersection / denominator if denominator > 0 else 0
                self.item_similarity[(item1, item2)] = similarity
                self.item_similarity[(item2, item1)] = similarity

    def recommend(self, user, top_k=5):
        """Recommend items for user based on ItemKNN."""
        user_items = self.graph.get_user_items(user)

        scores = defaultdict(float)

        # Aggregate scores from user's items
        for item in user_items:
            for candidate_item in self.graph.item_to_users.keys():
                if candidate_item not in user_items:  # Don't recommend already-seen
                    sim = self.item_similarity.get((item, candidate_item), 0)
                    scores[candidate_item] += sim

        # Sort and return top-K
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_items[:top_k]]


# Example
item_knn = ItemKNNGraph(graph)
item_knn.compute_similarities()

recommendations = item_knn.recommend(user=1, top_k=3)
print(f"Recommendations for user 1: {recommendations}")
```

---

## Random Walks on Graphs

### DeepWalk

**Paper**: Perozzi et al., "DeepWalk: Online Learning of Social Representations" (KDD 2014)

**Idea**: Generate random walks on graph, treat as "sentences", apply Word2Vec.

**Algorithm**:
1. For each node, generate $N$ random walks of length $L$
2. Treat walks as sentences (e.g., `[1, 3, 5, 2, 7]`)
3. Train Word2Vec (Skip-gram) on walks
4. Get node embeddings

**Result**: Nodes that co-occur in walks have similar embeddings.

---

### Random Walk Generation

```python
import random

def random_walk(graph, start_node, walk_length):
    """Generate random walk starting from start_node."""
    walk = [start_node]
    current = start_node

    for _ in range(walk_length - 1):
        neighbors = list(graph[current])
        if len(neighbors) == 0:
            break
        current = random.choice(neighbors)
        walk.append(current)

    return walk


# Example graph (adjacency list)
graph = {
    1: [2, 3],
    2: [1, 3, 4],
    3: [1, 2, 5],
    4: [2],
    5: [3]
}

walk = random_walk(graph, start_node=1, walk_length=10)
print(f"Random walk: {walk}")
# Output: [1, 3, 2, 4, 2, 1, 3, 5, 3, 2]
```

---

### DeepWalk Implementation

```python
from gensim.models import Word2Vec

class DeepWalk:
    def __init__(self, graph, embedding_dim=128, walk_length=10, num_walks=80):
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.model = None

    def generate_walks(self):
        """Generate random walks for all nodes."""
        walks = []
        nodes = list(self.graph.keys())

        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = random_walk(self.graph, node, self.walk_length)
                walks.append([str(n) for n in walk])  # Convert to strings for Word2Vec

        return walks

    def train(self):
        """Train DeepWalk (Word2Vec on walks)."""
        walks = self.generate_walks()

        # Train Word2Vec
        self.model = Word2Vec(
            sentences=walks,
            vector_size=self.embedding_dim,
            window=5,
            min_count=0,
            sg=1,  # Skip-gram
            workers=4,
            epochs=5
        )

    def get_embedding(self, node):
        """Get embedding for node."""
        return self.model.wv[str(node)]

    def similarity(self, node1, node2):
        """Compute similarity between two nodes."""
        return self.model.wv.similarity(str(node1), str(node2))


# Example
deepwalk = DeepWalk(graph, embedding_dim=64, walk_length=10, num_walks=80)
deepwalk.train()

# Get embedding for node 1
emb = deepwalk.get_embedding(1)
print(f"Node 1 embedding (first 10 dims): {emb[:10]}")

# Similarity between nodes
sim = deepwalk.similarity(1, 3)
print(f"Similarity(1, 3): {sim:.3f}")
```

---

### Node2Vec

**Paper**: Grover & Leskovec, "node2vec: Scalable Feature Learning for Networks" (KDD 2016)

**Extension of DeepWalk**: Biased random walks.

**Parameters**:
- $p$: Return parameter (go back to previous node)
- $q$: In-out parameter (explore vs. exploit)

**Intuition**:
- Low $p$: Stay local (BFS-like)
- Low $q$: Explore distant nodes (DFS-like)

**Use in RecSys**: Generate user/item embeddings from interaction graph.

---

## Graph Algorithms for RecSys

### PageRank

**Idea**: Important nodes have many incoming edges from important nodes.

**Formula**:
$$PR(v) = (1 - d) + d \sum_{u \in N(v)} \frac{PR(u)}{|N(u)|}$$

where $d$ = damping factor (0.85), $N(v)$ = nodes pointing to $v$.

**Application**: Rank items by importance (popular items have high PageRank).

---

### Personalized PageRank

**Extension**: Personalize for each user.

**Idea**: Random walk with restart to user's items.

**Algorithm**:
1. Start at user's items
2. Random walk on graph
3. With probability $\alpha$, restart at user's items
4. Items visited most often → recommended

**Advantage**: Personalizes PageRank to user.

---

## Graph Metrics

### Degree Centrality

**Definition**: Number of edges connected to node.

$$C_D(v) = |N(v)|$$

**For users**: Number of items user has interacted with (activity level).
**For items**: Number of users who interacted (popularity).

---

### Clustering Coefficient

**Definition**: How connected a node's neighbors are.

$$C(v) = \frac{\text{number of triangles involving } v}{\text{number of possible triangles}}$$

**Application**: Dense clusters → community detection.

---

## Summary

**Key Takeaways**:
1. **Graphs for RecSys**: Users and items as nodes, interactions as edges
2. **Graph types**: Bipartite (user-item), knowledge graphs, social graphs
3. **Representation**: Adjacency matrix (small), adjacency list (large)
4. **Collaborative filtering on graphs**: ItemKNN using graph similarity
5. **Random walks**: DeepWalk, Node2Vec for node embeddings
6. **Graph algorithms**: PageRank, Personalized PageRank

**When to use**:
- **Rich relational data**: Social networks, knowledge graphs
- **Cold start**: Leverage graph structure (user's friends' preferences)
- **Explainability**: Graph paths explain recommendations

**Next**: Graph Neural Networks (GNNs) - deep learning on graphs.

---

## References

1. **Perozzi, B., Al-Rfou, R., & Skiena, S. (2014)**. "DeepWalk: Online Learning of Social Representations". *KDD*.
   - **DeepWalk** algorithm

2. **Grover, A., & Leskovec, J. (2016)**. "node2vec: Scalable Feature Learning for Networks". *KDD*.
   - **Node2Vec** for biased random walks

3. **Page, L., et al. (1999)**. "The PageRank Citation Ranking: Bringing Order to the Web". *Stanford Technical Report*.
   - **PageRank** algorithm

4. **Sarwar, B., et al. (2001)**. "Item-based Collaborative Filtering Recommendation Algorithms". *WWW*.
   - **ItemKNN**, applicable to graphs

5. **Hamilton, W. L., Ying, R., & Leskovec, J. (2017)**. "Representation Learning on Graphs: Methods and Applications". *IEEE Data Engineering Bulletin*.
   - Survey of graph representation learning

---

## Practice Problems

### Problem 1: Jaccard Similarity

**Given**:
```
User 1 items: {A, B, C}
User 2 items: {B, C, D}
```

**Compute**: Jaccard similarity.

**Solution**:
```
Intersection: {B, C} → 2
Union: {A, B, C, D} → 4

Jaccard = 2 / 4 = 0.5
```

---

### Problem 2: Random Walk

**Given graph**:
```
1 → 2, 3
2 → 1, 4
3 → 1
4 → 2
```

**Generate**: Random walk of length 5 starting from node 1.

**Solution** (example):
```
Start: 1
Step 1: Choose from {2, 3} → 2
Step 2: Choose from {1, 4} → 4
Step 3: Choose from {2} → 2
Step 4: Choose from {1, 4} → 1
Step 5: Choose from {2, 3} → 3

Walk: [1, 2, 4, 2, 1, 3]
```

---

### Problem 3: PageRank (1 iteration)

**Given**:
```
Graph:
  1 → 2
  2 → 1, 3
  3 → 1

Initial PageRank: PR(1) = PR(2) = PR(3) = 1/3
Damping factor d = 0.85
```

**Compute**: PR(1) after 1 iteration.

**Solution**:
```
PR(1) = (1 - 0.85) + 0.85 * (PR(2)/2 + PR(3)/1)
     = 0.15 + 0.85 * ((1/3)/2 + (1/3)/1)
     = 0.15 + 0.85 * (1/6 + 1/3)
     = 0.15 + 0.85 * (1/2)
     = 0.15 + 0.425
     = 0.575
```
