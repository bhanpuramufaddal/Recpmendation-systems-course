# Week 7: Graph Neural Networks - Graph Basics

## The Opening Problem: Why Graphs? What Does Matrix Factorization Miss?

*Before diving into graphs, let me show you a failure case that will change how you think about recommendations.*

**The Setup**: Consider this user-item interaction pattern:

```
Users:     Alice ---[bought]---> Laptop
           Alice ---[bought]---> Laptop Bag
           Bob   ---[bought]---> Laptop
           Bob   ---[bought]---> Mouse
           Carol ---[bought]---> Mouse
           Carol ---[bought]---> Keyboard
```

**Question**: Should we recommend "Keyboard" to Alice?

**Matrix Factorization's Answer**: Let's see what MF learns.

The user-item matrix:
```
           Laptop  Bag  Mouse  Keyboard
Alice        1      1     0       0
Bob          1      0     1       0
Carol        0      0     1       1
```

**MF computes**:
- Alice-Bob similarity: Share "Laptop" $\rightarrow$ somewhat similar
- Bob-Carol similarity: Share "Mouse" $\rightarrow$ somewhat similar
- Alice-Carol similarity: Share nothing! $\rightarrow$ zero overlap

**MF's recommendation for Alice**: Maybe "Mouse" (because Bob bought it)

**But wait...** Look at the **chain of connections**:

$$\text{Alice} \xrightarrow{\text{bought same}} \text{Bob} \xrightarrow{\text{bought same}} \text{Carol} \xrightarrow{\text{bought}} \text{Keyboard}$$

*This is a 3-hop relationship!* Alice is connected to Keyboard through two intermediaries.

**MF completely misses this!** Why?

---

### The Multi-Hop Problem

**Matrix Factorization limitation**: MF only captures **direct** relationships through the dot product $\mathbf{u}_u^T \mathbf{v}_i$.

**What MF computes**:
$$\hat{r}_{ui} = \mathbf{u}_u^T \mathbf{v}_i = \text{(1-hop similarity)}$$

**What graphs capture**:
- 1-hop: Alice $\rightarrow$ Laptop (direct)
- 2-hop: Alice $\rightarrow$ Bob (share Laptop) $\rightarrow$ Mouse
- 3-hop: Alice $\rightarrow$ Bob $\rightarrow$ Carol $\rightarrow$ Keyboard
- n-hop: Arbitrarily long chains!

**The Graph Advantage**: By modeling the interaction graph, we can propagate information across multiple hops.

*Think about it*: In social networks, "friends of friends" matter. In e-commerce, "users who bought what you bought also bought X" chains matter. MF flattens this rich structure into simple dot products.

---

### A Concrete Numerical Example: Where MF Fails

**Setup**: 4 users (A, B, C, D) and 4 items (1, 2, 3, 4)

**Interactions** (forming a chain):
```
A--[1]--B--[2]--C--[3]--D--[4]
```

User A bought item 1, User B bought items 1 and 2, User C bought items 2 and 3, User D bought items 3 and 4.

**Matrix**:
```
     Item1  Item2  Item3  Item4
A      1      0      0      0
B      1      1      0      0
C      0      1      1      0
D      0      0      1      1
```

**MF with k=2**:
- A and D have **zero overlap** in the matrix
- Their user vectors will be trained on completely different items
- $\text{sim}(A, D) \approx 0$ in latent space

**Graph perspective**:
- A is 3 hops from item 4: A $\rightarrow$ B $\rightarrow$ C $\rightarrow$ D $\rightarrow$ item 4
- Information should flow: A likes item 1, which B also likes, B likes item 2, which C also likes, etc.

**Key insight**: *Graphs preserve transitivity. MF doesn't.*

---

## Learning Objectives

By the end of this section, you will:
- Understand graphs for recommendation (nodes, edges, types)
- Master graph representation and storage
- Learn random walk-based methods (DeepWalk, Node2Vec)
- **Derive the connection between random walks and Word2Vec**
- **Work through PageRank step by step from the "random surfer" model**
- **Calculate Node2Vec transition probabilities with specific p/q values**
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

### The Intuition: Walking = Sampling Neighborhoods

*Before we get into algorithms, let me give you the key intuition.*

**Analogy: Exploring a New City**

Imagine you're a tourist in a new city with no map. How do you discover interesting places?

**Strategy 1**: Visit one landmark, then randomly follow a nearby street to another landmark, and repeat.

After many walks, you'll notice:
- **Frequently visited places** are probably important (central, well-connected)
- **Places visited together** are probably similar (in the same neighborhood)
- **Places you never reach** are probably distant or isolated

*This is exactly what random walks do on graphs!*

---

### What Random Walks Capture

**A random walk** is a sequence of nodes: $(v_0, v_1, v_2, \ldots, v_L)$

At each step, we randomly pick a neighbor:
$$P(v_{t+1} = j | v_t = i) = \frac{A_{ij}}{\sum_k A_{ik}}$$

**What co-occurrence in walks means**:
- If nodes A and B appear together in many walks, they're **structurally similar**
- They might be:
  - Direct neighbors (1-hop)
  - In the same community (multi-hop)
  - Playing similar "roles" in the graph

**Key insight**: *Random walks automatically capture both local (1-hop) and global (multi-hop) structure!*

---

### Another Analogy: Following Trails in a Forest

*Let me give you another way to think about this.*

Imagine a forest with trails connecting clearings:

```
    Clearing A -------- Clearing B
         |                  |
         |                  |
    Clearing C -------- Clearing D -------- Clearing E
```

If you start at A and wander randomly:
- You'll often visit B and C (direct neighbors)
- You'll sometimes reach D (2 hops)
- You'll occasionally reach E (3 hops)

After 1000 random walks starting from A:
- B and C: visited ~300 times each
- D: visited ~150 times
- E: visited ~50 times

*The visit frequency encodes structural similarity!*

---

## DeepWalk: The Connection to Word2Vec

### The Word2Vec Intuition

*Before connecting to graphs, let me quickly explain Word2Vec.*

**Word2Vec's key insight**: Words that appear in similar **contexts** have similar meanings.

**Example sentences**:
- "The **cat** sat on the mat"
- "The **dog** sat on the rug"

**Observation**: "cat" and "dog" have similar contexts ("The __ sat on the ___")

**Word2Vec learns**: $\text{vec}(\text{cat}) \approx \text{vec}(\text{dog})$

**The training objective (Skip-gram)**:

Given a word $w$, predict its context words $c$:
$$\max_{w,c} \log P(c | w) = \log \frac{\exp(\mathbf{w}^T \mathbf{c})}{\sum_{c'} \exp(\mathbf{w}^T \mathbf{c}')}$$

*In plain English*: Learn word vectors such that a word can predict its neighbors in the sentence.

---

### Random Walks Are "Sentences" on Graphs!

*Here's the key connection that makes DeepWalk work.*

**Observation**: A random walk is like a "sentence" of nodes.

**Sentence**: "The cat sat on the mat"
**Walk**: $(v_1, v_3, v_5, v_2, v_7)$

**Word co-occurrence** $\leftrightarrow$ **Node co-occurrence in walks**

| Word2Vec | DeepWalk |
|----------|----------|
| Word | Node |
| Sentence | Random walk |
| Context window | Walk window |
| Similar words | Similar nodes |

---

### Step-by-Step: Deriving the DeepWalk Objective

*Let's derive exactly how DeepWalk applies Word2Vec to graphs.*

**Step 1: Generate random walks**

For each node $v$, generate $N$ random walks of length $L$:
$$W_v^{(1)}, W_v^{(2)}, \ldots, W_v^{(N)}$$

Example walks from node A:
- Walk 1: (A, B, D, E, D)
- Walk 2: (A, C, D, B, A)
- Walk 3: (A, B, A, C, D)

**Step 2: Define the "language model" objective**

Given a walk $(v_0, v_1, \ldots, v_L)$, maximize:
$$\prod_{t=0}^{L} P(v_{t-w}, \ldots, v_{t+w} | v_t)$$

where $w$ = context window size (typically 5-10).

**Step 3: Apply Skip-gram**

For each node $v_t$ in a walk, predict its context nodes:

$$\mathcal{L} = \sum_{\text{walks}} \sum_{t=0}^{L} \sum_{-w \leq j \leq w, j \neq 0} \log P(v_{t+j} | v_t)$$

**Step 4: Softmax probability**

$$P(v_c | v_t) = \frac{\exp(\mathbf{v}_c^T \mathbf{v}_t)}{\sum_{v' \in V} \exp(\mathbf{v}'^T \mathbf{v}_t)}$$

**Step 5: Negative sampling (for efficiency)**

The denominator sums over all nodes (expensive!). Instead:

$$\log P(v_c | v_t) \approx \log \sigma(\mathbf{v}_c^T \mathbf{v}_t) + \sum_{i=1}^{k} \mathbb{E}_{v_n \sim P_n} [\log \sigma(-\mathbf{v}_n^T \mathbf{v}_t)]$$

where $\sigma(x) = 1/(1 + e^{-x})$ and $k$ negative samples are drawn.

---

### DeepWalk Algorithm Summary

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

## Node2Vec: Biased Random Walks

### The Limitation of Uniform Random Walks

*DeepWalk uses uniform random walks, but this misses something important.*

**Question**: Should we explore broadly (like BFS) or deeply (like DFS)?

```
     [A] ---- [B] ---- [C]
      |        |        |
     [D]      [E]      [F]
      |        |        |
     [G]      [H]      [I]
```

**BFS-like exploration** (from A):
- Visits: A, B, D, E, C, G, ...
- Captures **local community structure**
- Good for: Similar roles, same neighborhood

**DFS-like exploration** (from A):
- Visits: A, B, C, F, I, ...
- Captures **structural equivalence**
- Good for: Bridge nodes, similar positions in different communities

*Can we control this?*

---

### Node2Vec: The p and q Parameters

**Paper**: Grover & Leskovec, "node2vec: Scalable Feature Learning for Networks" (KDD 2016)

**Key idea**: Bias the random walk using two parameters:

- $p$: **Return parameter** - likelihood of returning to previous node
- $q$: **In-out parameter** - inward vs outward exploration

---

### Understanding p and q Intuitively

**Setup**: We're at node $v$, came from node $t$. Next node is $x$.

```
        [t] -------- [v] -------- [x1] (distance 0 from t)
                      |
                     [x2] -------- [x3] (distances 1 and 2 from t)
```

**Transition probability** from $v$ to $x$ depends on distance from $t$:

$$\pi_{vx} = \alpha_{pq}(t, x) \cdot w_{vx}$$

where:

$$\alpha_{pq}(t, x) = \begin{cases}
\frac{1}{p} & \text{if } d_{tx} = 0 \text{ (return to } t \text{)} \\
1 & \text{if } d_{tx} = 1 \text{ (same distance from } t \text{)} \\
\frac{1}{q} & \text{if } d_{tx} = 2 \text{ (move away from } t \text{)}
\end{cases}$$

---

### Intuition for Different p and q Values

**Low p (e.g., p=0.5)**: High probability of returning
- $\alpha = 1/0.5 = 2$ for return
- Walker stays local, like **BFS**
- Explores immediate neighborhood thoroughly

**High p (e.g., p=2)**: Low probability of returning
- $\alpha = 1/2 = 0.5$ for return
- Walker moves away, less backtracking

**Low q (e.g., q=0.5)**: Encourages moving outward
- $\alpha = 1/0.5 = 2$ for distance-2 nodes
- Walker explores far, like **DFS**
- Discovers global structure

**High q (e.g., q=2)**: Discourages moving outward
- $\alpha = 1/2 = 0.5$ for distance-2 nodes
- Walker stays close to origin

---

### Complete Numerical Example: Biased Walk on a 5-Node Graph

*Let me walk you through a complete example with specific numbers.*

**Graph**:
```
    [1] -------- [2] -------- [3]
     |            |
     |            |
    [4] -------- [5]
```

**Adjacency**:
- Node 1: neighbors = {2, 4}
- Node 2: neighbors = {1, 3, 5}
- Node 3: neighbors = {2}
- Node 4: neighbors = {1, 5}
- Node 5: neighbors = {2, 4}

**Scenario**: We're at node 2, and we came from node 1.

**Parameters**: $p = 0.5$, $q = 2$

**Step 1: Identify neighbors of current node (2)**

Neighbors of 2: {1, 3, 5}

**Step 2: Compute distance of each neighbor from previous node (1)**

| Neighbor | Distance from node 1 | Category |
|----------|---------------------|----------|
| 1 | 0 (is node 1) | Return |
| 3 | 2 (1-2-3) | Move away |
| 5 | 2 (1-4-5 or 1-2-5) | Move away |

**Step 3: Compute unnormalized weights**

$$w_{\text{unnorm}}(1) = \frac{1}{p} = \frac{1}{0.5} = 2$$

$$w_{\text{unnorm}}(3) = \frac{1}{q} = \frac{1}{2} = 0.5$$

$$w_{\text{unnorm}}(5) = \frac{1}{q} = \frac{1}{2} = 0.5$$

**Step 4: Normalize to get probabilities**

$$\text{Sum} = 2 + 0.5 + 0.5 = 3$$

$$P(\text{next} = 1 | \text{current} = 2, \text{prev} = 1) = \frac{2}{3} = 0.667$$

$$P(\text{next} = 3 | \text{current} = 2, \text{prev} = 1) = \frac{0.5}{3} = 0.167$$

$$P(\text{next} = 5 | \text{current} = 2, \text{prev} = 1) = \frac{0.5}{3} = 0.167$$

**Step 5: Sample next node**

With probability 0.667, we return to node 1.
With probability 0.167 each, we go to node 3 or 5.

*Notice*: Low $p$ makes returning very likely (BFS-like behavior)!

---

### Another Example: Different Parameters

**Same graph, same position (at 2, came from 1)**

**New parameters**: $p = 2$, $q = 0.5$

**Step 3 (new weights)**:

$$w_{\text{unnorm}}(1) = \frac{1}{p} = \frac{1}{2} = 0.5$$

$$w_{\text{unnorm}}(3) = \frac{1}{q} = \frac{1}{0.5} = 2$$

$$w_{\text{unnorm}}(5) = \frac{1}{q} = \frac{1}{0.5} = 2$$

**Step 4 (new probabilities)**:

$$\text{Sum} = 0.5 + 2 + 2 = 4.5$$

$$P(\text{next} = 1) = \frac{0.5}{4.5} = 0.111$$

$$P(\text{next} = 3) = \frac{2}{4.5} = 0.444$$

$$P(\text{next} = 5) = \frac{2}{4.5} = 0.444$$

*Notice*: High $p$ and low $q$ make outward exploration much more likely (DFS-like behavior)!

---

### Tracing a Complete Walk

*Let me trace a walk step-by-step with $p=0.5$, $q=2$.*

**Start**: Node 1

**Walk so far**: [1]

**Step 1**: At node 1, no previous (first step is uniform)
- Neighbors: {2, 4}
- $P(2) = P(4) = 0.5$
- **Sample**: 2

**Walk**: [1, 2]

**Step 2**: At node 2, previous = 1
- Computed above: $P(1) = 0.667$, $P(3) = 0.167$, $P(5) = 0.167$
- **Sample**: 1 (most likely)

**Walk**: [1, 2, 1]

**Step 3**: At node 1, previous = 2
- Neighbors: {2, 4}
- Distance from 2: d(2,2)=0, d(2,4)=2
- Weights: $w(2) = 1/0.5 = 2$, $w(4) = 1/2 = 0.5$
- Probabilities: $P(2) = 2/2.5 = 0.8$, $P(4) = 0.5/2.5 = 0.2$
- **Sample**: 2

**Walk**: [1, 2, 1, 2]

*With low $p$, the walk oscillates back and forth! It's exploring the local neighborhood intensively.*

---

### Node2Vec Use in RecSys

**Generate user/item embeddings** from the interaction graph:

1. Build bipartite graph: users $\leftrightarrow$ items
2. Choose $p$ and $q$ based on desired behavior:
   - Low $p$, high $q$: Capture local preferences (users with similar items)
   - High $p$, low $q$: Capture global patterns (cross-community connections)
3. Generate biased random walks
4. Train Skip-gram on walks
5. Use embeddings for downstream tasks (recommendation, link prediction)

---

## PageRank: From "Random Surfer" to Algorithm

### The Random Surfer Model

*Let me derive PageRank from first principles using a simple intuition.*

**Imagine**: You're surfing the web randomly.

**Your behavior**:
1. You're on a webpage
2. With probability $d$ (e.g., 0.85), you click a random link on the page
3. With probability $1-d$ (e.g., 0.15), you get bored and jump to a random page

**Question**: After surfing forever, what fraction of time do you spend on each page?

*This fraction is the PageRank score!*

---

### Step-by-Step Derivation

**Setup**: $N$ webpages, $A$ = adjacency matrix where $A_{ij} = 1$ if page $i$ links to page $j$.

**Step 1: Define transition probabilities**

Let $O_i = \sum_j A_{ij}$ = number of outgoing links from page $i$.

Probability of going from $i$ to $j$ (by clicking a link):
$$M_{ij} = \frac{A_{ij}}{O_i}$$

**Step 2: Add the "teleportation"**

With probability $1-d$, the surfer teleports to a random page:
$$P(\text{at page } j \text{ next step}) = d \cdot \sum_i PR(i) \cdot M_{ij} + \frac{1-d}{N}$$

**Step 3: Write as a fixed-point equation**

Let $PR(v)$ be the PageRank of page $v$. At equilibrium:

$$PR(v) = \frac{1-d}{N} + d \sum_{u \in N_{in}(v)} \frac{PR(u)}{O_u}$$

where $N_{in}(v)$ = pages linking TO $v$.

**In words**: PageRank of $v$ = (base probability) + (sum of PageRank flowing in from neighbors)

---

### The PageRank Formula (Simplified)

For recommendation systems, we often write:

$$PR(v) = (1 - d) + d \sum_{u \in N(v)} \frac{PR(u)}{|N(u)|}$$

where:
- $d$ = damping factor (typically 0.85)
- $N(v)$ = nodes pointing to $v$
- $|N(u)|$ = out-degree of node $u$

---

### Complete Numerical Example: 4-Page Web

*Let me walk through PageRank computation step by step.*

**Web Graph**:
```
    [A] ←------ [B]
     |           ↑
     ↓           |
    [C] ------→ [D]
     ↑           |
     └-----------┘
```

**Edges**: A→C, B→A, C→D, D→B, D→C

**Adjacency** (row = from, col = to):
```
     A  B  C  D
A [  0  0  1  0 ]
B [  1  0  0  0 ]
C [  0  0  0  1 ]
D [  0  1  1  0 ]
```

**Parameters**: $d = 0.85$, $N = 4$

**Out-degrees**: $O_A = 1$, $O_B = 1$, $O_C = 1$, $O_D = 2$

---

**Iteration 0 (Initialize)**:

$$PR^{(0)}(A) = PR^{(0)}(B) = PR^{(0)}(C) = PR^{(0)}(D) = \frac{1}{4} = 0.25$$

---

**Iteration 1**:

$$PR^{(1)}(A) = \frac{1-0.85}{4} + 0.85 \cdot \frac{PR^{(0)}(B)}{O_B}$$
$$= 0.0375 + 0.85 \cdot \frac{0.25}{1} = 0.0375 + 0.2125 = 0.25$$

$$PR^{(1)}(B) = \frac{1-0.85}{4} + 0.85 \cdot \frac{PR^{(0)}(D)}{O_D}$$
$$= 0.0375 + 0.85 \cdot \frac{0.25}{2} = 0.0375 + 0.10625 = 0.14375$$

$$PR^{(1)}(C) = \frac{1-0.85}{4} + 0.85 \cdot \left(\frac{PR^{(0)}(A)}{O_A} + \frac{PR^{(0)}(D)}{O_D}\right)$$
$$= 0.0375 + 0.85 \cdot \left(\frac{0.25}{1} + \frac{0.25}{2}\right) = 0.0375 + 0.85 \cdot 0.375 = 0.35625$$

$$PR^{(1)}(D) = \frac{1-0.85}{4} + 0.85 \cdot \frac{PR^{(0)}(C)}{O_C}$$
$$= 0.0375 + 0.85 \cdot \frac{0.25}{1} = 0.0375 + 0.2125 = 0.25$$

---

**After Iteration 1**: $[A=0.25, B=0.14, C=0.36, D=0.25]$

*Notice*: Page C has the highest PageRank! It receives links from both A and D.

---

**Iteration 2**:

$$PR^{(2)}(A) = 0.0375 + 0.85 \cdot \frac{0.14375}{1} = 0.16$$

$$PR^{(2)}(B) = 0.0375 + 0.85 \cdot \frac{0.25}{2} = 0.14$$

$$PR^{(2)}(C) = 0.0375 + 0.85 \cdot \left(\frac{0.25}{1} + \frac{0.25}{2}\right) = 0.36$$

$$PR^{(2)}(D) = 0.0375 + 0.85 \cdot \frac{0.35625}{1} = 0.34$$

**After Iteration 2**: $[A=0.16, B=0.14, C=0.36, D=0.34]$

---

**Convergence** (after ~10-20 iterations):

$$PR(A) \approx 0.17, \quad PR(B) \approx 0.14, \quad PR(C) \approx 0.35, \quad PR(D) \approx 0.34$$

**Final ranking**: C > D > A > B

*Interpretation*: Page C is most "important" because it receives quality links from multiple sources.

---

### PageRank in RecSys

**Application**: Rank items by importance (popular items have high PageRank).

**How it works for recommendation**:
1. Build item-item graph (edge if items co-purchased)
2. Compute PageRank of each item
3. High PageRank = influential/popular item

**Personalized PageRank** (next section): Start walks from user's items, not uniformly.

---

## Personalized PageRank

### Extension: Personalize for Each User

**Idea**: Random walk with restart to user's items.

**Algorithm**:
1. Start at user's items
2. Random walk on graph
3. With probability $\alpha$, restart at user's items
4. Items visited most often → recommended

**Formula**:
$$PPR_u(v) = (1 - \alpha) \cdot \mathbf{1}_{v \in I_u} + \alpha \sum_{w \in N(v)} \frac{PPR_u(w)}{|N(w)|}$$

where $I_u$ = items user $u$ has interacted with.

**Advantage**: Personalizes PageRank to user's specific preferences.

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

## What Can Go Wrong?

*Let me warn you about common failure modes with graph-based methods.*

### Failure Mode 1: Disconnected Components

**Problem**: If the graph has disconnected components, random walks can't bridge them.

**Symptoms**:
- Users in one component never get items from another component
- Cold users/items in isolated subgraphs get no recommendations
- Embeddings for disconnected components are unrelated

**Example**:
```
Component 1:        Component 2:
  U1 -- I1            U3 -- I3
  U2 -- I2            U4 -- I4
```
Users U1/U2 will never be recommended I3/I4!

**Solutions**:
- Add weak connections between components (e.g., via attributes)
- Use content features to bridge
- Check for connectivity before training

---

### Failure Mode 2: Popularity Bias Amplification

**Problem**: PageRank and random walks favor highly-connected nodes.

**Symptoms**:
- Popular items dominate all recommendations
- Long-tail items never visited in walks
- Rich-get-richer effect

**Cause**: Random walks are more likely to visit high-degree nodes.

**Math**: For a node with degree $k$, expected visits $\propto k$.

**Solutions**:
- Degree normalization in transition probabilities
- Subsampling popular nodes
- Inverse frequency weighting
- Personalized PageRank (starts from user's items)

---

### Failure Mode 3: Short Walk Lengths

**Problem**: Walks too short to capture multi-hop relationships.

**Symptoms**:
- Only direct neighbors get similar embeddings
- Long-range dependencies missed
- Same results as item-item CF

**Example**: With walk length 3, you can only reach 2-hop neighbors consistently.

**Solutions**:
- Increase walk length (10-80 typical)
- Increase number of walks per node
- Use higher-order methods (GNNs with more layers)

---

### Failure Mode 4: Wrong p/q Parameters (Node2Vec)

**Problem**: p and q don't match the task.

**Symptoms**:
- With wrong parameters, embeddings don't capture desired structure
- BFS-like when you need DFS, or vice versa

**Guidelines**:
- **Homophily tasks** (similar nodes should cluster): Low p, high q
- **Structural equivalence tasks** (similar roles across communities): High p, low q
- **When unsure**: p=1, q=1 (uniform random walks, like DeepWalk)

**Solutions**:
- Grid search over p, q on validation set
- Visualize embeddings to check clustering
- Try both extremes and compare

---

### Failure Mode 5: Ignoring Edge Weights

**Problem**: Treating all edges equally when they have different strengths.

**Symptoms**:
- A user who bought 10 copies of a book treated same as one who bought 1
- Implicit feedback signal lost
- Click and purchase weighted equally

**Solutions**:
- Weight edges by interaction strength
- Normalize by total interactions
- Use weighted random walks

---

### Failure Mode 6: Static Graphs for Dynamic Data

**Problem**: Graph embeddings learned once, but interactions change over time.

**Symptoms**:
- Recommendations become stale
- New items never get recommended
- User preference shifts ignored

**Solutions**:
- Periodic retraining
- Online/incremental updates
- Temporal graph methods

---

## Summary

**Key Takeaways**:
1. **Graphs for RecSys**: Users and items as nodes, interactions as edges
2. **Why graphs beat MF**: Capture multi-hop relationships, transitive similarity
3. **Graph types**: Bipartite (user-item), knowledge graphs, social graphs
4. **Representation**: Adjacency matrix (small), adjacency list (large)
5. **Random walks**: "Walking = sampling neighborhoods" - captures local and global structure
6. **DeepWalk**: Random walks + Word2Vec = node embeddings
7. **Node2Vec**: Biased walks with p (return) and q (in-out) parameters
8. **PageRank**: "Random surfer" model - important nodes have many incoming links from important nodes

**When to use**:
- **Rich relational data**: Social networks, knowledge graphs
- **Cold start**: Leverage graph structure (user's friends' preferences)
- **Explainability**: Graph paths explain recommendations
- **Multi-hop patterns**: When direct overlap is sparse

**What can go wrong**:
- Disconnected components (no cross-component recommendations)
- Popularity bias amplification (popular items dominate)
- Short walk lengths (miss long-range dependencies)
- Wrong p/q parameters (wrong structural patterns captured)
- Ignoring edge weights (lose signal strength)
- Static graphs for dynamic data (stale recommendations)

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

---

### Problem 4: Node2Vec Transition Probabilities

**Given graph**:
```
    [A] ---- [B] ---- [C]
             |
            [D]
```

**Current position**: B
**Previous position**: A
**Parameters**: p = 2, q = 0.5

**Compute**: Transition probabilities to each neighbor of B.

**Solution**:
```
Neighbors of B: {A, C, D}

Distance from A (previous):
- A: distance 0 (return)
- C: distance 2 (A-B-C, not direct)
- D: distance 2 (A-B-D, not direct)

Unnormalized weights:
- w(A) = 1/p = 1/2 = 0.5
- w(C) = 1/q = 1/0.5 = 2
- w(D) = 1/q = 1/0.5 = 2

Sum = 0.5 + 2 + 2 = 4.5

Probabilities:
- P(A) = 0.5/4.5 = 0.111
- P(C) = 2/4.5 = 0.444
- P(D) = 2/4.5 = 0.444

Interpretation: With these parameters, the walk strongly prefers
to move outward (C or D) rather than return (A).
```

---

### Problem 5: Why Graphs Beat MF (Conceptual)

**Given** the user-item interactions from the opening:
```
Alice → Laptop, Laptop Bag
Bob → Laptop, Mouse
Carol → Mouse, Keyboard
```

**Question**: Explain why graph-based methods would recommend "Keyboard" to Alice, but MF likely wouldn't.

**Solution**:
```
Graph perspective:
1. Alice bought Laptop
2. Bob also bought Laptop (1-hop connection)
3. Bob bought Mouse
4. Carol also bought Mouse (2-hop from Alice)
5. Carol bought Keyboard (3-hop from Alice)

The graph captures the chain: Alice → Bob → Carol → Keyboard

MF perspective:
- Alice and Carol have ZERO overlap in their item vectors
- MF learns latent factors from direct interactions only
- Alice's factors trained on {Laptop, Bag}
- Carol's factors trained on {Mouse, Keyboard}
- No gradient signal connects them during training

Key insight: Graph methods propagate information transitively
through the network. MF only captures direct co-occurrence patterns.
```
