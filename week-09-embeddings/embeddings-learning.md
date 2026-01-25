# Week 9: Learning User and Item Embeddings

## Overview

**Embeddings** map discrete entities (users, items) into continuous vector spaces where similar entities are close together. This enables:
- **Efficient similarity computation** (dot product vs. complex features)
- **Transfer learning** (pre-compute, reuse across tasks)
- **Representation learning** (discover latent patterns)

**Key techniques**:
- **Item2Vec**: Adapt Word2Vec to recommendation (items as "words", sessions as "sentences")
- **User embeddings**: Aggregate item interactions into user representations
- **Evaluation**: Intrinsic (analogy tasks) and extrinsic (downstream RecSys metrics)

**Used by**: Spotify, Airbnb, Pinterest, Alibaba

This document covers embedding learning fundamentals for recommendation systems.

---

## Learning Objectives

By the end of this section, you will:
- Understand Word2Vec and its adaptation to recommendations
- Implement Item2Vec for learning item embeddings
- Master Skip-gram and CBOW architectures
- Apply negative sampling for efficient training
- Evaluate embedding quality

---

## From Word2Vec to Item2Vec

### Word2Vec Intuition

**Core idea**: Words appearing in similar contexts have similar meanings.

**Example**:
```
Sentence 1: "The cat sat on the mat"
Sentence 2: "The dog sat on the rug"

Context similarity:
- "cat" and "dog" appear in similar contexts → similar embeddings
- "mat" and "rug" appear after "on the" → similar embeddings
```

**Objective**: Learn word embeddings such that:
$$P(\text{word} | \text{context}) \text{ is maximized}$$

---

### Item2Vec Adaptation

**Analogy**:
```
Word2Vec                    Item2Vec
---------                   ---------
Word       ←→               Item
Sentence   ←→               User session/sequence
Context    ←→               Co-purchased/co-viewed items
```

**Example (E-commerce)**:
```
Session 1: [laptop, mouse, keyboard, monitor]
Session 2: [laptop, backpack, keyboard, USB drive]

Insight:
- "laptop" and "keyboard" co-occur → similar embeddings
- "mouse" and "monitor" appear with "laptop" → peripherals cluster
```

**Objective**: Learn item embeddings such that items co-occurring in sessions are close.

---

## Word2Vec Architectures

### 1. Skip-Gram

**Task**: Predict context words given center word.

**Architecture**:
```
Input: Center word (one-hot)
       ↓
Embedding Layer (W_in)
       ↓
Item Embedding (d-dim)
       ↓
Output Layer (W_out)
       ↓
Softmax over context words
```

**Objective**:
$$\mathcal{L} = -\sum_{(w,c) \in D} \log P(c | w)$$

where:
- $w$ = center word
- $c$ = context word
- $P(c|w) = \frac{\exp(\mathbf{v}_w^T \mathbf{v}_c)}{\sum_{c' \in V} \exp(\mathbf{v}_w^T \mathbf{v}_{c'})}$

**Problem**: Softmax over full vocabulary is slow!

---

### 2. CBOW (Continuous Bag of Words)

**Task**: Predict center word given context words.

**Architecture**:
```
Input: Context words (average embeddings)
       ↓
Average Embedding
       ↓
Output Layer
       ↓
Softmax over center word
```

**Objective**:
$$\mathcal{L} = -\sum_{(c,w) \in D} \log P(w | c_1, \ldots, c_k)$$

where context = average of context word embeddings.

**Comparison**:
| Skip-Gram | CBOW |
|-----------|------|
| Slow (multiple predictions) | Fast (single prediction) |
| Better for rare words | Better for frequent words |
| Used more often | Less common |

---

## Negative Sampling

### Problem with Softmax

**Standard softmax**:
$$P(c|w) = \frac{\exp(\mathbf{v}_w^T \mathbf{v}_c)}{\sum_{c' \in V} \exp(\mathbf{v}_w^T \mathbf{v}_{c'})}$$

**Issue**: Denominator sums over **all items** → $O(|V|)$ per update!

**Example**: Netflix has 10K+ movies → 10K exponentials per gradient step!

---

### Negative Sampling Solution

**Idea**: Instead of normalizing over all items, sample a few **negative** examples.

**Binary classification formulation**:
- **Positive pair**: (center word, actual context word) → label = 1
- **Negative pairs**: (center word, random words) → label = 0

**Objective** (per positive pair):
$$\mathcal{L} = -\log \sigma(\mathbf{v}_w^T \mathbf{v}_c) - \sum_{i=1}^k \mathbb{E}_{c_n \sim P_n} [\log \sigma(-\mathbf{v}_w^T \mathbf{v}_{c_n})]$$

where:
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ = sigmoid
- $k$ = number of negative samples (typically 5-20)
- $P_n$ = negative sampling distribution (often $P(w)^{3/4}$ to oversample rare words)

**Benefit**: Only $O(k)$ instead of $O(|V|)$ → 100x-1000x speedup!

---

### Negative Sampling Distribution

**Question**: How to sample negatives?

**Options**:
1. **Uniform**: $P(i) = \frac{1}{|V|}$ → biased toward rare items
2. **Popularity**: $P(i) \propto \text{count}(i)$ → biased toward popular items
3. **Smoothed (recommended)**: $P(i) \propto [\text{count}(i)]^{0.75}$

**Why 0.75?** Balances rare and popular items.

**Example**:
```
Item A: count = 100  → P(A) ∝ 100^0.75 = 31.6
Item B: count = 10   → P(B) ∝ 10^0.75 = 5.6
Ratio: 31.6 / 5.6 = 5.6 (vs. 10 with uniform, 100 with popularity)
```

---

## Item2Vec Implementation

### Skip-Gram with Negative Sampling

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Item2Vec(nn.Module):
    def __init__(self, num_items, embedding_dim=128):
        """
        num_items: Total number of items in catalog
        embedding_dim: Dimension of item embeddings
        """
        super().__init__()

        # Embedding matrices
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.context_embeddings = nn.Embedding(num_items, embedding_dim)

        # Initialize with small random values
        self.item_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.context_embeddings.weight.data.zero_()

    def forward(self, center_items, context_items, negative_items):
        """
        center_items: (batch,) - center item IDs
        context_items: (batch,) - positive context item IDs
        negative_items: (batch, num_negatives) - negative item IDs
        """
        # Get embeddings
        center_emb = self.item_embeddings(center_items)  # (batch, dim)
        context_emb = self.context_embeddings(context_items)  # (batch, dim)
        neg_emb = self.context_embeddings(negative_items)  # (batch, num_neg, dim)

        # Positive score
        pos_score = (center_emb * context_emb).sum(dim=1)  # (batch,)
        pos_loss = -F.logsigmoid(pos_score).mean()

        # Negative scores
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze()  # (batch, num_neg)
        neg_loss = -F.logsigmoid(-neg_score).mean()

        return pos_loss + neg_loss

    def get_item_embeddings(self):
        """Return learned item embeddings"""
        return self.item_embeddings.weight.data.cpu().numpy()


# Example usage
num_items = 10000
model = Item2Vec(num_items, embedding_dim=128)

# Sample batch
batch_size = 256
center_items = torch.randint(0, num_items, (batch_size,))
context_items = torch.randint(0, num_items, (batch_size,))
negative_items = torch.randint(0, num_items, (batch_size, 5))  # 5 negatives

# Forward pass
loss = model(center_items, context_items, negative_items)
print(f"Loss: {loss.item():.4f}")
```

---

### Creating Training Data from Sessions

```python
class SessionDataset(Dataset):
    def __init__(self, sessions, window_size=5, num_negatives=5, item_counts=None):
        """
        sessions: List of lists [[item1, item2, ...], [item5, item6, ...], ...]
        window_size: Context window size
        num_negatives: Number of negative samples per positive
        item_counts: Dict of item frequencies (for negative sampling)
        """
        self.pairs = []
        self.window_size = window_size
        self.num_negatives = num_negatives

        # Build vocabulary
        all_items = [item for session in sessions for item in session]
        self.num_items = max(all_items) + 1

        # Compute negative sampling distribution
        if item_counts is None:
            item_counts = {item: all_items.count(item) for item in set(all_items)}

        # Smoothed distribution (count^0.75)
        counts = np.array([item_counts.get(i, 0) for i in range(self.num_items)])
        self.neg_probs = np.power(counts, 0.75)
        self.neg_probs /= self.neg_probs.sum()

        # Generate (center, context) pairs
        for session in sessions:
            for i, center in enumerate(session):
                # Context window
                start = max(0, i - window_size)
                end = min(len(session), i + window_size + 1)

                for j in range(start, end):
                    if i != j:
                        context = session[j]
                        self.pairs.append((center, context))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]

        # Sample negatives
        negatives = np.random.choice(
            self.num_items,
            size=self.num_negatives,
            replace=False,
            p=self.neg_probs
        )

        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(context, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long)
        )


# Example: Create dataset from sessions
sessions = [
    [10, 25, 42, 53],  # User 1's session
    [10, 42, 100],     # User 2's session
    [25, 53, 78, 100], # User 3's session
]

dataset = SessionDataset(sessions, window_size=2, num_negatives=5)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

print(f"Training pairs: {len(dataset)}")
print(f"Vocabulary size: {dataset.num_items}")
```

---

### Training Loop

```python
def train_item2vec(sessions, embedding_dim=128, epochs=10, lr=0.025):
    """
    Complete training pipeline for Item2Vec.
    """
    # Create dataset
    dataset = SessionDataset(sessions, window_size=5, num_negatives=5)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2)

    # Initialize model
    model = Item2Vec(dataset.num_items, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for center, context, negatives in dataloader:
            optimizer.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Extract embeddings
    item_embeddings = model.get_item_embeddings()
    return item_embeddings, model


# Train on example sessions
embeddings, model = train_item2vec(sessions, embedding_dim=64, epochs=5)
print(f"Learned embeddings shape: {embeddings.shape}")
```

---

## User Embeddings

### Aggregation Strategies

**Goal**: Represent users as embeddings based on their interaction history.

**Methods**:

**1. Average of item embeddings**:
$$\mathbf{u} = \frac{1}{|I_u|} \sum_{i \in I_u} \mathbf{v}_i$$

where $I_u$ = items user $u$ interacted with.

**2. Weighted average** (by recency or rating):
$$\mathbf{u} = \frac{\sum_{i \in I_u} w_i \mathbf{v}_i}{\sum_{i \in I_u} w_i}$$

**3. Learned aggregation** (neural network):
```python
class UserEncoder(nn.Module):
    def __init__(self, item_embeddings, hidden_dim=128):
        super().__init__()
        self.item_embeddings = item_embeddings
        embed_dim = item_embeddings.weight.shape[1]

        # Attention-based aggregation
        self.attention = nn.Linear(embed_dim, 1)
        self.fc = nn.Linear(embed_dim, hidden_dim)

    def forward(self, item_ids):
        """
        item_ids: (batch, num_items) - user's item history
        """
        # Get item embeddings
        item_embs = self.item_embeddings(item_ids)  # (batch, num_items, dim)

        # Compute attention weights
        attn_scores = self.attention(item_embs).squeeze(-1)  # (batch, num_items)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, num_items, 1)

        # Weighted sum
        user_emb = (item_embs * attn_weights).sum(dim=1)  # (batch, dim)

        # Project to user space
        user_emb = self.fc(user_emb)  # (batch, hidden_dim)

        return user_emb
```

---

### Implementation: User Embeddings

```python
def compute_user_embeddings(user_histories, item_embeddings, method='average'):
    """
    user_histories: Dict {user_id: [item1, item2, ...]}
    item_embeddings: np.array (num_items, embed_dim)
    method: 'average', 'weighted', or 'attention'
    """
    user_embeddings = {}

    for user_id, items in user_histories.items():
        if len(items) == 0:
            # Cold start: use zero vector
            user_embeddings[user_id] = np.zeros(item_embeddings.shape[1])
            continue

        if method == 'average':
            # Simple average
            user_emb = item_embeddings[items].mean(axis=0)

        elif method == 'weighted':
            # Recency-weighted (more recent = higher weight)
            weights = np.exp(np.arange(len(items)) / len(items))
            weights /= weights.sum()
            user_emb = (item_embeddings[items] * weights[:, np.newaxis]).sum(axis=0)

        user_embeddings[user_id] = user_emb

    return user_embeddings


# Example
user_histories = {
    1: [10, 25, 42],
    2: [10, 100],
    3: [25, 53, 78],
}

user_embs = compute_user_embeddings(user_histories, embeddings, method='average')
print(f"User 1 embedding shape: {user_embs[1].shape}")
```

---

## Evaluation

### 1. Intrinsic Evaluation

**Analogy tasks** (from Word2Vec):
$$\mathbf{v}_{king} - \mathbf{v}_{man} + \mathbf{v}_{woman} \approx \mathbf{v}_{queen}$$

**Recommendation analogy**:
```
"laptop" - "electronics" + "fashion" ≈ "dress"
"The Godfather" - "crime" + "sci-fi" ≈ "Blade Runner"
```

**Implementation**:
```python
def analogy(embeddings, item_ids, a, b, c, top_k=5):
    """
    Find items d such that: a - b + c ≈ d

    Example: laptop - electronics + fashion ≈ ?
    """
    # Get embeddings
    v_a = embeddings[a]
    v_b = embeddings[b]
    v_c = embeddings[c]

    # Compute target vector
    target = v_a - v_b + v_c

    # Find nearest neighbors
    similarities = embeddings @ target

    # Exclude a, b, c
    similarities[[a, b, c]] = -np.inf

    # Top-K
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [(item_ids[i], similarities[i]) for i in top_indices]


# Example
result = analogy(embeddings, item_ids=list(range(len(embeddings))),
                 a=10, b=25, c=42, top_k=5)
print("Analogy results:", result)
```

---

### 2. Similarity Tasks

**Nearest neighbors**: Find similar items.

```python
def find_similar_items(embeddings, item_id, top_k=10):
    """
    Find K most similar items to item_id.
    """
    # Normalize embeddings
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Cosine similarity
    similarities = normed @ normed[item_id]

    # Exclude self
    similarities[item_id] = -1

    # Top-K
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [(i, similarities[i]) for i in top_indices]


# Example
similar = find_similar_items(embeddings, item_id=10, top_k=5)
print("Similar items to item 10:", similar)
```

---

### 3. Extrinsic Evaluation

**Downstream task**: Use embeddings as features in RecSys model.

**Metrics**:
- Ranking: NDCG, MAP, Precision@K
- Prediction: RMSE, MAE

**Example**:
```python
from sklearn.metrics.pairwise import cosine_similarity

def recommend_items(user_emb, item_embeddings, top_k=10):
    """
    Recommend items based on user embedding.
    """
    # Cosine similarity
    scores = cosine_similarity([user_emb], item_embeddings)[0]

    # Top-K
    top_indices = np.argsort(scores)[-top_k:][::-1]

    return top_indices, scores[top_indices]


# Evaluate on test set
def evaluate_recommendations(user_embeddings, item_embeddings, test_interactions):
    """
    test_interactions: Dict {user_id: [held-out items]}
    """
    from sklearn.metrics import ndcg_score

    ndcgs = []
    for user_id, true_items in test_interactions.items():
        if user_id not in user_embeddings:
            continue

        # Get recommendations
        user_emb = user_embeddings[user_id]
        scores = cosine_similarity([user_emb], item_embeddings)[0]

        # Compute NDCG
        true_relevance = np.zeros(len(item_embeddings))
        true_relevance[true_items] = 1

        ndcg = ndcg_score([true_relevance], [scores])
        ndcgs.append(ndcg)

    return np.mean(ndcgs)
```

---

## Advanced Techniques

### 1. Metapath2Vec (Heterogeneous Graphs)

**Problem**: User-item graph has **multiple** entity types (users, items, categories, brands).

**Solution**: Random walks along **metapaths**.

**Metapath example** (E-commerce):
```
User → Item → Category → Item → User
```

**Algorithm**:
1. Define metapath schema
2. Sample random walks following metapath
3. Apply Skip-gram to walks

---

### 2. Node2Vec (Flexible Random Walks)

**Idea**: Control exploration vs. exploitation in random walks.

**Parameters**:
- $p$ (return parameter): Likelihood to return to previous node
- $q$ (in-out parameter): BFS vs. DFS

**Effect**:
- Low $q$: DFS (local structure) → homophily (similar nodes)
- High $q$: BFS (global structure) → structural equivalence (same role)

---

### 3. Graph Convolutional Networks (GCNs)

**Limitation of Item2Vec**: Only uses co-occurrence, ignores graph structure.

**GCN approach**: Aggregate neighbor embeddings.

$$\mathbf{h}_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \frac{\mathbf{W}^{(l)} \mathbf{h}_j^{(l)}}{\sqrt{d_i d_j}}\right)$$

**See Week 7** for full GNN coverage.

---

## Production Considerations

### 1. Incremental Updates

**Problem**: New items added daily → need to update embeddings.

**Solutions**:
- **Retrain from scratch**: Expensive (days)
- **Fine-tune**: Initialize new items with category average, train on recent data
- **Online learning**: Update embeddings in real-time (complex)

---

### 2. Cold Start

**New items** (no interactions):
- Use **content features** (title, category) to initialize embedding
- Average embeddings of similar items (based on metadata)

**New users**:
- Use **demographic** embeddings
- Ask for initial preferences

---

### 3. Scaling to Billions

**Challenge**: 1B items × 128 dim × 4 bytes = 512 GB!

**Solutions**:
- **Quantization**: Store embeddings in int8 (4x compression)
- **Dimensionality reduction**: Use PCA to reduce to 64 or 32 dims
- **Distributed storage**: Store embeddings across multiple machines (Redis, Cassandra)

---

## Summary

**Key Takeaways**:
1. **Item2Vec**: Adapt Word2Vec to learn item embeddings from sessions
2. **Skip-gram + Negative Sampling**: Efficient training at scale
3. **User embeddings**: Aggregate item embeddings (average, weighted, attention)
4. **Evaluation**: Intrinsic (analogy, similarity) and extrinsic (downstream RecSys)

**Best Practices**:
- Embedding dim: 64-256
- Window size: 5-10
- Negative samples: 5-20
- Smoothed negative sampling: $P(i) \propto [\text{count}(i)]^{0.75}$
- Training: 5-10 epochs, learning rate 0.01-0.1

**When to use**:
- **Large catalogs**: Millions of items
- **Session data**: User sequences (clicks, purchases)
- **Transfer learning**: Pre-train embeddings, fine-tune for specific tasks

**Next**: Pre-training strategies and self-supervised learning.

---

## References

1. **Mikolov, T., et al. (2013)**. "Efficient Estimation of Word Representations in Vector Space". *ICLR*.
   - **Word2Vec**, Skip-gram, CBOW

2. **Barkan, O., & Koenigstein, N. (2016)**. "Item2Vec: Neural Item Embedding for Collaborative Filtering". *MLSP*.
   - **Item2Vec** adaptation

3. **Grover, A., & Leskovec, J. (2016)**. "node2vec: Scalable Feature Learning for Networks". *KDD*.
   - **Node2Vec** for graph embeddings

4. **Grbovic, M., et al. (2015)**. "E-commerce in Your Inbox: Product Recommendations at Scale". *KDD*.
   - **Airbnb** embedding-based search

5. **Vasile, F., et al. (2016)**. "Meta-Prod2Vec: Product Embeddings Using Side-Information for Recommendation". *RecSys*.
   - **Metadata-aware** Item2Vec

---

## Practice Problems

### Problem 1: Skip-Gram Loss

**Given**:
- Center item: $i = 5$
- Context item: $c = 10$ (positive)
- Negative items: $n_1 = 3, n_2 = 7$
- Embeddings: $\mathbf{v}_5 = [1, 0], \mathbf{v}_{10} = [0.8, 0.6], \mathbf{v}_3 = [-0.6, 0.8], \mathbf{v}_7 = [0, -1]$

**Compute**: Negative sampling loss.

**Solution**:
```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

v5 = np.array([1, 0])
v10 = np.array([0.8, 0.6])
v3 = np.array([-0.6, 0.8])
v7 = np.array([0, -1])

# Positive score
pos_score = np.dot(v5, v10)  # 1*0.8 + 0*0.6 = 0.8
pos_loss = -np.log(sigmoid(pos_score))

# Negative scores
neg_score1 = np.dot(v5, v3)  # 1*(-0.6) + 0*0.8 = -0.6
neg_score2 = np.dot(v5, v7)  # 1*0 + 0*(-1) = 0
neg_loss = -(np.log(sigmoid(-neg_score1)) + np.log(sigmoid(-neg_score2)))

total_loss = pos_loss + neg_loss
print(f"Loss: {total_loss:.4f}")
# Output: Loss ≈ 1.87
```

---

### Problem 2: User Embedding

**Given**:
- User interacted with items: [1, 5, 10]
- Item embeddings:
  - $\mathbf{v}_1 = [0.5, 0.5]$
  - $\mathbf{v}_5 = [1.0, 0.0]$
  - $\mathbf{v}_{10} = [0.0, 1.0]$

**Compute**: User embedding (average method).

**Solution**:
```python
v1 = np.array([0.5, 0.5])
v5 = np.array([1.0, 0.0])
v10 = np.array([0.0, 1.0])

user_emb = (v1 + v5 + v10) / 3
print(f"User embedding: {user_emb}")
# Output: [0.5, 0.5]
```

---

### Problem 3: Similar Items

**Given**:
- Item embeddings (normalized):
  - $\mathbf{v}_1 = [0.6, 0.8]$
  - $\mathbf{v}_2 = [0.8, 0.6]$
  - $\mathbf{v}_3 = [-0.6, 0.8]$
  - $\mathbf{v}_4 = [0.0, 1.0]$

**Find**: Most similar item to item 1.

**Solution**:
```python
v1 = np.array([0.6, 0.8])
v2 = np.array([0.8, 0.6])
v3 = np.array([-0.6, 0.8])
v4 = np.array([0.0, 1.0])

# Cosine similarities
sim_12 = np.dot(v1, v2)  # 0.6*0.8 + 0.8*0.6 = 0.96
sim_13 = np.dot(v1, v3)  # 0.6*(-0.6) + 0.8*0.8 = 0.28
sim_14 = np.dot(v1, v4)  # 0.6*0 + 0.8*1.0 = 0.80

print(f"Most similar: Item 2 (similarity = {sim_12:.2f})")
```
