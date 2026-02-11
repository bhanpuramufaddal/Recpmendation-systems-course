# Week 9: Learning User and Item Embeddings

## Opening Problem: Why Can't We Just Use One-Hot Vectors?

*"Before we dive into embeddings, let me pose a fundamental question: You have 1 million items in your catalog. Why not just represent each item as a one-hot vector?"*

**The Naive Approach: One-Hot Encoding**

With 1M items, each item becomes a 1,000,000-dimensional vector:
```
Item 0:     [1, 0, 0, 0, ..., 0]  (1M dimensions)
Item 1:     [0, 1, 0, 0, ..., 0]
Item 999999: [0, 0, 0, 0, ..., 1]
```

**Three Fatal Problems**:

**Problem 1: Dimensionality Explosion**
```
1M items x 1M dimensions x 4 bytes = 4 TB just for item representations!

For comparison:
- Netflix: ~15,000 movies → 225MB one-hot matrix
- Amazon: ~350M products → 490 PB (!!)
- YouTube: ~800M videos → 2.56 EB (!!!)
```

**Problem 2: No Similarity Structure**

*"Here's the critical insight. What's the cosine similarity between any two different one-hot vectors?"*

$$\text{sim}(\mathbf{e}_i, \mathbf{e}_j) = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{||\mathbf{e}_i|| \cdot ||\mathbf{e}_j||} = \frac{0}{1 \cdot 1} = 0 \quad \text{for } i \neq j$$

**Every item is equally dissimilar to every other item!**

- "The Godfather" is as similar to "Goodfellas" as it is to "Finding Nemo"
- A laptop is as similar to a mouse as it is to a refrigerator
- **This violates our fundamental intuition about items**

**Problem 3: No Generalization**

*"If a user likes 'laptop', can we infer anything about 'keyboard'?"*

With one-hot: **No.** The representations share nothing.

---

**The Embedding Solution**

*"What if, instead of 1M dimensions, we used just 256 dimensions?"*

```
Item 0 (laptop):    [0.23, -0.45, 0.12, ..., 0.78]  (256 dimensions)
Item 1 (keyboard):  [0.25, -0.41, 0.15, ..., 0.72]  (close to laptop!)
Item 2 (dress):     [-0.82, 0.33, -0.56, ..., -0.21]  (far from laptop)
```

**Benefits**:
1. **Compact**: 1M items x 256 dims x 4 bytes = **1 GB** (vs 4 TB)
2. **Similarity structure**: Similar items have similar vectors
3. **Generalization**: "laptop" knowledge transfers to "keyboard"

---

### Socratic Moment: The Compression Puzzle

*"Wait - if we have 1M items, why is a 256-dimensional embedding enough?"*

**Think about it**: 256 real-valued dimensions can encode $10^{256}$ unique points (in theory). But we only need to distinguish 1M items = $10^6$ points.

**The deeper insight**: We're not just encoding identity - we're encoding **relationships**:
- Similar items should be close
- Dissimilar items should be far
- Items share latent factors (genre, style, brand, etc.)

**With 256 dimensions**, each dimension can capture a latent concept:
- Dimension 1: "formality" (formal clothes vs casual)
- Dimension 2: "tech-related" (gadgets vs non-gadgets)
- Dimension 47: "winter-appropriate" (for fashion)
- ...and so on

*"The embedding space becomes a semantic map of your item catalog."*

---

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

## From Word2Vec to Item2Vec: The Intuition

### The Language Connection

*"The key insight that revolutionized NLP - and later recommendations - came from a simple observation about language."*

**Distributional Hypothesis** (Firth, 1957): "You shall know a word by the company it keeps."

**Example**:
```
"The _____ sat on the mat."

Words that fit: cat, dog, child, professor, robot
Words that don't fit: democracy, photosynthesis, algorithm
```

Words appearing in similar contexts have similar meanings. **This is learnable!**

### Word2Vec: The Breakthrough

**Core idea**: Words appearing in similar contexts have similar meanings.

**Example**:
```
Sentence 1: "The cat sat on the mat"
Sentence 2: "The dog sat on the rug"

Context similarity:
- "cat" and "dog" appear in similar contexts -> similar embeddings
- "mat" and "rug" appear after "on the" -> similar embeddings
```

**Objective**: Learn word embeddings such that:
$$P(\text{word} | \text{context}) \text{ is maximized}$$

---

### Item2Vec Adaptation

*"Here's where it gets exciting for recommendations. What if items are like words, and user sessions are like sentences?"*

**Analogy**:
```
Word2Vec                    Item2Vec
---------                   ---------
Word       <->              Item
Sentence   <->              User session/sequence
Context    <->              Co-purchased/co-viewed items
```

**Example (E-commerce)**:
```
Session 1: [laptop, mouse, keyboard, monitor]
Session 2: [laptop, backpack, keyboard, USB drive]

Insight:
- "laptop" and "keyboard" co-occur -> similar embeddings
- "mouse" and "monitor" appear with "laptop" -> peripherals cluster
```

**Objective**: Learn item embeddings such that items co-occurring in sessions are close.

---

## Skip-Gram Derivation for Items: Step by Step

*"Let me walk you through exactly how we derive the Skip-gram objective for item embeddings."*

### Step 1: Define the Data

**Given**: User sessions (sequences of items)
```
Session 1: [A, B, C, D, E]
Session 2: [B, C, F, G]
Session 3: [A, C, D, H]
```

**Goal**: Learn embedding $\mathbf{v}_i \in \mathbb{R}^d$ for each item $i$.

---

### Step 2: Define Context

**Context window size** = 2 (items within 2 positions).

**For item C in Session 1 [A, B, C, D, E]**:
```
Center item: C
Context items: {A, B, D, E}  (2 items on each side)
```

**Training pairs** (center, context):
```
(C, A), (C, B), (C, D), (C, E)
```

---

### Step 3: Formulate the Objective

**Question**: Given center item $w$, what's the probability of seeing context item $c$?

**Intuition**: If $w$ and $c$ have similar embeddings, probability should be high.

**Softmax formulation**:
$$P(c | w) = \frac{\exp(\mathbf{v}_w^T \mathbf{v}_c)}{\sum_{c' \in V} \exp(\mathbf{v}_w^T \mathbf{v}_{c'})}$$

where:
- $\mathbf{v}_w$ = embedding of center item
- $\mathbf{v}_c$ = embedding of context item
- $V$ = vocabulary (all items)

---

### Step 4: Maximum Likelihood

**Objective**: Maximize log-likelihood over all (center, context) pairs:
$$\mathcal{L} = \sum_{(w,c) \in D} \log P(c | w)$$

Expanding:
$$\mathcal{L} = \sum_{(w,c) \in D} \left[ \mathbf{v}_w^T \mathbf{v}_c - \log \sum_{c' \in V} \exp(\mathbf{v}_w^T \mathbf{v}_{c'}) \right]$$

**Problem**: The sum over all items $V$ is expensive! For 1M items, we need 1M exponentials per training example.

---

### Step 5: Negative Sampling (The Solution)

*"Instead of computing the full softmax, what if we reformulated this as a binary classification problem?"*

**Reformulation**:
- Given (center, context) pair, is it **real** (from data) or **fake** (randomly sampled)?

**New objective**:
$$\mathcal{L} = \log \sigma(\mathbf{v}_w^T \mathbf{v}_c) + \sum_{i=1}^k \mathbb{E}_{c_n \sim P_n} [\log \sigma(-\mathbf{v}_w^T \mathbf{v}_{c_n})]$$

where:
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ = sigmoid
- $k$ = number of negative samples (typically 5-20)
- $P_n$ = negative sampling distribution

**Interpretation**:
- First term: Push positive pairs together (high dot product)
- Second term: Push negative pairs apart (low dot product)

---

### Step 6: Gradient Computation

**For positive pair** $(w, c)$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_w} = (1 - \sigma(\mathbf{v}_w^T \mathbf{v}_c)) \cdot \mathbf{v}_c$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_c} = (1 - \sigma(\mathbf{v}_w^T \mathbf{v}_c)) \cdot \mathbf{v}_w$$

**For negative pair** $(w, c_n)$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_w} = -\sigma(\mathbf{v}_w^T \mathbf{v}_{c_n}) \cdot \mathbf{v}_{c_n}$$

**Update rule** (SGD):
$$\mathbf{v}_w \leftarrow \mathbf{v}_w + \alpha \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{v}_w}$$

---

## Numerical Walkthrough: 5 Items, 3-Dimensional Embeddings

*"Let's trace through a complete example with actual numbers."*

### Setup

**Items**: A, B, C, D, E

**Sessions**:
```
Session 1: [A, B, C]
Session 2: [A, B, D]
Session 3: [C, D, E]
```

**Observation**: Items A, B frequently co-occur. Items C, D, E frequently co-occur.

**Goal**: Learn 3D embeddings where {A, B} cluster together and {C, D, E} cluster together.

---

### Initial Random Embeddings

```python
import numpy as np
np.random.seed(42)

# Random initialization (small values)
embeddings = {
    'A': np.array([0.12, -0.08, 0.15]),
    'B': np.array([-0.05, 0.11, -0.09]),
    'C': np.array([0.08, -0.03, 0.07]),
    'D': np.array([-0.10, 0.06, 0.13]),
    'E': np.array([0.03, -0.12, -0.05])
}

# Initial similarities (dot products)
print("Initial similarities:")
print(f"sim(A, B) = {np.dot(embeddings['A'], embeddings['B']):.4f}")  # -0.0273
print(f"sim(A, C) = {np.dot(embeddings['A'], embeddings['C']):.4f}")  # 0.0203
print(f"sim(C, D) = {np.dot(embeddings['C'], embeddings['D']):.4f}")  # -0.0071
```

**Initial state**: All similarities near 0 (random, unstructured).

---

### Training Step: Process Pair (A, B)

**From Session 1**: Center = A, Context = B (positive pair)

```python
# Current embeddings
v_A = np.array([0.12, -0.08, 0.15])
v_B = np.array([-0.05, 0.11, -0.09])

# Compute similarity and sigmoid
dot_product = np.dot(v_A, v_B)  # -0.0273
sigmoid_pos = 1 / (1 + np.exp(-dot_product))  # 0.4932

# Gradient for positive pair
grad_A = (1 - sigmoid_pos) * v_B  # [−0.0253, 0.0558, −0.0457]
grad_B = (1 - sigmoid_pos) * v_A  # [0.0609, −0.0406, 0.0761]

# Update (learning rate = 0.1)
lr = 0.1
v_A_new = v_A + lr * grad_A  # [0.1175, -0.0744, 0.1454]
v_B_new = v_B + lr * grad_B  # [-0.0439, 0.1059, -0.0824]

# New similarity
new_dot = np.dot(v_A_new, v_B_new)  # 0.0050
print(f"sim(A, B): {dot_product:.4f} -> {new_dot:.4f}")  # Increased!
```

**After one update**: A and B moved slightly closer (similarity increased from -0.027 to 0.005).

---

### After 100 Epochs

*"After processing all pairs multiple times, the embeddings converge:"*

```python
# Converged embeddings (simplified illustration)
final_embeddings = {
    'A': np.array([0.85, 0.45, -0.28]),
    'B': np.array([0.78, 0.52, -0.35]),   # Close to A!
    'C': np.array([-0.42, 0.73, 0.54]),
    'D': np.array([-0.38, 0.68, 0.62]),   # Close to C!
    'E': np.array([-0.45, 0.65, 0.58])    # Close to C, D!
}

print("Final similarities:")
print(f"sim(A, B) = {np.dot(final_embeddings['A'], final_embeddings['B']):.4f}")  # 0.9975 (high!)
print(f"sim(C, D) = {np.dot(final_embeddings['C'], final_embeddings['D']):.4f}")  # 0.9892 (high!)
print(f"sim(A, C) = {np.dot(final_embeddings['A'], final_embeddings['C']):.4f}")  # 0.0234 (low!)
```

**Result**:
- Items that co-occur (A-B, C-D-E) have high similarity (~0.99)
- Items that don't co-occur (A-C) have low similarity (~0.02)

---

### Visualization

```
        Dimension 2
            ^
            |
        C   |   D
          \ | /
           \|/  E
    --------|---------> Dimension 1
           /|\
          / | \
         A  |  B
            |
```

*"The embedding space has naturally organized items into clusters based on co-occurrence patterns!"*

---

## Word2Vec Architectures

### 1. Skip-Gram

**Task**: Predict context words given center word.

**Architecture**:
```
Input: Center word (one-hot)
       |
Embedding Layer (W_in)
       |
Item Embedding (d-dim)
       |
Output Layer (W_out)
       |
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
       |
Average Embedding
       |
Output Layer
       |
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

**Issue**: Denominator sums over **all items** -> $O(|V|)$ per update!

**Example**: Netflix has 10K+ movies -> 10K exponentials per gradient step!

---

### Negative Sampling Solution

**Idea**: Instead of normalizing over all items, sample a few **negative** examples.

**Binary classification formulation**:
- **Positive pair**: (center word, actual context word) -> label = 1
- **Negative pairs**: (center word, random words) -> label = 0

**Objective** (per positive pair):
$$\mathcal{L} = -\log \sigma(\mathbf{v}_w^T \mathbf{v}_c) - \sum_{i=1}^k \mathbb{E}_{c_n \sim P_n} [\log \sigma(-\mathbf{v}_w^T \mathbf{v}_{c_n})]$$

where:
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ = sigmoid
- $k$ = number of negative samples (typically 5-20)
- $P_n$ = negative sampling distribution (often $P(w)^{3/4}$ to oversample rare words)

**Benefit**: Only $O(k)$ instead of $O(|V|)$ -> 100x-1000x speedup!

---

### Negative Sampling Distribution

**Question**: How to sample negatives?

**Options**:
1. **Uniform**: $P(i) = \frac{1}{|V|}$ -> biased toward rare items
2. **Popularity**: $P(i) \propto \text{count}(i)$ -> biased toward popular items
3. **Smoothed (recommended)**: $P(i) \propto [\text{count}(i)]^{0.75}$

**Why 0.75?** Balances rare and popular items.

**Example**:
```
Item A: count = 100  -> P(A) is proportional to 100^0.75 = 31.6
Item B: count = 10   -> P(B) is proportional to 10^0.75 = 5.6
Ratio: 31.6 / 5.6 = 5.6 (vs. 10 with uniform, 100 with popularity)
```

---

## Embedding Initialization: Why It Matters

*"Should we start with random embeddings or something smarter? Let me show you why this choice matters."*

### Random Initialization

**Standard approach**: Initialize embeddings with small random values.

```python
# Small random initialization
embedding_dim = 128
num_items = 10000

# Option 1: Uniform
embeddings = np.random.uniform(-0.5/embedding_dim, 0.5/embedding_dim,
                                (num_items, embedding_dim))

# Option 2: Normal
embeddings = np.random.normal(0, 0.01, (num_items, embedding_dim))
```

**Why small values?**
- Large initial values -> large gradients -> unstable training
- Small values -> gradual, stable learning

---

### Numerical Example: Initialization Impact

*"Let me show you how initialization affects training dynamics."*

**Scenario**: 3 items, 2D embeddings

**Bad Initialization (large values)**:
```python
# Large random values
v_A = np.array([5.2, -4.8])
v_B = np.array([-3.1, 6.7])

dot_product = np.dot(v_A, v_B)  # -48.28
sigmoid = 1 / (1 + np.exp(-dot_product))  # ~0 (saturated!)

# Gradient is nearly zero!
gradient = (1 - sigmoid) * v_B  # [~-3.1, ~6.7] but scaled by ~1
```

**Problem**: With extreme dot products, sigmoid saturates and gradients vanish.

**Good Initialization (small values)**:
```python
# Small random values
v_A = np.array([0.05, -0.04])
v_B = np.array([-0.03, 0.06])

dot_product = np.dot(v_A, v_B)  # -0.0039
sigmoid = 1 / (1 + np.exp(-dot_product))  # 0.499 (in linear region!)

# Healthy gradient
gradient = (1 - sigmoid) * v_B  # [-0.015, 0.030]
```

**Result**: Gradients flow properly, learning is stable.

---

### Pre-trained Initialization

*"What if we have related embeddings from another task?"*

**Scenario**: You have Word2Vec embeddings for product names.

```python
# Pre-trained word embeddings for product names
word2vec = load_word2vec_model('product_names.bin')

# Initialize item embeddings from product name embeddings
def initialize_from_pretrained(item_names, word2vec):
    embeddings = {}
    for item_id, name in item_names.items():
        words = name.lower().split()
        valid_words = [w for w in words if w in word2vec]
        if valid_words:
            # Average word embeddings for item name
            embeddings[item_id] = np.mean([word2vec[w] for w in valid_words], axis=0)
        else:
            # Fall back to random
            embeddings[item_id] = np.random.normal(0, 0.01, word2vec.vector_size)
    return embeddings

# Items: {"laptop123": "Dell XPS 15 Laptop", "mouse456": "Wireless Gaming Mouse"}
item_embs = initialize_from_pretrained(item_names, word2vec)
```

**Benefits**:
1. **Faster convergence**: Start closer to good solution
2. **Better cold start**: New items with good names have reasonable embeddings
3. **Semantic structure**: Similar names -> similar initial embeddings

**When to use pre-trained**:
- Rich text metadata available
- Pre-trained model on similar domain
- Limited interaction data

---

### Training Dynamics Comparison

```
Loss
^
|
|  Random Init
|  ____________
| /
|/
|________ Pre-trained Init
|        \
|         \_____________
+-------------------------> Epochs

Random:     Epochs to converge: 50
Pre-trained: Epochs to converge: 15
```

**Observation**: Pre-trained initialization converges 3x faster in this example.

---

## Embedding Combination Strategies for Users

### The User Representation Problem

*"We've learned item embeddings. But how do we represent users?"*

**Given**: User $u$ interacted with items $I_u = \{i_1, i_2, ..., i_n\}$

**Goal**: Compute user embedding $\mathbf{u}$

---

### Strategy 1: Average Pooling

**Formula**:
$$\mathbf{u} = \frac{1}{|I_u|} \sum_{i \in I_u} \mathbf{v}_i$$

**Example**:
```python
# User interacted with items [laptop, keyboard, mouse]
v_laptop   = np.array([0.8, 0.2, -0.3, 0.5])
v_keyboard = np.array([0.7, 0.3, -0.2, 0.4])
v_mouse    = np.array([0.6, 0.4, -0.1, 0.3])

user_emb = (v_laptop + v_keyboard + v_mouse) / 3
# [0.7, 0.3, -0.2, 0.4]
```

**Pros**:
- Simple, fast
- Works well for homogeneous interests

**Cons**:
- All items weighted equally
- Recent items same weight as old items
- Outliers affect representation

---

### Strategy 2: Attention-Based Aggregation

*"What if some items are more important than others for defining the user?"*

**Formula**:
$$\mathbf{u} = \sum_{i \in I_u} \alpha_i \mathbf{v}_i$$

where attention weights:
$$\alpha_i = \frac{\exp(f(\mathbf{v}_i))}{\sum_{j \in I_u} \exp(f(\mathbf{v}_j))}$$

**Example**:
```python
# Attention scores (learned)
def attention_score(v):
    # Simple: dot product with learned query vector
    query = np.array([0.5, 0.5, 0.5, 0.5])  # learned
    return np.dot(v, query)

scores = {
    'laptop':   attention_score(v_laptop),    # 0.6
    'keyboard': attention_score(v_keyboard),  # 0.6
    'mouse':    attention_score(v_mouse)      # 0.6
}

# Softmax
exp_scores = {k: np.exp(v) for k, v in scores.items()}
total = sum(exp_scores.values())
weights = {k: v/total for k, v in exp_scores.items()}
# {'laptop': 0.33, 'keyboard': 0.33, 'mouse': 0.33}

# If laptop had higher score (e.g., 1.2):
# {'laptop': 0.45, 'keyboard': 0.275, 'mouse': 0.275}
```

**Pros**:
- Learns item importance
- Can focus on relevant items

**Cons**:
- More parameters to learn
- Needs sufficient data

---

### When to Use Each Strategy

*"How do you choose? Let me give you a decision framework."*

**Use Average Pooling when**:
1. **Homogeneous behavior**: User has consistent interests
2. **Limited data**: Not enough to learn attention weights
3. **Speed critical**: Inference needs to be very fast
4. **Baseline**: Always start here, then try attention

**Use Attention when**:
1. **Diverse interests**: User has multiple, distinct interests
2. **Sufficient data**: >100 interactions per user on average
3. **Sequential context**: Recent items more relevant
4. **Target-aware**: Different items relevant for different recommendations

---

### Mathematical Derivation: When Is Attention Optimal?

*"Let's formalize when attention helps."*

**Setup**: User has interacted with items from 2 distinct clusters:
- Cluster A: {laptop, keyboard} (tech)
- Cluster B: {dress, shoes} (fashion)

**Target**: Recommend tech items.

**Average Pooling**:
$$\mathbf{u}_{avg} = \frac{1}{4}(\mathbf{v}_{laptop} + \mathbf{v}_{keyboard} + \mathbf{v}_{dress} + \mathbf{v}_{shoes})$$

This is the centroid - somewhere between tech and fashion clusters.

**Attention (with tech query)**:
$$\mathbf{u}_{attn} = 0.4 \cdot \mathbf{v}_{laptop} + 0.4 \cdot \mathbf{v}_{keyboard} + 0.1 \cdot \mathbf{v}_{dress} + 0.1 \cdot \mathbf{v}_{shoes}$$

This is much closer to the tech cluster.

**Result**: For recommending tech items, $\mathbf{u}_{attn}$ will have higher similarity to tech candidates than $\mathbf{u}_{avg}$.

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

## What Can Go Wrong: Common Embedding Pitfalls

*"Let me walk you through the failure modes I've seen in production embedding systems."*

### 1. Embedding Collapse

**Symptom**: All embeddings converge to similar vectors.

**Example**:
```python
# After training, check embedding variance
emb_var = np.var(embeddings, axis=0).mean()
print(f"Embedding variance: {emb_var}")

# Healthy: 0.01 - 0.1
# Collapsed: < 0.001  # All embeddings nearly identical!
```

**Causes**:
- Learning rate too high
- Too few negative samples
- Imbalanced data (one item appears in 50% of sessions)

**Diagnosis**:
```python
# Check pairwise similarities
from sklearn.metrics.pairwise import cosine_similarity

sims = cosine_similarity(embeddings)
avg_sim = sims.mean()
print(f"Average pairwise similarity: {avg_sim}")

# Healthy: 0.0 - 0.2
# Collapsed: > 0.8  # Everything similar to everything!
```

**Solutions**:
- Reduce learning rate
- Increase negative samples (5 -> 15)
- Add L2 regularization
- Use temperature scaling in loss

---

### 2. Cold Items Problem

**Symptom**: New items have poor embeddings (random or zero).

**Example**:
```python
# New item added yesterday
new_item_id = 50001

# Only 3 interactions -> embedding barely trained
new_item_emb = embeddings[new_item_id]

# Similarity to popular items
popular_item_emb = embeddings[123]  # 10000 interactions
sim = cosine_similarity([new_item_emb], [popular_item_emb])[0][0]
print(f"Similarity: {sim}")  # Essentially random!
```

**Why it happens**:
- Embeddings are learned from co-occurrence
- No co-occurrence -> no learning signal
- New items have few/no sessions

**Solutions**:
1. **Content-based initialization**:
```python
def initialize_cold_item(item_features, content_model):
    """Use item metadata to initialize embedding."""
    # Use title, category, brand to create initial embedding
    content_emb = content_model.encode(item_features)
    return content_emb
```

2. **Similar item transfer**:
```python
def transfer_embedding(new_item, item_catalog, embeddings):
    """Copy embedding from most similar existing item."""
    # Find similar items by category/brand
    similar_items = find_similar_by_metadata(new_item, item_catalog)
    # Average their embeddings
    return np.mean([embeddings[i] for i in similar_items], axis=0)
```

3. **Popularity-biased initialization**:
```python
def popularity_init(embeddings, item_counts):
    """Initialize with weighted average of popular items."""
    weights = np.array([item_counts.get(i, 1) for i in range(len(embeddings))])
    weights = weights / weights.sum()
    return (embeddings * weights[:, np.newaxis]).sum(axis=0)
```

---

### 3. Popularity Bias in Embeddings

*"This is subtle but critical."*

**Symptom**: Popular items dominate the embedding space.

**Example**:
```python
# Popular item appears in 50% of sessions
# Rare item appears in 0.1% of sessions

# Popular item is "close" to everything (it's in context with everything!)
popular_sims = cosine_similarity([embeddings[popular_item]], embeddings)[0]
print(f"Popular item avg similarity: {popular_sims.mean()}")  # High!

rare_sims = cosine_similarity([embeddings[rare_item]], embeddings)[0]
print(f"Rare item avg similarity: {rare_sims.mean()}")  # Low (but not bad!)
```

**Why it happens**:
- Negative sampling distribution affects learning
- Popular items sampled as negatives more often
- Embeddings pushed away from popular items

**Solutions**:

1. **Smoothed negative sampling** (already discussed):
```python
# Use count^0.75 instead of raw counts
neg_probs = np.power(counts, 0.75)
neg_probs /= neg_probs.sum()
```

2. **Inverse propensity weighting**:
```python
def weighted_loss(pos_score, neg_scores, item_counts, center_item, context_item):
    """Weight loss inversely by item popularity."""
    # Less weight for popular item pairs
    weight = 1.0 / np.sqrt(item_counts[center_item] * item_counts[context_item])
    pos_loss = -weight * np.log(sigmoid(pos_score))
    neg_loss = -np.mean([np.log(sigmoid(-s)) for s in neg_scores])
    return pos_loss + neg_loss
```

3. **Popularity-debiased embeddings** (post-hoc):
```python
def debias_embeddings(embeddings, item_counts, alpha=0.5):
    """Remove popularity component from embeddings."""
    # Compute popularity embedding (weighted average)
    weights = np.array([item_counts.get(i, 1) for i in range(len(embeddings))])
    weights = weights / weights.sum()
    pop_emb = (embeddings * weights[:, np.newaxis]).sum(axis=0)

    # Remove popularity component
    debiased = embeddings - alpha * pop_emb
    return debiased
```

---

### 4. Dimension Selection: Too Few vs. Too Many

*"How do you choose embedding dimension? This is more art than science."*

**Too Few Dimensions (e.g., 16 for 1M items)**:

**Symptoms**:
- High loss that doesn't decrease
- Similar items don't cluster
- Poor downstream performance

```python
# Dimension 16 -> only 16 "concepts" to encode 1M items
# Information bottleneck!
```

**Too Many Dimensions (e.g., 1024 for 10K items)**:

**Symptoms**:
- Overfitting (train loss low, validation high)
- Embeddings don't generalize
- Slow training and inference

```python
# Dimension 1024 -> 10M parameters for 10K items
# More parameters than data points!
```

---

### Dimension Selection Guidelines

**Rule of thumb**:
$$d \approx 4 \times \sqrt[4]{|V|}$$

**Examples**:
| Items | Suggested Dim | Common Choice |
|-------|---------------|---------------|
| 1K    | 28            | 32            |
| 10K   | 50            | 64            |
| 100K  | 89            | 128           |
| 1M    | 159           | 256           |
| 10M   | 283           | 256-512       |

**Empirical Approach**:
```python
# Train with different dimensions, measure downstream task
dimensions = [32, 64, 128, 256, 512]
results = {}

for dim in dimensions:
    model = train_item2vec(sessions, embedding_dim=dim)
    embeddings = model.get_item_embeddings()

    # Evaluate on downstream task (e.g., recommendation accuracy)
    ndcg = evaluate_recommendations(embeddings, test_set)
    results[dim] = ndcg

# Plot dimension vs. performance
# Usually: increases, plateaus, sometimes decreases (overfitting)
```

---

### 5. Embedding Drift Over Time

**Symptom**: Embeddings trained on old data don't work for new user behavior.

**Example**:
```python
# Embeddings trained in January
# User behavior changes in December (holiday shopping!)

# Tech enthusiast in January: laptop, keyboard, monitor
# Same user in December: toys, gift cards, wrapping paper

# Old embeddings don't capture seasonal interests
```

**Solutions**:
1. **Regular retraining**: Monthly or weekly full retrain
2. **Incremental updates**: Fine-tune on recent data
3. **Time-aware embeddings**: Add temporal features

```python
def time_aware_embedding(user_history, item_embeddings, timestamps, current_time):
    """Weight items by recency."""
    recency_weights = []
    for t in timestamps:
        days_ago = (current_time - t).days
        weight = np.exp(-days_ago / 30)  # 30-day decay
        recency_weights.append(weight)

    weights = np.array(recency_weights)
    weights /= weights.sum()

    user_emb = (item_embeddings[user_history] * weights[:, np.newaxis]).sum(axis=0)
    return user_emb
```

---

## Evaluation

### 1. Intrinsic Evaluation

**Analogy tasks** (from Word2Vec):
$$\mathbf{v}_{king} - \mathbf{v}_{man} + \mathbf{v}_{woman} \approx \mathbf{v}_{queen}$$

**Recommendation analogy**:
```
"laptop" - "electronics" + "fashion" approximately equals "dress"
"The Godfather" - "crime" + "sci-fi" approximately equals "Blade Runner"
```

**Implementation**:
```python
def analogy(embeddings, item_ids, a, b, c, top_k=5):
    """
    Find items d such that: a - b + c is close to d

    Example: laptop - electronics + fashion is close to ?
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
User -> Item -> Category -> Item -> User
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
- Low $q$: DFS (local structure) -> homophily (similar nodes)
- High $q$: BFS (global structure) -> structural equivalence (same role)

---

### 3. Graph Convolutional Networks (GCNs)

**Limitation of Item2Vec**: Only uses co-occurrence, ignores graph structure.

**GCN approach**: Aggregate neighbor embeddings.

$$\mathbf{h}_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \frac{\mathbf{W}^{(l)} \mathbf{h}_j^{(l)}}{\sqrt{d_i d_j}}\right)$$

**See Week 7** for full GNN coverage.

---

## Production Considerations

### 1. Incremental Updates

**Problem**: New items added daily -> need to update embeddings.

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

**Challenge**: 1B items x 128 dim x 4 bytes = 512 GB!

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
# Output: Loss is approximately 1.87
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

---

### Problem 4: Embedding Dimension Justification

**Question**: You have 500,000 items. Your manager wants to use 32-dimensional embeddings to save storage. Is this a good idea?

**Analysis**:

Using the rule of thumb: $d \approx 4 \times \sqrt[4]{|V|}$
$$d \approx 4 \times \sqrt[4]{500000} = 4 \times 26.6 \approx 107$$

Recommended: **128 dimensions** (power of 2 for efficiency).

**32 dimensions is likely too small because**:
1. Information bottleneck: 32 dims cannot capture 500K item relationships
2. Empirically: Precision@K typically drops significantly below 64 dims for catalogs >100K
3. Storage savings are marginal: 500K x 32 x 4 bytes = 64 MB vs 500K x 128 x 4 bytes = 256 MB
   - Only 192 MB difference - worth it for better recommendations

**Recommendation**: Use at least 128 dimensions. The 192 MB extra storage is negligible compared to recommendation quality improvements.

---

### Problem 5: Diagnosing Embedding Collapse

**Given**: You trained Item2Vec on 100K items with 256-dimensional embeddings. After training, you observe:
- Average pairwise cosine similarity: 0.85
- All items have similar similarity to query item

**Questions**:
1. What problem does this indicate?
2. What are 3 potential causes?
3. How would you fix it?

**Solution**:

1. **Problem**: Embedding collapse - all embeddings converged to similar vectors

2. **Potential causes**:
   - Learning rate too high (gradients too large)
   - Too few negative samples (insufficient repulsion)
   - Highly imbalanced data (one item dominates)

3. **Fixes**:
   ```python
   # 1. Reduce learning rate
   lr = 0.001  # instead of 0.01

   # 2. Increase negative samples
   num_negatives = 20  # instead of 5

   # 3. Add L2 regularization
   weight_decay = 0.01
   optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

   # 4. Use temperature scaling
   temperature = 0.07
   pos_score = torch.dot(v_center, v_context) / temperature
   ```
