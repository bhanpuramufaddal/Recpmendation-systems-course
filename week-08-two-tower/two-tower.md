# Week 8: Two-Tower Models - Architecture & Design

## Overview

**Two-tower models** (also called **dual encoder** or **siamese networks**) separately encode users and items into embeddings, enabling **fast candidate retrieval** at scale.

**Key idea**:
- **User tower**: $\mathbf{u} = f_{\text{user}}(\text{user features})$
- **Item tower**: $\mathbf{v} = f_{\text{item}}(\text{item features})$
- **Score**: $\text{score}(u, i) = \mathbf{u}^T \mathbf{v}$ (dot product)

**Critical property**: User and item embeddings computed **independently** - can precompute item embeddings - fast retrieval.

**Used by**: YouTube, Pinterest, Spotify, LinkedIn, TikTok

This document covers two-tower architecture fundamentals.

---

## Learning Objectives

By the end of this section, you will:
- Understand two-tower architecture and its benefits
- Master user and item encoding strategies
- Implement two-tower models in PyTorch
- Apply negative sampling and in-batch negatives
- Optimize two-tower models for production

---

## Opening Problem: Why Can't We Just Score All Items?

*"Let me ask you a question that should bother you: Why don't we just run our best ranking model on every item in the catalog?"*

### The Brute Force Fantasy

Imagine you have a sophisticated neural ranking model $f(u, i, \text{context})$ that takes user features, item features, and context, then outputs a relevance score. This model has learned rich cross-feature interactions - exactly what we want for accurate ranking.

**Why not just apply it to everything?**

Let's do the math for real systems:

### YouTube Scale Calculation

| Metric | Value |
|--------|-------|
| Monthly active users | 2.5 billion |
| Videos in catalog | 800 million |
| User-video pairs | $2.5B \times 800M = 2 \times 10^{18}$ |
| Ranking model latency | 1ms per item |
| Time to score all items for one user | $800M \times 1ms = 800,000$ seconds |
| **That's...** | **9.3 days per user!** |

And users expect recommendations in **under 200 milliseconds**.

### Netflix Scale

| Metric | Value |
|--------|-------|
| Subscribers | 230 million |
| Titles | 17,000 |
| User-title pairs | $230M \times 17K = 3.9 \times 10^{12}$ |
| Scoring at 100K/second | 39 million seconds |
| **That's...** | **1.2 years** for one global refresh |

*"Can you see why brute force fails? The problem isn't computational power - it's fundamental combinatorics. We need a different approach entirely."*

---

## Why Two-Tower?

### The Scalability Problem

**Traditional approach**: Score all user-item pairs with complex model.

$$\text{score}(u, i) = f(u, i, \text{context})$$

**Problem**: At inference, need to score **millions of items** per user - slow!

**Example** (YouTube):
- 2B users
- 800M videos
- Scoring all pairs: $2B \times 800M = 1.6 \times 10^{18}$ scores!
- Even at 1M scores/second: **50 years** per user!

---

### The Key Insight: Decomposition Enables Precomputation

*"Here's the insight that makes large-scale recommendations possible. What if we could decompose the scoring function?"*

#### Step-by-Step Derivation

**Step 1: The Decomposition Assumption**

Instead of a general function $f(u, i)$, we constrain ourselves to:

$$\text{score}(u, i) = \mathbf{u}^T \mathbf{v}$$

where $\mathbf{u} \in \mathbb{R}^d$ depends **only on user features** and $\mathbf{v} \in \mathbb{R}^d$ depends **only on item features**.

**Step 2: The Independence Property**

This constraint gives us something magical - **independence**:

$$\mathbf{u} = f_{\text{user}}(\text{user}_\text{features})$$
$$\mathbf{v} = f_{\text{item}}(\text{item}_\text{features})$$

*"Notice what's NOT in these equations? The user tower has no item features. The item tower has no user features. This is not a bug - it's the key feature!"*

**Step 3: Why Independence Enables Precomputation**

Since $\mathbf{v}$ depends only on item features (which change infrequently):
- Compute all item embeddings **offline** (e.g., nightly batch job)
- Store 800M embeddings $\times$ 128 dimensions = ~400GB
- Index with ANN structure (FAISS, ScaNN)

Since $\mathbf{u}$ depends only on user features:
- Compute **at request time** (just one embedding!)
- ~2ms to encode user

**Step 4: The Retrieval Trick**

Finding top-K items is now:

$$\text{TopK}(u) = \arg\max_{i \in \text{all items}} \mathbf{u}^T \mathbf{v}_i$$

This is **Maximum Inner Product Search (MIPS)** - solved efficiently by ANN algorithms!

| Operation | Time |
|-----------|------|
| Compute user embedding | 2ms |
| ANN search (800M items) | 5ms |
| **Total** | **<10ms** |

*"We went from 9 days to 10 milliseconds. That's a speedup of about 80 million times. All because we accepted the constraint of decomposition."*

---

### Two-Tower Solution

**Decomposition**:
$$\text{score}(u, i) = \mathbf{u}^T \mathbf{v}$$

where $\mathbf{u}, \mathbf{v}$ are learned embeddings.

**Benefits**:

**1. Precomputation**:
- Compute item embeddings **offline** (once per day)
- Store in index (FAISS, ScaNN)

**2. Fast retrieval**:
- At inference: Compute user embedding $\mathbf{u}$
- Query ANN index: Find nearest items to $\mathbf{u}$
- **Latency**: <10ms (vs. minutes with full model)

**3. Scalability**:
- Can handle billions of items
- Distributed storage/retrieval

---

## Numerical Walkthrough: Complete Example

*"Let me work through a complete example so you can see exactly how the math works. We'll use small numbers, but the principles scale."*

### Setup: 3 Users, 5 Items

**User features** (demographics + history summary):
| User | Age (norm) | Watch hours (norm) | Likes comedy | Likes action |
|------|-----------|-------------------|--------------|--------------|
| $u_1$ | 0.3 | 0.8 | 1 | 0 |
| $u_2$ | 0.7 | 0.4 | 0 | 1 |
| $u_3$ | 0.5 | 0.6 | 1 | 1 |

**Item features** (content + metadata):
| Item | Duration (norm) | Is comedy | Is action | Popularity |
|------|----------------|-----------|-----------|------------|
| $i_1$ | 0.2 | 1 | 0 | 0.9 |
| $i_2$ | 0.8 | 0 | 1 | 0.7 |
| $i_3$ | 0.5 | 1 | 1 | 0.5 |
| $i_4$ | 0.3 | 0 | 0 | 0.3 |
| $i_5$ | 0.6 | 1 | 0 | 0.8 |

### Step 1: User Tower Forward Pass

Suppose our trained user tower is a simple 2-layer network producing 3D embeddings:

$$\mathbf{u} = \text{normalize}(W_2 \cdot \text{ReLU}(W_1 \cdot \mathbf{x}_u + b_1) + b_2)$$

For user $u_1$ with features $[0.3, 0.8, 1, 0]$:

```
Input: [0.3, 0.8, 1.0, 0.0]
Hidden (after ReLU): [0.7, 1.2, 0.5]
Output (before norm): [0.6, 0.8, 0.4]
After L2 norm: [0.55, 0.73, 0.37]
```

**All user embeddings** (after normalization):

| User | $\mathbf{u}$ |
|------|--------------|
| $u_1$ | $[0.55, 0.73, 0.37]$ |
| $u_2$ | $[0.71, 0.41, 0.58]$ |
| $u_3$ | $[0.63, 0.57, 0.52]$ |

### Step 2: Item Tower Forward Pass

Similarly, the item tower produces:

| Item | $\mathbf{v}$ |
|------|--------------|
| $i_1$ | $[0.58, 0.69, 0.43]$ |
| $i_2$ | $[0.67, 0.33, 0.67]$ |
| $i_3$ | $[0.60, 0.52, 0.60]$ |
| $i_4$ | $[0.45, 0.45, 0.77]$ |
| $i_5$ | $[0.54, 0.76, 0.36]$ |

### Step 3: Score Computation (Dot Products)

*"Now we compute ALL user-item scores with simple dot products."*

$$\text{score}(u_1, i_1) = \mathbf{u}_1^T \mathbf{v}_1 = 0.55 \times 0.58 + 0.73 \times 0.69 + 0.37 \times 0.43$$

$$= 0.319 + 0.504 + 0.159 = \mathbf{0.982}$$

**Full Score Matrix**:

|  | $i_1$ | $i_2$ | $i_3$ | $i_4$ | $i_5$ |
|--|-------|-------|-------|-------|-------|
| $u_1$ | **0.98** | 0.85 | 0.93 | 0.86 | **0.99** |
| $u_2$ | 0.84 | **0.99** | **0.96** | 0.82 | 0.90 |
| $u_3$ | 0.93 | 0.93 | **0.96** | 0.93 | 0.93 |

### Step 4: Ranking

**Top-2 recommendations per user**:
- $u_1$: $i_5$ (0.99), $i_1$ (0.98) - Comedy lover gets comedies!
- $u_2$: $i_2$ (0.99), $i_3$ (0.96) - Action lover gets action!
- $u_3$: $i_3$ (0.96), tied rest - Mixed taste gets comedy-action hybrid!

*"Notice how the learned embeddings captured preferences even though we never explicitly programmed 'comedy lovers should get comedies'!"*

---

## Architecture

### General Two-Tower Framework

```
User Features            Item Features
(demographics, history)  (content, metadata)
       |                        |
  User Tower              Item Tower
  (Neural Net)            (Neural Net)
       |                        |
User Embedding          Item Embedding
    u (d-dim)               v (d-dim)
       |                        |
         Dot Product: u^T v
              |
           Score
```

---

### Architecture Intuition: Why Separate Towers?

*"Students often ask: why not share weights between towers? Let's think carefully about what information flows where."*

#### What Information CAN Flow

**User Tower receives**:
- User demographics (age, location, gender)
- Aggregated interaction history (avg embedding of watched items)
- Session context (time, device)

**Item Tower receives**:
- Item content (title embedding, description)
- Item metadata (category, creator, duration)
- Item statistics (popularity, avg rating)

#### What Information CANNOT Flow

**Critical constraint**: No cross-tower information at encoding time!

| This is FORBIDDEN | Why it breaks precomputation |
|-------------------|------------------------------|
| "Does user $u$ like this item's category?" | Need to know item at user encoding time |
| "Is this item popular among users like $u$?" | Need to know user at item encoding time |
| "Did $u$'s friends like item $i$?" | Requires joint computation |

*"Can you see the tradeoff? We sacrifice expressiveness for scalability. The model cannot learn 'User A likes sci-fi only when it's from the 2010s' because that requires knowing both the user and item simultaneously."*

#### Socratic Check

*"What would happen if we allowed the user tower to see the specific item being scored?"*

**Answer**: We'd lose precomputation! The item embedding would depend on the user, so we'd need to recompute all 800M embeddings per request. We're back to 9 days.

---

### User Tower

**Input**: User features
- Demographics: age, gender, location
- History: past clicks, watches, purchases (aggregated)
- Context: time of day, device, season

**Architecture**: Multi-layer perceptron (MLP)

```
User Features (sparse + dense)
       |
  Embedding Layers (for categorical)
       |
  Concatenate
       |
  MLP (FC layers + ReLU)
       |
  User Embedding (d-dim, L2 normalized)
```

---

### Item Tower

**Input**: Item features
- Content: title, description, tags
- Metadata: category, price, brand
- Popularity: click count, rating

**Architecture**: Similar MLP (can share structure with user tower)

```
Item Features
       |
  Embedding Layers
       |
  Concatenate
       |
  MLP
       |
  Item Embedding (d-dim, L2 normalized)
```

---

## Implementation

### Basic Two-Tower Model

```python
import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, user_feature_dim, item_feature_dim, embedding_dim=128, hidden_dims=[256, 128]):
        super().__init__()

        # User tower
        user_layers = []
        input_dim = user_feature_dim
        for hidden_dim in hidden_dims:
            user_layers.append(nn.Linear(input_dim, hidden_dim))
            user_layers.append(nn.ReLU())
            user_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        user_layers.append(nn.Linear(input_dim, embedding_dim))
        self.user_tower = nn.Sequential(*user_layers)

        # Item tower
        item_layers = []
        input_dim = item_feature_dim
        for hidden_dim in hidden_dims:
            item_layers.append(nn.Linear(input_dim, hidden_dim))
            item_layers.append(nn.ReLU())
            item_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        item_layers.append(nn.Linear(input_dim, embedding_dim))
        self.item_tower = nn.Sequential(*item_layers)

    def forward(self, user_features, item_features):
        """
        user_features: (batch, user_feature_dim)
        item_features: (batch, item_feature_dim)
        """
        # Encode
        user_emb = self.user_tower(user_features)  # (batch, embedding_dim)
        item_emb = self.item_tower(item_features)  # (batch, embedding_dim)

        # L2 normalize
        user_emb = nn.functional.normalize(user_emb, p=2, dim=1)
        item_emb = nn.functional.normalize(item_emb, p=2, dim=1)

        # Dot product (element-wise multiply + sum)
        scores = (user_emb * item_emb).sum(dim=1)  # (batch,)

        return scores, user_emb, item_emb


# Example
user_feature_dim = 50  # e.g., demographics + aggregated history
item_feature_dim = 30  # e.g., title embedding + metadata

model = TwoTowerModel(user_feature_dim, item_feature_dim, embedding_dim=128)

# Sample batch
batch_size = 32
user_features = torch.randn(batch_size, user_feature_dim)
item_features = torch.randn(batch_size, item_feature_dim)

scores, user_embs, item_embs = model(user_features, item_features)
print(f"Scores shape: {scores.shape}")  # (32,)
print(f"User embeddings shape: {user_embs.shape}")  # (32, 128)
```

---

## Training Strategies

### 1. Pointwise Loss

**Positive samples**: (user, item) pairs user interacted with.

**Loss**: Binary cross-entropy

$$\mathcal{L}_{\text{pointwise}} = -\sum_{(u,i)} [y_{ui} \log(\sigma(\mathbf{u}^T \mathbf{v})) + (1 - y_{ui}) \log(1 - \sigma(\mathbf{u}^T \mathbf{v}))]$$

**Problem**: Need explicit negatives (items user didn't interact with).

---

### 2. Pairwise Loss (Triplet Loss)

**Idea**: User embedding should be closer to positive item than negative item.

$$\mathcal{L}_{\text{triplet}} = \sum_{(u, i^+, i^-)} \max(0, \text{margin} + d(\mathbf{u}, \mathbf{v}_{i^-}) - d(\mathbf{u}, \mathbf{v}_{i^+}))$$

where:
- $\mathbf{u}$ = user embedding vector
- $\mathbf{v}_{i^+}$ = positive item embedding (item user interacted with)
- $\mathbf{v}_{i^-}$ = negative item embedding (item user didn't interact with)
- $d(\cdot, \cdot)$ = distance function measuring dissimilarity between embeddings
  - Common choices:
    - Euclidean distance: $d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2$
    - Cosine distance: $d(\mathbf{a}, \mathbf{b}) = 1 - \frac{\mathbf{a}^T \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$
    - Dot product distance: $d(\mathbf{a}, \mathbf{b}) = 1 - \mathbf{a}^T \mathbf{b}$ (when embeddings are normalized)
- $\text{margin}$ = minimum separation between positive and negative distances (typical values: 0.1-0.5)

**Intuition**: Loss is zero when positive item is closer than negative item by at least the margin. Otherwise, loss encourages increasing the gap.

**Implementation**:
```python
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, user_emb, pos_item_emb, neg_item_emb):
        """
        user_emb: (batch, dim)
        pos_item_emb: (batch, dim)
        neg_item_emb: (batch, dim)
        """
        pos_score = (user_emb * pos_item_emb).sum(dim=1)  # (batch,)
        neg_score = (user_emb * neg_item_emb).sum(dim=1)  # (batch,)

        loss = torch.relu(self.margin + neg_score - pos_score).mean()
        return loss
```

---

### 3. In-Batch Negatives (Efficient!)

**Key insight**: Within a batch, other users' positive items are negatives for current user.

**Example batch**:
```
User 1 -> Item A (positive)
User 2 -> Item B (positive)
User 3 -> Item C (positive)

For User 1:
  Positive: Item A
  Negatives: Item B, Item C (from batch)
```

**Advantage**: No need to sample negatives explicitly - more efficient.

**Loss** (Sampled SoftMax):
$$\mathcal{L} = -\log \frac{\exp(\mathbf{u}_i^T \mathbf{v}_i)}{\sum_{j \in \text{batch}} \exp(\mathbf{u}_i^T \mathbf{v}_j)}$$

---

### In-Batch Negatives: Derivation and Efficiency Analysis

*"Let me show you why in-batch negatives is such a clever trick. It's not just convenient - it's mathematically efficient."*

#### The Naive Approach

Without in-batch negatives, for batch size $B$ with $K$ negatives each:

| Operation | Count |
|-----------|-------|
| Positive pairs | $B$ |
| Negative samples needed | $B \times K$ |
| Total item embeddings | $B + B \times K$ |
| Forward passes through item tower | $B \times (1 + K)$ |

For $B = 256$ and $K = 100$: **25,856 item embeddings** per batch.

#### The In-Batch Trick

With in-batch negatives:

| Operation | Count |
|-----------|-------|
| Positive pairs | $B$ |
| Negative samples | $B \times (B-1)$ (free!) |
| Total item embeddings | $B$ (already computed!) |
| Forward passes through item tower | $B$ |

For $B = 256$: Only **256 item embeddings**, but **65,280 negative pairs**!

#### The Math

We compute the score matrix:

$$S = U \cdot V^T \in \mathbb{R}^{B \times B}$$

where:
- $U \in \mathbb{R}^{B \times d}$ = user embeddings (batch)
- $V \in \mathbb{R}^{B \times d}$ = item embeddings (batch)

For user $i$:
- $S_{ii}$ = positive score (diagonal)
- $S_{ij}$ for $j \neq i$ = negative scores (off-diagonal)

**Loss for user $i$**:

$$\mathcal{L}_i = -\log \frac{\exp(S_{ii} / \tau)}{\sum_{j=1}^{B} \exp(S_{ij} / \tau)}$$

*"Can you see why this is elegant? One matrix multiplication gives us ALL scores. The cross-entropy loss treats it as a $B$-way classification: 'Which of these $B$ items did user $i$ actually click?'"*

#### Efficiency Comparison

| Metric | Naive ($K=100$) | In-Batch ($B=256$) |
|--------|-----------------|-------------------|
| Negatives per user | 100 | 255 |
| Item tower forwards | 25,856 | 256 |
| **Speedup** | 1x | **~100x** |

---

### In-Batch Negatives Implementation

```python
def in_batch_negatives_loss(user_embs, item_embs, temperature=0.1):
    """
    Compute in-batch negatives loss.

    user_embs: (batch, dim)
    item_embs: (batch, dim)
    temperature: scaling factor
    """
    batch_size = user_embs.size(0)

    # Compute all pairwise scores
    scores = torch.mm(user_embs, item_embs.T) / temperature  # (batch, batch)

    # Labels: diagonal (user i paired with item i)
    labels = torch.arange(batch_size).to(user_embs.device)

    # Cross-entropy loss
    loss = nn.functional.cross_entropy(scores, labels)

    return loss


# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch in data_loader:
        user_features, item_features = batch

        # Forward
        scores, user_embs, item_embs = model(user_features, item_features)

        # Loss (in-batch negatives)
        loss = in_batch_negatives_loss(user_embs, item_embs, temperature=0.1)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Why it works**: Batch size = 256 - each user has 255 negatives - strong learning signal.

---

## What Can Go Wrong: Failure Modes

*"Two-tower models are powerful but fragile. Let me walk you through the ways they fail - understanding these will make you a better practitioner."*

### Failure Mode 1: Embedding Collapse

**Symptom**: All embeddings converge to the same point (or small cluster).

**What happens**:
- Model finds "lazy" solution: map everything to same embedding
- All scores become identical (~0 after normalization)
- Loss appears low but recommendations are random

**Root causes**:
- Temperature too high (gradients too weak)
- Learning rate too high (overshooting)
- Not enough negatives
- Missing L2 normalization

**Detection**:
```python
# Check embedding diversity
def check_collapse(embeddings):
    # Compute pairwise cosine similarities
    normed = F.normalize(embeddings, dim=1)
    sim_matrix = normed @ normed.T

    # Off-diagonal mean should be low (diverse embeddings)
    mask = ~torch.eye(len(embeddings), dtype=bool)
    avg_sim = sim_matrix[mask].mean()

    if avg_sim > 0.9:
        print(f"WARNING: Possible collapse! Avg similarity: {avg_sim:.3f}")
    return avg_sim
```

**Solutions**:
- Lower temperature (0.05-0.1)
- Add uniformity loss (push embeddings apart)
- Increase batch size
- Gradient clipping

---

### Failure Mode 2: Feature Leakage

**Symptom**: Model performs great offline but fails online.

**What happens**:
- Training data contains features that implicitly encode the label
- Model learns to "cheat" instead of learning preferences

**Examples**:
| Leaked Feature | Why It's Problematic |
|----------------|---------------------|
| Item click count during training period | Includes future clicks we're trying to predict |
| "Trending" flag computed on full data | Uses information from test period |
| User's average rating for this category | Computed including the target item |

**The temporal leakage trap**:
```python
# WRONG: User history includes items from "future"
user_history = get_all_interactions(user_id)  # Includes today's clicks!
user_embedding = encode_user(user_history)
predict(user_embedding, today_items)  # Data leakage!

# CORRECT: Point-in-time features
user_history = get_interactions_before(user_id, timestamp)
user_embedding = encode_user(user_history)
predict(user_embedding, items_at_timestamp)
```

**Detection**: Large gap between offline and online metrics.

**Solutions**:
- Strict temporal splits (never use future data)
- Feature engineering audits
- Ablation studies (remove suspicious features)

---

### Failure Mode 3: Cold Items / Cold Users

**Symptom**: New items never get recommended; new users get poor recommendations.

**What happens for cold items**:
- Item has no interaction history
- If item features are weak, embedding is essentially random
- ANN retrieval never surfaces it

**What happens for cold users**:
- User has no history (or very little)
- User embedding defaults to "average user"
- Gets generic popular items (popularity bias)

**The math of cold start**:

For a new item $i_{\text{new}}$ with only content features $\mathbf{x}_i$:

$$\mathbf{v}_{\text{new}} = f_{\text{item}}(\mathbf{x}_i, \underbrace{\mathbf{0}}_{\text{no engagement history}})$$

If item tower relies heavily on engagement features, $\mathbf{v}_{\text{new}}$ is under-determined.

**Solutions**:

| Strategy | Description |
|----------|-------------|
| **Content bootstrapping** | Use rich content features (BERT embeddings of title/description) |
| **Exploration** | Bandits to surface cold items to gather feedback |
| **Hybrid retrieval** | Separate retrieval path for new items |
| **Feature imputation** | Use category averages for missing engagement stats |

---

### Failure Mode 4: Popularity Bias Amplification

**Symptom**: Model only recommends popular items; long-tail items never surface.

**What happens**:
- Popular items appear more in training data
- Model learns to score them higher
- They get recommended more - more clicks - more training signal
- **Feedback loop!**

**The math**:

$$P(\text{item } i \text{ recommended}) \propto P(\text{item } i \text{ in training})^{\alpha}$$

where $\alpha > 1$ due to the feedback loop.

**Detection**:
```python
# Gini coefficient of recommendation frequency
def recommendation_gini(rec_counts):
    sorted_counts = np.sort(rec_counts)
    n = len(sorted_counts)
    cumulative = np.cumsum(sorted_counts)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_counts))) / (n * np.sum(sorted_counts)) - (n+1)/n
    return gini

# Gini > 0.8 suggests severe popularity bias
```

**Solutions**:
- Inverse propensity scoring (IPS) in loss
- Popularity-stratified sampling
- Explicit diversity in retrieval
- Separate retrieval path for long-tail

---

### Failure Mode 5: Stale Embeddings

**Symptom**: Recommendations become less relevant over time.

**What happens**:
- Item embeddings computed in batch job (e.g., daily)
- Item features change (new reviews, price changes, trending status)
- Embeddings don't reflect current state

**Example timeline**:
```
Day 1, 00:00: Compute item embeddings
Day 1, 14:00: Item goes viral on social media
Day 1, 14:01-23:59: Stale embedding, item under-recommended
Day 2, 00:00: Finally updated!
```

**Solutions**:
- More frequent refresh (hourly for dynamic features)
- Real-time embedding updates for trending items
- Hybrid: static base + dynamic delta

---

### Failure Mode 6: Sampling Bias in In-Batch Negatives

**Symptom**: Model underperforms on rare items.

**What happens**:
- In-batch negatives samples from **training distribution**
- Popular items appear in more batches - harder negatives for them
- Rare items have easier negatives - less learning signal

*"Can you see the irony? We wanted more negatives to train better, but we got biased negatives that hurt tail items."*

**The math**:

If item $i$ appears with frequency $p_i$ in training:
- Probability of $i$ as negative: $\propto p_i$
- Popular items: over-represented as negatives
- Rare items: under-represented

**Solutions**:
- **Logit correction** (YouTube paper): Subtract $\log(p_i)$ from logits
- **Mixed negatives**: Some in-batch + some uniformly sampled
- **Popularity-weighted sampling**: Upsample rare items in batches

```python
def corrected_in_batch_loss(user_embs, item_embs, item_frequencies, temperature=0.1):
    scores = torch.mm(user_embs, item_embs.T) / temperature

    # Log correction for sampling bias
    log_correction = torch.log(item_frequencies + 1e-8)
    corrected_scores = scores - log_correction.unsqueeze(0)

    labels = torch.arange(len(user_embs)).to(user_embs.device)
    return F.cross_entropy(corrected_scores, labels)
```

---

## Feature Engineering

### User Features

**1. Demographic**:
- Age, gender, location (categorical - embeddings)

**2. Historical Interactions**:
- Aggregate past clicks/watches:
  - Average embedding of past items
  - Count of interactions per category
  - Time since last interaction

**3. Contextual**:
- Time of day (morning, afternoon, evening)
- Device type (mobile, desktop, TV)
- Session duration

---

### Item Features

**1. Content**:
- Title/description (BERT embeddings)
- Tags/categories (multi-hot encoding)

**2. Metadata**:
- Creator/brand
- Price (numerical)
- Release date (time-based features)

**3. Engagement Signals**:
- Total clicks, views
- Average rating
- Click-through rate

---

## Normalization

### Why Normalize Embeddings?

**Dot product without normalization**:
$$\mathbf{u}^T \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)$$

**Problem**: Magnitude affects score (high-norm vectors dominate).

**Solution**: L2 normalize - only angle matters.

$$\text{score} = \frac{\mathbf{u}}{\|\mathbf{u}\|} \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|} = \cos(\theta)$$

**Implementation**:
```python
user_emb = nn.functional.normalize(user_emb, p=2, dim=1)
item_emb = nn.functional.normalize(item_emb, p=2, dim=1)
```

---

## Temperature Scaling

**Problem**: Dot products can be small (close to 0) - hard to optimize.

**Solution**: Scale by temperature $\tau$.

$$\text{logits} = \frac{\mathbf{u}^T \mathbf{v}}{\tau}$$

**Effect**:
- Small $\tau$ (e.g., 0.05): Sharper distribution (more confident)
- Large $\tau$ (e.g., 1.0): Smoother distribution (less confident)

**Typical**: $\tau = 0.07$ (from CLIP paper).

*"What would happen if you used $\tau = 0.001$? The softmax would become nearly one-hot - extreme overconfidence. What about $\tau = 10$? Uniform distribution - no learning signal. The sweet spot is problem-dependent."*

---

## Production Deployment

### Offline: Item Embedding Computation

```python
# Compute embeddings for all items (e.g., daily batch job)
def compute_item_embeddings(model, all_items):
    """
    all_items: DataFrame with item features
    """
    model.eval()
    item_embeddings = {}

    with torch.no_grad():
        for _, item in all_items.iterrows():
            item_features = preprocess_item(item)  # Convert to tensor
            _, item_emb = model.item_tower(item_features)
            item_emb = nn.functional.normalize(item_emb, p=2, dim=1)
            item_embeddings[item['id']] = item_emb.cpu().numpy()

    return item_embeddings

# Save to file
import pickle
with open('item_embeddings.pkl', 'wb') as f:
    pickle.dump(item_embeddings, f)
```

---

### Online: User Embedding + ANN Retrieval

```python
# At inference time
def recommend(user_features, top_k=10):
    """
    Real-time recommendation.
    """
    # 1. Compute user embedding
    user_tensor = preprocess_user(user_features)
    with torch.no_grad():
        user_emb, _ = model.user_tower(user_tensor)
        user_emb = nn.functional.normalize(user_emb, p=2, dim=1)

    # 2. Query ANN index (FAISS)
    distances, indices = faiss_index.search(user_emb.cpu().numpy(), top_k)

    # 3. Return top-K items
    recommended_items = [item_id_map[idx] for idx in indices[0]]

    return recommended_items
```

**Latency**: <10ms (user encoding ~2ms, ANN search ~5ms).

---

## Advantages & Limitations

### Advantages

1. **Scalable**: Can handle billions of items
2. **Fast**: Precompute item embeddings - fast retrieval
3. **Flexible**: Can add new items easily (just encode)
4. **Interpretable**: Embeddings can be visualized

---

### Limitations

1. **Limited expressiveness**: Dot product less expressive than full cross-features
2. **No feature crosses**: Can't model user-item interactions (e.g., "user likes sci-fi + movies from 2010s")
3. **Cold start**: New users with no history - weak embeddings

**Solution**: Use two-tower for **candidate retrieval** (stage 1), then **ranking model** (stage 2) for fine-grained scoring.

---

## Socratic Review Questions

*"Before we wrap up, let me pose some questions to test your understanding. Think about these carefully."*

1. **Why must the towers be independent?**
   - What breaks if the user tower sees item features?
   - What's the complexity difference?

2. **Can you see why temperature matters for in-batch negatives?**
   - What happens with $\tau \to 0$?
   - What happens with $\tau \to \infty$?

3. **What would happen if we trained without L2 normalization?**
   - How would popular items behave?
   - Could the model exploit this?

4. **Why is sampling bias correction important?**
   - Which items suffer most from in-batch negative bias?
   - How does the correction help?

5. **When would a two-tower model fail completely?**
   - Consider recommendation scenarios where user-item feature interaction is essential.

---

## Summary

**Key Takeaways**:
1. **Two-tower**: Separate user and item encoding - fast retrieval
2. **Scalability**: Precompute item embeddings - query with ANN
3. **Training**: In-batch negatives (efficient), triplet loss
4. **Normalization**: L2 normalize embeddings (only angle matters)
5. **Production**: Two-stage (retrieval + ranking)

**Best Practices**:
- Embedding dim: 128-256
- Hidden layers: [512, 256, 128]
- Dropout: 0.2
- Temperature: 0.05-0.1
- Batch size: 256-1024 (for in-batch negatives)

**When to use**:
- **Candidate retrieval**: First stage (1000 candidates from millions)
- **Large scale**: Billions of items
- **Real-time**: Need <10ms latency

**Next**: YouTube's two-tower recommendation system (real-world case study).

---

## References

1. **Covington, P., Adams, J., & Sargin, E. (2016)**. "Deep Neural Networks for YouTube Recommendations". *RecSys*.
   - **YouTube's two-tower** architecture

2. **Yi, X., et al. (2019)**. "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations". *RecSys*.
   - **In-batch negatives**, YouTube

3. **Radford, A., et al. (2021)**. "Learning Transferable Visual Models From Natural Language Supervision". *ICML*.
   - **CLIP** (two-tower for vision-language), temperature scaling

4. **Huang, P.-S., et al. (2020)**. "Embedding-based Retrieval in Facebook Search". *KDD*.
   - **Facebook's two-tower** for search

5. **Zhai, A., et al. (2020)**. "Visual Discovery at Pinterest". *WWW*.
   - **Pinterest's two-tower** for image recommendations

---

## Practice Problems

### Problem 1: Dot Product Score

**Given**:
```
User embedding: [0.6, 0.8] (L2 normalized)
Item embedding: [0.8, 0.6] (L2 normalized)
```

**Compute**: Dot product score.

**Solution**:
```
score = 0.6 * 0.8 + 0.8 * 0.6
     = 0.48 + 0.48
     = 0.96
```

---

### Problem 2: In-Batch Negatives

**Given batch**:
```
User 1 -> Item A
User 2 -> Item B
User 3 -> Item C

User embeddings: U = [[u1], [u2], [u3]]  (3 x d)
Item embeddings: V = [[vA], [vB], [vC]]  (3 x d)
```

**Compute**: Score matrix (all user-item pairs).

**Solution**:
```
Scores = U @ V^T  (3 x 3 matrix)

Scores[i, j] = u_i^T v_j

For User 1:
  - score(u1, vA) = diagonal (positive)
  - score(u1, vB), score(u1, vC) = off-diagonal (negatives)
```

---

### Problem 3: Temperature Scaling

**Given**:
```
Raw dot product: 0.8
Temperature tau = 0.1
```

**Compute**: Scaled logit.

**Solution**:
```
logit = 0.8 / 0.1 = 8.0
```

**Effect**: Amplifies differences (makes distribution sharper).

---

### Problem 4: Complexity Analysis

**Scenario**: You have 1M items and batch size 512.

**Questions**:
a) How many negatives per user with in-batch negatives?
b) How many total scores computed per batch?
c) What's the speedup vs. explicit 100 negatives per user?

**Solution**:
```
a) 512 - 1 = 511 negatives per user

b) 512 users x 512 items = 262,144 scores per batch

c) Explicit: 512 x (1 + 100) = 51,712 item embeddings
   In-batch: 512 item embeddings
   Speedup: 51,712 / 512 = ~101x
```

---

### Problem 5: Failure Mode Diagnosis

**Scenario**: Your two-tower model has these symptoms:
- Offline recall@100: 0.45
- Online recall@100: 0.12
- Popular items dominate recommendations

**Diagnose**: Which failure modes are likely? What would you check?

**Solution**:
```
Likely failure modes:
1. Feature leakage (large offline/online gap)
   - Check: Temporal correctness of features

2. Popularity bias (popular items dominate)
   - Check: Gini coefficient of recommendations
   - Check: Sampling distribution in training

3. Stale embeddings (if online has temporal component)
   - Check: Embedding refresh frequency

Diagnostic steps:
1. Audit feature pipelines for temporal leakage
2. Plot recommendation frequency vs. item popularity
3. Compare offline metrics with strict temporal split
```
