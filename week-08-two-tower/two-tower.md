# Week 8: Two-Tower Models - Architecture & Design

## Overview

**Two-tower models** (also called **dual encoder** or **siamese networks**) separately encode users and items into embeddings, enabling **fast candidate retrieval** at scale.

**Key idea**:
- **User tower**: $\mathbf{u} = f_{\text{user}}(\text{user features})$
- **Item tower**: $\mathbf{v} = f_{\text{item}}(\text{item features})$
- **Score**: $\text{score}(u, i) = \mathbf{u}^T \mathbf{v}$ (dot product)

**Critical property**: User and item embeddings computed **independently** → can precompute item embeddings → fast retrieval.

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

## Why Two-Tower?

### The Scalability Problem

**Traditional approach**: Score all user-item pairs with complex model.

$$\text{score}(u, i) = f(u, i, \text{context})$$

**Problem**: At inference, need to score **millions of items** per user → slow!

**Example** (YouTube):
- 2B users
- 800M videos
- Scoring all pairs: $2B \times 800M = 1.6 \times 10^{18}$ scores!
- Even at 1M scores/second: **50 years** per user!

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

## Architecture

### General Two-Tower Framework

```
User Features            Item Features
(demographics, history)  (content, metadata)
       ↓                        ↓
  User Tower              Item Tower
  (Neural Net)            (Neural Net)
       ↓                        ↓
User Embedding          Item Embedding
    u (d-dim)               v (d-dim)
       ↓                        ↓
         Dot Product: u^T v
              ↓
           Score
```

---

### User Tower

**Input**: User features
- Demographics: age, gender, location
- History: past clicks, watches, purchases (aggregated)
- Context: time of day, device, season

**Architecture**: Multi-layer perceptron (MLP)

```
User Features (sparse + dense)
       ↓
  Embedding Layers (for categorical)
       ↓
  Concatenate
       ↓
  MLP (FC layers + ReLU)
       ↓
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
       ↓
  Embedding Layers
       ↓
  Concatenate
       ↓
  MLP
       ↓
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

$$\mathcal{L}_{\text{triplet}} = \sum_{(u, i^+, i^-)} \max(0, \margin + d(\mathbf{u}, \mathbf{v}_{i^-}) - d(\mathbf{u}, \mathbf{v}_{i^+}))$$

where $d$ = distance (e.g., $1 - \mathbf{u}^T \mathbf{v}$).

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
User 1 → Item A (positive)
User 2 → Item B (positive)
User 3 → Item C (positive)

For User 1:
  Positive: Item A
  Negatives: Item B, Item C (from batch)
```

**Advantage**: No need to sample negatives explicitly → more efficient.

**Loss** (Sampled SoftMax):
$$\mathcal{L} = -\log \frac{\exp(\mathbf{u}_i^T \mathbf{v}_i)}{\sum_{j \in \text{batch}} \exp(\mathbf{u}_i^T \mathbf{v}_j)}$$

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

**Why it works**: Batch size = 256 → each user has 255 negatives → strong learning signal.

---

## Feature Engineering

### User Features

**1. Demographic**:
- Age, gender, location (categorical → embeddings)

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

**Solution**: L2 normalize → only angle matters.

$$\text{score} = \frac{\mathbf{u}}{\|\mathbf{u}\|} \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|} = \cos(\theta)$$

**Implementation**:
```python
user_emb = nn.functional.normalize(user_emb, p=2, dim=1)
item_emb = nn.functional.normalize(item_emb, p=2, dim=1)
```

---

## Temperature Scaling

**Problem**: Dot products can be small (close to 0) → hard to optimize.

**Solution**: Scale by temperature $\tau$.

$$\text{logits} = \frac{\mathbf{u}^T \mathbf{v}}{\tau}$$

**Effect**:
- Small $\tau$ (e.g., 0.05): Sharper distribution (more confident)
- Large $\tau$ (e.g., 1.0): Smoother distribution (less confident)

**Typical**: $\tau = 0.07$ (from CLIP paper).

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
2. **Fast**: Precompute item embeddings → fast retrieval
3. **Flexible**: Can add new items easily (just encode)
4. **Interpretable**: Embeddings can be visualized

---

### Limitations

1. **Limited expressiveness**: Dot product less expressive than full cross-features
2. **No feature crosses**: Can't model user-item interactions (e.g., "user likes sci-fi + movies from 2010s")
3. **Cold start**: New users with no history → weak embeddings

**Solution**: Use two-tower for **candidate retrieval** (stage 1), then **ranking model** (stage 2) for fine-grained scoring.

---

## Summary

**Key Takeaways**:
1. **Two-tower**: Separate user and item encoding → fast retrieval
2. **Scalability**: Precompute item embeddings → query with ANN
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
User 1 → Item A
User 2 → Item B
User 3 → Item C

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
Temperature τ = 0.1
```

**Compute**: Scaled logit.

**Solution**:
```
logit = 0.8 / 0.1 = 8.0
```

**Effect**: Amplifies differences (makes distribution sharper).
