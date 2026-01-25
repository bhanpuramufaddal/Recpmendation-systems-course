# Week 8: Two-Tower Models and Large-Scale Retrieval - Practice Problems

## Overview
Master two-tower architectures, approximate nearest neighbor search, YouTube's recommendation system, and multi-task learning for production-scale retrieval.

---

## Problem 1: Two-Tower Architecture Design
**Difficulty:** Medium
**Topics:** Two-tower models, separate encoding

Design a two-tower model for movie recommendations:
- **User tower:** Encodes user ID, demographics, watch history
- **Item tower:** Encodes movie ID, genre, director, year

**Tasks:**
1. Design the network architecture for each tower
2. What activation function should the final layer use?
3. Why use dot product instead of concatenation + MLP?
4. How many parameters does this model have?

**Learning Outcomes:**
- Design two-tower architectures
- Understand dot product scoring
- Calculate model complexity

---

## Problem 2: In-Batch Negative Sampling
**Difficulty:** Medium
**Topics:** Training efficiency, negative sampling

**Standard approach:** For each positive (user, item), sample K negatives
**In-batch negatives:** Use other items in the batch as negatives

Given batch size = 256:
1. How many negatives does each example get?
2. Calculate computational savings vs. explicit sampling
3. What is the risk of in-batch negatives?
4. How would you ensure diverse negatives?

**Learning Outcomes:**
- Optimize training efficiency
- Understand in-batch sampling
- Recognize potential biases

---

## Problem 3: ANN Index Selection
**Difficulty:** Hard
**Topics:** FAISS, ScaNN, HNSW, index structures

You have 10M items, 512-dim embeddings, latency budget 10ms:

| Method | Build Time | Search Time | Recall@100 | Memory |
|--------|------------|-------------|------------|--------|
| Flat (exact) | 0s | 2000ms | 100% | 20GB |
| IVF | 1h | 5ms | 95% | 5GB |
| HNSW | 3h | 3ms | 98% | 8GB |
| PQ | 30min | 2ms | 90% | 2GB |

**Questions:**
1. Which index would you choose for production?
2. How would you tune recall vs. latency?
3. What is product quantization (PQ)?
4. When would you use GPU FAISS vs. CPU HNSW?

**Learning Outcomes:**
- Choose appropriate ANN indices
- Trade off recall, latency, memory
- Understand index mechanics

---

## Problem 4: YouTube Recommendation Architecture
**Difficulty:** Medium
**Topics:** Candidate generation, ranking, two-stage systems

**YouTube system:**
1. **Candidate generation:** Reduce millions → hundreds (fast, approximate)
2. **Ranking:** Re-rank hundreds → dozens (slow, accurate)

**Questions:**
1. Why two stages instead of one end-to-end model?
2. What features are used in each stage?
3. How do you train the candidate generator?
4. How do you handle the cold start problem for new videos?

**Learning Outcomes:**
- Understand two-stage architectures
- Design candidate generation
- Optimize for scale

---

## Problem 5: Multi-Task Learning
**Difficulty:** Hard
**Topics:** MMOE, hard parameter sharing, task balancing

**Objectives to optimize:**
- Click-through rate (CTR)
- Watch time
- User satisfaction (like/dislike)

**Architecture:**
```
Shared layers → Task-specific towers → Multiple outputs
```

**Questions:**
1. How do you balance losses from different tasks?
2. What is the "seesaw problem" in multi-task learning?
3. Design the loss function: L = w1×L_CTR + w2×L_watchtime + w3×L_satisfaction
4. How do you choose weights w1, w2, w3?

**Learning Outcomes:**
- Design multi-task architectures
- Balance competing objectives
- Handle task conflicts

---

## Problem 6: Serving Latency Optimization
**Difficulty:** Medium
**Topics:** Model serving, latency, optimization

**Latency breakdown:**
- Feature extraction: 20ms
- User tower forward pass: 15ms
- Item tower forward pass: 5ms (pre-computed)
- ANN search: 10ms
- Total: 50ms

**Tasks:**
1. Which component should you optimize first?
2. How can you reduce user tower latency?
3. Why pre-compute item tower but not user tower?
4. Design a caching strategy for user embeddings

**Learning Outcomes:**
- Profile model latency
- Optimize serving pipeline
- Design caching strategies

---

## Problem 7: Hard Negative Mining
**Difficulty:** Hard
**Topics:** Negative sampling, hard negatives, training

**Standard negatives:** Random items user didn't interact with
**Hard negatives:** Items with high predicted score but no interaction

**Questions:**
1. Why are hard negatives more informative?
2. How do you mine hard negatives efficiently?
3. What is the risk of using only hard negatives?
4. Design a mixed sampling strategy (easy + hard)

**Learning Outcomes:**
- Implement hard negative mining
- Balance training difficulty
- Improve model discrimination

---

## Problem 8: Cross-Domain Two-Tower
**Difficulty:** Hard
**Topics:** Transfer learning, domain adaptation

Train on movies, deploy on TV shows:
1. Can you reuse the user tower?
2. Can you reuse the item tower?
3. How would you fine-tune for the new domain?
4. What if TV shows have different features than movies?

**Learning Outcomes:**
- Transfer across domains
- Adapt two-tower models
- Handle feature mismatch

---

## Programming Exercises

### Exercise 1: Implement Basic Two-Tower Model
**Dataset:** MovieLens 1M

```python
import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=128):
        super(TwoTowerModel, self).__init__()

        # User tower
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Item tower
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.item_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, user_ids, item_ids):
        user_vec = self.user_mlp(self.user_emb(user_ids))
        item_vec = self.item_mlp(self.item_emb(item_ids))

        # Dot product
        scores = torch.sum(user_vec * item_vec, dim=-1)
        return scores

    def get_user_embedding(self, user_id):
        return self.user_mlp(self.user_emb(user_id))

    def get_item_embeddings(self):
        all_items = torch.arange(self.n_items)
        return self.item_mlp(self.item_emb(all_items))
```

**Training:**
- In-batch negatives
- Loss: Softmax cross-entropy
- Metrics: Recall@20, NDCG@20

---

### Exercise 2: Build FAISS Index for Fast Retrieval
**Dataset:** MovieLens 1M
**Task:** Pre-compute item embeddings and build ANN index

```python
import faiss
import numpy as np

# Get item embeddings from trained model
model.eval()
item_embeddings = model.get_item_embeddings().detach().cpu().numpy()

# Normalize for cosine similarity
faiss.normalize_L2(item_embeddings)

# Build index
d = item_embeddings.shape[1]  # Dimension
index = faiss.IndexFlatIP(d)  # Inner product (cosine after normalization)
index.add(item_embeddings)

# Search
def recommend(user_id, k=10):
    user_emb = model.get_user_embedding(user_id).detach().cpu().numpy()
    faiss.normalize_L2(user_emb.reshape(1, -1))

    distances, indices = index.search(user_emb.reshape(1, -1), k)
    return indices[0], distances[0]
```

**Experiment:** Compare FAISS methods: Flat, IVF, HNSW

---

### Exercise 3: Multi-Task Two-Tower
**Dataset:** MovieLens with engagement labels (watch, like, share)

```python
class MultiTaskTwoTower(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=128):
        super(MultiTaskTwoTower, self).__init__()

        # Shared towers
        self.user_tower = UserTower(n_users, embedding_dim)
        self.item_tower = ItemTower(n_items, embedding_dim)

        # Task-specific heads
        self.watch_head = nn.Linear(64, 1)
        self.like_head = nn.Linear(64, 1)
        self.share_head = nn.Linear(64, 1)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_tower(user_ids)
        item_vec = self.item_tower(item_ids)

        interaction = user_vec * item_vec

        watch_prob = torch.sigmoid(self.watch_head(interaction))
        like_prob = torch.sigmoid(self.like_head(interaction))
        share_prob = torch.sigmoid(self.share_head(interaction))

        return watch_prob, like_prob, share_prob
```

**Loss:** L = w1×BCE(watch) + w2×BCE(like) + w3×BCE(share)

---

### Exercise 4: Implement YouTube DNN Architecture
**Dataset:** YouTube-like data (video watches with context)
**Task:** Replicate the candidate generation model

**Features:**
- Watch history (average embeddings)
- Search history (average embeddings)
- Demographics (age, gender, geography)
- Context (time, device)

```python
class YouTubeDNN(nn.Module):
    def __init__(self, n_videos, n_search_tokens, embedding_dim=256):
        super(YouTubeDNN, self).__init__()

        self.video_emb = nn.Embedding(n_videos, embedding_dim)
        self.search_emb = nn.Embedding(n_search_tokens, embedding_dim)

        # User tower
        input_dim = embedding_dim * 2 + 10  # watch + search + demographics
        self.user_tower = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # Video tower (just embedding + MLP)
        self.video_tower = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, watch_history, search_history, demographics, video_id):
        # Average embeddings
        watch_emb = torch.mean(self.video_emb(watch_history), dim=1)
        search_emb = torch.mean(self.search_emb(search_history), dim=1)

        # Concatenate features
        user_features = torch.cat([watch_emb, search_emb, demographics], dim=-1)
        user_vec = self.user_tower(user_features)

        video_vec = self.video_tower(self.video_emb(video_id))

        score = torch.sum(user_vec * video_vec, dim=-1)
        return score
```

---

### Exercise 5: Hard Negative Mining
**Dataset:** MovieLens 1M
**Task:** Sample hard negatives during training

```python
def mine_hard_negatives(model, user_id, n_hard=5, n_candidates=100):
    # Sample candidate negatives
    all_items = set(range(n_items))
    pos_items = set(user_positive_items[user_id])
    neg_candidates = list(all_items - pos_items)
    candidates = np.random.choice(neg_candidates, n_candidates, replace=False)

    # Score candidates
    user_emb = model.get_user_embedding(user_id)
    candidate_embs = model.item_emb(torch.tensor(candidates))
    scores = torch.sum(user_emb * candidate_embs, dim=-1)

    # Select top-scoring as hard negatives
    hard_neg_indices = torch.topk(scores, n_hard).indices
    hard_negatives = candidates[hard_neg_indices.cpu().numpy()]

    return hard_negatives

# Training with hard negatives
for epoch in range(n_epochs):
    for user, pos_item in train_data:
        # Easy negatives (random)
        easy_negatives = sample_random_negatives(user, n=3)

        # Hard negatives (high-scoring but not clicked)
        hard_negatives = mine_hard_negatives(model, user, n=2)

        # Combine
        negatives = easy_negatives + hard_negatives

        # Compute loss
        # ... training step ...
```

---

## Discussion Questions

1. **One Tower vs. Two Towers:** When would you use a single unified model instead of two towers?

2. **Feature Engineering:** What features should go in user tower vs. item tower? What about contextual features (time, device)?

3. **Embedding Drift:** Item embeddings can become stale. How often should you recompute them?

4. **Diversity:** Two-tower models maximize similarity (dot product). How do you introduce diversity?

5. **Privacy:** User tower processes personal data. How do you ensure privacy in serving?

6. **A/B Testing:** How would you A/B test a new two-tower model against production baseline?

7. **Bias:** Popular items get more training signal. How do you prevent popularity bias in two-tower models?

8. **Real-time:** Can you update user embeddings in real-time as they interact? What are the challenges?

---

## Challenge Problem: Streaming Two-Tower Updates

**Difficulty:** Very Hard
**Topics:** Online learning, streaming updates, embedding drift

**Problem:** User preferences change. How do you update embeddings in real-time?

**Approach:**
1. **Static item tower:** Pre-computed, updated daily
2. **Dynamic user tower:** Updated in real-time based on recent interactions

**Architecture:**
```python
class StreamingUserTower(nn.Module):
    def __init__(self, base_model):
        self.base_model = base_model  # Pre-trained
        self.short_term_encoder = nn.GRU(embedding_dim, embedding_dim)

    def forward(self, user_id, recent_interactions):
        # Long-term (static)
        user_emb_static = self.base_model.get_user_embedding(user_id)

        # Short-term (dynamic)
        item_seq = self.item_emb(recent_interactions)
        _, user_emb_dynamic = self.short_term_encoder(item_seq)

        # Combine
        user_emb = 0.7 * user_emb_static + 0.3 * user_emb_dynamic
        return user_emb
```

**Evaluation:** Test on streaming data, measure adaptation speed

---

## References

### Papers
1. Covington, P., et al. (2016). "Deep neural networks for YouTube recommendations". RecSys.
2. Yi, X., et al. (2019). "Sampling-bias-corrected neural modeling for large corpus item recommendations". RecSys.
3. Ma, J., et al. (2018). "Modeling task relationships in multi-task learning with multi-gate mixture-of-experts". KDD.

### Libraries
- FAISS: https://github.com/facebookresearch/faiss
- ScaNN: https://github.com/google-research/google-research/tree/master/scann
- Annoy: https://github.com/spotify/annoy

---

*Return to [Week 8 Main Page](README.md)*
