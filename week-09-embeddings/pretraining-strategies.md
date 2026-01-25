# Week 9: Pre-training Strategies for Recommendations

## Overview

**Pre-training** enables models to learn general representations from large datasets, then **fine-tune** on specific tasks. This approach has revolutionized NLP (BERT, GPT) and computer vision (ResNet, CLIP), and is now transforming recommendations.

**Key benefits**:
- **Transfer learning**: Leverage large-scale data (web logs, public datasets)
- **Cold start**: Pre-trained models work better with limited data
- **Multi-task**: Single pre-trained model → multiple downstream tasks

**Pre-training paradigms**:
1. **Self-supervised learning**: Learn from unlabeled data (masking, contrastive)
2. **Transfer learning**: Pre-train on source domain → fine-tune on target
3. **Multi-task learning**: Jointly train on multiple objectives

This document covers pre-training strategies for recommendation systems.

---

## Learning Objectives

By the end of this section, you will:
- Understand self-supervised learning for RecSys
- Implement contrastive learning (SimCLR, MoCo)
- Apply transfer learning across domains
- Master pre-training tasks (masking, next-item prediction)
- Fine-tune pre-trained models

---

## Self-Supervised Learning

### Core Idea

**Supervised learning**: Requires labels (ratings, clicks) → expensive, sparse.

**Self-supervised learning**: Create "pseudo-labels" from data structure → free, abundant.

**Example pretext tasks**:
- **Masked prediction**: Mask items in sequence → predict masked items
- **Next-item prediction**: Given past items → predict next item
- **Contrastive learning**: Pull similar users/items together, push apart dissimilar ones

---

### Why Self-Supervised?

**Problem**: User-item interaction data is sparse.

**Example** (Netflix):
```
Users: 200M
Items: 10K movies
Possible interactions: 200M × 10K = 2T
Actual interactions: ~20B (1% density)
```

**Solution**: Use **implicit structure** in data:
- Temporal patterns (session sequences)
- Co-occurrence (items viewed together)
- User/item features (metadata)

---

## Contrastive Learning

### Intuition

**Goal**: Learn representations where similar samples are close, dissimilar samples are far.

**Key idea**:
$$\text{similarity}(\text{user}, \text{positive item}) > \text{similarity}(\text{user}, \text{negative item})$$

**Framework**:
1. **Positive pairs**: (user, interacted item), (item A, item B from same session)
2. **Negative pairs**: (user, random item), (item A, random item B)
3. **Objective**: Maximize similarity of positives, minimize similarity of negatives

---

### SimCLR for Recommendations

**SimCLR** (Simple Contrastive Learning of Representations) adapted to RecSys.

**Original (vision)**: Augment image → learn invariance to augmentations.

**Recommendation adaptation**: Augment user/item representations.

**User augmentations**:
- Dropout random interactions
- Temporal crop (use recent vs. old history)
- Feature masking

**Item augmentations**:
- Mask features (title, category)
- Perturb embeddings with noise

---

### SimCLR Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128, hidden_dim=256):
        super().__init__()

        # Embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Projection head (for contrastive learning)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def encode_user(self, user_ids, dropout_rate=0.1):
        """
        Encode users with optional dropout augmentation.
        """
        user_emb = self.user_embeddings(user_ids)

        # Augmentation: dropout
        if self.training and dropout_rate > 0:
            mask = torch.rand_like(user_emb) > dropout_rate
            user_emb = user_emb * mask

        # Project to contrastive space
        user_proj = self.projection(user_emb)
        return F.normalize(user_proj, dim=1)

    def encode_item(self, item_ids, noise_std=0.1):
        """
        Encode items with optional noise augmentation.
        """
        item_emb = self.item_embeddings(item_ids)

        # Augmentation: add Gaussian noise
        if self.training and noise_std > 0:
            noise = torch.randn_like(item_emb) * noise_std
            item_emb = item_emb + noise

        # Project to contrastive space
        item_proj = self.projection(item_emb)
        return F.normalize(item_proj, dim=1)


def nt_xent_loss(anchor, positive, negatives, temperature=0.1):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

    anchor: (batch, dim)
    positive: (batch, dim)
    negatives: (batch, num_neg, dim) or (num_neg, dim)
    temperature: scaling factor
    """
    batch_size = anchor.size(0)

    # Positive similarity
    pos_sim = (anchor * positive).sum(dim=1) / temperature  # (batch,)

    # Negative similarities
    if negatives.dim() == 3:
        # (batch, num_neg, dim)
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze() / temperature  # (batch, num_neg)
    else:
        # (num_neg, dim) - shared negatives
        neg_sim = torch.mm(anchor, negatives.T) / temperature  # (batch, num_neg)

    # Logits: [positive, negatives]
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch, 1 + num_neg)

    # Labels: positive is at index 0
    labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

    # Cross-entropy
    loss = F.cross_entropy(logits, labels)

    return loss


# Example usage
model = ContrastiveRecommender(num_users=1000, num_items=5000, embedding_dim=128)

# Sample batch
user_ids = torch.randint(0, 1000, (64,))
pos_item_ids = torch.randint(0, 5000, (64,))
neg_item_ids = torch.randint(0, 5000, (64, 10))  # 10 negatives per user

# Encode
user_repr = model.encode_user(user_ids, dropout_rate=0.2)
pos_item_repr = model.encode_item(pos_item_ids, noise_std=0.1)
neg_item_repr = model.encode_item(neg_item_ids.view(-1)).view(64, 10, -1)

# Compute loss
loss = nt_xent_loss(user_repr, pos_item_repr, neg_item_repr, temperature=0.1)
print(f"Contrastive loss: {loss.item():.4f}")
```

---

### MoCo (Momentum Contrast)

**Problem with SimCLR**: Need large batch sizes (e.g., 8192) for many negatives → memory intensive.

**MoCo solution**: Maintain a **queue** of negative samples.

**Key components**:
1. **Encoder** (query): Updated via backprop
2. **Momentum encoder** (key): Updated slowly (EMA of encoder)
3. **Queue**: Store past embeddings as negatives

**Update rule**:
$$\theta_{\text{momentum}} \leftarrow m \cdot \theta_{\text{momentum}} + (1 - m) \cdot \theta_{\text{encoder}}$$

where $m \approx 0.999$ (slow update).

---

### MoCo Implementation

```python
class MoCoRecommender(nn.Module):
    def __init__(self, num_items, embedding_dim=128, queue_size=4096, momentum=0.999):
        super().__init__()

        # Query encoder (updated via backprop)
        self.encoder_q = nn.Embedding(num_items, embedding_dim)

        # Key encoder (updated via momentum)
        self.encoder_k = nn.Embedding(num_items, embedding_dim)
        self.encoder_k.weight.data.copy_(self.encoder_q.weight.data)
        self.encoder_k.requires_grad_(False)  # No gradient

        # Queue for negatives
        self.register_buffer("queue", torch.randn(queue_size, embedding_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.queue_size = queue_size
        self.momentum = momentum

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update queue with new keys
        """
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # Replace oldest entries
        self.queue[ptr:ptr + batch_size] = keys

        # Move pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, query_items, key_items, temperature=0.07):
        """
        query_items: (batch,) - anchor items
        key_items: (batch,) - positive items
        """
        # Query embeddings
        q = self.encoder_q(query_items)  # (batch, dim)
        q = F.normalize(q, dim=1)

        # Key embeddings (no gradient)
        with torch.no_grad():
            self._momentum_update_key_encoder()

            k = self.encoder_k(key_items)  # (batch, dim)
            k = F.normalize(k, dim=1)

        # Positive logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # (batch, 1)

        # Negative logits (from queue)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.T])  # (batch, queue_size)

        # Logits: [positive, negatives]
        logits = torch.cat([l_pos, l_neg], dim=1) / temperature  # (batch, 1 + queue_size)

        # Labels: positive is at index 0
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)

        # Update queue
        self._dequeue_and_enqueue(k)

        # Loss
        loss = F.cross_entropy(logits, labels)

        return loss


# Example
model = MoCoRecommender(num_items=5000, embedding_dim=128, queue_size=4096)
query_items = torch.randint(0, 5000, (256,))
key_items = torch.randint(0, 5000, (256,))

loss = model(query_items, key_items, temperature=0.07)
print(f"MoCo loss: {loss.item():.4f}")
```

---

## Masked Prediction

### BERT4Rec (Masked Item Prediction)

**Inspiration**: BERT masks tokens in text → predict masked tokens.

**Adaptation**: Mask items in user sequence → predict masked items.

**Example**:
```
Original sequence: [item1, item2, item3, item4, item5]
Masked sequence:   [item1, [MASK], item3, [MASK], item5]
Task: Predict item2 and item4
```

**Advantages**:
- **Bidirectional**: Use both past and future context
- **Self-supervised**: No need for explicit labels

---

### BERT4Rec Implementation

```python
class BERT4Rec(nn.Module):
    def __init__(self, num_items, max_len=50, embedding_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.num_items = num_items
        self.max_len = max_len

        # Special tokens
        self.mask_token = num_items  # [MASK] token
        self.pad_token = num_items + 1  # [PAD] token

        # Embeddings
        self.item_embeddings = nn.Embedding(num_items + 2, embedding_dim, padding_idx=self.pad_token)
        self.position_embeddings = nn.Embedding(max_len, embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.output_layer = nn.Linear(embedding_dim, num_items)

    def forward(self, sequences, masked_positions=None):
        """
        sequences: (batch, seq_len) - item IDs with some masked
        masked_positions: (batch, num_masked) - positions of masked items
        """
        batch_size, seq_len = sequences.size()

        # Positional encoding
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        item_emb = self.item_embeddings(sequences)  # (batch, seq_len, dim)
        pos_emb = self.position_embeddings(positions)  # (batch, seq_len, dim)

        # Combined embedding
        x = item_emb + pos_emb

        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, dim)

        # Predict masked items
        if masked_positions is not None:
            # Extract embeddings at masked positions
            masked_emb = x[torch.arange(batch_size).unsqueeze(1), masked_positions]  # (batch, num_masked, dim)

            # Predict
            logits = self.output_layer(masked_emb)  # (batch, num_masked, num_items)
        else:
            # Predict all positions
            logits = self.output_layer(x)  # (batch, seq_len, num_items)

        return logits


def create_masked_sequence(sequence, mask_prob=0.15, mask_token=5000):
    """
    Randomly mask items in sequence.

    Returns:
        masked_seq: Sequence with masked items
        masked_positions: Positions of masked items
        masked_labels: Original items at masked positions
    """
    seq_len = len(sequence)
    num_mask = max(1, int(seq_len * mask_prob))

    # Random positions to mask
    masked_positions = torch.randperm(seq_len)[:num_mask]

    # Create masked sequence
    masked_seq = sequence.clone()
    masked_labels = sequence[masked_positions].clone()
    masked_seq[masked_positions] = mask_token

    return masked_seq, masked_positions, masked_labels


# Example
model = BERT4Rec(num_items=5000, max_len=50, embedding_dim=128)

# User sequence
sequence = torch.tensor([10, 25, 42, 100, 250])

# Create masked version
masked_seq, masked_pos, labels = create_masked_sequence(sequence, mask_prob=0.4, mask_token=5000)
print(f"Original: {sequence}")
print(f"Masked: {masked_seq}")
print(f"Masked positions: {masked_pos}")
print(f"Labels: {labels}")

# Forward pass
logits = model(masked_seq.unsqueeze(0), masked_pos.unsqueeze(0))
print(f"Logits shape: {logits.shape}")  # (1, num_masked, num_items)

# Loss
loss = F.cross_entropy(logits.view(-1, 5000), labels)
print(f"Masked LM loss: {loss.item():.4f}")
```

---

## Transfer Learning

### Cross-Domain Transfer

**Scenario**: Pre-train on **large source domain** → fine-tune on **small target domain**.

**Example**:
- Source: Amazon product recommendations (millions of users/items)
- Target: Niche e-commerce site (thousands of users/items)

**Benefit**: Leverage large-scale data to improve cold-start performance.

---

### Transfer Learning Pipeline

**Step 1: Pre-training** (source domain)
```python
# Pre-train on large dataset
source_model = TwoTowerModel(user_dim=100, item_dim=50, embedding_dim=128)

# Train on source domain
for epoch in range(epochs):
    for batch in source_dataloader:
        loss = source_model(batch)
        loss.backward()
        optimizer.step()

# Save pre-trained weights
torch.save(source_model.state_dict(), 'pretrained_model.pth')
```

**Step 2: Fine-tuning** (target domain)
```python
# Initialize target model with pre-trained weights
target_model = TwoTowerModel(user_dim=80, item_dim=40, embedding_dim=128)

# Load pre-trained weights (partial)
pretrained_dict = torch.load('pretrained_model.pth')
model_dict = target_model.state_dict()

# Filter out size mismatches
pretrained_dict = {k: v for k, v in pretrained_dict.items()
                   if k in model_dict and v.size() == model_dict[k].size()}

model_dict.update(pretrained_dict)
target_model.load_state_dict(model_dict)

# Fine-tune on target domain (use smaller learning rate!)
optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-4)

for epoch in range(fine_tune_epochs):
    for batch in target_dataloader:
        loss = target_model(batch)
        loss.backward()
        optimizer.step()
```

---

### Domain Adaptation Strategies

**1. Feature-level adaptation**:
- Align feature distributions (Maximum Mean Discrepancy, adversarial)

**2. Embedding-level adaptation**:
- Map source embeddings to target space

**3. Task-level adaptation**:
- Multi-task learning (joint source + target objectives)

---

## Multi-Task Pre-Training

### Motivation

**Single task**: Predict clicks → model optimizes only for clicks.

**Multi-task**: Predict clicks + watch time + ratings → richer representations.

**Benefits**:
- **Regularization**: Prevents overfitting to single task
- **Transfer**: Shared representations across tasks
- **Efficiency**: Single model serves multiple objectives

---

### Multi-Task Architecture

```python
class MultiTaskRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128, hidden_dim=256):
        super().__init__()

        # Shared embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Task-specific heads
        self.click_head = nn.Linear(hidden_dim, 1)  # Binary (click or not)
        self.rating_head = nn.Linear(hidden_dim, 5)  # Multi-class (1-5 stars)
        self.watch_time_head = nn.Linear(hidden_dim, 1)  # Regression (minutes)

    def forward(self, user_ids, item_ids):
        # Embeddings
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)

        # Concatenate
        x = torch.cat([user_emb, item_emb], dim=1)

        # Shared encoding
        shared_repr = self.shared_encoder(x)

        # Task-specific predictions
        click_logits = self.click_head(shared_repr).squeeze()
        rating_logits = self.rating_head(shared_repr)
        watch_time = self.watch_time_head(shared_repr).squeeze()

        return click_logits, rating_logits, watch_time


def multi_task_loss(click_logits, rating_logits, watch_time,
                     click_labels, rating_labels, watch_time_labels,
                     weights=[1.0, 1.0, 1.0]):
    """
    Compute weighted multi-task loss.
    """
    # Click prediction (binary cross-entropy)
    click_loss = F.binary_cross_entropy_with_logits(click_logits, click_labels.float())

    # Rating prediction (cross-entropy)
    rating_loss = F.cross_entropy(rating_logits, rating_labels)

    # Watch time prediction (MSE)
    watch_loss = F.mse_loss(watch_time, watch_time_labels)

    # Weighted combination
    total_loss = weights[0] * click_loss + weights[1] * rating_loss + weights[2] * watch_loss

    return total_loss, click_loss, rating_loss, watch_loss


# Example
model = MultiTaskRecommender(num_users=1000, num_items=5000)

user_ids = torch.randint(0, 1000, (64,))
item_ids = torch.randint(0, 5000, (64,))

# Forward
click_logits, rating_logits, watch_time = model(user_ids, item_ids)

# Labels
click_labels = torch.randint(0, 2, (64,))
rating_labels = torch.randint(0, 5, (64,))
watch_time_labels = torch.rand(64) * 60

# Loss
total_loss, click_loss, rating_loss, watch_loss = multi_task_loss(
    click_logits, rating_logits, watch_time,
    click_labels, rating_labels, watch_time_labels
)

print(f"Total loss: {total_loss.item():.4f}")
print(f"  Click: {click_loss.item():.4f}, Rating: {rating_loss.item():.4f}, Watch: {watch_loss.item():.4f}")
```

---

## Fine-Tuning Strategies

### 1. Full Fine-Tuning

**Approach**: Update all parameters on target task.

**Pros**: Maximum flexibility, can adapt fully to target domain.
**Cons**: Risk of overfitting with small target data.

```python
# All parameters trainable
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

---

### 2. Frozen Embeddings

**Approach**: Freeze embeddings, only fine-tune task-specific layers.

**Pros**: Prevents catastrophic forgetting, faster training.
**Cons**: Less flexibility.

```python
# Freeze embeddings
model.user_embeddings.weight.requires_grad = False
model.item_embeddings.weight.requires_grad = False

# Only task-specific layers trainable
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)
```

---

### 3. Layer-wise Learning Rates

**Approach**: Use different learning rates for different layers.

**Intuition**: Early layers learn general features → small LR; later layers learn task-specific → large LR.

```python
optimizer = torch.optim.Adam([
    {'params': model.user_embeddings.parameters(), 'lr': 1e-5},
    {'params': model.item_embeddings.parameters(), 'lr': 1e-5},
    {'params': model.encoder.parameters(), 'lr': 1e-4},
    {'params': model.task_head.parameters(), 'lr': 1e-3}
])
```

---

### 4. Gradual Unfreezing

**Approach**: Progressively unfreeze layers during fine-tuning.

**Schedule**:
1. Train only task head (5 epochs)
2. Unfreeze top encoder layer (5 epochs)
3. Unfreeze all layers (10 epochs)

```python
# Phase 1: Freeze encoder
for param in model.encoder.parameters():
    param.requires_grad = False

train(epochs=5)

# Phase 2: Unfreeze top layer
for param in model.encoder[-1].parameters():
    param.requires_grad = True

train(epochs=5)

# Phase 3: Unfreeze all
for param in model.parameters():
    param.requires_grad = True

train(epochs=10)
```

---

## Summary

**Key Takeaways**:
1. **Self-supervised learning**: Learn from unlabeled data (masking, contrastive)
2. **Contrastive learning**: SimCLR (augmentations), MoCo (queue)
3. **Masked prediction**: BERT4Rec (bidirectional context)
4. **Transfer learning**: Pre-train on large domain → fine-tune on target
5. **Multi-task**: Learn shared representations across objectives

**Best Practices**:
- **Pre-training**: Large batch (SimCLR) or queue (MoCo)
- **Augmentations**: Dropout, noise, feature masking
- **Temperature**: 0.05-0.1 for contrastive loss
- **Fine-tuning**: Small LR (1e-4), layer-wise rates

**When to use**:
- **Transfer learning**: Limited target data, large source data available
- **Multi-task**: Multiple objectives (clicks, ratings, watch time)
- **Self-supervised**: Abundant unlabeled data (logs, sessions)

**Next**: Large language models for recommendations.

---

## References

1. **Chen, T., et al. (2020)**. "A Simple Framework for Contrastive Learning of Visual Representations". *ICML*.
   - **SimCLR** framework

2. **He, K., et al. (2020)**. "Momentum Contrast for Unsupervised Visual Representation Learning". *CVPR*.
   - **MoCo** with queue

3. **Sun, F., et al. (2019)**. "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer". *CIKM*.
   - **BERT4Rec** masked prediction

4. **Zhou, K., et al. (2020)**. "S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization". *CIKM*.
   - **Self-supervised** RecSys

5. **Ma, J., et al. (2019)**. "Learning Disentangled Representations for Recommendation". *NeurIPS*.
   - **Multi-task** disentangled learning

---

## Practice Problems

### Problem 1: Contrastive Loss

**Given**:
- Anchor: $\mathbf{a} = [1, 0]$ (normalized)
- Positive: $\mathbf{p} = [0.8, 0.6]$ (normalized)
- Negatives: $\mathbf{n}_1 = [0, 1], \mathbf{n}_2 = [-0.6, 0.8]$ (normalized)
- Temperature: $\tau = 0.1$

**Compute**: NT-Xent loss.

**Solution**:
```python
import numpy as np

a = np.array([1, 0])
p = np.array([0.8, 0.6])
n1 = np.array([0, 1])
n2 = np.array([-0.6, 0.8])
tau = 0.1

# Similarities
sim_pos = np.dot(a, p) / tau  # 0.8 / 0.1 = 8.0
sim_n1 = np.dot(a, n1) / tau  # 0.0 / 0.1 = 0.0
sim_n2 = np.dot(a, n2) / tau  # -0.6 / 0.1 = -6.0

# Softmax denominator
logits = [sim_pos, sim_n1, sim_n2]
exp_sum = np.sum(np.exp(logits))

# Loss
loss = -np.log(np.exp(sim_pos) / exp_sum)
print(f"NT-Xent loss: {loss:.4f}")
# Output: ~0.0009 (very low because positive is much higher)
```

---

### Problem 2: Masked Prediction

**Given sequence**: [10, 25, 42, 100, 250]

**Mask 40%** of items (positions 1 and 3).

**Compute**: Masked sequence and labels.

**Solution**:
```python
sequence = [10, 25, 42, 100, 250]
mask_positions = [1, 3]  # Positions to mask
MASK_TOKEN = 9999

masked_seq = sequence.copy()
labels = []

for pos in mask_positions:
    labels.append(sequence[pos])
    masked_seq[pos] = MASK_TOKEN

print(f"Original: {sequence}")
print(f"Masked: {masked_seq}")
print(f"Labels: {labels}")

# Output:
# Original: [10, 25, 42, 100, 250]
# Masked: [10, 9999, 42, 9999, 250]
# Labels: [25, 100]
```

---

### Problem 3: Fine-Tuning Learning Rates

**Given**:
- Source domain: 1M users, trained for 100 epochs
- Target domain: 10K users
- Model: 3 layers (embeddings, encoder, head)

**Design**: Layer-wise learning rate schedule for fine-tuning.

**Solution**:
```python
# Intuition: Lower layers (embeddings) should change slowly
# Higher layers (task head) can adapt more quickly

learning_rates = {
    'embeddings': 1e-5,    # Very slow (general features)
    'encoder': 1e-4,       # Moderate (domain features)
    'task_head': 1e-3      # Fast (task-specific)
}

optimizer = torch.optim.Adam([
    {'params': model.embeddings.parameters(), 'lr': learning_rates['embeddings']},
    {'params': model.encoder.parameters(), 'lr': learning_rates['encoder']},
    {'params': model.task_head.parameters(), 'lr': learning_rates['task_head']}
])

print(f"Embedding LR: {learning_rates['embeddings']}")
print(f"Encoder LR: {learning_rates['encoder']}")
print(f"Head LR: {learning_rates['task_head']}")
```
