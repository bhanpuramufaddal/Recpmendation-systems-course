# Week 5: Neural Collaborative Filtering - Training

## Overview

Training neural collaborative filtering models requires careful consideration of **loss functions**, **negative sampling**, **optimization**, and **regularization**.

**Key challenges**:
- **Implicit feedback**: No explicit ratings, only positive signals (clicks, views)
- **Class imbalance**: Far more negative samples than positive
- **Scalability**: Millions of users and items

This document covers practical techniques for training high-performance neural CF models.

---

## Learning Objectives

By the end of this section, you will:
- Master loss functions for recommendation (pointwise, pairwise, listwise)
- Implement negative sampling strategies
- Optimize training with Adam, learning rate schedules, and regularization
- Build end-to-end training pipelines
- Debug common training issues

---

## Loss Functions

### 1. Pointwise Loss (Binary Cross-Entropy)

**Scenario**: Predict whether user will interact with item (binary classification).

**Loss**:
$$\mathcal{L}_{\text{BCE}} = -\sum_{(u,i) \in \mathcal{D}} [y_{ui} \log(\hat{y}_{ui}) + (1 - y_{ui}) \log(1 - \hat{y}_{ui})]$$

where:
- $y_{ui} = 1$ if user $u$ interacted with item $i$ (positive)
- $y_{ui} = 0$ otherwise (negative)
- $\hat{y}_{ui}$ = model's predicted probability

**Implementation**:
```python
import torch
import torch.nn as nn

# Model predictions
predictions = torch.tensor([0.9, 0.3, 0.7, 0.2])  # Predicted probabilities

# Ground truth labels
labels = torch.tensor([1.0, 0.0, 1.0, 0.0])  # 1 = positive, 0 = negative

# Binary cross-entropy loss
criterion = nn.BCELoss()
loss = criterion(predictions, labels)

print(f"BCE Loss: {loss.item():.4f}")
```

**Pros**: Simple, widely used
**Cons**: Treats positive and negative samples equally (problematic with class imbalance)

---

### 2. Pairwise Loss (BPR - Bayesian Personalized Ranking)

**Paper**: Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)

**Scenario**: User should rank positive items higher than negative items.

**Assumption**: User prefers observed items over unobserved items.

**Loss** (for a user $u$):
$$\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j) \in D_S} \log \sigma(\hat{y}_{ui} - \hat{y}_{uj})$$

where:
- $i$ = positive item (user interacted)
- $j$ = negative item (user didn't interact)
- $\sigma$ = sigmoid function
- $\hat{y}_{ui}, \hat{y}_{uj}$ = model scores

**Intuition**: Maximize gap between positive and negative scores.

---

### BPR Implementation

```python
import torch
import torch.nn.functional as F

class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_scores, neg_scores):
        """
        pos_scores: (batch,) - scores for positive items
        neg_scores: (batch,) - scores for negative items
        """
        # BPR loss: -log(sigmoid(pos - neg))
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return loss


# Example
pos_scores = torch.tensor([0.9, 0.8, 0.75])  # User likes these items
neg_scores = torch.tensor([0.3, 0.4, 0.5])   # User hasn't seen these

bpr_loss = BPRLoss()
loss = bpr_loss(pos_scores, neg_scores)
print(f"BPR Loss: {loss.item():.4f}")
```

**Pros**: Better for implicit feedback, focuses on ranking
**Cons**: Requires sampling negative items

---

### 3. Listwise Loss (ListNet, SoftMax)

**Scenario**: Rank entire list of items.

**SoftMax Loss**:
$$\mathcal{L}_{\text{SoftMax}} = -\log \frac{\exp(\hat{y}_{ui})}{\sum_{j \in \mathcal{I}} \exp(\hat{y}_{uj})}$$

where $\mathcal{I}$ = all items.

**Problem**: Summing over all items is expensive (millions of items).

**Solution**: Sampled SoftMax (sum over sampled negatives instead).

---

### Sampled SoftMax Implementation

```python
class SampledSoftMaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_scores, neg_scores):
        """
        pos_scores: (batch, 1) - scores for positive items
        neg_scores: (batch, n_negatives) - scores for sampled negative items
        """
        # Concatenate positive and negative scores
        all_scores = torch.cat([pos_scores, neg_scores], dim=1)  # (batch, 1 + n_negatives)

        # SoftMax loss (positive is first column, index 0)
        loss = F.cross_entropy(all_scores, torch.zeros(all_scores.size(0), dtype=torch.long))
        return loss


# Example
pos_scores = torch.tensor([[0.9], [0.8], [0.85]])  # (3, 1)
neg_scores = torch.tensor([[0.3, 0.2, 0.4, 0.1],
                            [0.5, 0.3, 0.2, 0.4],
                            [0.4, 0.3, 0.5, 0.2]])  # (3, 4) - 4 negatives per sample

ssm_loss = SampledSoftMaxLoss()
loss = ssm_loss(pos_scores, neg_scores)
print(f"Sampled SoftMax Loss: {loss.item():.4f}")
```

**Pros**: Mimics ranking objective
**Cons**: Sensitive to number of negatives sampled

---

## Negative Sampling

### The Negative Sampling Problem

**Implicit feedback challenge**: Only positive signals (clicks, purchases). No explicit negatives.

**Assumption**: Unobserved items = negative (user not interested).

**Problem**: Too many negatives (millions of items, user interacted with <100).

**Solution**: Sample a subset of negatives for training.

---

### 1. Uniform Random Sampling

**Approach**: Sample negatives uniformly at random from items user hasn't interacted with.

```python
import numpy as np

def sample_negatives_uniform(user_id, pos_items, all_items, num_negatives=4):
    """
    Sample negative items uniformly at random.

    user_id: User ID
    pos_items: Set of items user has interacted with
    all_items: Set of all items
    num_negatives: Number of negatives to sample
    """
    # Candidate negatives (items user hasn't seen)
    negative_candidates = list(all_items - pos_items)

    # Sample
    negatives = np.random.choice(negative_candidates, size=num_negatives, replace=False)

    return negatives.tolist()


# Example
all_items = set(range(1000))  # 1000 items
user_pos_items = {10, 25, 100, 250, 500}  # User interacted with 5 items

negatives = sample_negatives_uniform(user_id=1, pos_items=user_pos_items,
                                      all_items=all_items, num_negatives=4)
print(f"Sampled negatives: {negatives}")
```

**Pros**: Simple, unbiased
**Cons**: May sample very obscure items (not informative)

---

### 2. Popularity-Based Sampling

**Observation**: Popular items are more likely to be shown to user → if not clicked, strong negative signal.

**Approach**: Sample negatives proportional to item popularity.

$$P(\text{sample item } i) \propto (\text{popularity of } i)^\alpha$$

where $\alpha \in [0, 1]$ controls skew ($\alpha=0$ → uniform, $\alpha=1$ → proportional to popularity).

```python
def sample_negatives_popularity(user_id, pos_items, item_popularity, num_negatives=4, alpha=0.75):
    """
    Sample negatives proportional to item popularity.

    item_popularity: dict {item_id: popularity_count}
    alpha: Exponent for popularity (0.75 is common)
    """
    # Candidate negatives
    negative_candidates = [item for item in item_popularity if item not in pos_items]

    # Compute sampling probabilities
    popularities = np.array([item_popularity[item] for item in negative_candidates])
    probs = np.power(popularities, alpha)
    probs /= probs.sum()  # Normalize

    # Sample
    negatives = np.random.choice(negative_candidates, size=num_negatives, replace=False, p=probs)

    return negatives.tolist()


# Example
item_popularity = {i: np.random.randint(1, 1000) for i in range(1000)}  # Random popularity

negatives_pop = sample_negatives_popularity(user_id=1, pos_items=user_pos_items,
                                             item_popularity=item_popularity, num_negatives=4, alpha=0.75)
print(f"Popularity-based negatives: {negatives_pop}")
```

**Pros**: More informative negatives
**Cons**: Biased toward popular items

---

### 3. Hard Negative Mining

**Observation**: Easy negatives (very different from user profile) don't help learning.

**Approach**: Sample negatives similar to positive items (hard to distinguish).

**Process**:
1. Compute scores for all candidate negatives
2. Sample negatives with highest scores (model thinks user will like, but user didn't interact)

```python
def sample_hard_negatives(user_id, pos_items, all_items, model, num_negatives=4, top_k=100):
    """
    Sample hard negatives (high model scores, but not interacted).

    model: Trained model to compute scores
    top_k: Consider top-K highest-scoring negatives as candidates
    """
    import torch

    # Candidate negatives
    negative_candidates = list(all_items - pos_items)

    # Compute scores for all candidates
    user_tensor = torch.tensor([user_id] * len(negative_candidates))
    item_tensor = torch.tensor(negative_candidates)

    with torch.no_grad():
        scores = model(user_tensor, item_tensor).numpy()

    # Select top-K highest-scoring negatives
    top_indices = np.argsort(scores)[::-1][:top_k]
    hard_candidates = [negative_candidates[i] for i in top_indices]

    # Randomly sample from hard candidates
    negatives = np.random.choice(hard_candidates, size=num_negatives, replace=False)

    return negatives.tolist()


# (Requires trained model - pseudo-code example)
# negatives_hard = sample_hard_negatives(user_id=1, pos_items=user_pos_items,
#                                         all_items=all_items, model=trained_model, num_negatives=4)
```

**Pros**: Improves model on difficult cases
**Cons**: Requires model inference (slower), can lead to overfitting if too hard

---

## Optimization

### 1. Optimizer Choice

**Adam** (Adaptive Moment Estimation) is the most common choice.

**Advantages**:
- Adaptive learning rates per parameter
- Works well with sparse gradients (common in recommendation)
- Less sensitive to learning rate tuning

```python
import torch.optim as optim

model = NeuMF(n_users=10000, n_items=5000, embedding_dim=64)

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training step
for batch in data_loader:
    user_ids, item_ids, labels = batch

    optimizer.zero_grad()
    predictions = model(user_ids, item_ids)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
```

**Hyperparameters**:
- `lr`: Learning rate (typical: 0.001 - 0.0001)
- `weight_decay`: L2 regularization (typical: 1e-5 - 1e-6)

---

### 2. Learning Rate Schedules

**Problem**: Fixed learning rate may be suboptimal.

**Solution**: Decay learning rate over time.

**a) Step Decay**

```python
from torch.optim.lr_scheduler import StepLR

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Reduce LR by factor of 0.5 every 10 epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(50):
    train_one_epoch(model, optimizer, data_loader)
    scheduler.step()  # Update learning rate
    print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()}")
```

---

**b) ReduceLROnPlateau**

Reduce LR when validation metric plateaus.

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

for epoch in range(50):
    train_loss = train_one_epoch(model, optimizer, data_loader)
    val_loss = evaluate(model, val_loader)

    # Reduce LR if val_loss doesn't improve for 5 epochs
    scheduler.step(val_loss)
```

---

**c) Cosine Annealing**

Smoothly decrease LR following cosine curve.

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

for epoch in range(50):
    train_one_epoch(model, optimizer, data_loader)
    scheduler.step()
```

---

### 3. Gradient Clipping

**Problem**: Gradients can explode (very large values) → unstable training.

**Solution**: Clip gradients to maximum norm.

```python
import torch.nn.utils as nn_utils

# Training loop
for batch in data_loader:
    optimizer.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()

    # Clip gradients
    nn_utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

    optimizer.step()
```

---

## Regularization

### 1. Dropout

**Idea**: Randomly drop neurons during training to prevent overfitting.

```python
class NeuMFWithDropout(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, dropout=0.2):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),  # Dropout layer
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        concat = torch.cat([user_emb, item_emb], dim=-1)
        output = self.mlp(concat)
        return output.squeeze()
```

**Typical dropout rate**: 0.2 - 0.5

---

### 2. L2 Regularization (Weight Decay)

**Idea**: Penalize large weights.

**Loss**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \sum_{\theta \in \Theta} \theta^2$$

**Implementation**: Use `weight_decay` in optimizer.

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

---

### 3. Batch Normalization

**Idea**: Normalize activations to stabilize training.

```python
self.mlp = nn.Sequential(
    nn.Linear(embedding_dim * 2, 128),
    nn.BatchNorm1d(128),  # Batch normalization
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)
```

**Benefit**: Faster convergence, less sensitive to initialization.

---

## Full Training Pipeline

### End-to-End Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Dataset
class ImplicitFeedbackDataset(Dataset):
    def __init__(self, interactions, all_items, num_negatives=4):
        """
        interactions: list of (user_id, item_id) tuples (positive samples)
        all_items: set of all item IDs
        num_negatives: number of negative samples per positive
        """
        self.interactions = interactions
        self.all_items = all_items
        self.num_negatives = num_negatives

        # Build user -> positive items mapping
        self.user_pos_items = {}
        for user_id, item_id in interactions:
            if user_id not in self.user_pos_items:
                self.user_pos_items[user_id] = set()
            self.user_pos_items[user_id].add(item_id)

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user_id, pos_item = self.interactions[idx]

        # Sample negatives
        neg_items = []
        user_pos = self.user_pos_items[user_id]
        negative_candidates = list(self.all_items - user_pos)

        neg_items = np.random.choice(negative_candidates, size=self.num_negatives, replace=False)

        # Return user, positive item, negative items
        return user_id, pos_item, neg_items.tolist()


# Training function
def train_epoch(model, data_loader, optimizer, criterion, device='cpu'):
    model.train()
    total_loss = 0

    for batch in data_loader:
        user_ids, pos_items, neg_items_list = batch

        user_ids = user_ids.to(device)
        pos_items = pos_items.to(device)

        # Positive scores
        pos_scores = model(user_ids, pos_items)

        # Negative scores (for each negative)
        batch_size = user_ids.size(0)
        num_negatives = len(neg_items_list)

        neg_scores_all = []
        for i in range(num_negatives):
            neg_items = torch.tensor([neg_items_list[j][i] for j in range(batch_size)]).to(device)
            neg_scores = model(user_ids, neg_items)
            neg_scores_all.append(neg_scores)

        neg_scores = torch.stack(neg_scores_all, dim=1)  # (batch, num_negatives)

        # BPR loss (average over negatives)
        loss = 0
        for i in range(num_negatives):
            loss += criterion(pos_scores, neg_scores[:, i])
        loss /= num_negatives

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


# Main training loop
def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, device='cpu'):
    model = model.to(device)

    criterion = BPRLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = evaluate_model(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model


# Evaluation
def evaluate_model(model, data_loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            user_ids, pos_items, neg_items_list = batch
            user_ids = user_ids.to(device)
            pos_items = pos_items.to(device)

            pos_scores = model(user_ids, pos_items)

            batch_size = user_ids.size(0)
            num_negatives = len(neg_items_list)

            neg_scores_all = []
            for i in range(num_negatives):
                neg_items = torch.tensor([neg_items_list[j][i] for j in range(batch_size)]).to(device)
                neg_scores = model(user_ids, neg_items)
                neg_scores_all.append(neg_scores)

            neg_scores = torch.stack(neg_scores_all, dim=1)

            loss = sum(criterion(pos_scores, neg_scores[:, i]) for i in range(num_negatives)) / num_negatives
            total_loss += loss.item()

    return total_loss / len(data_loader)


# Example usage
# interactions = [(user_id, item_id), ...]  # Your data
# all_items = set([item_id for _, item_id in interactions])

# dataset = ImplicitFeedbackDataset(interactions, all_items, num_negatives=4)
# train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

# model = NeuMF(n_users=10000, n_items=5000, embedding_dim=64)
# trained_model = train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001)
```

---

## Debugging Training

### Common Issues

**1. Loss not decreasing**:
- Check learning rate (too high or too low)
- Check data pipeline (labels correct?)
- Verify model can overfit small subset (sanity check)

**2. Loss exploding (NaN)**:
- Gradient clipping
- Lower learning rate
- Check for numerical instability in model

**3. Overfitting (train loss ↓, val loss ↑)**:
- Add dropout
- Increase weight decay
- More data
- Early stopping

**4. Underfitting (both train and val loss high)**:
- Increase model capacity (more layers, larger embeddings)
- Train longer
- Reduce regularization

---

## Summary

**Key Takeaways**:
1. **Loss functions**: BCE (pointwise), BPR (pairwise), Sampled SoftMax (listwise)
2. **Negative sampling**: Uniform, popularity-based, hard negatives
3. **Optimization**: Adam with learning rate schedules
4. **Regularization**: Dropout, L2, batch normalization
5. **Training pipeline**: Data loading, training loop, early stopping

**Best Practices**:
- Use BPR loss for implicit feedback
- Sample 4-10 negatives per positive
- Adam optimizer with LR = 0.001
- Dropout 0.2-0.5, weight decay 1e-5
- Early stopping with patience 5-10

**Next**: Week 6: Sequential recommendations (modeling user sequences with RNNs and Transformers).

---

## References

1. **Rendle, S., et al. (2009)**. "BPR: Bayesian Personalized Ranking from Implicit Feedback". *UAI*.
   - **BPR loss** for pairwise ranking

2. **He, X., et al. (2017)**. "Neural Collaborative Filtering". *WWW*.
   - Training NCF models

3. **Kingma, D. P., & Ba, J. (2015)**. "Adam: A Method for Stochastic Optimization". *ICLR*.
   - **Adam optimizer**

4. **Srivastava, N., et al. (2014)**. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting". *JMLR*.
   - **Dropout** regularization

5. **Ioffe, S., & Szegedy, C. (2015)**. "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift". *ICML*.
   - **Batch normalization**
