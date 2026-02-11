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

### Deriving BPR Loss from Bayesian First Principles

*Let me walk you through where this loss function actually comes from. Understanding the derivation will help you see why BPR is so effective for implicit feedback.*

#### Step 1: The Probabilistic Model

We start with a simple question: **What is the probability that user $u$ prefers item $i$ over item $j$?**

We model this as:
$$P(i >_u j) = \sigma(\hat{y}_{uij})$$

where:
- $i >_u j$ means "user $u$ prefers item $i$ over item $j$"
- $\hat{y}_{uij} = \hat{y}_{ui} - \hat{y}_{uj}$ = score difference
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ = sigmoid function

*Why sigmoid?* The sigmoid maps any real number to $(0, 1)$, which is perfect for probabilities:
- When $\hat{y}_{ui} \gg \hat{y}_{uj}$: $\sigma(\hat{y}_{uij}) \to 1$ (high confidence user prefers $i$)
- When $\hat{y}_{ui} \ll \hat{y}_{uj}$: $\sigma(\hat{y}_{uij}) \to 0$ (low confidence)
- When $\hat{y}_{ui} = \hat{y}_{uj}$: $\sigma(0) = 0.5$ (50-50)

---

#### Step 2: Building the Likelihood Function

For implicit feedback, we observe:
- User $u$ interacted with item $i$ (positive)
- User $u$ did NOT interact with item $j$ (negative)

**Key assumption**: The user prefers $i$ over $j$:
$$i >_u j$$

Given training data $D_S = \{(u, i, j) : i \in I_u^+, j \in I \setminus I_u^+\}$

The **likelihood** of observing this preference data is:
$$L(\Theta | D_S) = \prod_{(u,i,j) \in D_S} P(i >_u j | \Theta)$$

where $\Theta$ represents all model parameters (embeddings).

*Pause and think: Why do we multiply probabilities?* Because we assume pairwise preferences are independent. The probability of observing ALL preferences is the product of individual probabilities.

---

#### Step 3: Maximum Likelihood Estimation

We want to find parameters $\Theta$ that **maximize** the likelihood:
$$\Theta^* = \arg\max_\Theta L(\Theta | D_S)$$

**Problem**: Products of many small probabilities lead to numerical underflow.

**Solution**: Take the logarithm! Since $\log$ is monotonic, maximizing $\log L$ is equivalent to maximizing $L$.

$$\log L(\Theta | D_S) = \sum_{(u,i,j) \in D_S} \log P(i >_u j | \Theta)$$

Substituting our model:
$$\log L = \sum_{(u,i,j) \in D_S} \log \sigma(\hat{y}_{ui} - \hat{y}_{uj})$$

---

#### Step 4: Adding Regularization (The Bayesian Part)

From a Bayesian perspective, we add a **prior** on the parameters:
$$P(\Theta) \propto \exp(-\lambda \|\Theta\|^2)$$

This says "we believe parameters should be small" (Gaussian prior with zero mean).

The **posterior** becomes:
$$P(\Theta | D_S) \propto L(\Theta | D_S) \cdot P(\Theta)$$

Taking the log:
$$\log P(\Theta | D_S) = \sum_{(u,i,j) \in D_S} \log \sigma(\hat{y}_{uij}) - \lambda \|\Theta\|^2 + \text{const}$$

---

#### Step 5: The Final BPR Loss

To turn maximization into minimization (as is convention in ML), we negate:

$$\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j) \in D_S} \log \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda \|\Theta\|^2$$

**That's it!** This is exactly the BPR loss we started with.

**Key insight**: BPR loss is the **negative log-likelihood** of pairwise preferences under a Bayesian model with Gaussian prior on parameters.

---

#### Visual Understanding of the Sigmoid in BPR

```
                           P(i >_u j)
                              1.0 ┤                    ────────
                                  │                 ──/
                                  │               /
                              0.5 ┤─────────────/─────────────
                                  │           /
                                  │        ──
                              0.0 ┤────────
                                  └──────┴──────┴──────┴──────
                                      -4     0     4
                                    ŷ_ui - ŷ_uj (score difference)
```

**Gradient behavior**:
- When $\hat{y}_{ui} - \hat{y}_{uj}$ is very negative: Model is "wrong" → gradient is large → big update
- When $\hat{y}_{ui} - \hat{y}_{uj}$ is positive: Model is "right" → gradient is small → small update

*This is exactly what we want!* The model focuses learning effort on pairs it gets wrong.

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

**Observation**: Popular items are more likely to be shown to user - if not clicked, strong negative signal.

**Approach**: Sample negatives proportional to item popularity.

$$P(\text{sample item } i) \propto (\text{popularity of } i)^\alpha$$

where $\alpha \in [0, 1]$ controls skew ($\alpha=0$ - uniform, $\alpha=1$ - proportional to popularity).

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

### Hard Negative Mining: Deep Dive

*Why does hard negative mining work? And when can it backfire? Let me explain the intuition.*

#### Why Random Negatives Are Often Too Easy

Consider a user who loves sci-fi movies. They've watched: Inception, Interstellar, Blade Runner.

**Random negative sampling might give**:
- "The Notebook" (romance)
- "Cooking with Julia" (cooking show)
- "Barney & Friends" (kids' show)

*These are "easy" negatives.* The model quickly learns: "sci-fi > romance, cooking, kids' shows."

**But this teaches the model nothing useful!** When recommending, we need to distinguish between:
- Inception vs. Arrival (both sci-fi)
- Interstellar vs. Gravity (both space movies)

#### Example: Gradient Analysis

Consider BPR loss gradient:
$$\frac{\partial \mathcal{L}}{\partial \theta} \propto \sigma(-\hat{y}_{uij}) \cdot (\text{embedding terms})$$

**Easy negative** (cooking show): $\hat{y}_{ui} - \hat{y}_{uj} = 5.0 - 0.1 = 4.9$
$$\sigma(-4.9) \approx 0.007$$
*Gradient is tiny! Model barely learns.*

**Hard negative** (similar sci-fi movie): $\hat{y}_{ui} - \hat{y}_{uj} = 5.0 - 4.8 = 0.2$
$$\sigma(-0.2) \approx 0.45$$
*Gradient is ~60x larger! Model learns much more.*

#### The "Too Hard" Problem

*But be careful!* There's a danger if negatives are TOO hard.

**False negative problem**: Some "negatives" are actually items the user would love but just hasn't seen yet.

If you always sample the hardest negatives:
1. Model predicts item X is great for user
2. X becomes a "hard negative" (user hasn't clicked)
3. Model is pushed to rank X lower
4. But X was actually perfect for the user!

**Solution**: Don't sample THE hardest negatives. Sample from top-K (e.g., top 100) with some randomness.

```
        Gradient magnitude vs. Negative difficulty

        High ┤      ╱╲
             │     ╱  ╲
             │    ╱    ╲ ← Sweet spot: hard but not too hard
             │   ╱      ╲
             │  ╱        ╲
        Low  ┤─╱──────────╲─────
             └────────────────────
               Easy      Hard    Too Hard
             (random)   (top-K)  (top-1)
```

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

---

### Why Do We Need Learning Rate Decay?

*Let me build intuition for why learning rate schedules are essential.*

#### The "Large Steps Early, Small Steps Late" Principle

**Analogy**: Finding your seat in a dark theater.

1. **Early training** (entering the theater): You need to move quickly across the room. Big steps are efficient.

2. **Late training** (near your seat): You need to navigate carefully between seats. Big steps would make you bump into things!

**Mathematically**:
- **Early**: Loss landscape is "steep" - large gradients point toward good minima. Large LR helps traverse quickly.
- **Late**: Near a minimum - loss landscape is "flat". Large LR causes oscillation around the minimum instead of settling in.

#### What Happens Without LR Decay

```
        Loss
          │╲
          │ ╲
          │  ╲    ← Good progress early
          │   ╲
          │    ╲
          │     ╲  ╱╲  ╱╲  ╱╲  ← Oscillation! Can't converge
          │      ╲╱  ╲╱  ╲╱
          └──────────────────────
                                Epochs

        With constant LR = 0.01
```

#### What Happens With LR Decay

```
        Loss
          │╲
          │ ╲
          │  ╲    ← Same good progress early
          │   ╲
          │    ╲
          │     ╲
          │      ────────────  ← Smooth convergence!
          └──────────────────────
                                Epochs

        With LR decay: 0.01 → 0.001 → 0.0001
```

---

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

*Why cosine?* The cosine schedule decreases slowly at first, then faster in the middle, then slowly again near the minimum. This "warm" ending helps the model settle into a good minimum.

---

### 3. Gradient Clipping

**Problem**: Gradients can explode (very large values) - unstable training.

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

## What Can Go Wrong: A Training Troubleshooting Guide

*Training neural recommenders is tricky. Let me walk you through common failure modes at each training stage.*

### Stage 1: Data Loading & Preprocessing

| What You See | What's Happening | How to Fix |
|-------------|------------------|------------|
| Training is extremely slow | Data loading is bottleneck | Increase `num_workers` in DataLoader, use prefetching |
| Memory error during batching | Batch size too large for embeddings | Reduce batch size or use gradient accumulation |
| Model always predicts same value | All negatives are too easy (e.g., all zeros) | Check negative sampling is working correctly |
| NaN in first batch | User/item ID out of embedding range | Verify ID mapping, check embedding vocabulary size |

### Stage 2: Forward Pass Issues

| What You See | What's Happening | How to Fix |
|-------------|------------------|------------|
| All predictions are ~0.5 | Sigmoid saturating at midpoint | Check embedding initialization, reduce initial LR |
| All predictions are 0 or 1 | Sigmoid saturating at extremes | Embeddings too large - add weight decay or reduce embedding dim |
| Predictions don't change across users | User embedding not being used | Debug forward pass, print intermediate values |
| Very different train/inference predictions | Dropout not disabled in eval mode | Call `model.eval()` before inference |

### Stage 3: Loss & Gradient Issues

| What You See | What's Happening | How to Fix |
|-------------|------------------|------------|
| Loss stays at initial value | Learning rate too small OR gradient not flowing | Increase LR 10x, check for frozen parameters |
| Loss immediately goes to NaN | Learning rate too large OR numerical instability | Reduce LR, add gradient clipping, use log-sigmoid |
| Loss oscillates wildly | Learning rate too large | Reduce LR, add warmup |
| Gradients are all zero | Dead ReLU or vanishing gradient | Check for layers with zero gradients, use LeakyReLU |
| Gradients explode (>1000) | Unstable training | Add gradient clipping (max_norm=1.0) |

### Stage 4: Training Dynamics

| What You See | What's Happening | How to Fix |
|-------------|------------------|------------|
| Train loss decreasing, val loss increasing | Overfitting | Add dropout, increase regularization, early stopping |
| Both losses high and not improving | Underfitting | Increase model capacity, train longer, reduce regularization |
| Loss plateaus early | Stuck in local minimum OR LR too small | Try LR warmup, increase LR, add momentum |
| Loss decreases then suddenly spikes | Learning rate too high for current loss landscape | Use LR decay, ReduceLROnPlateau |

---

## Debugging Training: A Systematic Checklist

*When training goes wrong, follow this systematic debugging procedure.*

### Step 1: Sanity Checks (Do These First!)

- [ ] **Overfit on tiny dataset**: Can your model achieve near-zero loss on 10 samples?
  - If NO: Bug in model or loss function
  - If YES: Model works, issue is with data or hyperparameters

```python
# Sanity check: overfit 10 samples
tiny_loader = DataLoader(train_data[:10], batch_size=10)
for epoch in range(100):
    loss = train_epoch(model, tiny_loader, optimizer, criterion)
    print(f"Epoch {epoch}: Loss = {loss:.4f}")
# Should get loss < 0.1
```

- [ ] **Check data labels**: Print 10 random (user, pos_item, neg_item) triplets. Do they make sense?

- [ ] **Verify shapes**: Print shapes at each step of forward pass.

### Step 2: Gradient Health Check

```python
# After backward(), before optimizer.step()
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm = {grad_norm:.6f}")
            if grad_norm == 0:
                print(f"  WARNING: Zero gradient!")
            if grad_norm > 100:
                print(f"  WARNING: Exploding gradient!")
```

### Step 3: Learning Rate Finder

*Not sure what learning rate to use? Run a learning rate range test.*

```python
def find_lr(model, train_loader, criterion, start_lr=1e-7, end_lr=1, num_iter=100):
    """Plot loss vs learning rate to find optimal LR."""
    lrs, losses = [], []
    lr = start_lr
    lr_multiplier = (end_lr / start_lr) ** (1 / num_iter)

    model_copy = copy.deepcopy(model)
    optimizer = optim.Adam(model_copy.parameters(), lr=lr)

    for i, batch in enumerate(train_loader):
        if i >= num_iter:
            break

        loss = compute_loss(model_copy, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        lrs.append(lr)
        losses.append(loss.item())

        # Increase LR
        lr *= lr_multiplier
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Plot: Choose LR where loss decreases fastest
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.show()
```

**Rule of thumb**: Choose LR about 10x smaller than where loss starts increasing.

---

## Comprehensive Debugging Reference Table

| Symptom | Likely Cause | Diagnostic | Solution |
|---------|--------------|------------|----------|
| **Loss not decreasing** | LR too small | Check gradient norms (should be > 0) | Increase LR by 10x |
| **Loss not decreasing** | Gradients vanishing | Check gradients at each layer | Use skip connections, batch norm |
| **Loss not decreasing** | Bug in loss function | Manually compute loss for one example | Fix loss implementation |
| **Loss exploding (NaN)** | LR too large | Loss spikes before NaN | Reduce LR, add gradient clipping |
| **Loss exploding (NaN)** | Numerical instability | Check for log(0) or div by zero | Use log-sigmoid, add epsilon |
| **Loss exploding (NaN)** | Bad initialization | Check embedding magnitudes | Use xavier/kaiming init |
| **Overfitting** | Model too complex | Val loss >> Train loss | Add dropout, reduce capacity |
| **Overfitting** | Not enough regularization | Val loss diverges after N epochs | Increase weight decay |
| **Overfitting** | Training too long | Val loss was good, then got worse | Use early stopping |
| **Underfitting** | Model too simple | Both losses plateau high | Increase embedding dim, add layers |
| **Underfitting** | Too much regularization | Train loss plateaus high | Reduce dropout, weight decay |
| **Underfitting** | Not enough training | Loss still decreasing at end | Train more epochs |
| **Predictions all same** | Dead embeddings | All users/items have same embedding | Check initialization, reduce regularization |
| **Predictions all same** | Sigmoid saturation | Predictions stuck at 0 or 1 | Reduce embedding magnitude |
| **Predictions all same** | Negative sampling broken | All negatives same or very easy | Debug sampling function |
| **Training very slow** | Data loading bottleneck | GPU utilization < 50% | Increase num_workers, use SSD |
| **Training very slow** | Large sparse operations | Profile shows sparse ops slow | Use dense batching strategies |

---

## Summary

**Key Takeaways**:
1. **Loss functions**: BCE (pointwise), BPR (pairwise), Sampled SoftMax (listwise)
2. **Negative sampling**: Uniform, popularity-based, hard negatives
3. **Optimization**: Adam with learning rate schedules
4. **Regularization**: Dropout, L2, batch normalization
5. **Training pipeline**: Data loading, training loop, early stopping
6. **BPR derivation**: Comes from maximizing log-likelihood of pairwise preferences
7. **Hard negatives**: Important for learning, but "too hard" can hurt

**Best Practices**:
- Use BPR loss for implicit feedback
- Sample 4-10 negatives per positive
- Adam optimizer with LR = 0.001
- Dropout 0.2-0.5, weight decay 1e-5
- Early stopping with patience 5-10
- Always run sanity check: can model overfit 10 samples?

**Next**: Week 6: Sequential recommendations (modeling user sequences with RNNs and Transformers).

---

## Self-Check Questions

Before moving on, make sure you can answer these questions:

1. **Derive the BPR gradient**: Starting from $\mathcal{L} = -\log \sigma(\hat{y}_{ui} - \hat{y}_{uj})$, what is $\frac{\partial \mathcal{L}}{\partial \hat{y}_{ui}}$?

2. **Hard negative intuition**: If your model predicts score 0.9 for item A (positive) and 0.1 for item B (negative), what is the BPR gradient magnitude? What if B scored 0.85?

3. **Debugging**: Your loss is stuck at 0.693. What does this suggest? (Hint: what's $-\log(0.5)$?)

4. **Learning rate**: You're training and loss oscillates between 0.5 and 0.8 every epoch. What should you try?

5. **Sanity check failure**: Your model can't overfit 10 samples even after 1000 epochs. List 3 things to check.

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
