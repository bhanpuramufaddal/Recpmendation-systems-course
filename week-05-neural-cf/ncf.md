# Week 5: Neural Collaborative Filtering (NCF)

## Overview

Neural Collaborative Filtering (NCF) replaces the linear inner product in matrix factorization with a deep neural network, enabling non-linear modeling of user-item interactions. This breakthrough paper (He et al., 2017) demonstrated that neural networks could significantly outperform traditional CF methods.

**Key Innovation**: Learn arbitrary functions from data instead of assuming linear interaction.

---

## Motivation: Limitations of Matrix Factorization

### Matrix Factorization Recap

**Prediction**:
$$\hat{r}_{ui} = \mathbf{p}_u^T \mathbf{q}_i = \sum_{k=1}^K p_{uk} \cdot q_{ik}$$

**Assumption**: Rating is a **linear combination** of latent factor interactions.

**Problem**: Real-world interactions are often **non-linear**.

---

### Example: Non-Linear Interaction

**Scenario**: Movie recommendation

**Latent factors** (learned by MF):
- Factor 1: Action level
- Factor 2: Complexity

**User preferences**:
- Alice: Loves **high action AND high complexity** (e.g., Inception)
- Bob: Loves **high action OR low complexity** (e.g., Fast & Furious OR Finding Nemo)

**MF limitation**:
- Linear combination can't capture "AND" vs "OR" logic
- Neural networks can learn these non-linear patterns

---

## Neural Collaborative Filtering Framework

### High-Level Architecture

```
User ID ──> User Embedding ──┐
                              ├──> Neural Network ──> Prediction
Item ID ──> Item Embedding ──┘
```

**Key idea**: Replace inner product with multi-layer neural network.

---

### Mathematical Formulation

**MF (linear)**:
$$\hat{y}_{ui} = \mathbf{p}_u^T \mathbf{q}_i$$

**NCF (non-linear)**:
$$\hat{y}_{ui} = f(\mathbf{p}_u, \mathbf{q}_i | \Theta)$$

where:
- $f$ = neural network
- $\Theta$ = network parameters (weights, biases)

---

## NCF Components

### 1. Generalized Matrix Factorization (GMF)

**MF as a neural network**:

```
User Embedding: p_u ∈ ℝ^k
Item Embedding: q_i ∈ ℝ^k
    ↓
Element-wise Product: p_u ⊙ q_i  (⊙ = Hadamard product)
    ↓
Linear Layer: h^T (p_u ⊙ q_i)
    ↓
Sigmoid: σ(h^T (p_u ⊙ q_i))
```

**Formula**:
$$\hat{y}_{ui}^{GMF} = \sigma(\mathbf{h}^T (\mathbf{p}_u \odot \mathbf{q}_i))$$

**Note**: If $\mathbf{h} = \mathbf{1}$ (all ones), this is standard MF.

---

### 2. Multi-Layer Perceptron (MLP)

**Deep neural network** on concatenated embeddings:

```
User Embedding: p_u ∈ ℝ^k
Item Embedding: q_i ∈ ℝ^k
    ↓
Concatenate: [p_u, q_i] ∈ ℝ^{2k}
    ↓
Dense Layer 1: ReLU(W_1[p_u, q_i] + b_1)
    ↓
Dense Layer 2: ReLU(W_2 h_1 + b_2)
    ↓
...
    ↓
Dense Layer L: ReLU(W_L h_{L-1} + b_L)
    ↓
Output Layer: σ(w^T h_L)
```

**Formula**:
$$\mathbf{z}_1 = \text{concat}(\mathbf{p}_u, \mathbf{q}_i)$$
$$\mathbf{h}_l = \text{ReLU}(\mathbf{W}_l \mathbf{h}_{l-1} + \mathbf{b}_l), \quad l = 1, \ldots, L$$
$$\hat{y}_{ui}^{MLP} = \sigma(\mathbf{w}^T \mathbf{h}_L)$$

**Capacity**: Can learn **arbitrary non-linear functions**.

---

### 3. NeuMF (Neural Matrix Factorization)

**Combine GMF and MLP**:

```
        User ID          Item ID
           |                |
    ┌──────┴──────┐   ┌────┴─────┐
    |             |   |          |
User Emb (GMF) User Emb (MLP) Item Emb (MLP) Item Emb (GMF)
    |             |   |          |
    └──────┬──────┘   └────┬─────┘
           |               |
    Element-wise       Concatenate
       Product             |
           |            MLP Layers
           |               |
           └───────┬───────┘
                   |
               Concatenate
                   |
              Output Layer
                   |
              Prediction
```

**Formula**:
$$\phi^{GMF} = \mathbf{p}_u^{GMF} \odot \mathbf{q}_i^{GMF}$$
$$\phi^{MLP} = \text{MLP}(\mathbf{p}_u^{MLP}, \mathbf{q}_i^{MLP})$$
$$\hat{y}_{ui} = \sigma(\mathbf{h}^T [\phi^{GMF}, \phi^{MLP}])$$

**Benefits**:
- **GMF**: Captures linear interactions (like MF)
- **MLP**: Captures non-linear interactions
- **Fusion**: Best of both worlds

---

## Implementation

### PyTorch Code

```python
import torch
import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=64, layers=[64, 32, 16, 8]):
        """
        NeuMF: Neural Matrix Factorization.

        Args:
            n_users: Number of users
            n_items: Number of items
            n_factors: Embedding dimension for GMF
            layers: MLP layer sizes
        """
        super(NeuMF, self).__init__()

        # GMF embeddings
        self.user_embedding_gmf = nn.Embedding(n_users, n_factors)
        self.item_embedding_gmf = nn.Embedding(n_items, n_factors)

        # MLP embeddings (larger for richer representation)
        self.user_embedding_mlp = nn.Embedding(n_users, layers[0]//2)
        self.item_embedding_mlp = nn.Embedding(n_items, layers[0]//2)

        # MLP layers
        self.mlp_layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.mlp_layers.append(nn.Linear(layers[i], layers[i+1]))

        # Output layer
        self.output_layer = nn.Linear(n_factors + layers[-1], 1)

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings and weights."""
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

        for layer in self.mlp_layers:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, user_indices, item_indices):
        """
        Forward pass.

        Args:
            user_indices: Tensor of user IDs
            item_indices: Tensor of item IDs

        Returns:
            Predictions (0-1 range)
        """
        # GMF path
        user_emb_gmf = self.user_embedding_gmf(user_indices)
        item_emb_gmf = self.item_embedding_gmf(item_indices)
        gmf_vector = user_emb_gmf * item_emb_gmf  # Element-wise product

        # MLP path
        user_emb_mlp = self.user_embedding_mlp(user_indices)
        item_emb_mlp = self.item_embedding_mlp(item_indices)
        mlp_vector = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)

        for layer in self.mlp_layers:
            mlp_vector = torch.relu(layer(mlp_vector))

        # Concatenate GMF and MLP
        concat_vector = torch.cat([gmf_vector, mlp_vector], dim=-1)

        # Output
        prediction = self.sigmoid(self.output_layer(concat_vector))

        return prediction.squeeze()
```

---

### Training Loop

```python
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class RatingDataset(Dataset):
    """Dataset for implicit feedback (binary labels)."""
    def __init__(self, user_ids, item_ids, labels):
        self.users = torch.LongTensor(user_ids)
        self.items = torch.LongTensor(item_ids)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

def train_ncf(model, train_loader, val_loader, epochs=20, lr=0.001):
    """Train NeuMF model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # Binary cross-entropy for implicit feedback

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        for users, items, labels in train_loader:
            users, items, labels = users.to(device), items.to(device), labels.to(device)

            # Forward
            predictions = model(users, items)
            loss = criterion(predictions, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for users, items, labels in val_loader:
                users, items, labels = users.to(device), items.to(device), labels.to(device)
                predictions = model(users, items)
                val_loss += criterion(predictions, labels).item()

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss/len(train_loader):.4f}, "
              f"Val Loss = {val_loss/len(val_loader):.4f}")

# Example usage
n_users = 1000
n_items = 500

model = NeuMF(n_users, n_items, n_factors=64, layers=[128, 64, 32, 16])

# Assume train_data, val_data are prepared (user_ids, item_ids, labels)
train_dataset = RatingDataset(train_user_ids, train_item_ids, train_labels)
val_dataset = RatingDataset(val_user_ids, val_item_ids, val_labels)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

train_ncf(model, train_loader, val_loader, epochs=20, lr=0.001)
```

---

## Negative Sampling

### Challenge: Implicit Feedback

**No explicit ratings**, only positive signals (clicks, views, purchases).

**Problem**: All unobserved items treated as negative → extreme class imbalance.

**Solution**: Sample negative examples.

---

### Sampling Strategy

**For each positive (user, item) pair**:
1. Sample $k$ negative items (items user didn't interact with)
2. Label: Positive = 1, Negative = 0
3. Train to distinguish positive from negative

**Typical $k$**: 4-10 negatives per positive

---

### Code

```python
import random

def sample_negatives(user_id, positive_items, all_items, n_neg=4):
    """
    Sample negative items for a user.

    Args:
        user_id: User ID
        positive_items: Set of items user interacted with
        all_items: Set of all items
        n_neg: Number of negative samples

    Returns:
        List of negative item IDs
    """
    negative_items = list(all_items - positive_items)
    return random.sample(negative_items, min(n_neg, len(negative_items)))

# Build training data
train_data = []

for user in users:
    positive_items = user_interactions[user]  # Set of items user liked

    # Add positive examples
    for item in positive_items:
        train_data.append((user, item, 1))  # Label = 1

    # Sample and add negative examples
    negatives = sample_negatives(user, positive_items, all_items, n_neg=4)
    for item in negatives:
        train_data.append((user, item, 0))  # Label = 0
```

---

## Pre-Training Strategy

### Motivation

**GMF and MLP have different objectives** in NeuMF.

**Problem**: Training from scratch may not converge well.

**Solution**: Pre-train GMF and MLP separately, then combine.

---

### Process

**Step 1**: Train GMF alone
```python
model_gmf = GMF(n_users, n_items, n_factors=64)
train(model_gmf, data, epochs=20)
```

**Step 2**: Train MLP alone
```python
model_mlp = MLP(n_users, n_items, layers=[128, 64, 32, 16])
train(model_mlp, data, epochs=20)
```

**Step 3**: Initialize NeuMF with pre-trained weights
```python
model_neumf = NeuMF(n_users, n_items, n_factors=64, layers=[128, 64, 32, 16])

# Transfer GMF embeddings
model_neumf.user_embedding_gmf.weight = model_gmf.user_embedding.weight
model_neumf.item_embedding_gmf.weight = model_gmf.item_embedding.weight

# Transfer MLP embeddings
model_neumf.user_embedding_mlp.weight = model_mlp.user_embedding.weight
model_neumf.item_embedding_mlp.weight = model_mlp.item_embedding.weight

# Fine-tune
train(model_neumf, data, epochs=10, lr=0.0001)  # Lower LR for fine-tuning
```

**Benefits**:
- Faster convergence
- Better performance (2-3% improvement reported in paper)

---

## Experimental Results (from Paper)

### Datasets

1. **MovieLens 1M**: 1M ratings, 6K users, 3.7K movies
2. **Pinterest**: 1.5M interactions, 55K users, 1.5M pins

### Metrics

- **HR@10** (Hit Rate @ 10): % of test items in top-10
- **NDCG@10**: Normalized Discounted Cumulative Gain

### Performance

| Method | MovieLens HR@10 | MovieLens NDCG@10 | Pinterest HR@10 |
|--------|-----------------|-------------------|-----------------|
| **ItemPop** (popularity) | 0.471 | 0.263 | 0.419 |
| **BPR-MF** | 0.682 | 0.416 | 0.566 |
| **eALS** (MF variant) | 0.685 | 0.419 | 0.571 |
| **GMF** | 0.705 | 0.432 | 0.591 |
| **MLP** | 0.692 | 0.425 | 0.582 |
| **NeuMF** (no pre-train) | 0.716 | 0.441 | 0.603 |
| **NeuMF (pre-train)** | **0.726** | **0.445** | **0.613** |

**Key findings**:
- NeuMF beats MF by ~5-7%
- Pre-training helps (+1-2%)
- GMF alone competitive with MF variants

---

## When to Use NCF

### Advantages

✅ **Non-linear interactions**: Can model complex user-item relationships

✅ **State-of-the-art (2017)**: Significantly outperformed MF

✅ **Flexible**: Easy to add features, modify architecture

✅ **End-to-end**: Learns embeddings and interaction function jointly

---

### Disadvantages

❌ **Slower training**: Neural networks require more epochs than MF

❌ **More hyperparameters**: Layer sizes, dropout, learning rate, etc.

❌ **Overfitting risk**: Needs regularization (dropout, weight decay)

❌ **Interpretability**: Harder to explain than MF

---

### Recommendations

**Use NCF when**:
- Large dataset (millions of interactions)
- Complex user-item relationships suspected
- Computational resources available (GPU)
- State-of-the-art performance critical

**Stick with MF when**:
- Small dataset (<100K interactions)
- Interpretability important
- Limited computation
- Baseline needed

---

## Extensions and Variants

### 1. **Neural CF with Side Information**

Add user/item features:
```python
# Concatenate features with embeddings
user_vector = torch.cat([user_emb, user_age, user_gender], dim=-1)
item_vector = torch.cat([item_emb, item_category, item_price], dim=-1)
```

---

### 2. **Attention Mechanisms**

Weight different latent factors differently:
```python
attention_weights = softmax(W_attention @ user_emb)
weighted_emb = attention_weights * user_emb
```

---

### 3. **Deep Crossing**

Microsoft's variant using deep networks for feature crosses.

---

## Comparison with Other Methods

| Method | Linearity | Interpretability | Performance | Training Time |
|--------|-----------|------------------|-------------|---------------|
| **MF** | Linear | High | Good | Fast |
| **GMF** | Linear (generalized) | Medium | Good | Fast |
| **MLP** | Non-linear | Low | Very Good | Medium |
| **NeuMF** | Non-linear | Low | Best | Slow |

---

## Summary

**Neural Collaborative Filtering (NCF)**:
- Replaces MF's inner product with neural network
- **GMF**: Generalized MF (linear)
- **MLP**: Deep network (non-linear)
- **NeuMF**: Fusion of GMF + MLP (best performance)

**Key techniques**:
- Negative sampling for implicit feedback
- Pre-training for better initialization
- Batch training with Adam optimizer

**Performance**: 5-7% improvement over traditional MF

**Trade-offs**: Better accuracy vs. slower training, less interpretable

**Impact**: Opened the door for deep learning in recommendation systems

**Next**: See **deep-variants.md** for AutoRec, VAE-CF, and other deep models.

---

## References

1. **He, X., et al. (2017)**. "Neural collaborative filtering". *WWW 2017*, 173-182.
   - **Primary source**: Original NCF paper

2. **He, X., et al. (2017)**. "Outer product-based neural collaborative filtering". *IJCAI*.
   - Variant using outer product instead of concatenation

3. **Code**: https://github.com/hexiangnan/neural_collaborative_filtering
   - Official TensorFlow implementation
