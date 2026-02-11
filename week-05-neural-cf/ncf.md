# Week 5: Neural Collaborative Filtering (NCF)

## Opening: Why Does Matrix Factorization Fail?

*Before we dive into neural networks, let me show you exactly where matrix factorization breaks down.*

### The XOR Problem: A Pattern MF Cannot Capture

**Scenario**: Consider a simple recommendation with 2 latent factors.

**User Alice's preferences** (ground truth):
| Factor 1 (Action) | Factor 2 (Romance) | Alice Likes? |
|:-:|:-:|:-:|
| Low (0) | Low (0) | No |
| Low (0) | High (1) | Yes |
| High (1) | Low (0) | Yes |
| High (1) | High (1) | No |

*Can you see the pattern?* Alice likes movies that are **either** action **or** romance, but **not both** and **not neither**. This is the classic XOR (exclusive OR) pattern.

**What happens when MF tries to learn this?**

MF predicts: $\hat{r} = \mathbf{p}_u^T \mathbf{q}_i = p_1 \cdot q_1 + p_2 \cdot q_2$

Let's try to find embeddings that work:
- For Alice to like "High Action, Low Romance" ($q = [1, 0]$): We need $p_1 \cdot 1 + p_2 \cdot 0 = p_1 > 0$ → $p_1$ must be positive
- For Alice to like "Low Action, High Romance" ($q = [0, 1]$): We need $p_1 \cdot 0 + p_2 \cdot 1 = p_2 > 0$ → $p_2$ must be positive
- For Alice to dislike "High Action, High Romance" ($q = [1, 1]$): We need $p_1 \cdot 1 + p_2 \cdot 1 = p_1 + p_2 < 0$

**But wait!** If $p_1 > 0$ and $p_2 > 0$, then $p_1 + p_2 > 0$. **Contradiction!**

*This is mathematically impossible with a dot product.* The inner product is fundamentally a **linear** operation, and XOR is a **non-linear** pattern.

**Concrete numbers**:
- Best MF can do: $\mathbf{p}_{Alice} = [0.5, 0.5]$
- Predictions:
  - Action only: $0.5 \times 1 + 0.5 \times 0 = 0.5$ (want high)
  - Romance only: $0.5 \times 0 + 0.5 \times 1 = 0.5$ (want high)
  - Both: $0.5 \times 1 + 0.5 \times 1 = 1.0$ (want LOW! But it's highest!)
  - Neither: $0.5 \times 0 + 0.5 \times 0 = 0.0$ (want low, correct)

**Error rate**: MF gets 1 out of 4 predictions wrong (25% error), and worse, it's maximally confident on the wrong prediction!

*Notice that* the problem isn't lack of data or poor optimization—it's a **fundamental limitation** of the linear inner product.

**The solution**: Replace the dot product with a function that can learn non-linear patterns. Enter: Neural Collaborative Filtering.

---

## Overview

Neural Collaborative Filtering (NCF) replaces the linear inner product in matrix factorization with a deep neural network, enabling non-linear modeling of user-item interactions. This breakthrough paper (He et al., 2017) demonstrated that neural networks could significantly outperform traditional CF methods.

**Key Innovation**: Learn arbitrary functions from data instead of assuming linear interaction.

---

## NCF Framework: The Big Picture

### High-Level Architecture

```
User ID ──> User Embedding ──┐
                              ├──> Neural Network ──> Prediction
Item ID ──> Item Embedding ──┘
```

**The key insight**: Replace the fixed inner product with a learnable function.

### Mathematical Formulation

**MF (linear)**:
$$\hat{y}_{ui} = \mathbf{p}_u^T \mathbf{q}_i$$

**NCF (non-linear)**:
$$\hat{y}_{ui} = f(\mathbf{p}_u, \mathbf{q}_i | \Theta)$$

where:
- $f$ = neural network (can learn any function!)
- $\Theta$ = network parameters (weights, biases)

*What happens if* we choose $f$ to be a dot product? We get MF back! So NCF is strictly more expressive.

---

## NCF Components

### 1. Generalized Matrix Factorization (GMF)

*Let's build up to neural networks step by step. First, let's "neuralize" MF.*

**Architecture**:

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

where:
- $\hat{y}_{ui}^{GMF}$ = predicted score for user $u$ and item $i$ using GMF
- $\mathbf{p}_u \in \mathbb{R}^k$ = user $u$'s embedding vector
- $\mathbf{q}_i \in \mathbb{R}^k$ = item $i$'s embedding vector
- $\odot$ = element-wise (Hadamard) product: $(\mathbf{p}_u \odot \mathbf{q}_i)_j = p_{uj} \cdot q_{ij}$
- $\mathbf{h} \in \mathbb{R}^k$ = learnable weight vector (assigns importance to each dimension)
- $\sigma(\cdot)$ = sigmoid activation function: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- $k$ = embedding dimension

---

### The GMF-to-MF Equivalence Proof

*Let me prove to you that GMF is a true generalization of MF.*

**Claim**: When $\mathbf{h} = [1, 1, \ldots, 1]$ (all ones) and we remove the sigmoid, GMF reduces exactly to MF.

**Proof**:

**Step 1**: Write out GMF prediction (ignoring sigmoid for now):
$$\hat{y}_{ui}^{GMF} = \mathbf{h}^T (\mathbf{p}_u \odot \mathbf{q}_i)$$

**Step 2**: Expand the Hadamard product:
$$\mathbf{p}_u \odot \mathbf{q}_i = \begin{bmatrix} p_{u1} \cdot q_{i1} \\ p_{u2} \cdot q_{i2} \\ \vdots \\ p_{uk} \cdot q_{ik} \end{bmatrix}$$

**Step 3**: Apply the dot product with $\mathbf{h}$:
$$\mathbf{h}^T (\mathbf{p}_u \odot \mathbf{q}_i) = \sum_{j=1}^k h_j \cdot (p_{uj} \cdot q_{ij})$$

**Step 4**: Set $\mathbf{h} = [1, 1, \ldots, 1]$:
$$\sum_{j=1}^k 1 \cdot (p_{uj} \cdot q_{ij}) = \sum_{j=1}^k p_{uj} \cdot q_{ij} = \mathbf{p}_u^T \mathbf{q}_i$$

**This is exactly the MF prediction formula!** $\blacksquare$

**Numerical verification**:
- Let $\mathbf{p}_u = [0.5, -0.3, 0.8]$ and $\mathbf{q}_i = [0.2, 0.6, 0.4]$
- MF: $\hat{y} = 0.5 \times 0.2 + (-0.3) \times 0.6 + 0.8 \times 0.4 = 0.1 - 0.18 + 0.32 = 0.24$
- GMF with $\mathbf{h} = [1, 1, 1]$:
  - Hadamard: $[0.5 \times 0.2, -0.3 \times 0.6, 0.8 \times 0.4] = [0.1, -0.18, 0.32]$
  - Dot with $\mathbf{h}$: $1 \times 0.1 + 1 \times (-0.18) + 1 \times 0.32 = 0.24$ ✓

*Can you see why* this equivalence matters? It means GMF can do everything MF can do, **plus more** (by learning $\mathbf{h}$).

**What does learnable $\mathbf{h}$ add?**

$\mathbf{h}$ is a **learnable importance vector**. It says:
- "Dimension 1 matters a lot for prediction" (high $h_1$)
- "Dimension 5 is less important" (low $h_5$)

In MF, all dimensions contribute equally. In GMF, the model learns which dimensions matter most.

---

### 2. Multi-Layer Perceptron (MLP)

*Now let's add true non-linearity.*

**Architecture**:

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

**Formula derivation**:

**Step 1** - Concatenate embeddings:
$$\mathbf{z}_1 = \text{concat}(\mathbf{p}_u, \mathbf{q}_i) = \begin{bmatrix} \mathbf{p}_u \\ \mathbf{q}_i \end{bmatrix} \in \mathbb{R}^{2k}$$

*Why concatenate instead of Hadamard product?* Concatenation preserves all information about both vectors separately. The network can then learn **any** interaction pattern, not just element-wise.

**Step 2** - Apply hidden layers:
$$\mathbf{h}_l = \text{ReLU}(\mathbf{W}_l \mathbf{h}_{l-1} + \mathbf{b}_l), \quad l = 1, \ldots, L$$

*Why ReLU?* ReLU introduces non-linearity: $\text{ReLU}(x) = \max(0, x)$. Without it, stacking layers would still give a linear function!

**Step 3** - Final prediction:
$$\hat{y}_{ui}^{MLP} = \sigma(\mathbf{w}^T \mathbf{h}_L)$$

**Capacity**: By the Universal Approximation Theorem, this can learn **any** continuous function given enough neurons!

---

#### Layer Sizing Intuition: The Funnel Architecture

*Why do layer sizes typically decrease? (e.g., 128 → 64 → 32 → 16)*

**The Funnel Metaphor**:

```
Layer 0: [64-dim user] + [64-dim item] = 128 dimensions
         "Everything about user and item"
              ↓
Layer 1: 64 dimensions
         "The most important interaction patterns"
              ↓
Layer 2: 32 dimensions
         "Compressed, essential features"
              ↓
Layer 3: 16 dimensions
         "The core signal: will user like item?"
              ↓
Output: 1 dimension (probability)
```

**Why this works**:

1. **Information bottleneck**: Force the network to identify what matters
2. **Regularization**: Fewer parameters in later layers → less overfitting
3. **Hierarchical abstraction**: Early layers: raw patterns; late layers: decision-relevant features

---

### 3. NeuMF (Neural Matrix Factorization)

*Here's the key insight: GMF captures linear interactions, MLP captures non-linear ones. Why not use both?*

**Architecture**:

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

**Formula derivation**:

**Step 1** - GMF path (linear interactions):
$$\phi^{GMF} = \mathbf{p}_u^{GMF} \odot \mathbf{q}_i^{GMF}$$

*Why separate embeddings?* GMF and MLP have different goals. GMF wants embeddings optimized for element-wise multiplication; MLP wants them optimized for concatenation. Sharing would be a compromise.

**Step 2** - MLP path (non-linear interactions):
$$\phi^{MLP} = \text{MLP}(\text{concat}(\mathbf{p}_u^{MLP}, \mathbf{q}_i^{MLP}))$$

**Step 3** - Fusion (best of both worlds):
$$\hat{y}_{ui} = \sigma(\mathbf{h}^T [\phi^{GMF}; \phi^{MLP}])$$

where $[\cdot ; \cdot]$ denotes concatenation.

*Notice that* the final layer learns how to weight the GMF vs MLP contributions. If your data has mostly linear patterns, it will weight GMF higher. If complex patterns dominate, MLP gets more weight.

---

## Complete Numerical Walkthrough

*Let's trace through NeuMF with actual numbers. I'll use small dimensions so you can verify by hand.*

### Setup

**Users**: Alice (ID=0), Bob (ID=1)
**Items**: Movie A (ID=0), Movie B (ID=1), Movie C (ID=2)
**Embedding dimension**: $k=2$ for GMF, MLP uses 2+2=4 input

### Learned Embeddings (after training)

**GMF Embeddings**:
| | Dim 1 | Dim 2 |
|---|-------|-------|
| Alice (GMF) | 0.8 | 0.3 |
| Bob (GMF) | -0.2 | 0.9 |
| Movie A (GMF) | 0.5 | 0.4 |
| Movie B (GMF) | 0.1 | 0.7 |
| Movie C (GMF) | 0.6 | -0.2 |

**MLP Embeddings**:
| | Dim 1 | Dim 2 |
|---|-------|-------|
| Alice (MLP) | 0.4 | 0.6 |
| Bob (MLP) | 0.7 | -0.1 |
| Movie A (MLP) | 0.3 | 0.5 |
| Movie B (MLP) | 0.8 | 0.2 |
| Movie C (MLP) | -0.3 | 0.9 |

### Prediction for (Alice, Movie A)

**Step 1: GMF Path**

Compute element-wise product:
$$\phi^{GMF} = \mathbf{p}_{Alice}^{GMF} \odot \mathbf{q}_{A}^{GMF} = \begin{bmatrix} 0.8 \\ 0.3 \end{bmatrix} \odot \begin{bmatrix} 0.5 \\ 0.4 \end{bmatrix} = \begin{bmatrix} 0.8 \times 0.5 \\ 0.3 \times 0.4 \end{bmatrix} = \begin{bmatrix} 0.40 \\ 0.12 \end{bmatrix}$$

**Step 2: MLP Path**

Concatenate embeddings:
$$\mathbf{z}_0 = [\mathbf{p}_{Alice}^{MLP}; \mathbf{q}_{A}^{MLP}] = [0.4, 0.6, 0.3, 0.5]$$

Apply Layer 1 (let's say $\mathbf{W}_1 \in \mathbb{R}^{2 \times 4}$, $\mathbf{b}_1 \in \mathbb{R}^2$):

Assume:
$$\mathbf{W}_1 = \begin{bmatrix} 0.5 & 0.3 & -0.2 & 0.4 \\ 0.1 & -0.4 & 0.6 & 0.2 \end{bmatrix}, \quad \mathbf{b}_1 = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}$$

Compute:
$$\mathbf{W}_1 \mathbf{z}_0 + \mathbf{b}_1 = \begin{bmatrix} 0.5(0.4) + 0.3(0.6) + (-0.2)(0.3) + 0.4(0.5) + 0.1 \\ 0.1(0.4) + (-0.4)(0.6) + 0.6(0.3) + 0.2(0.5) + (-0.1) \end{bmatrix}$$

$$= \begin{bmatrix} 0.20 + 0.18 - 0.06 + 0.20 + 0.1 \\ 0.04 - 0.24 + 0.18 + 0.10 - 0.1 \end{bmatrix} = \begin{bmatrix} 0.62 \\ -0.02 \end{bmatrix}$$

Apply ReLU: $\mathbf{h}_1 = \text{ReLU}([0.62, -0.02]) = [0.62, 0.0]$

*Notice that* the second neuron "died" (became 0) because its pre-activation was negative. This is ReLU doing its job—creating sparsity.

**Step 3: Fusion**

Concatenate GMF and MLP outputs:
$$\mathbf{z}_{final} = [\phi^{GMF}; \mathbf{h}_1] = [0.40, 0.12, 0.62, 0.0]$$

Apply output layer (let's say $\mathbf{h}_{out} = [0.5, 0.3, 0.4, 0.2]$):
$$\text{logit} = \mathbf{h}_{out}^T \mathbf{z}_{final} = 0.5(0.40) + 0.3(0.12) + 0.4(0.62) + 0.2(0.0)$$
$$= 0.20 + 0.036 + 0.248 + 0 = 0.484$$

Apply sigmoid:
$$\hat{y}_{Alice, A} = \sigma(0.484) = \frac{1}{1 + e^{-0.484}} = \frac{1}{1 + 0.616} = 0.619$$

**Interpretation**: NeuMF predicts Alice has a 61.9% probability of liking Movie A.

---

### Comparison: What Would Pure MF Predict?

Using the GMF embeddings with $\mathbf{h} = [1, 1]$:
$$\hat{y}_{MF} = \sigma(\mathbf{p}_{Alice}^{GMF} \cdot \mathbf{q}_{A}^{GMF}) = \sigma(0.8 \times 0.5 + 0.3 \times 0.4) = \sigma(0.52) = 0.627$$

*Can you see why* NeuMF might differ? The MLP path contributes additional signal (the 0.62 from the first hidden unit) that pure MF doesn't capture.

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

### The Problem: Implicit Feedback Has No Negatives

*Let me show you the challenge concretely.*

**Scenario**: Netflix with 10,000 movies. User Alice has watched 50 movies.

**The data we have**:
- 50 positive examples: (Alice, Movie_1) = 1, (Alice, Movie_2) = 1, ...
- 9,950 "unknowns": All movies Alice hasn't watched

**Naive approach**: Treat all unwatched as negative. Train on all 10,000 pairs.

**Why this fails (with numbers)**:

1. **Extreme imbalance**: 50 positives vs 9,950 negatives (0.5% positive!)
   - A model that always predicts 0 gets 99.5% accuracy!
   - But it's useless for recommendation

2. **False negatives**: Alice hasn't watched "Inception" — does she dislike it, or just hasn't discovered it?
   - Many "negatives" are potential positives
   - Training on them as negatives corrupts the model

3. **Computational cost**: 10,000 pairs × 1M users = 10 billion training examples per epoch!

### The Solution: Sample Negatives

**For each positive (user, item) pair**:
1. Sample $k$ random items the user hasn't interacted with
2. Label: Positive = 1, Sampled = 0
3. Train to distinguish positive from sampled negatives

**Typical $k$**: 4-10 negatives per positive

### The Intuition: Contrastive Learning

*Think of it like a multiple choice test.*

**Question**: Which movie did Alice actually watch?

```
A. Inception (positive - she watched it)    ← Correct answer
B. Random Movie #4521 (negative - didn't watch)
C. Random Movie #892 (negative)
D. Random Movie #7723 (negative)
E. Random Movie #156 (negative)
```

**Training objective**: Make the model rank A above B, C, D, E.

*Notice that* we're not claiming Alice dislikes B, C, D, E — just that she **definitely** watched A. The model learns to identify positive signals.

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

### Why Pre-Train?

**GMF and MLP have different optimization landscapes.** Training NeuMF from scratch can get stuck in poor local minima.

**Analogy**: Imagine learning to play piano and violin simultaneously vs. learning each separately first, then combining for a duet.

### Process

**Step 1**: Train GMF alone (until convergence)
```python
model_gmf = GMF(n_users, n_items, n_factors=64)
train(model_gmf, data, epochs=20)
```

**Step 2**: Train MLP alone (until convergence)
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

# Fine-tune with lower learning rate
train(model_neumf, data, epochs=10, lr=0.0001)
```

**Benefits**: Faster convergence, 2-3% improvement in final metrics.

---

## Experimental Results (from Paper)

### Datasets

1. **MovieLens 1M**: 1M ratings, 6K users, 3.7K movies
2. **Pinterest**: 1.5M interactions, 55K users, 1.5M pins

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
- Pre-training adds +1-2%
- GMF alone already beats classical MF (learned $\mathbf{h}$ helps!)

---

## What Can Go Wrong: Failure Modes and Solutions

### Failure Mode 1: Overfitting with Deep MLP

**Symptoms**:
- Training loss decreases steadily
- Validation loss decreases, then starts **increasing**
- Test HR@10 much lower than validation HR@10
- Model memorizes training pairs but fails on new pairs

**Concrete example**:
```
Epoch 10: Train Loss = 0.15, Val Loss = 0.28
Epoch 20: Train Loss = 0.05, Val Loss = 0.35  ← Gap widening!
Epoch 30: Train Loss = 0.01, Val Loss = 0.45  ← Severe overfitting
```

**Causes**:
- MLP too deep/wide for dataset size
- Insufficient regularization
- Too many epochs without early stopping

**Solutions**:
1. **Add dropout** (0.2-0.5 between layers):
```python
self.dropout = nn.Dropout(0.3)
mlp_vector = self.dropout(torch.relu(layer(mlp_vector)))
```

2. **Use weight decay** (L2 regularization):
```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

3. **Reduce network size**: Try [64, 32, 16] instead of [256, 128, 64, 32]

4. **Early stopping**: Stop when validation loss hasn't improved for 5 epochs

---

### Failure Mode 2: Embedding Dimension Mismatch

**Symptoms**:
- GMF path dominates (or is ignored)
- MLP path dominates (or is ignored)
- Performance similar to using just one path

**Concrete example**:
```
GMF embedding dim: 8
MLP final output dim: 64

After fusion: [8 GMF dims, 64 MLP dims] = 72 dims
The 8 GMF dimensions get "drowned out" by 64 MLP dimensions!
```

**Causes**:
- GMF and MLP contribute unequal signal to final prediction
- Optimization favors one path over the other

**Solutions**:
1. **Balance dimensions**: GMF dim ≈ MLP final layer dim
```python
# Good: balanced
n_factors = 32  # GMF
layers = [128, 64, 32]  # MLP ends at 32

# Bad: imbalanced
n_factors = 8   # GMF
layers = [256, 128, 64]  # MLP ends at 64
```

2. **Monitor path contributions**: Check gradient magnitudes for each path

3. **Try separate learning rates**: Lower LR for the dominant path

---

### Failure Mode 3: Wrong Negative Sampling Ratio

**Symptoms (too few negatives, k=1)**:
- Model overpredicts positive class
- Many false positives in recommendations
- Precision very low, recall high

**Symptoms (too many negatives, k=20)**:
- Model underpredicts positive class
- Training is slow (too many samples per epoch)
- Model becomes too conservative

**Concrete example**:
```
Dataset: 50 positives per user, 10,000 items

k=1: Train on 50 pos + 50 neg = 100 samples
  → 50% positive class in training
  → Model thinks positives are common
  → Overpredicts!

k=20: Train on 50 pos + 1000 neg = 1050 samples
  → 4.8% positive class in training
  → Model thinks positives are rare
  → Underpredicts!
```

**Solutions**:
1. **Start with k=4** (paper's default)
2. **Tune k based on dataset**: Sparser datasets may need fewer negatives
3. **Monitor positive prediction rate**: Should be ~5-10%, not 50% or 0.5%
4. **Consider popularity-based negative sampling**: Sample popular items as negatives (harder negatives, better learning)

---

### Failure Mode 4: Cold Start Amplification

**Symptoms**:
- Works well for active users, poorly for new users
- Item recommendations stuck on popular items for cold users
- Embeddings for cold users/items are essentially random

**Causes**:
- Neural networks amplify the cold start problem
- MF at least has regularization pushing embeddings toward zero
- NCF's random initialization stays random without updates

**Solutions**:
1. **Default to popularity** for cold users (hybrid system)
2. **Use content features** for cold items (add side information)
3. **Regularize embeddings** toward zero or average:
```python
reg_loss = 0.01 * (user_emb.norm() + item_emb.norm())
total_loss = bce_loss + reg_loss
```

4. **Minimum interaction threshold**: Only include users/items with ≥5 interactions

---

## When to Use NCF

### Advantages

- **Non-linear interactions**: Can model complex user-item relationships (XOR, AND, OR patterns)
- **State-of-the-art (2017)**: Significantly outperformed MF
- **Flexible**: Easy to add features, modify architecture
- **End-to-end**: Learns embeddings and interaction function jointly

### Disadvantages

- **Slower training**: Neural networks require more epochs than MF
- **More hyperparameters**: Layer sizes, dropout, learning rate, negative sampling ratio
- **Overfitting risk**: Needs regularization (dropout, weight decay)
- **Interpretability**: Harder to explain than MF ("which factors matter?")

### Decision Guide

**Use NCF when**:
- Large dataset (millions of interactions)
- Complex user-item relationships suspected
- Computational resources available (GPU)
- State-of-the-art performance critical

**Stick with MF when**:
- Small dataset (<100K interactions)
- Interpretability important
- Limited computation
- Baseline needed quickly

---

## Summary

**Neural Collaborative Filtering (NCF)**:
- Replaces MF's inner product with neural network
- **GMF**: Generalized MF with learnable importance weights
- **MLP**: Deep network for non-linear patterns
- **NeuMF**: Fusion of GMF + MLP (best performance)

**Key techniques**:
- Negative sampling for implicit feedback
- Pre-training for better initialization
- Batch training with Adam optimizer

**The fundamental insight**: MF assumes $\hat{y} = \mathbf{p}^T \mathbf{q}$, which is linear. NCF learns $\hat{y} = f(\mathbf{p}, \mathbf{q})$, which can be anything. This strictly greater expressiveness enables capturing patterns like XOR that MF fundamentally cannot represent.

**Next**: See **deep-variants.md** for AutoRec, VAE-CF, and other deep models.

---

## References

1. **He, X., et al. (2017)**. "Neural collaborative filtering". *WWW 2017*, 173-182.
   - **Primary source**: Original NCF paper

2. **He, X., et al. (2017)**. "Outer product-based neural collaborative filtering". *IJCAI*.
   - Variant using outer product instead of concatenation

3. **Code**: https://github.com/hexiangnan/neural_collaborative_filtering
   - Official TensorFlow implementation
