# Week 5: Neural Collaborative Filtering - Deep Variants

## The Problem: What Non-Linear Pattern Can't Matrix Factorization Capture?

*Before diving into neural networks, let me show you exactly where matrix factorization fails.*

**Consider this scenario**: A user has complex preferences that MF struggles with.

**User Alice's preferences**:
- Loves **Nolan** films (Inception, Interstellar, Tenet)
- Loves **Sci-Fi** genre
- But only when **combined**! She doesn't like generic sci-fi or non-sci-fi Nolan films as much

**The MF model**:
$$\hat{r}_{ui} = \mathbf{p}_u^T \mathbf{q}_i = \sum_{k=1}^K p_{uk} q_{ik}$$

*Can you see the problem?*

MF computes a weighted **sum** of factor contributions. It can't express: "I like this **if and only if** both conditions are met."

**Example with k=2 factors**:
- Factor 1: "Nolan-ness"
- Factor 2: "Sci-Fi-ness"

| Movie | Factor 1 (Nolan) | Factor 2 (Sci-Fi) | MF Score | Alice's Actual Preference |
|-------|------------------|-------------------|----------|---------------------------|
| Inception | 0.9 | 0.9 | 1.62 | **LOVES** (5 stars) |
| Interstellar | 0.9 | 0.85 | 1.53 | **LOVES** (5 stars) |
| Dunkirk | 0.9 | 0.1 | 0.90 | **MEH** (3 stars) |
| Avatar | 0.1 | 0.95 | 0.95 | **MEH** (3 stars) |
| The Notebook | 0.1 | 0.1 | 0.18 | **DISLIKES** (1 star) |

**The problem**: Dunkirk and Avatar get similar MF scores (~0.9), but Alice has very different opinions!

**What we need**: A model that can learn: "High score **only if** both factors are high" — this is a non-linear (AND) relationship.

**The solution**: Neural networks can learn arbitrary non-linear functions!

---

## Overview

**Neural Collaborative Filtering (NCF)** replaces the inner product in matrix factorization with neural networks, enabling the model to learn complex, non-linear user-item interactions.

**Evolution**:
- **Matrix Factorization** (2006): $\hat{r}_{ui} = \mathbf{p}_u^T \mathbf{q}_i$ (linear)
- **Neural CF** (2017): $\hat{r}_{ui} = f(\mathbf{p}_u, \mathbf{q}_i; \Theta)$ (non-linear with neural network $f$)

This document covers deep learning variants of collaborative filtering that power modern recommendation systems.

---

## Learning Objectives

By the end of this section, you will:
- Understand deep learning architectures for collaborative filtering
- Master GMF, MLP, and NeuMF frameworks
- **See exactly when GMF reduces to MF**
- Implement Wide & Deep, DeepFM, and DCN models
- **Understand memorization vs generalization trade-off**
- Compare architectures and know when to use each
- Apply deep CF to real-world problems

---

## From Matrix Factorization to Neural Networks

### The Limitations in Detail

**Standard MF**:
$$\hat{r}_{ui} = \mathbf{p}_u^T \mathbf{q}_i = \sum_{k=1}^K p_{uk} q_{ik}$$

**Three key limitations**:

1. **Linear interactions only**: The inner product is a linear function. Even with many factors, we can only capture linear combinations.

2. **Fixed interaction pattern**: Each factor $k$ contributes $p_{uk} \cdot q_{ik}$ — always multiplication, never "if-then" logic.

3. **No feature engineering**: Can't easily incorporate side information (user age, item category, time of day).

---

### A Concrete Non-Linear Pattern MF Can't Capture

*Let me show you mathematically why MF fails on the "AND" pattern.*

**The true preference function** (what we want to model):
$$f(x_1, x_2) = \begin{cases} 1 & \text{if } x_1 > 0.5 \text{ AND } x_2 > 0.5 \\ 0 & \text{otherwise} \end{cases}$$

**What MF can express**:
$$\hat{f}(x_1, x_2) = w_1 x_1 + w_2 x_2$$

No matter what weights $w_1, w_2$ we choose, we can't capture the "AND" relationship!

| $x_1$ | $x_2$ | True $f$ | Best linear $\hat{f}$ | Error |
|-------|-------|----------|----------------------|-------|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0.5 | 0.5 |
| 0 | 1 | 0 | 0.5 | 0.5 |
| 1 | 1 | 1 | 1.0 | 0 |

**With a neural network** (even one hidden layer):
$$\hat{f}(x_1, x_2) = \sigma(w \cdot \text{ReLU}(W[x_1, x_2]^T + b))$$

This CAN learn the AND pattern!

---

## 1. Generalized Matrix Factorization (GMF)

### The Key Insight: Weighted Element-wise Product

**GMF idea**: Instead of summing all factor contributions equally, let the model **learn which factors matter more**.

$$\mathbf{h} = \mathbf{p}_u \odot \mathbf{q}_i$$
$$\hat{y}_{ui} = a(\mathbf{h}^T \mathbf{w})$$

where:
- $\odot$ = element-wise product
- $\mathbf{w}$ = learned weight vector (one weight per factor)
- $a$ = activation (sigmoid for implicit feedback)

---

### Step-by-Step: How GMF Generalizes MF

*Let me show you the exact connection.*

**Standard MF**:
$$\hat{r}_{ui} = \mathbf{p}_u^T \mathbf{q}_i = \sum_{k=1}^K p_{uk} \cdot q_{ik}$$

Every factor contributes equally (weight = 1).

**GMF**:
$$\hat{y}_{ui} = \sum_{k=1}^K w_k \cdot (p_{uk} \cdot q_{ik}) = \sum_{k=1}^K w_k \cdot h_k$$

Each factor $k$ has a learnable importance weight $w_k$.

**When does GMF = MF?**

If $\mathbf{w} = [1, 1, ..., 1]^T$ (all ones) and $a$ is identity:
$$\hat{y}_{ui} = \sum_{k=1}^K 1 \cdot (p_{uk} \cdot q_{ik}) = \mathbf{p}_u^T \mathbf{q}_i$$

This is exactly MF!

**So GMF is strictly more expressive**: It can learn to ignore certain factors (set $w_k \approx 0$) or emphasize others (large $w_k$).

---

### Numerical Example: GMF vs MF

**Setup**: 3 users, 4 items, k=2 factors

**User embeddings** $P$:
| User | Factor 1 | Factor 2 |
|------|----------|----------|
| Alice | 0.8 | 0.3 |
| Bob | 0.2 | 0.9 |

**Item embeddings** $Q$:
| Item | Factor 1 | Factor 2 |
|------|----------|----------|
| Action Movie | 0.9 | 0.1 |
| Romance | 0.1 | 0.8 |

**MF Predictions** (equal weights):
- Alice × Action: $0.8 \times 0.9 + 0.3 \times 0.1 = 0.72 + 0.03 = 0.75$
- Alice × Romance: $0.8 \times 0.1 + 0.3 \times 0.8 = 0.08 + 0.24 = 0.32$

**GMF Predictions** with learned weights $\mathbf{w} = [2.0, 0.5]$:
- Alice × Action: $2.0 \times (0.8 \times 0.9) + 0.5 \times (0.3 \times 0.1) = 1.44 + 0.015 = 1.455$
- Alice × Romance: $2.0 \times (0.8 \times 0.1) + 0.5 \times (0.3 \times 0.8) = 0.16 + 0.12 = 0.28$

**What the weights learned**: Factor 1 is more important for predictions (weight 2.0 vs 0.5).

---

### Implementation

```python
import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64):
        super().__init__()

        # Embeddings (these are p_u and q_i)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # The key difference from MF: learnable weights for each factor
        # Output layer: h -> scalar (the w vector)
        self.fc = nn.Linear(embedding_dim, 1)

        # Initialize
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # (batch, embedding_dim) = p_u
        item_emb = self.item_embedding(item_ids)  # (batch, embedding_dim) = q_i

        # Element-wise product: h = p_u ⊙ q_i
        interaction = user_emb * item_emb  # (batch, embedding_dim)

        # Weighted sum + activation: a(h^T w)
        output = self.fc(interaction)  # (batch, 1)
        output = torch.sigmoid(output)  # (batch, 1)

        return output.squeeze()


# Example usage
model = GMF(n_users=10000, n_items=5000, embedding_dim=64)

# Sample batch: 4 user-item pairs
user_ids = torch.tensor([1, 5, 10, 20])
item_ids = torch.tensor([100, 250, 300, 450])

# Forward pass
predictions = model(user_ids, item_ids)
print(f"Predictions: {predictions}")  # Probabilities in [0, 1]
```

---

## 2. Multi-Layer Perceptron (MLP)

### The Idea: Let a Neural Network Learn the Interaction

*What if instead of defining the interaction (element-wise product), we let the network learn it?*

**MLP approach**: Concatenate embeddings, pass through hidden layers.

$$\mathbf{z}_1 = [\mathbf{p}_u, \mathbf{q}_i]$$
$$\mathbf{z}_{l+1} = a(\mathbf{W}_l^T \mathbf{z}_l + \mathbf{b}_l)$$
$$\hat{y}_{ui} = \sigma(\mathbf{h}^T \mathbf{z}_L)$$

**Why concatenation?**
- Element-wise product (GMF) assumes factors interact one-to-one
- Concatenation lets Factor 1 of user interact with ANY factor of item

---

### The Tower Structure

*Visualize what's happening:*

```
User ID: 42              Item ID: 1337
    ↓                        ↓
Embedding lookup        Embedding lookup
    ↓                        ↓
[0.3, -0.1, 0.8, ...]   [0.5, 0.2, -0.4, ...]
           ↓                ↓
           └──── Concatenate ────┘
                     ↓
            [0.3, -0.1, 0.8, ..., 0.5, 0.2, -0.4, ...]
                     ↓
                Hidden Layer 1 (256 units) + ReLU
                     ↓
                Hidden Layer 2 (128 units) + ReLU
                     ↓
                Hidden Layer 3 (64 units) + ReLU
                     ↓
                Output Layer (1 unit) + Sigmoid
                     ↓
                 Prediction: 0.87
```

---

### Implementation

```python
class MLP(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, hidden_layers=[128, 64, 32]):
        super().__init__()

        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # MLP layers
        mlp_modules = []
        input_dim = embedding_dim * 2  # Concatenated user + item

        for hidden_dim in hidden_layers:
            mlp_modules.append(nn.Linear(input_dim, hidden_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_modules)

        # Output layer
        self.fc = nn.Linear(hidden_layers[-1], 1)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, user_ids, item_ids):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # (batch, embedding_dim)
        item_emb = self.item_embedding(item_ids)  # (batch, embedding_dim)

        # Concatenate: [p_u, q_i]
        interaction = torch.cat([user_emb, item_emb], dim=-1)  # (batch, 2*embedding_dim)

        # Pass through MLP
        mlp_output = self.mlp(interaction)  # (batch, hidden_layers[-1])

        # Prediction
        output = self.fc(mlp_output)  # (batch, 1)
        output = torch.sigmoid(output)

        return output.squeeze()


# Example
model_mlp = MLP(n_users=10000, n_items=5000, embedding_dim=64, hidden_layers=[128, 64, 32])
predictions = model_mlp(user_ids, item_ids)
print(f"MLP Predictions: {predictions}")
```

---

## 3. Neural Matrix Factorization (NeuMF)

### The Winning Combination

**Paper**: He et al., "Neural Collaborative Filtering" (WWW 2017)

*Key insight*: GMF and MLP have complementary strengths:
- **GMF**: Good at modeling **linear** interactions (when MF patterns work)
- **MLP**: Good at modeling **non-linear** interactions (complex patterns)

*Why not combine them?*

---

### Architecture Diagram

```
User u                           Item i
  ↓                                ↓
┌─────────────────────────────────────────────────────┐
│          Embedding Layer (separate for each path)   │
└─────────────────────────────────────────────────────┘
  ↓                                              ↓
GMF Embeddings                          MLP Embeddings
(p_u^GMF, q_i^GMF)                     (p_u^MLP, q_i^MLP)
  ↓                                              ↓
Element-wise                               Concatenate
  Product                                   [p_u, q_i]
  ↓                                              ↓
  h_GMF                                        MLP
(embedding_dim)                              Layers
  ↓                                              ↓
  │                                          h_MLP
  │                                     (last_hidden)
  │                                              │
  └──────────────── Concatenate ─────────────────┘
                         ↓
                  [h_GMF, h_MLP]
                         ↓
                   Final Layer
                         ↓
                   Prediction
```

**Key design**: Separate embeddings for GMF and MLP paths. This gives each path freedom to learn different representations!

---

### The Math

$$\mathbf{h}_{GMF} = \mathbf{p}_u^{GMF} \odot \mathbf{q}_i^{GMF}$$

$$\mathbf{h}_{MLP} = \text{MLP}([\mathbf{p}_u^{MLP}, \mathbf{q}_i^{MLP}])$$

$$\hat{y}_{ui} = \sigma(\mathbf{h}^T [\mathbf{h}_{GMF}, \mathbf{h}_{MLP}])$$

---

### Implementation

```python
class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, mlp_hidden_layers=[128, 64, 32]):
        super().__init__()

        # GMF embeddings (separate from MLP!)
        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)

        # MLP embeddings (separate from GMF!)
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)

        # MLP layers
        mlp_modules = []
        input_dim = embedding_dim * 2

        for hidden_dim in mlp_hidden_layers:
            mlp_modules.append(nn.Linear(input_dim, hidden_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_modules)

        # Final layer: combines GMF output + MLP output
        final_dim = embedding_dim + mlp_hidden_layers[-1]
        self.fc = nn.Linear(final_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, user_ids, item_ids):
        # ===== GMF path =====
        user_emb_gmf = self.user_embedding_gmf(user_ids)
        item_emb_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_emb_gmf * item_emb_gmf  # Element-wise product

        # ===== MLP path =====
        user_emb_mlp = self.user_embedding_mlp(user_ids)
        item_emb_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # ===== Combine =====
        combined = torch.cat([gmf_output, mlp_output], dim=-1)

        # ===== Final prediction =====
        output = self.fc(combined)
        output = torch.sigmoid(output)

        return output.squeeze()


# Example
model_neumf = NeuMF(n_users=10000, n_items=5000, embedding_dim=64, mlp_hidden_layers=[128, 64, 32])
predictions = model_neumf(user_ids, item_ids)
print(f"NeuMF Predictions: {predictions}")
```

**Result**: NeuMF outperforms both GMF and MLP alone (reported in paper: ~5% improvement on HR@10).

---

## 4. Wide & Deep Learning

### The Core Insight: Memorization vs Generalization

**Paper**: Cheng et al., "Wide & Deep Learning for Recommender Systems" (2016) - Google Play

*Before I show you the architecture, let me explain the key insight with an example.*

**Scenario**: Recommending apps on Google Play

**User profile**:
- Recently installed: "Uber", "Lyft"
- Query: "food delivery"

**Memorization** (Wide):
- "Users who installed Uber and searched 'food delivery' installed DoorDash"
- Specific pattern from data → high precision
- But fails for new patterns!

**Generalization** (Deep):
- "Uber/Lyft are transportation apps → user likes on-demand services"
- "Food delivery is also on-demand → recommend GrubHub, Postmates too"
- Abstracts to general patterns → better recall
- But might miss specific correlations!

**Solution**: Combine both!

---

### Wide vs Deep: Concrete Example

**User**: Installed ["Uber", "Lyft"], searched "food delivery"

**Wide Component** (Cross-product features):
```
Feature: "installed_Uber AND installed_Lyft AND query_food_delivery"
→ Memorizes: Users with EXACTLY this pattern prefer DoorDash

Prediction: 0.95 for DoorDash (seen this exact pattern before)
Prediction: 0.10 for GrubHub (never seen this exact pattern)
```

**Deep Component** (Learned embeddings):
```
Uber embedding: [0.8, 0.3, -0.1, ...]  (on-demand, transportation)
Lyft embedding: [0.75, 0.35, -0.05, ...]  (similar!)
"food delivery" embedding: [0.7, 0.5, 0.2, ...]

→ Generalizes: "On-demand services" cluster together

Prediction: 0.70 for DoorDash
Prediction: 0.65 for GrubHub
Prediction: 0.60 for Postmates
```

**Combined**:
- DoorDash: 0.95 × 0.5 + 0.70 × 0.5 = **0.825** (High! Both agree)
- GrubHub: 0.10 × 0.5 + 0.65 × 0.5 = **0.375** (Lower, but still reasonable)

---

### Architecture

$$\hat{y} = \sigma(\mathbf{w}_{\text{wide}}^T \mathbf{x}_{\text{wide}} + \mathbf{w}_{\text{deep}}^T a^{(l_f)} + b)$$

```
                Input Features
                     ↓
    ┌────────────────┴───────────────┐
    ↓                                ↓
WIDE COMPONENT                  DEEP COMPONENT
(Sparse features)               (Dense features)
    ↓                                ↓
Linear Model:                   Embedding Layer
w^T × x                              ↓
    ↓                           Hidden Layers
    ↓                                ↓
    └────────── + (Add) ─────────────┘
                    ↓
                 Sigmoid
                    ↓
               Prediction
```

---

### What Can Go Wrong with Wide & Deep?

**Failure Mode 1: Over-memorization**
- Wide component dominates → poor generalization
- Symptoms: Great training accuracy, poor on new patterns
- Solution: Reduce wide feature complexity, increase deep capacity

**Failure Mode 2: Feature Engineering Hell**
- Wide component needs cross-product features
- Someone has to decide which crosses to include!
- Solution: Use DeepFM (learns crosses automatically)

**Failure Mode 3: Unbalanced Training**
- Wide and deep train at different speeds
- Solution: Careful learning rate tuning, separate optimizers

---

### Implementation

```python
class WideAndDeep(nn.Module):
    def __init__(self, wide_dim, deep_dims, hidden_layers=[128, 64, 32]):
        super().__init__()

        # Wide component (linear model on cross-product features)
        self.wide = nn.Linear(wide_dim, 1)

        # Deep component (MLP on embeddings)
        deep_modules = []
        input_dim = deep_dims

        for hidden_dim in hidden_layers:
            deep_modules.append(nn.Linear(input_dim, hidden_dim))
            deep_modules.append(nn.ReLU())
            deep_modules.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        self.deep = nn.Sequential(*deep_modules)
        self.deep_fc = nn.Linear(hidden_layers[-1], 1)

    def forward(self, wide_features, deep_features):
        # Wide path: simple linear
        wide_output = self.wide(wide_features)  # (batch, 1)

        # Deep path: neural network
        deep_output = self.deep(deep_features)  # (batch, hidden_layers[-1])
        deep_output = self.deep_fc(deep_output)  # (batch, 1)

        # Combine (sum, then sigmoid)
        output = wide_output + deep_output  # (batch, 1)
        output = torch.sigmoid(output)

        return output.squeeze()


# Example
wide_dim = 100   # Cross-product features (e.g., "installed_Uber AND query_food")
deep_dim = 50    # Dense features (embeddings)

model_wd = WideAndDeep(wide_dim=wide_dim, deep_dims=deep_dim, hidden_layers=[128, 64, 32])

# Sample features
wide_feat = torch.randn(32, wide_dim)  # Batch of 32, sparse features
deep_feat = torch.randn(32, deep_dim)  # Batch of 32, dense embeddings

predictions = model_wd(wide_feat, deep_feat)
print(f"Wide & Deep Predictions shape: {predictions.shape}")
```

**Business impact**: Google reported Wide & Deep improved app acquisition by 3.9% (A/B test on Google Play).

---

## 5. DeepFM: Learning Feature Crosses Automatically

### The Problem DeepFM Solves

*Wide & Deep requires manual feature engineering for the wide component. What if we could learn the crosses automatically?*

**DeepFM insight**: Use Factorization Machines (FM) to **automatically** learn 2nd-order feature interactions!

---

### The FM Component: Efficient 2nd-Order Interactions

**Naive approach** to feature crosses:
$$y = w_0 + \sum_i w_i x_i + \sum_i \sum_{j>i} w_{ij} x_i x_j$$

**Problem**: $O(n^2)$ parameters for the $w_{ij}$ terms!

**FM trick**: Factor the interaction weights
$$w_{ij} = \langle \mathbf{v}_i, \mathbf{v}_j \rangle$$

where $\mathbf{v}_i \in \mathbb{R}^k$ is a latent vector for feature $i$.

**Now we only need** $O(nk)$ parameters!

---

### The FM Computation Trick: O(kn) Instead of O(kn²)

*This is a beautiful mathematical trick. Let me derive it step by step.*

**We want to compute**:
$$\sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

**The trick**: Use the identity
$$\frac{1}{2}\left[(\sum_i \mathbf{v}_i x_i)^2 - \sum_i (\mathbf{v}_i x_i)^2\right] = \sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

*Why does this work?* Let me expand:

$$(\sum_i \mathbf{v}_i x_i)^2 = \sum_i \sum_j \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

This includes:
- $i = j$ terms: $\sum_i \langle \mathbf{v}_i, \mathbf{v}_i \rangle x_i^2 = \sum_i (\mathbf{v}_i x_i)^2$
- $i < j$ terms: $\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$
- $i > j$ terms: Same as $i < j$ (symmetric)

So:
$$(\sum_i \mathbf{v}_i x_i)^2 = \sum_i (\mathbf{v}_i x_i)^2 + 2\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

Rearranging:
$$\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2}\left[(\sum_i \mathbf{v}_i x_i)^2 - \sum_i (\mathbf{v}_i x_i)^2\right]$$

**Complexity**:
- Naive: $O(kn^2)$ (loop over all pairs)
- With trick: $O(kn)$ (sum once, square, subtract)

---

### DeepFM Architecture

```
        Sparse Input Features
               ↓
    ┌──────────┴──────────┐
    ↓                     ↓
FM Component          Deep Component
    ↓                     ↓
┌───────────────┐    ┌───────────────┐
│ Linear: Σw_i  │    │  Embeddings   │
│    +          │    │      ↓        │
│ 2nd-order:    │    │    MLP        │
│ Σ<v_i,v_j>    │    │      ↓        │
└───────────────┘    └───────────────┘
        ↓                     ↓
        └───────── + ─────────┘
                  ↓
               Sigmoid
                  ↓
             Prediction
```

**Key advantage over Wide & Deep**: The embeddings are **shared** between FM and Deep components!

---

### Implementation

```python
class DeepFM(nn.Module):
    def __init__(self, n_features, embedding_dim=10, hidden_layers=[128, 64]):
        super().__init__()

        # Embeddings (SHARED between FM and Deep)
        self.feature_embeddings = nn.Embedding(n_features, embedding_dim)

        # FM: Linear part (1st order)
        self.fm_linear = nn.Embedding(n_features, 1)

        # FM: 2nd-order part uses the shared embeddings

        # Deep: MLP
        # Input: all field embeddings concatenated
        mlp_input_dim = n_features * embedding_dim
        mlp_modules = []

        input_dim = mlp_input_dim
        for hidden_dim in hidden_layers:
            mlp_modules.append(nn.Linear(input_dim, hidden_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        self.deep = nn.Sequential(*mlp_modules)
        self.deep_fc = nn.Linear(hidden_layers[-1], 1)

    def forward(self, feature_indices):
        """
        feature_indices: (batch, n_fields) - indices of non-zero features
        """
        batch_size = feature_indices.size(0)
        n_fields = feature_indices.size(1)

        # Get embeddings: (batch, n_fields, emb_dim)
        embeddings = self.feature_embeddings(feature_indices)

        # ===== FM Component =====

        # 1st order: sum of linear terms
        fm_linear = self.fm_linear(feature_indices).sum(dim=1)  # (batch, 1)

        # 2nd order: using the efficient trick
        # Sum of embeddings: (batch, emb_dim)
        sum_of_emb = embeddings.sum(dim=1)
        # Square of sum: (batch, emb_dim)
        sum_of_emb_squared = sum_of_emb ** 2

        # Sum of squares: (batch, emb_dim)
        sum_of_emb_sq = (embeddings ** 2).sum(dim=1)

        # FM 2nd order: 0.5 * (square_of_sum - sum_of_squares)
        fm_2nd_order = 0.5 * (sum_of_emb_squared - sum_of_emb_sq).sum(dim=1, keepdim=True)  # (batch, 1)

        # ===== Deep Component =====

        # Flatten embeddings for MLP input
        deep_input = embeddings.view(batch_size, -1)  # (batch, n_fields * emb_dim)
        deep_output = self.deep(deep_input)
        deep_output = self.deep_fc(deep_output)  # (batch, 1)

        # ===== Combine =====
        output = fm_linear + fm_2nd_order + deep_output
        output = torch.sigmoid(output)

        return output.squeeze()


# Example
n_fields = 10  # Number of feature fields (e.g., user_id, item_id, category, ...)
n_features = 1000  # Total vocabulary size across all fields

model_deepfm = DeepFM(n_features=n_features, embedding_dim=10, hidden_layers=[128, 64])
feature_indices = torch.randint(0, n_features, (32, n_fields))  # 32 samples
predictions = model_deepfm(feature_indices)
print(f"DeepFM Predictions: {predictions.shape}")
```

---

### What Can Go Wrong with DeepFM?

**Failure Mode 1: Embedding Dimension Mismatch**
- Same embeddings used for FM (needs small $k$) and Deep (needs large $k$)
- Solution: Use different embedding dimensions, or increase FM dimension

**Failure Mode 2: Feature Field Explosion**
- Too many feature fields → huge MLP input
- Solution: Field-weighted pooling, attention over fields

**Failure Mode 3: Overfitting on Small Data**
- FM 2nd-order terms scale with $n^2$ features
- Solution: Strong regularization, dropout

---

## 6. Deep & Cross Network (DCN)

### The Innovation: Explicit Feature Crosses

**Paper**: Wang et al., "Deep & Cross Network for Ad Click Predictions" (2017) - Google

**Key insight**: Instead of learning arbitrary non-linear functions, explicitly model **feature crosses** at each layer.

**Cross layer**:
$$\mathbf{x}_{l+1} = \mathbf{x}_0 \mathbf{x}_l^T \mathbf{w}_l + \mathbf{b}_l + \mathbf{x}_l$$

*What does this mean?*
- $\mathbf{x}_0$: Original input features
- $\mathbf{x}_l$: Features at layer $l$
- $\mathbf{x}_0 \mathbf{x}_l^T$: Outer product (creates all pairwise crosses)
- $\mathbf{w}_l$: Learns which crosses are important

**After $L$ cross layers**: Model captures up to $(L+1)$-order feature interactions!

---

### Summary: Comparison of Architectures

| Model | Key Idea | Pros | Cons | Use Case |
|-------|----------|------|------|----------|
| **GMF** | Element-wise product | Simple, interpretable | Limited expressiveness | Baseline, understanding |
| **MLP** | Concatenate + MLP | Learns non-linear | Needs more data | General |
| **NeuMF** | GMF + MLP | Best of both | More complex | Implicit feedback |
| **Wide & Deep** | Memorize + generalize | Good for sparse + dense | Needs feature eng | App stores |
| **DeepFM** | FM + Deep | Auto feature crosses | Computationally expensive | CTR prediction |
| **DCN** | Explicit crosses | Efficient bounded-degree | Complex architecture | Ad systems |

---

## When to Use Which?

*Let me give you a decision framework.*

**Decision tree**:

```
Start here
    │
    ▼
Do you have only collaborative signals (user-item IDs)?
    ├── YES → Use NeuMF (or GMF as baseline)
    │
    └── NO → Do you have many categorical features?
                ├── YES → Is manual feature engineering feasible?
                │           ├── YES → Wide & Deep
                │           └── NO → DeepFM
                │
                └── NO → Use basic MLP or NeuMF
```

**Practical recommendations**:
1. **Always start simple**: GMF or MLP as baseline
2. **Implicit feedback + IDs only**: NeuMF
3. **Rich categorical features + no time for feature eng**: DeepFM
4. **Production with established feature eng pipeline**: Wide & Deep
5. **Need explicit high-order interactions**: DCN

---

## Summary

**Key Takeaways**:

1. **MF limitation**: Can only capture linear interactions
2. **GMF**: MF with learnable factor weights (when $\mathbf{w}=[1,1,...,1]$, equals MF)
3. **MLP**: Concatenate embeddings, learn arbitrary non-linear functions
4. **NeuMF**: Combines GMF (linear) + MLP (non-linear) paths
5. **Wide & Deep**: Memorization (wide) + generalization (deep)
6. **DeepFM**: Automatic 2nd-order feature crosses via FM trick
7. **DCN**: Explicit bounded-degree feature crosses

**The key insight**: Different architectures capture different types of patterns. Choose based on your data and requirements!

**Next**: Training neural CF models (loss functions, negative sampling, optimization).

---

## References

1. **He, X., et al. (2017)**. "Neural Collaborative Filtering". *WWW*.
   - **NCF, GMF, MLP, NeuMF** foundations

2. **Cheng, H.-T., et al. (2016)**. "Wide & Deep Learning for Recommender Systems". *RecSys*.
   - Google's Wide & Deep architecture

3. **Guo, H., et al. (2017)**. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction". *IJCAI*.
   - DeepFM model, FM computational trick

4. **Wang, R., et al. (2017)**. "Deep & Cross Network for Ad Click Predictions". *ADKDD*.
   - DCN architecture

5. **Covington, P., Adams, J., & Sargin, E. (2016)**. "Deep Neural Networks for YouTube Recommendations". *RecSys*.
   - YouTube's deep learning system
