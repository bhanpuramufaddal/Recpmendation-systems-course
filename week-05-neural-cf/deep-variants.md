# Week 5: Neural Collaborative Filtering - Deep Variants

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
- Implement Wide & Deep, DeepFM, and DCN models
- Compare architectures and choose the right one
- Apply deep CF to real-world problems

---

## From Matrix Factorization to Neural Networks

### Limitations of Matrix Factorization

**Standard MF**:
$$\hat{r}_{ui} = \mathbf{p}_u^T \mathbf{q}_i = \sum_{k=1}^K p_{uk} q_{ik}$$

**Problems**:
1. **Linear interactions**: Inner product assumes linear relationship
2. **Fixed capacity**: Embedding dimension $K$ limits expressiveness
3. **No side features**: Cannot incorporate user/item metadata

**Example where MF fails**:
```
User likes:   [Action, Sci-Fi, Nolan]
Item 1:       [Action, Sci-Fi, Nolan] → High score ✓
Item 2:       [Action, Drama, Nolan]  → Medium score (should be high!)
Item 3:       [Comedy, Romance, Other] → Low score ✓

Issue: MF can't capture "Must have Nolan" preference (non-linear).
```

---

## Neural Collaborative Filtering (NCF) Framework

### The NCF Paradigm

**Paper**: He et al., "Neural Collaborative Filtering" (WWW 2017)

**Key idea**: Replace inner product with neural network.

**General framework**:
$$\hat{y}_{ui} = f(\mathbf{p}_u, \mathbf{q}_i | \mathbf{p}_u, \mathbf{q}_i, \Theta)$$

where $f$ is a neural network with parameters $\Theta$.

**Architecture**:
```
User u              Item i
   ↓                  ↓
User Embedding    Item Embedding
   ↓                  ↓
      Neural Network
           ↓
      Prediction
```

---

## 1. Generalized Matrix Factorization (GMF)

### Architecture

**Idea**: Element-wise product instead of inner product.

$$\mathbf{h} = \mathbf{p}_u \odot \mathbf{q}_i$$
$$\hat{y}_{ui} = a(\mathbf{h}^T \mathbf{w})$$

where:
- $\odot$ = element-wise product
- $\mathbf{w}$ = learned weight vector
- $a$ = activation (sigmoid for implicit feedback)

**Difference from MF**: Allows different weighting for each latent dimension.

---

### Implementation

```python
import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64):
        super().__init__()

        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Output layer
        self.fc = nn.Linear(embedding_dim, 1)

        # Initialize
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # (batch, embedding_dim)
        item_emb = self.item_embedding(item_ids)  # (batch, embedding_dim)

        # Element-wise product
        interaction = user_emb * item_emb  # (batch, embedding_dim)

        # Prediction
        output = self.fc(interaction)  # (batch, 1)
        output = torch.sigmoid(output)  # (batch, 1)

        return output.squeeze()


# Example usage
model = GMF(n_users=10000, n_items=5000, embedding_dim=64)

# Sample batch
user_ids = torch.tensor([1, 5, 10, 20])
item_ids = torch.tensor([100, 250, 300, 450])

# Forward pass
predictions = model(user_ids, item_ids)
print(f"Predictions: {predictions}")  # Probabilities in [0, 1]
```

---

## 2. Multi-Layer Perceptron (MLP)

### Architecture

**Idea**: Concatenate embeddings, pass through MLP to learn complex interactions.

$$\mathbf{z}_1 = [\mathbf{p}_u, \mathbf{q}_i]$$
$$\mathbf{z}_{l+1} = a(\mathbf{W}_l^T \mathbf{z}_l + \mathbf{b}_l)$$
$$\hat{y}_{ui} = \sigma(\mathbf{h}^T \mathbf{z}_L)$$

where:
- $\mathbf{z}_1$ = concatenation of user and item embeddings
- $\mathbf{z}_l$ = hidden layer $l$
- $L$ = number of layers

**Advantage**: Can learn non-linear interactions.

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

        # Concatenate
        interaction = torch.cat([user_emb, item_emb], dim=-1)  # (batch, 2*embedding_dim)

        # MLP
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

### Architecture

**Paper**: He et al., "Neural Collaborative Filtering" (WWW 2017)

**Idea**: Combine GMF and MLP paths.

**Two paths**:
1. **GMF path**: Element-wise product (captures linear interactions)
2. **MLP path**: Multi-layer perceptron (captures non-linear interactions)

**Final prediction**: Concatenate outputs, pass through final layer.

$$\hat{y}_{ui} = \sigma(\mathbf{h}^T [\text{GMF}(\mathbf{p}_u, \mathbf{q}_i), \text{MLP}(\mathbf{p}_u, \mathbf{q}_i)])$$

**Architecture diagram**:
```
User Embedding (GMF)   User Embedding (MLP)
       ↓                        ↓
Item Embedding (GMF)   Item Embedding (MLP)
       ↓                        ↓
Element-wise Product      Concatenation
       ↓                        ↓
       ↓                       MLP
       ↓                        ↓
       └────── Concatenate ─────┘
                 ↓
            Final Layer
                 ↓
            Prediction
```

---

### Implementation

```python
class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, mlp_hidden_layers=[128, 64, 32]):
        super().__init__()

        # GMF embeddings
        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)

        # MLP embeddings (separate)
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

        # Final layer
        final_dim = embedding_dim + mlp_hidden_layers[-1]
        self.fc = nn.Linear(final_dim, 1)

        # Initialize
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
        # GMF path
        user_emb_gmf = self.user_embedding_gmf(user_ids)
        item_emb_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_emb_gmf * item_emb_gmf  # Element-wise product

        # MLP path
        user_emb_mlp = self.user_embedding_mlp(user_ids)
        item_emb_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Concatenate GMF and MLP outputs
        combined = torch.cat([gmf_output, mlp_output], dim=-1)

        # Final prediction
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

### Architecture

**Paper**: Cheng et al., "Wide & Deep Learning for Recommender Systems" (2016) - Google

**Use case**: Google Play Store app recommendations

**Idea**: Combine **memorization** (Wide) and **generalization** (Deep).

**Two components**:
1. **Wide component**: Linear model with cross-product features (memorization)
2. **Deep component**: Deep neural network (generalization)

$$\hat{y} = \sigma(\mathbf{w}_{\text{wide}}^T \mathbf{x}_{\text{wide}} + \mathbf{w}_{\text{deep}}^T a^{(l_f)} + b)$$

where $a^{(l_f)}$ = final activation of deep network.

---

### Implementation

```python
class WideAndDeep(nn.Module):
    def __init__(self, wide_dim, deep_dims, hidden_layers=[128, 64, 32]):
        super().__init__()

        # Wide component (linear)
        self.wide = nn.Linear(wide_dim, 1)

        # Deep component (MLP)
        deep_modules = []
        input_dim = deep_dims

        for hidden_dim in hidden_layers:
            deep_modules.append(nn.Linear(input_dim, hidden_dim))
            deep_modules.append(nn.ReLU())
            deep_modules.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        self.deep = nn.Sequential(*deep_modules)
        self.deep_fc = nn.Linear(hidden_layers[-1], 1)

        # Final layer (combine wide and deep)
        self.fc = nn.Linear(2, 1)  # Combine wide output + deep output

    def forward(self, wide_features, deep_features):
        # Wide path
        wide_output = self.wide(wide_features)  # (batch, 1)

        # Deep path
        deep_output = self.deep(deep_features)  # (batch, hidden_layers[-1])
        deep_output = self.deep_fc(deep_output)  # (batch, 1)

        # Combine
        combined = torch.cat([wide_output, deep_output], dim=-1)  # (batch, 2)
        output = self.fc(combined)  # (batch, 1)
        output = torch.sigmoid(output)

        return output.squeeze()


# Example
wide_dim = 100  # Cross-product features
deep_dim = 50   # Dense features (user/item embeddings)

model_wd = WideAndDeep(wide_dim=wide_dim, deep_dims=deep_dim, hidden_layers=[128, 64, 32])

# Sample features
wide_feat = torch.randn(32, wide_dim)  # Batch of 32
deep_feat = torch.randn(32, deep_dim)

predictions = model_wd(wide_feat, deep_feat)
print(f"Wide & Deep Predictions shape: {predictions.shape}")
```

**Business impact**: Google reported Wide & Deep improved app acquisition by 3.9% (A/B test on Google Play).

---

## 5. DeepFM

### Architecture

**Paper**: Guo et al., "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction" (IJCAI 2017)

**Use case**: Click-through rate prediction (Huawei app store)

**Idea**: Combine Factorization Machines (FM) and Deep Neural Networks.

**Two components**:
1. **FM component**: Captures 2nd-order feature interactions
2. **Deep component**: Captures high-order interactions

**FM component** (2nd-order interactions):
$$y_{\text{FM}} = \langle \mathbf{w}, \mathbf{x} \rangle + \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

**Deep component**: Standard MLP

**Final**: $\hat{y} = \sigma(y_{\text{FM}} + y_{\text{Deep}})$

---

### Implementation (Simplified)

```python
class DeepFM(nn.Module):
    def __init__(self, n_features, embedding_dim=10, hidden_layers=[128, 64]):
        super().__init__()

        # Embeddings (shared between FM and Deep)
        self.feature_embeddings = nn.Embedding(n_features, embedding_dim)

        # FM: Linear part
        self.fm_linear = nn.Embedding(n_features, 1)

        # Deep: MLP
        mlp_input_dim = n_features * embedding_dim
        mlp_modules = []

        for hidden_dim in hidden_layers:
            mlp_modules.append(nn.Linear(mlp_input_dim, hidden_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2))
            mlp_input_dim = hidden_dim

        self.deep = nn.Sequential(*mlp_modules)
        self.deep_fc = nn.Linear(hidden_layers[-1], 1)

    def forward(self, feature_indices):
        """
        feature_indices: (batch, n_fields) - indices of features
        """
        # Embeddings
        embeddings = self.feature_embeddings(feature_indices)  # (batch, n_fields, emb_dim)

        # FM: Linear part
        fm_linear = self.fm_linear(feature_indices).sum(dim=1)  # (batch, 1)

        # FM: 2nd-order interactions
        sum_of_square = torch.pow(embeddings.sum(dim=1), 2)  # (batch, emb_dim)
        square_of_sum = torch.pow(embeddings, 2).sum(dim=1)  # (batch, emb_dim)
        fm_interactions = 0.5 * (sum_of_square - square_of_sum).sum(dim=1, keepdim=True)  # (batch, 1)

        # Deep: MLP
        deep_input = embeddings.view(embeddings.size(0), -1)  # (batch, n_fields * emb_dim)
        deep_output = self.deep(deep_input)
        deep_output = self.deep_fc(deep_output)  # (batch, 1)

        # Combine
        output = fm_linear + fm_interactions + deep_output
        output = torch.sigmoid(output)

        return output.squeeze()


# Example
model_deepfm = DeepFM(n_features=1000, embedding_dim=10, hidden_layers=[128, 64])
feature_indices = torch.randint(0, 1000, (32, 10))  # 32 samples, 10 feature fields
predictions = model_deepfm(feature_indices)
print(f"DeepFM Predictions: {predictions.shape}")
```

---

## 6. Deep & Cross Network (DCN)

### Architecture

**Paper**: Wang et al., "Deep & Cross Network for Ad Click Predictions" (ADKDD 2017) - Google

**Idea**: Explicitly model feature crosses at each layer.

**Cross Network**: Learns bounded-degree feature interactions efficiently.

**Cross layer**:
$$\mathbf{x}_{l+1} = \mathbf{x}_0 \mathbf{x}_l^T \mathbf{w}_l + \mathbf{b}_l + \mathbf{x}_l$$

**Deep Network**: Standard MLP

**Final**: Combine cross and deep outputs.

---

### Implementation (Simplified)

```python
class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, x):
        """
        x0: (batch, input_dim) - initial input
        x: (batch, input_dim) - current layer input
        """
        # x0 * x^T * w + b + x
        xw = torch.sum(x * self.weight, dim=1, keepdim=True)  # (batch, 1)
        cross = x0 * xw + self.bias + x  # (batch, input_dim)
        return cross


class DCN(nn.Module):
    def __init__(self, input_dim, num_cross_layers=3, deep_layers=[128, 64]):
        super().__init__()

        # Cross network
        self.cross_layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(num_cross_layers)
        ])

        # Deep network
        deep_modules = []
        for hidden_dim in deep_layers:
            deep_modules.append(nn.Linear(input_dim, hidden_dim))
            deep_modules.append(nn.ReLU())
            deep_modules.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        self.deep = nn.Sequential(*deep_modules)

        # Final layer
        final_dim = input_dim + num_cross_layers * input_dim  # Simplified
        self.fc = nn.Linear(len(deep_layers[-1:]) + input_dim, 1)

    def forward(self, x):
        # Cross network
        x0 = x
        x_cross = x
        for cross_layer in self.cross_layers:
            x_cross = cross_layer(x0, x_cross)

        # Deep network
        x_deep = self.deep(x)

        # Combine
        combined = torch.cat([x_cross, x_deep], dim=-1)
        output = self.fc(combined)
        output = torch.sigmoid(output)

        return output.squeeze()
```

---

## Comparison of Architectures

| Model | Key Idea | Pros | Cons | Use Case |
|-------|----------|------|------|----------|
| **GMF** | Element-wise product | Simple, interpretable | Limited expressiveness | Baseline |
| **MLP** | Multi-layer perceptron | Learns non-linear interactions | Needs more data | General |
| **NeuMF** | GMF + MLP | Best of both | More complex | Implicit feedback |
| **Wide & Deep** | Memorization + generalization | Good for sparse + dense features | Requires feature engineering | Google Play |
| **DeepFM** | FM + Deep | Automatic feature crosses | Computationally expensive | CTR prediction |
| **DCN** | Explicit feature crossing | Efficient crosses | Complex architecture | Ad systems |

---

## When to Use Which?

**Decision tree**:

1. **Simple baseline needed?** → GMF
2. **Implicit feedback (clicks, views)?** → NeuMF
3. **Sparse features + need memorization?** → Wide & Deep
4. **CTR prediction with categorical features?** → DeepFM
5. **Need explicit feature crosses?** → DCN

**General recommendation**: Start with **NeuMF** (good balance of performance and complexity).

---

## Summary

**Key Takeaways**:
1. **NCF framework**: Replace inner product with neural networks
2. **GMF**: Element-wise product (generalized MF)
3. **MLP**: Multi-layer perceptron (non-linear)
4. **NeuMF**: Combines GMF + MLP (best performance)
5. **Wide & Deep**: Memorization + generalization (Google)
6. **DeepFM**: FM + Deep (automatic feature crosses)
7. **DCN**: Explicit feature crossing (efficient)

**Best Practices**:
- Start simple (GMF or MLP)
- Use NeuMF for implicit feedback
- Use Wide & Deep or DeepFM for categorical features
- Optimize hyperparameters (embedding dim, layers, dropout)

**Next**: Training neural CF models (loss functions, negative sampling, optimization).

---

## References

1. **He, X., et al. (2017)**. "Neural Collaborative Filtering". *WWW*.
   - **NCF, GMF, MLP, NeuMF** foundations

2. **Cheng, H.-T., et al. (2016)**. "Wide & Deep Learning for Recommender Systems". *RecSys*.
   - Google's Wide & Deep architecture

3. **Guo, H., et al. (2017)**. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction". *IJCAI*.
   - DeepFM model

4. **Wang, R., et al. (2017)**. "Deep & Cross Network for Ad Click Predictions". *ADKDD*.
   - DCN architecture

5. **Covington, P., Adams, J., & Sargin, E. (2016)**. "Deep Neural Networks for YouTube Recommendations". *RecSys*.
   - YouTube's deep learning system
