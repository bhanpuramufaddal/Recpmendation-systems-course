# Week 5: From Matrix Factorization to Neural Networks

## Overview

This document provides the **conceptual bridge** from classical matrix factorization to deep learning-based recommendation systems. We'll explore why neural networks are powerful for recommendations, how MF can be viewed as a shallow neural network, and what additional capabilities deep learning brings.

**Key question**: If matrix factorization works well, why do we need neural networks?

---

## Learning Objectives

By the end of this section, you will:
- Understand MF as a shallow neural network
- Recognize the limitations of linear interactions
- Appreciate the expressive power of deep models
- Know when to use MF vs. neural approaches
- Be prepared for advanced models (NCF, AutoRec, VAE-CF)

---

## Matrix Factorization: A Neural Perspective

### Standard MF Formula

$$\hat{r}_{ui} = \mathbf{u}_u^T \mathbf{v}_i = \sum_{f=1}^k u_{uf} \cdot v_{fi}$$

where:
- $\mathbf{u}_u \in \mathbb{R}^k$: User embedding
- $\mathbf{v}_i \in \mathbb{R}^k$: Item embedding
- $k$: Embedding dimension

---

### MF as a Neural Network

**Architecture**:

```
Input Layer:    [user_id]  [item_id]
                    ↓           ↓
Embedding Layer: [u_u]      [v_i]     (k-dim vectors)
                    ↓           ↓
Interaction:      dot product (u_u · v_i)
                    ↓
Output:          prediction
```

**In neural network terms**:
1. **Input**: One-hot encoded user and item IDs
2. **Embedding layer**: Lookup tables for $U$ and $V$
3. **Interaction function**: Dot product (element-wise multiply + sum)
4. **Output**: Predicted rating

**Key insight**: MF is a **one-layer neural network** with a fixed interaction function (dot product).

---

### PyTorch Implementation

```python
import torch
import torch.nn as nn

class MatrixFactorizationNN(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        # Embedding layers (lookup tables)
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)

        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        # Look up embeddings
        user_emb = self.user_embedding(user_ids)  # (batch, k)
        item_emb = self.item_embedding(item_ids)  # (batch, k)

        # Dot product interaction
        prediction = (user_emb * item_emb).sum(dim=1)  # (batch,)

        return prediction

# Usage
model = MatrixFactorizationNN(n_users=1000, n_items=5000, n_factors=50)
user_ids = torch.LongTensor([0, 1, 2])
item_ids = torch.LongTensor([10, 20, 30])
predictions = model(user_ids, item_ids)
print(predictions)  # Predicted ratings
```

**Observation**: This is identical to classical MF, just expressed in NN notation!

---

## The Limitation: Linear Interactions

### What's Wrong with Dot Product?

**Dot product**:
$$\hat{r}_{ui} = \sum_{f=1}^k u_{uf} \cdot v_{fi}$$

**Problem**: This is a **linear combination** of latent factors.

**Limitation**: Can only model simple patterns.

---

### Example: The Cold-Start Problem

**Scenario**: Predict rating for (User A, Item X)

**User A's embedding**:
```
u_A = [0.8, 0.2, -0.3]  # Likes action (f1), dislikes romance (f3)
```

**Item X's embedding**:
```
v_X = [0.9, 0.1, 0.5]   # Action movie (f1), some romance (f3)
```

**Prediction**:
$$\hat{r}_{A,X} = 0.8 \cdot 0.9 + 0.2 \cdot 0.1 + (-0.3) \cdot 0.5 = 0.72 + 0.02 - 0.15 = 0.59$$

**Issue**: What if the interaction between factors is non-linear?

Example:
- User A likes action movies **unless** they have romance
- Dot product can't capture this "action AND NOT romance" pattern
- Needs **non-linear interaction**: $f(\text{action}) \cdot g(\text{not romance})$

---

### Mathematical Limitation

**Linear model** (MF):
$$\hat{r}_{ui} = \mathbf{u}_u^T \mathbf{v}_i$$

**Can represent**:
- Additive effects: "I like action (0.8) + comedy (0.2)"
- Simple correlations

**Cannot represent**:
- Conjunctions: "I like action AND comedy together"
- Disjunctions: "I like either action OR romance"
- Complex patterns: "I like action if by director X, else prefer drama"

**Universal Approximation Theorem**: A neural network with non-linear activations can approximate any function!

---

## The Neural Advantage: Non-Linear Interactions

### Multi-Layer Perceptron (MLP)

**Idea**: Replace dot product with a learned non-linear function.

**Architecture**:

```
Input:        [user_id]  [item_id]
                 ↓           ↓
Embedding:    [u_u]      [v_i]
                 ↓           ↓
Concat:       [u_u ; v_i]  (2k-dim)
                 ↓
Hidden Layer: ReLU(W1 · [u;v] + b1)
                 ↓
Hidden Layer: ReLU(W2 · h1 + b2)
                 ↓
Output:       W3 · h2 + b3
                 ↓
            prediction
```

**Key difference**: Multiple layers with non-linear activations (ReLU, tanh, sigmoid)

---

### Why Non-Linearity Matters

**MF** (linear):
$$\hat{r}_{ui} = \mathbf{u}_u^T \mathbf{v}_i$$

**MLP** (non-linear):
$$\hat{r}_{ui} = f_L(\ldots f_2(f_1([

\mathbf{u}_u; \mathbf{v}_i])))$$

where $f_l(\mathbf{x}) = \sigma(W_l \mathbf{x} + \mathbf{b}_l)$ and $\sigma$ is ReLU or other non-linearity.

**Expressiveness**:
- **MF**: Hyperplane decision boundary
- **MLP**: Arbitrary decision boundary (complex shapes)

**Example**:
- MF: "If action > 0.5 → like"
- MLP: "If (action > 0.5 AND comedy > 0.3) OR (drama > 0.7 AND director == Nolan) → like"

---

### Empirical Evidence

**He et al., 2017** (Neural Collaborative Filtering paper):

**MovieLens dataset** (1M ratings):
- **MF (k=64)**: NDCG@10 = 0.6722
- **MLP (4 layers, 64→32→16→8)**: NDCG@10 = 0.6916
- **Improvement**: **2.9%**

**Even better**: Combine both!
- **NeuMF** (MF + MLP fusion): NDCG@10 = 0.7177
- **Improvement over MF**: **6.8%**

---

## Learned Similarity Functions

### Dot Product (MF)

$$\text{similarity}(\mathbf{u}_u, \mathbf{v}_i) = \mathbf{u}_u^T \mathbf{v}_i$$

**Assumptions**:
- Linear combination
- Each dimension contributes independently
- Same importance for all factor dimensions

---

### Neural Network (Learned Similarity)

$$\text{similarity}(\mathbf{u}_u, \mathbf{v}_i) = \phi(\mathbf{u}_u, \mathbf{v}_i; \Theta)$$

where $\phi$ is a neural network with parameters $\Theta$.

**Flexibility**:
- Non-linear combinations
- Adaptive weighting of dimensions
- Complex interaction patterns

**Example**:
```python
def learned_similarity(u, v):
    concat = torch.cat([u, v], dim=1)
    h1 = torch.relu(W1 @ concat + b1)
    h2 = torch.relu(W2 @ h1 + b2)
    score = W3 @ h2 + b3
    return score
```

---

## When MF is Sufficient

### MF Works Well When:

1. **Simple patterns**: User preferences are linearly separable
2. **Dense data**: Many ratings per user/item
3. **Latency-critical**: MF is faster (simple dot product)
4. **Interpretability**: Latent factors are somewhat interpretable
5. **Small datasets**: Few parameters, less overfitting

**Example**: Netflix Prize (2009)
- Matrix Factorization (ensemble of MF variants) won
- Deep learning wasn't mature yet
- Data was dense enough for MF to excel

---

## When Neural Networks Shine

### Neural Approaches Better When:

1. **Complex patterns**: Non-linear user preferences
2. **Sparse data**: NNs can learn from side information (features)
3. **Rich features**: User demographics, item attributes, context
4. **Heterogeneous data**: Text, images, audio (multi-modal)
5. **Large datasets**: Can leverage billions of interactions

**Example**: YouTube (2016)
- Billions of users, millions of videos
- Complex watching patterns (sequential, contextual)
- Deep neural networks (two-tower architecture)
- Traditional MF doesn't scale to this complexity

---

## The Best of Both Worlds: Hybrid Models

### NeuMF (Neural Matrix Factorization)

**Idea**: Combine MF (linear) + MLP (non-linear)

**Architecture**:

```
        User Embedding (GMF)    User Embedding (MLP)
                ↓                       ↓
        Item Embedding (GMF)    Item Embedding (MLP)
                ↓                       ↓
          Element-wise ×            Concatenate
                ↓                       ↓
              (MF path)              (MLP path)
                ↓                       ↓
                |                    Hidden Layers
                ↓                       ↓
                └───── Concatenate ─────┘
                            ↓
                      Output Layer
                            ↓
                       Prediction
```

**Formula**:
$$\hat{r}_{ui} = \sigma(\mathbf{h}^T [\mathbf{u}_u^{GMF} \odot \mathbf{v}_i^{GMF}; \phi_{MLP}(\mathbf{u}_u^{MLP}, \mathbf{v}_i^{MLP})])$$

where:
- $\odot$: Element-wise product (MF component)
- $\phi_{MLP}$: Multi-layer perceptron
- $\sigma$: Sigmoid activation
- Separate embeddings for GMF and MLP paths

**Result**: Captures both linear and non-linear patterns!

---

### Implementation Sketch

```python
class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=64, layers=[64, 32, 16, 8]):
        super().__init__()

        # GMF embeddings
        self.user_embedding_gmf = nn.Embedding(n_users, n_factors)
        self.item_embedding_gmf = nn.Embedding(n_items, n_factors)

        # MLP embeddings
        self.user_embedding_mlp = nn.Embedding(n_users, layers[0] // 2)
        self.item_embedding_mlp = nn.Embedding(n_items, layers[0] // 2)

        # MLP layers
        mlp_modules = []
        for i in range(len(layers) - 1):
            mlp_modules.append(nn.Linear(layers[i], layers[i+1]))
            mlp_modules.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_modules)

        # Final prediction layer
        self.output = nn.Linear(n_factors + layers[-1], 1)

    def forward(self, user_ids, item_ids):
        # GMF path
        user_gmf = self.user_embedding_gmf(user_ids)
        item_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_gmf * item_gmf  # Element-wise product

        # MLP path
        user_mlp = self.user_embedding_mlp(user_ids)
        item_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=1)
        mlp_output = self.mlp(mlp_input)

        # Concatenate and predict
        concat = torch.cat([gmf_output, mlp_output], dim=1)
        prediction = self.output(concat).squeeze()

        return prediction
```

---

## Feature Interactions

### MF: Fixed Interaction (Dot Product)

$$\text{interaction}(\mathbf{u}, \mathbf{v}) = \sum_{f=1}^k u_f \cdot v_f$$

**Problem**: All factor pairs interact the same way.

---

### Neural Networks: Learned Interactions

**Attention Mechanisms**:
- Learn which factors are important for each prediction
- Dynamic weighting based on context

**Cross-Network** (Deep & Cross Network):
- Explicit feature crossing at each layer
- Captures high-order interactions

**Example** (simplified attention):
```python
def attention_interaction(u, v):
    # Compute attention weights
    attention = softmax(u @ W @ v.T)

    # Weighted combination
    interaction = attention * (u * v)

    return interaction.sum()
```

---

## Comparison Summary

| Aspect | Matrix Factorization | Neural Networks |
|--------|---------------------|-----------------|
| **Interaction** | Dot product (fixed) | Learned (flexible) |
| **Expressiveness** | Linear combinations | Arbitrary non-linear functions |
| **Parameters** | $O(k \cdot (|U| + |I|))$ | $O(k \cdot (|U| + |I|) + L \cdot d^2)$ |
| **Training speed** | Fast (ALS or simple SGD) | Slower (deep networks) |
| **Inference speed** | Very fast (dot product) | Fast (GPU batch inference) |
| **Overfitting risk** | Lower (fewer params) | Higher (needs regularization) |
| **Data requirements** | Moderate | Large (billions preferred) |
| **Interpretability** | Somewhat interpretable factors | Black box |
| **Side information** | Difficult to incorporate | Natural (just add features) |
| **Best for** | Dense ratings, simple patterns | Sparse data, complex patterns |

---

## Evolution Timeline

```
2006: Netflix Prize begins
  └─> Basic MF dominates

2008: SVD++, Advanced MF variants
  └─> Implicit feedback, temporal dynamics

2009: Netflix Prize ends
  └─> Ensemble of MF models wins

2016: Deep Learning era begins
  └─> Neural Collaborative Filtering (NCF)
  └─> YouTube Deep Neural Networks
  └─> Autoencoders for Collaborative Filtering

2017-2024: Hybrid approaches
  └─> NeuMF, DeepFM, Wide & Deep
  └─> Transformers for Sequential Rec
  └─> Graph Neural Networks
  └─> LLMs for Recommendations
```

---

## Practical Guidelines

### Use Matrix Factorization When:
- Dataset is moderately sized (< 10M ratings)
- Patterns are relatively simple
- Need fast inference (latency < 10ms)
- Want some interpretability
- Limited computational resources

### Use Neural Networks When:
- Massive scale (billions of interactions)
- Complex user behavior (sequential, contextual)
- Rich side information available
- Multi-modal data (text, images, audio)
- Can afford computational cost

### Use Hybrid (NeuMF, Wide & Deep) When:
- Want best of both worlds
- Have large dataset + computational resources
- Need both memorization (MF) and generalization (NN)

---

## Next Steps

With this foundation, you're ready to explore:

1. **Neural Collaborative Filtering (NCF)** - ncf.md
   - Full implementation of NeuMF
   - Training strategies
   - Experimental results

2. **Deep Variants** - deep-variants.md
   - AutoRec (autoencoder-based CF)
   - VAE-CF (variational autoencoders)
   - CDAE (denoising autoencoders)

3. **Sequential Models** - Week 6
   - RNNs for recommendation (GRU4Rec)
   - Transformers (BERT4Rec, SASRec)

4. **Graph Neural Networks** - Week 7
   - LightGCN, PinSage
   - User-item bipartite graphs

---

## Summary

**Key Takeaways**:
1. **MF is a shallow neural network** with dot product interaction
2. **Neural networks add non-linearity** → more expressive
3. **Dot product is limiting** for complex patterns
4. **Hybrid models** (NeuMF) combine linear + non-linear
5. **Choice depends on** data size, pattern complexity, resources

**The shift from MF to Neural**:
- 2006-2015: MF dominated (Netflix Prize era)
- 2016-present: Neural networks dominate (deep learning era)
- Future: Hybrid approaches + foundation models (LLMs, multi-modal)

**Fundamental insight**: Recommendation is fundamentally about learning a good user-item similarity function. MF uses dot product; neural networks learn arbitrary similarity functions.

---

## References

1. **He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017)**. "Neural Collaborative Filtering". *WWW*.
   - Foundational NCF paper

2. **Cheng, H. T., et al. (2016)**. "Wide & Deep Learning for Recommender Systems". *DLRS Workshop*.
   - Google's hybrid approach

3. **Covington, P., Adams, J., & Sargin, E. (2016)**. "Deep Neural Networks for YouTube Recommendations". *RecSys*.
   - Industrial deep learning at scale

4. **Sedhain, S., et al. (2015)**. "AutoRec: Autoencoders Meet Collaborative Filtering". *WWW*.
   - Autoencoder-based approach

5. **Liang, D., et al. (2018)**. "Variational Autoencoders for Collaborative Filtering". *WWW*.
   - VAE-CF, probabilistic deep learning
