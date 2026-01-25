# Week 5: Neural Collaborative Filtering - Practice Problems

## Overview
These problems test your understanding of neural networks for recommendations, the NCF framework, autoencoders for CF, and deep learning training techniques. Focus on architecture design, negative sampling, and comparing neural vs. traditional methods.

---

## Problem 1: MF as a Neural Network
**Difficulty:** Easy
**Topics:** Matrix factorization, neural network equivalence

Standard matrix factorization predicts:
$$\hat{r}_{ui} = \mathbf{p}_u^T \mathbf{q}_i$$

**Task:** Express this as a neural network
1. Design the network architecture (input layer, embedding layers, output)
2. What is the activation function?
3. What is the loss function for rating prediction?
4. How many parameters does this network have?

**Hints:**
- Inputs: user ID and item ID (one-hot encoded)
- Embedding layers map IDs to latent vectors
- Element-wise product + sum = dot product
- Linear activation for rating prediction

**Learning Outcomes:**
- Understand MF as a shallow neural network
- Recognize limitations of linear models
- See the connection between traditional and neural CF

---

## Problem 2: NCF Architecture Design
**Difficulty:** Medium
**Topics:** Neural Collaborative Filtering, architecture design

The NeuMF architecture combines GMF (Generalized Matrix Factorization) and MLP:

```
GMF pathway: element-wise product of embeddings
MLP pathway: concatenate embeddings → MLP layers
Final: concatenate both pathways → output layer
```

**Given:**
- 1000 users, 500 items
- GMF embedding size: 32
- MLP embedding size: 64
- MLP layers: [128, 64, 32]

**Calculate:**
1. Total number of parameters in GMF pathway
2. Total number of parameters in MLP pathway
3. Total model parameters
4. Memory footprint (assume float32)

**Hints:**
- GMF: 2 embeddings (user + item)
- MLP: 2 embeddings + 3 dense layers + output layer
- Each dense layer: (input_dim × output_dim) + bias

**Learning Outcomes:**
- Understand NCF architecture
- Calculate model complexity
- Make capacity decisions

---

## Problem 3: Negative Sampling Strategies
**Difficulty:** Medium
**Topics:** Negative sampling, implicit feedback, training

For implicit feedback, you have positive interactions but no explicit negatives.

**Positive samples:** (user=1, item=5, label=1)

**Negative sampling strategies:**
1. **Uniform:** Sample random items user hasn't interacted with
2. **Popularity-based:** Sample popular items more frequently
3. **Hard negatives:** Sample items with high predicted scores but no interaction

**Questions:**
1. Why do we need negative samples for implicit feedback?
2. Compare the three strategies: pros and cons
3. How many negatives per positive should you sample?
4. What happens if you use too many negatives?

**Learning Outcomes:**
- Understand training with implicit feedback
- Choose appropriate negative sampling
- Balance positive and negative samples

---

## Problem 4: Autoencoder for Collaborative Filtering
**Difficulty:** Hard
**Topics:** AutoRec, autoencoders, reconstruction loss

AutoRec uses an autoencoder to learn user/item representations:

**Architecture:**
```
Input: user's rating vector (sparse, |I| dimensions)
Encoder: Dense layer → hidden representation
Decoder: Dense layer → reconstructed ratings
Loss: MSE on observed ratings only
```

**Given:**
- User vector: [5, ?, 3, ?, ?, 4, ?] (? = unobserved)
- Hidden size: 10
- Encoder: linear
- Decoder: linear

**Tasks:**
1. Write the forward pass equations
2. Why do we only compute loss on observed ratings?
3. How would you handle the unobserved ratings in the loss function?
4. Compare AutoRec complexity with standard MF

**Hints:**
- Mask unobserved ratings in loss computation
- Autoencoder learns a non-linear embedding
- Each user is encoded independently

**Learning Outcomes:**
- Understand autoencoder-based CF
- Handle missing data in neural networks
- Compare with MF

---

## Problem 5: Variational Autoencoder for CF (VAE-CF)
**Difficulty:** Hard
**Topics:** VAE, probabilistic models, KL divergence

VAE-CF models user preferences as a latent distribution:

**Objective:**
$$\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - KL(q(z|x) || p(z))$$

where:
- $x$ = user's interaction vector
- $z$ = latent user representation
- $q(z|x)$ = encoder (inference network)
- $p(x|z)$ = decoder (generative network)

**Questions:**
1. What does the KL term regularize?
2. Why is VAE-CF better than standard AutoRec for sparse data?
3. How do you sample from the latent distribution during training?
4. What is the "reparameterization trick"?

**Hints:**
- KL term prevents overfitting to training data
- VAE learns a probabilistic representation
- Reparameterization: $z = \mu + \sigma \odot \epsilon$, where $\epsilon \sim N(0,1)$

**Learning Outcomes:**
- Understand probabilistic deep learning
- Work with variational autoencoders
- Apply VAE to recommendations

---

## Problem 6: Training Deep Recommenders
**Difficulty:** Medium
**Topics:** Training techniques, regularization, optimization

You're training an NCF model and observe:
- Training loss decreases steadily
- Validation loss decreases, then increases after epoch 15
- Test NDCG@10 plateaus at epoch 12

**Questions:**
1. What is happening? Diagnose the problem.
2. What regularization techniques would you apply?
3. At which epoch should you stop training?
4. How would you implement early stopping?

**Regularization options:**
- Dropout (rate?)
- L2 regularization (λ?)
- Batch normalization
- Learning rate scheduling

**Learning Outcomes:**
- Recognize overfitting in deep models
- Apply regularization techniques
- Implement early stopping

---

## Problem 7: Point-wise vs. Pair-wise Loss
**Difficulty:** Medium
**Topics:** Loss functions, ranking, BPR

**Point-wise loss (binary cross-entropy):**
$$L_{point} = -\sum_{(u,i)} [y_{ui} \log \hat{y}_{ui} + (1-y_{ui}) \log(1-\hat{y}_{ui})]$$

**Pair-wise loss (BPR):**
$$L_{pair} = -\sum_{(u,i,j)} \log \sigma(\hat{y}_{ui} - \hat{y}_{uj})$$

where $i$ = positive item, $j$ = negative item

**Compare:**
1. What does each optimize for?
2. Which is better for ranking tasks (top-N recommendations)?
3. Computational complexity comparison
4. When would you prefer point-wise?

**Learning Outcomes:**
- Understand different training objectives
- Choose loss functions for ranking
- Recognize impact on recommendation quality

---

## Problem 8: Neural Network Expressiveness
**Difficulty:** Hard
**Topics:** Universal approximation, network depth

**Theorem:** A neural network with one hidden layer and sufficient neurons can approximate any continuous function.

**Questions:**
1. Why use deep networks (multiple layers) if one layer is universal?
2. For NCF, compare performance: 1-layer MLP vs. 3-layer MLP
3. What user-item interaction patterns can deep networks capture that MF cannot?
4. Give a concrete example of a non-linear interaction

**Example non-linear pattern:**
"User likes action movies OR comedies, but NOT action-comedies"

This requires: NOT(A AND B) = (NOT A) OR (NOT B), which is non-linear.

**Learning Outcomes:**
- Understand network depth benefits
- Recognize non-linear interactions
- Justify deep architectures

---

## Programming Exercises

### Exercise 1: Implement NCF from Scratch (PyTorch)
**Dataset:** MovieLens 100K
**Task:** Build the NeuMF architecture

**Implementation:**
```python
import torch
import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, gmf_dim=32, mlp_dims=[64, 32, 16]):
        super(NeuMF, self).__init__()

        # GMF embeddings
        self.user_emb_gmf = nn.Embedding(n_users, gmf_dim)
        self.item_emb_gmf = nn.Embedding(n_items, gmf_dim)

        # MLP embeddings
        self.user_emb_mlp = nn.Embedding(n_users, mlp_dims[0]//2)
        self.item_emb_mlp = nn.Embedding(n_items, mlp_dims[0]//2)

        # MLP layers
        self.mlp_layers = nn.ModuleList()
        for i in range(len(mlp_dims)-1):
            self.mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i+1]))
            self.mlp_layers.append(nn.ReLU())

        # Final prediction layer
        self.output = nn.Linear(gmf_dim + mlp_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        # GMF pathway
        user_gmf = self.user_emb_gmf(user)
        item_gmf = self.item_emb_gmf(item)
        gmf_output = user_gmf * item_gmf  # Element-wise product

        # MLP pathway
        user_mlp = self.user_emb_mlp(user)
        item_mlp = self.item_emb_mlp(item)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_output = mlp_input
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)

        # Concatenate and predict
        concat = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.sigmoid(self.output(concat))
        return prediction
```

**Training:**
- Loss: Binary Cross-Entropy
- Negative sampling: 4 negatives per positive
- Optimizer: Adam, lr=0.001
- Batch size: 256
- Epochs: 20 with early stopping

**Evaluation:** HR@10, NDCG@10 (target: HR@10 > 0.65)

---

### Exercise 2: Implement AutoRec
**Dataset:** MovieLens 100K
**Task:** Build item-based AutoRec

**Architecture:**
```python
class AutoRec(nn.Module):
    def __init__(self, n_items, hidden_dim=500):
        super(AutoRec, self).__init__()
        self.encoder = nn.Linear(n_items, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, n_items)

    def forward(self, x, mask):
        # x: user rating vector, mask: observed ratings
        hidden = torch.relu(self.encoder(x))
        reconstruction = self.decoder(hidden)
        return reconstruction

    def loss(self, reconstruction, target, mask):
        # Only compute MSE on observed ratings
        diff = (reconstruction - target) * mask
        return torch.sum(diff ** 2) / torch.sum(mask)
```

**Training:**
- For each user, treat rating vector as input/output
- Mask unobserved ratings in loss
- Compare with MF on RMSE

---

### Exercise 3: Negative Sampling Comparison
**Dataset:** MovieLens 1M (implicit: rating ≥4 = positive)
**Task:** Compare negative sampling strategies

**Strategies:**
1. **Uniform:** `negative_item = np.random.randint(0, n_items)`
2. **Popularity:** `negative_item = np.random.choice(items, p=popularity_dist)`
3. **Hard negatives:** Sample from top-K predictions of current model

**Implementation:**
```python
def sample_negatives(user, n_negatives, strategy='uniform'):
    positives = user_positive_items[user]
    negatives = []

    if strategy == 'uniform':
        candidates = list(set(range(n_items)) - set(positives))
        negatives = np.random.choice(candidates, n_negatives, replace=False)

    elif strategy == 'popularity':
        # Sample proportional to item popularity
        pass

    elif strategy == 'hard':
        # Predict scores for all items, sample high-scoring negatives
        pass

    return negatives
```

**Comparison:** Train NCF with each strategy, measure NDCG@10

---

### Exercise 4: Deep Matrix Factorization
**Dataset:** MovieLens 100K
**Task:** Replace MF dot product with a deep network

**Architecture:**
```python
class DeepMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=50):
        super(DeepMF, self).__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)

        # Deep interaction network
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, user, item):
        user_vec = self.user_emb(user)
        item_vec = self.item_emb(item)
        concat = torch.cat([user_vec, item_vec], dim=-1)

        x = torch.relu(self.fc1(concat))
        x = torch.relu(self.fc2(x))
        rating = self.fc3(x)
        return rating
```

**Experiment:** Compare DeepMF vs. standard MF on RMSE

---

### Exercise 5: Pre-training Strategy for NCF
**Dataset:** MovieLens 1M
**Task:** Implement pre-training as described in NCF paper

**Steps:**
1. Pre-train GMF pathway separately
2. Pre-train MLP pathway separately
3. Initialize NeuMF with pre-trained weights
4. Fine-tune entire NeuMF model

**Implementation:**
```python
# Step 1: Train GMF
gmf_model = GMF(n_users, n_items, embedding_dim=32)
train(gmf_model, data, epochs=10)

# Step 2: Train MLP
mlp_model = MLP(n_users, n_items, layers=[64, 32, 16])
train(mlp_model, data, epochs=10)

# Step 3: Initialize NeuMF
neumf_model = NeuMF(n_users, n_items)
neumf_model.user_emb_gmf.weight = gmf_model.user_emb.weight
neumf_model.item_emb_gmf.weight = gmf_model.item_emb.weight
neumf_model.user_emb_mlp.weight = mlp_model.user_emb.weight
neumf_model.item_emb_mlp.weight = mlp_model.item_emb.weight

# Step 4: Fine-tune
train(neumf_model, data, epochs=5, lr=0.0001)  # Lower LR
```

**Comparison:** NeuMF with vs. without pre-training

---

### Exercise 6: Batch Normalization and Dropout
**Dataset:** MovieLens 100K
**Task:** Study regularization effects

**Variants:**
1. No regularization
2. Dropout only (rate=0.5)
3. Batch normalization only
4. Both dropout and batch norm

**Architecture:**
```python
class RegularizedNCF(nn.Module):
    def __init__(self, use_bn=False, dropout_rate=0.0):
        # ... embeddings ...

        layers = []
        for i in range(len(mlp_dims)-1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i+1]))
            if use_bn:
                layers.append(nn.BatchNorm1d(mlp_dims[i+1]))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        self.mlp = nn.Sequential(*layers)
```

**Measure:**
- Training loss, validation loss
- Test NDCG@10
- Overfitting gap (train - test performance)

---

### Exercise 7: Learning Rate Scheduling
**Dataset:** MovieLens 1M
**Task:** Implement and compare LR schedules

**Schedules:**
1. Constant: lr = 0.001
2. Step decay: lr × 0.5 every 5 epochs
3. Exponential decay: lr × 0.95 every epoch
4. Cosine annealing

**Implementation:**
```python
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

# Step decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(20):
    train_epoch(model, train_loader, optimizer)
    scheduler.step()
```

**Analysis:** Plot learning curves for each schedule

---

## Discussion Questions

1. **Why Deep Learning?** NCF shows ~5% improvement over MF. Is this worth the added complexity? When would you still use MF?

2. **Interpretability:** Neural networks are black boxes. How do you explain NCF recommendations to users?

3. **Feature Integration:** How would you incorporate side information (user age, item genre) into NCF?

4. **Scalability:** NCF requires forward passes for all items at inference. How do you scale this to millions of items?

5. **Adversarial Examples:** Can you craft adversarial user behaviors that fool NCF? What defenses exist?

6. **Transfer Learning:** Can you pre-train NCF on one domain (movies) and transfer to another (books)?

7. **Online Learning:** How would you update NCF as new user interactions arrive? Full retraining or incremental updates?

8. **Fairness:** Do deep recommenders amplify popularity bias more than traditional CF? How would you measure and mitigate this?

---

## Challenge Problem: Attention-Based Neural CF

**Difficulty:** Very Hard
**Topics:** Attention mechanisms, interpretability, deep learning

**Task:** Extend NCF with attention to make it interpretable

**Key Idea:** Instead of simple element-wise product or concatenation, use attention to weight the importance of different latent dimensions.

**Architecture:**
```
User embedding: u = [u1, u2, ..., uk]
Item embedding: i = [i1, i2, ..., ik]

Attention scores: a_j = softmax(W × [u_j, i_j])
Weighted interaction: h = Σ a_j × (u_j ⊙ i_j)
Output: ŷ = σ(W_out × h)
```

**Benefits:**
- Interpretable: Attention scores show which dimensions matter
- Performance: Attention can focus on relevant features

**Implementation:**
```python
class AttentionNCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=32):
        super(AttentionNCF, self).__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)

        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.output = nn.Linear(embedding_dim, 1)

    def forward(self, user, item):
        user_vec = self.user_emb(user)
        item_vec = self.item_emb(item)

        # Compute attention for each dimension
        attention_input = torch.stack([user_vec, item_vec], dim=-1)
        attention_scores = torch.softmax(self.attention(attention_input), dim=1)

        # Weighted element-wise product
        interaction = attention_scores.squeeze(-1) * (user_vec * item_vec)

        # Predict
        prediction = torch.sigmoid(self.output(interaction))
        return prediction, attention_scores
```

**Evaluation:**
1. Compare with standard NCF on NDCG@10
2. Visualize attention scores for sample predictions
3. Identify which latent dimensions are most important

---

## References

### Papers
1. He, X., et al. (2017). "Neural collaborative filtering". WWW.
2. Sedhain, S., et al. (2015). "AutoRec: Autoencoders meet collaborative filtering". WWW.
3. Liang, D., et al. (2018). "Variational autoencoders for collaborative filtering". WWW.
4. He, X., & Chua, T. S. (2017). "Neural factorization machines for sparse predictive analytics". SIGIR.

### Libraries
- PyTorch: https://pytorch.org/
- TensorFlow/Keras: https://www.tensorflow.org/
- Cornac: https://cornac.readthedocs.io/ (includes NCF implementations)

### Datasets
- MovieLens: https://grouplens.org/datasets/movielens/
- Amazon Reviews: http://jmcauley.ucsd.edu/data/amazon/
- Yelp: https://www.yelp.com/dataset

---

*Return to [Week 5 Main Page](README.md)*
