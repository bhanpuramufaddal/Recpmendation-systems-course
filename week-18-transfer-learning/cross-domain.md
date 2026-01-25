# Week 18: Cross-Domain Recommendation

## Overview

**Cross-domain recommendation**: Transfer knowledge from data-rich domain (source) to data-sparse domain (target).

**Motivation**:
- **Cold start**: New domain has limited data
- **Data scarcity**: Some domains naturally have fewer interactions
- **Shared users**: Users active in multiple domains

**Example**:
- Source: Movie recommendations (millions of ratings)
- Target: Book recommendations (thousands of ratings)
- Transfer user preferences from movies to books

---

## Problem Formulation

### Notation

**Domains**:
- Source domain $\mathcal{D}_S$: Rich data
- Target domain $\mathcal{D}_T$: Sparse data

**Goal**: Improve target domain performance using source domain knowledge.

**Assumptions**:
1. **Shared users**: Some users active in both domains
2. **Transferable patterns**: Preferences correlate across domains (action movie fans → action book fans)

---

### Types of Transfer

**1. User-level transfer**:
- Transfer user representations
- Example: User embedding learned from movies used for books

**2. Item-level transfer**:
- Transfer item features or embeddings
- Example: Genre embeddings shared across movies and books

**3. Model-level transfer**:
- Pre-train model on source, fine-tune on target
- Example: Neural network trained on movies, adapted for books

---

## Shared User Representations

### Assumption

**Users have consistent preferences across domains.**

**Example**:
- User likes Sci-Fi movies → likely to like Sci-Fi books
- User dislikes Romance movies → likely to dislikes Romance books

---

### Cross-Domain Matrix Factorization

**Idea**: Share user latent factors across domains, domain-specific item factors.

**Model**:
$$R_S \approx U^T V_S \quad \text{(source domain)}$$
$$R_T \approx U^T V_T \quad \text{(target domain)}$$

where:
- $U$ = shared user factors
- $V_S$ = source item factors
- $V_T$ = target item factors

**Benefits**:
- User representations benefit from both domains
- Alleviates user cold start in target domain

---

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossDomainMF(nn.Module):
    def __init__(self, n_users, n_items_source, n_items_target, n_factors=64):
        super().__init__()

        # Shared user embeddings
        self.user_factors = nn.Embedding(n_users, n_factors)

        # Domain-specific item embeddings
        self.item_factors_source = nn.Embedding(n_items_source, n_factors)
        self.item_factors_target = nn.Embedding(n_items_target, n_factors)

        # Initialize
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors_source.weight, std=0.01)
        nn.init.normal_(self.item_factors_target.weight, std=0.01)

    def forward(self, user_ids, item_ids, domain='source'):
        """
        Predict ratings.

        Args:
            domain: 'source' or 'target'
        """
        user_emb = self.user_factors(user_ids)

        if domain == 'source':
            item_emb = self.item_factors_source(item_ids)
        else:
            item_emb = self.item_factors_target(item_ids)

        # Dot product
        prediction = (user_emb * item_emb).sum(dim=1)

        return prediction


# Training
model = CrossDomainMF(
    n_users=10000,
    n_items_source=5000,   # Movies
    n_items_target=2000,   # Books
    n_factors=64
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(10):
    # Train on source domain
    for user_ids, item_ids, ratings in source_loader:
        predictions = model(user_ids, item_ids, domain='source')
        loss = criterion(predictions, ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Train on target domain
    for user_ids, item_ids, ratings in target_loader:
        predictions = model(user_ids, item_ids, domain='target')
        loss = criterion(predictions, ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Source Loss = {loss_source:.4f}, Target Loss = {loss_target:.4f}")
```

---

## Transfer Mapping

### Codebook Approach

**Assumption**: Latent factors in different domains can be mapped to each other.

**Idea**: Learn mapping $W$ from source embeddings to target embeddings.

$$V_T = W \cdot V_S$$

**Training**:
1. Pre-train source domain: $R_S \approx U_S^T V_S$
2. Pre-train target domain: $R_T \approx U_T^T V_T$
3. Learn mapping on shared users/items

---

### Implementation

```python
class TransferMapping(nn.Module):
    def __init__(self, n_factors=64):
        super().__init__()

        # Linear mapping from source to target
        self.mapping = nn.Linear(n_factors, n_factors)

    def forward(self, source_embedding):
        """Map source embedding to target space."""
        return self.mapping(source_embedding)


# Training transfer mapping
def train_transfer_mapping(source_model, target_model, shared_users, n_epochs=10):
    """
    Learn mapping from source to target using shared users.
    """
    mapping = TransferMapping(n_factors=64)
    optimizer = torch.optim.Adam(mapping.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        for user_id in shared_users:
            # Get user embedding from source
            source_emb = source_model.user_factors(torch.tensor([user_id]))

            # Get user embedding from target
            target_emb = target_model.user_factors(torch.tensor([user_id]))

            # Map source to target
            mapped_emb = mapping(source_emb)

            # Loss: mapped embedding should match target embedding
            loss = F.mse_loss(mapped_emb, target_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return mapping


# Usage
source_model = load_source_model()  # Pre-trained on movies
target_model = load_target_model()  # Pre-trained on books

shared_users = get_shared_users()  # Users active in both domains

mapping = train_transfer_mapping(source_model, target_model, shared_users)

# Transfer knowledge for cold-start user
new_user_id = 12345
source_emb = source_model.user_factors(torch.tensor([new_user_id]))
target_emb = mapping(source_emb)

# Use target_emb for recommendations in target domain
```

---

## Embedding Alignment

### Canonical Correlation Analysis (CCA)

**Goal**: Find projections that maximize correlation between source and target embeddings.

**Idea**:
- Project source embeddings: $U_S' = W_S U_S$
- Project target embeddings: $U_T' = W_T U_T$
- Maximize correlation: $\text{corr}(U_S', U_T')$

---

### Implementation

```python
from sklearn.cross_decomposition import CCA
import numpy as np

def align_embeddings_cca(source_embeddings, target_embeddings, n_components=50):
    """
    Align source and target embeddings using CCA.

    Args:
        source_embeddings: [n_shared_users, n_factors]
        target_embeddings: [n_shared_users, n_factors]
        n_components: Dimensionality of aligned space
    """
    cca = CCA(n_components=n_components)

    # Fit CCA on shared users
    cca.fit(source_embeddings, target_embeddings)

    # Transform embeddings
    source_aligned, target_aligned = cca.transform(source_embeddings, target_embeddings)

    return cca, source_aligned, target_aligned


# Example
source_model = load_source_model()
target_model = load_target_model()

shared_users = get_shared_users()

# Extract embeddings for shared users
source_embs = source_model.user_factors.weight[shared_users].detach().numpy()
target_embs = target_model.user_factors.weight[shared_users].detach().numpy()

# Align embeddings
cca, source_aligned, target_aligned = align_embeddings_cca(source_embs, target_embs)

# Transfer for new user
new_user_emb_source = source_model.user_factors(torch.tensor([new_user_id])).detach().numpy()
new_user_emb_target = cca.transform(new_user_emb_source.reshape(1, -1))

# Use new_user_emb_target for recommendations
```

---

## Neural Transfer Networks

### Deep Transfer Learning

**Idea**: Use deep neural networks to learn complex mappings between domains.

**Architecture**:
1. **Shared layers**: Learn general representations
2. **Domain-specific layers**: Adapt to each domain

---

### Implementation

```python
class DeepCrossDomainModel(nn.Module):
    def __init__(self, n_users, n_items_source, n_items_target, embedding_dim=64):
        super().__init__()

        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_source = nn.Embedding(n_items_source, embedding_dim)
        self.item_embedding_target = nn.Embedding(n_items_target, embedding_dim)

        # Shared layers
        self.shared_fc1 = nn.Linear(embedding_dim * 2, 128)
        self.shared_fc2 = nn.Linear(128, 64)

        # Domain-specific layers
        self.source_fc = nn.Linear(64, 1)
        self.target_fc = nn.Linear(64, 1)

    def forward(self, user_ids, item_ids, domain='source'):
        user_emb = self.user_embedding(user_ids)

        if domain == 'source':
            item_emb = self.item_embedding_source(item_ids)
        else:
            item_emb = self.item_embedding_target(item_ids)

        # Concatenate user and item embeddings
        x = torch.cat([user_emb, item_emb], dim=1)

        # Shared layers
        x = torch.relu(self.shared_fc1(x))
        x = torch.relu(self.shared_fc2(x))

        # Domain-specific prediction
        if domain == 'source':
            output = self.source_fc(x)
        else:
            output = self.target_fc(x)

        return output.squeeze()


# Training
model = DeepCrossDomainModel(
    n_users=10000,
    n_items_source=5000,
    n_items_target=2000
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(10):
    # Alternate between source and target domains
    for (user_ids_s, item_ids_s, ratings_s), (user_ids_t, item_ids_t, ratings_t) in zip(source_loader, target_loader):
        # Source domain
        preds_source = model(user_ids_s, item_ids_s, domain='source')
        loss_source = criterion(preds_source, ratings_s)

        # Target domain
        preds_target = model(user_ids_t, item_ids_t, domain='target')
        loss_target = criterion(preds_target, ratings_t)

        # Combined loss
        loss = loss_source + loss_target

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Domain Adaptation Techniques

### Adversarial Domain Adaptation

**Idea**: Train model to be domain-invariant using adversarial training.

**Components**:
1. **Feature extractor**: Learns shared representations
2. **Domain classifier**: Tries to distinguish source from target
3. **Adversarial loss**: Feature extractor tries to fool domain classifier

**Goal**: Representations that work well in both domains.

---

### Implementation

```python
class DomainAdversarialRecommender(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64):
        super().__init__()

        # Feature extractor (shared)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Recommendation predictor
        self.predictor = nn.Linear(64, 1)

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        x = torch.cat([user_emb, item_emb], dim=1)

        # Extract features
        features = self.feature_extractor(x)

        # Recommendation prediction
        rating_pred = self.predictor(features).squeeze()

        # Domain prediction
        domain_pred = self.domain_classifier(features).squeeze()

        return rating_pred, domain_pred


# Training with adversarial loss
model = DomainAdversarialRecommender(n_users=10000, n_items=10000)

optimizer_main = torch.optim.Adam(
    list(model.user_embedding.parameters()) +
    list(model.item_embedding.parameters()) +
    list(model.feature_extractor.parameters()) +
    list(model.predictor.parameters()),
    lr=0.001
)

optimizer_domain = torch.optim.Adam(
    model.domain_classifier.parameters(),
    lr=0.001
)

criterion_rating = nn.MSELoss()
criterion_domain = nn.BCELoss()

for epoch in range(10):
    for (user_ids_s, item_ids_s, ratings_s), (user_ids_t, item_ids_t, ratings_t) in zip(source_loader, target_loader):
        # Combine source and target batches
        user_ids = torch.cat([user_ids_s, user_ids_t])
        item_ids = torch.cat([item_ids_s, item_ids_t])
        ratings = torch.cat([ratings_s, ratings_t])

        # Domain labels: 0 for source, 1 for target
        domain_labels = torch.cat([
            torch.zeros(len(user_ids_s)),
            torch.ones(len(user_ids_t))
        ])

        # Forward pass
        rating_preds, domain_preds = model(user_ids, item_ids)

        # Recommendation loss
        loss_rating = criterion_rating(rating_preds, ratings)

        # Domain classification loss
        loss_domain = criterion_domain(domain_preds, domain_labels)

        # Update domain classifier
        optimizer_domain.zero_grad()
        loss_domain.backward(retain_graph=True)
        optimizer_domain.step()

        # Adversarial loss for feature extractor
        # (reverse gradient to fool domain classifier)
        loss_adversarial = -loss_domain

        # Total loss for main model
        loss = loss_rating + 0.1 * loss_adversarial

        optimizer_main.zero_grad()
        loss.backward()
        optimizer_main.step()
```

---

## Evaluation

### Metrics

**1. Target domain performance**:
- NDCG, Recall@K on target domain test set

**2. Transfer effectiveness**:
- Compare with baseline (no transfer)
- Measure improvement: $\Delta = \text{NDCG}_{\text{transfer}} - \text{NDCG}_{\text{no transfer}}$

**3. Data efficiency**:
- Performance vs. target domain training data size

---

### Experimental Setup

```python
def evaluate_cross_domain(model, target_test_loader, k=10):
    """
    Evaluate cross-domain recommendation.
    """
    model.eval()

    ndcg_scores = []

    with torch.no_grad():
        for user_ids, item_ids_pos, item_ids_neg in target_test_loader:
            # Predict scores
            scores_pos = model(user_ids, item_ids_pos, domain='target')
            scores_neg = model(user_ids, item_ids_neg, domain='target')

            # Rank items
            all_scores = torch.cat([scores_pos, scores_neg], dim=1)
            ranked_indices = torch.argsort(all_scores, descending=True)

            # Compute NDCG
            ndcg = compute_ndcg(ranked_indices, k=k)
            ndcg_scores.append(ndcg)

    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    return avg_ndcg


# Compare baseline vs. transfer
baseline_model = train_baseline(target_domain_only=True)
transfer_model = train_cross_domain(source_and_target=True)

ndcg_baseline = evaluate_cross_domain(baseline_model, target_test_loader)
ndcg_transfer = evaluate_cross_domain(transfer_model, target_test_loader)

improvement = (ndcg_transfer - ndcg_baseline) / ndcg_baseline * 100

print(f"Baseline NDCG: {ndcg_baseline:.4f}")
print(f"Transfer NDCG: {ndcg_transfer:.4f}")
print(f"Improvement: {improvement:.2f}%")

# Output:
# Baseline NDCG: 0.3245
# Transfer NDCG: 0.3821
# Improvement: 17.74%
```

---

## Summary

**Key Takeaways**:
1. **Cross-domain MF**: Share user factors across domains
2. **Transfer mapping**: Learn linear/non-linear mapping between embeddings
3. **CCA**: Align embeddings using canonical correlation
4. **Deep transfer**: Shared layers + domain-specific layers
5. **Adversarial training**: Domain-invariant representations

**When to use**:
- Target domain has limited data
- Shared users/items across domains
- Preferences correlate across domains

**Best practices**:
- Start with simple shared-user MF
- Use adversarial training for distribution shift
- Evaluate on held-out target domain data

---

## Practice Problems

**Problem 1**: Implement cross-domain matrix factorization for MovieLens (source) → Amazon Books (target). Measure improvement vs. target-only baseline.

**Problem 2**: Design an experiment to test when cross-domain transfer helps vs. hurts. What characteristics make transfer effective?

**Problem 3**: Implement CCA-based embedding alignment. Compare with linear mapping (single matrix).

**Problem 4**: Use adversarial domain adaptation to transfer from music recommendations to podcast recommendations. Evaluate on shared users.

---

## References

1. **Pan, W., et al. (2010)**. "Transfer Learning in Collaborative Filtering for Sparsity Reduction". *AAAI*.

2. **Hu, L., et al. (2018)**. "CoNet: Collaborative Cross Networks for Cross-Domain Recommendation". *CIKM*.

3. **Ganin, Y., et al. (2016)**. "Domain-Adversarial Training of Neural Networks". *JMLR*.

4. **Zhu, F., et al. (2021)**. "Cross-Domain Recommendation: Challenges, Progress, and Prospects". *IJCAI*.
