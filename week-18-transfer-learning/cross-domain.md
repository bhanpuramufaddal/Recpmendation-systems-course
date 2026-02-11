# Week 18: Cross-Domain Recommendation

## The Opening Problem

*"Professor, I built a great book recommender at Goodreads. My boss wants me to use it for our new movie streaming service. Can I just... use the same model?"*

**The transfer question** - and it's one of the most valuable questions in recommendation systems.

Think about it: You've spent months building a book recommender. You have 50 million book ratings, carefully tuned embeddings, and a model that knows Alice loves dystopian fiction while Bob prefers historical biographies. Now your company launches a movie service with... 10,000 ratings.

**Can we transfer Alice's book preferences to movies?**

This is cross-domain recommendation - and it's both more nuanced and more powerful than you might expect.

---

## Socratic Exploration: What Actually Transfers?

*Before we dive into algorithms, let's think carefully about what we're attempting...*

**Question 1**: If Alice loves "1984" by George Orwell, what movies might she enjoy?

*Think about it...*

Probably dystopian films like "Blade Runner", "The Matrix", or "V for Vendetta". Her preference for dystopian themes might transfer!

**Question 2**: If Bob reads 3 books per week at bedtime, does that tell us how he watches movies?

*This is trickier...*

Not really. His book consumption pattern (bedtime reading) doesn't transfer to movie watching (maybe weekend binge-watching). **Consumption patterns don't transfer.**

**Question 3**: What book-reading preferences would predict movie preferences?

Let's categorize:

| **Transfers Well** | **Transfers Poorly** |
|-------------------|---------------------|
| Genre preferences | Consumption timing |
| Thematic interests | Format preferences |
| Complexity tolerance | Device preferences |
| Author style -> director style | Reading speed -> watch duration |
| Emotional tone preferences | Social context (solo vs group) |

**The Key Insight**: *Domain-invariant preferences* transfer. *Domain-specific behaviors* don't.

---

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

## Domain Overlap Analysis: When Is Transfer Possible?

*"Can we transfer from ANY domain to ANY other domain?"*

**Absolutely not.** Transfer requires specific conditions. Let's derive them.

### The Overlap Requirements

For transfer to work, we need **bridges** between domains:

#### Requirement 1: User Overlap

Let $\mathcal{U}_S$ be users in source domain, $\mathcal{U}_T$ be users in target domain.

**Bridge users**: $\mathcal{U}_B = \mathcal{U}_S \cap \mathcal{U}_T$

**Theorem (Informal)**: *Transfer effectiveness is proportional to bridge user coverage.*

$$\text{Transfer Quality} \propto \frac{|\mathcal{U}_B|}{|\mathcal{U}_T|}$$

**Why?** Bridge users are our Rosetta Stone - they tell us how source behavior maps to target behavior.

**Example**:
- 1000 target users, 100 bridge users = 10% coverage (limited transfer)
- 1000 target users, 800 bridge users = 80% coverage (strong transfer)

#### Requirement 2: Semantic Overlap

Even without shared users, domains might share **concepts**:

$$\text{Semantic Overlap} = \text{sim}(\mathcal{C}_S, \mathcal{C}_T)$$

where $\mathcal{C}_S$, $\mathcal{C}_T$ are concept spaces.

**Examples of High Semantic Overlap**:
- Books <-> Movies (genres, themes, stories)
- Music <-> Podcasts (audio preferences, mood)
- News <-> Blogs (topics, writing style)

**Examples of Low Semantic Overlap**:
- Books <-> Groceries (almost nothing)
- Movies <-> Financial products (nothing)

#### Requirement 3: Preference Correlation

Do preferences actually correlate across domains?

**Test**: For bridge users, compute correlation between ratings:

$$\rho = \text{corr}(\mathbf{r}^S_u, \mathbf{r}^T_u)$$

**If $\rho \approx 0$**: Transfer won't help (preferences are independent).

**If $\rho > 0$**: Transfer has potential.

### The Transfer Decision Framework

```
Is Transfer Worth It?

1. Bridge User Coverage > 5%?
   NO  -> Limited transfer benefit
   YES -> Continue

2. Semantic Overlap Exists?
   NO  -> Transfer likely harmful
   YES -> Continue

3. Preference Correlation > 0.3?
   NO  -> Transfer questionable
   YES -> Transfer recommended!
```

---

## Problem Formulation

### Notation

**Domains**:
- Source domain $\mathcal{D}_S$: Rich data
- Target domain $\mathcal{D}_T$: Sparse data

**Goal**: Improve target domain performance using source domain knowledge.

**Assumptions**:
1. **Shared users**: Some users active in both domains
2. **Transferable patterns**: Preferences correlate across domains (action movie fans -> action book fans)

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

## Embedding Alignment Derivation

*"Professor, HOW do we actually map embeddings from one domain to another?"*

Let's derive this step-by-step.

### The Core Problem

We have:
- Source embeddings: $\mathbf{U}_S \in \mathbb{R}^{n \times d}$ (n bridge users, d dimensions)
- Target embeddings: $\mathbf{U}_T \in \mathbb{R}^{n \times d}$ (same users in target domain)

The embeddings live in **different spaces** - a user's "action preference" dimension in movies might be dimension 3, but in books it might be dimension 47.

**Goal**: Find transformation $\mathbf{W}$ such that:

$$\mathbf{U}_S \cdot \mathbf{W} \approx \mathbf{U}_T$$

### Step 1: Formulate the Objective

We want $\mathbf{W}$ that minimizes reconstruction error:

$$\min_{\mathbf{W}} \|\mathbf{U}_S \mathbf{W} - \mathbf{U}_T\|_F^2$$

where $\|\cdot\|_F$ is the Frobenius norm.

### Step 2: Take the Derivative

Expand the objective:

$$\mathcal{L} = \|\mathbf{U}_S \mathbf{W} - \mathbf{U}_T\|_F^2 = \text{tr}((\mathbf{U}_S \mathbf{W} - \mathbf{U}_T)^T(\mathbf{U}_S \mathbf{W} - \mathbf{U}_T))$$

Taking derivative with respect to $\mathbf{W}$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = 2\mathbf{U}_S^T\mathbf{U}_S\mathbf{W} - 2\mathbf{U}_S^T\mathbf{U}_T$$

### Step 3: Solve for Optimal W

Setting derivative to zero:

$$\mathbf{U}_S^T\mathbf{U}_S\mathbf{W} = \mathbf{U}_S^T\mathbf{U}_T$$

$$\mathbf{W}^* = (\mathbf{U}_S^T\mathbf{U}_S)^{-1}\mathbf{U}_S^T\mathbf{U}_T$$

This is the **least-squares solution** - the pseudo-inverse!

$$\mathbf{W}^* = \mathbf{U}_S^+ \mathbf{U}_T$$

### Step 4: Regularization for Stability

In practice, $\mathbf{U}_S^T\mathbf{U}_S$ might be singular. Add regularization:

$$\mathbf{W}^* = (\mathbf{U}_S^T\mathbf{U}_S + \lambda \mathbf{I})^{-1}\mathbf{U}_S^T\mathbf{U}_T$$

**The Intuition**: $\mathbf{W}$ is a **rotation and scaling** that aligns the source embedding space with the target embedding space.

---

## Numerical Example: Embedding Alignment in Action

*Let's work through a concrete example with 3 users and 2-dimensional embeddings.*

### Setup

**Bridge Users**: Alice, Bob, Carol (active in both domains)

**Source Domain (Movies) Embeddings**:
```
           Action   Romance
Alice:    [0.8,     0.2]     <- Loves action
Bob:      [0.3,     0.9]     <- Loves romance
Carol:    [0.6,     0.5]     <- Balanced
```

$$\mathbf{U}_S = \begin{bmatrix} 0.8 & 0.2 \\ 0.3 & 0.9 \\ 0.6 & 0.5 \end{bmatrix}$$

**Target Domain (Books) Embeddings**:
```
           Thriller  Drama
Alice:    [0.7,     0.3]
Bob:      [0.2,     0.8]
Carol:    [0.5,     0.6]
```

$$\mathbf{U}_T = \begin{bmatrix} 0.7 & 0.3 \\ 0.2 & 0.8 \\ 0.5 & 0.6 \end{bmatrix}$$

### Step 1: Compute $\mathbf{U}_S^T\mathbf{U}_S$

$$\mathbf{U}_S^T\mathbf{U}_S = \begin{bmatrix} 0.8 & 0.3 & 0.6 \\ 0.2 & 0.9 & 0.5 \end{bmatrix} \begin{bmatrix} 0.8 & 0.2 \\ 0.3 & 0.9 \\ 0.6 & 0.5 \end{bmatrix}$$

$$= \begin{bmatrix} 0.64+0.09+0.36 & 0.16+0.27+0.30 \\ 0.16+0.27+0.30 & 0.04+0.81+0.25 \end{bmatrix} = \begin{bmatrix} 1.09 & 0.73 \\ 0.73 & 1.10 \end{bmatrix}$$

### Step 2: Compute $\mathbf{U}_S^T\mathbf{U}_T$

$$\mathbf{U}_S^T\mathbf{U}_T = \begin{bmatrix} 0.8 & 0.3 & 0.6 \\ 0.2 & 0.9 & 0.5 \end{bmatrix} \begin{bmatrix} 0.7 & 0.3 \\ 0.2 & 0.8 \\ 0.5 & 0.6 \end{bmatrix}$$

$$= \begin{bmatrix} 0.56+0.06+0.30 & 0.24+0.24+0.36 \\ 0.14+0.18+0.25 & 0.06+0.72+0.30 \end{bmatrix} = \begin{bmatrix} 0.92 & 0.84 \\ 0.57 & 1.08 \end{bmatrix}$$

### Step 3: Compute $\mathbf{W}^*$

$$\mathbf{W}^* = (\mathbf{U}_S^T\mathbf{U}_S)^{-1}\mathbf{U}_S^T\mathbf{U}_T$$

First, invert $\mathbf{U}_S^T\mathbf{U}_S$:

$$\det = 1.09 \times 1.10 - 0.73^2 = 1.199 - 0.533 = 0.666$$

$$(\mathbf{U}_S^T\mathbf{U}_S)^{-1} = \frac{1}{0.666}\begin{bmatrix} 1.10 & -0.73 \\ -0.73 & 1.09 \end{bmatrix} = \begin{bmatrix} 1.65 & -1.10 \\ -1.10 & 1.64 \end{bmatrix}$$

Finally:

$$\mathbf{W}^* = \begin{bmatrix} 1.65 & -1.10 \\ -1.10 & 1.64 \end{bmatrix} \begin{bmatrix} 0.92 & 0.84 \\ 0.57 & 1.08 \end{bmatrix} = \begin{bmatrix} 0.89 & 0.20 \\ -0.08 & 0.85 \end{bmatrix}$$

### Step 4: Transfer a New User!

**Dave** only has movie ratings: $\mathbf{u}_S^{\text{Dave}} = [0.9, 0.1]$ (action lover)

**Predicted book embedding**:

$$\mathbf{u}_T^{\text{Dave}} = \mathbf{u}_S^{\text{Dave}} \cdot \mathbf{W}^* = [0.9, 0.1] \begin{bmatrix} 0.89 & 0.20 \\ -0.08 & 0.85 \end{bmatrix}$$

$$= [0.9 \times 0.89 + 0.1 \times (-0.08), 0.9 \times 0.20 + 0.1 \times 0.85] = [0.79, 0.27]$$

**Interpretation**: Dave's movie preference for action maps to books with high thriller dimension (0.79) and low drama (0.27). **Makes sense!**

---

## Cold Domain Bootstrapping: Starting from Zero

*"What if my target domain has ZERO data? Can I still transfer?"*

Yes! This is **cold domain bootstrapping** - one of the most powerful applications of transfer learning.

### The Zero-Data Transfer Protocol

**Scenario**: You have a mature movie recommender. You're launching a book service tomorrow. No user has rated a single book yet.

**Strategy**:

#### Phase 1: Feature Mapping (No Target Data Required)

Map item features across domains:

```python
# Movie features -> Book features mapping
feature_mapping = {
    'action_movie': 'thriller_book',
    'romance_movie': 'romance_book',
    'scifi_movie': 'scifi_book',
    'drama_movie': 'literary_fiction',
    'comedy_movie': 'humor_book'
}

def transfer_item_features(movie_embedding, feature_map):
    """Transfer movie features to book space without any book data."""
    book_embedding = np.zeros(len(feature_map))
    for movie_feat, book_feat in feature_map.items():
        book_embedding[book_feat] = movie_embedding[movie_feat]
    return book_embedding
```

#### Phase 2: User Preference Transfer

For users with movie history, predict book preferences:

$$\hat{\mathbf{u}}_{\text{books}} = f(\mathbf{u}_{\text{movies}})$$

where $f$ is learned from semantic relationships (not data).

#### Phase 3: Cold Start Recommendations

```python
def cold_domain_recommend(user_id, n_recommendations=10):
    """Recommend books to user who has only rated movies."""

    # Get user's movie embedding
    movie_embedding = movie_model.get_user_embedding(user_id)

    # Transfer to book space (using feature mapping)
    book_embedding = transfer_user_embedding(movie_embedding)

    # Score all books
    book_scores = {}
    for book_id in all_books:
        book_features = get_book_features(book_id)
        score = np.dot(book_embedding, book_features)
        book_scores[book_id] = score

    # Return top recommendations
    return sorted(book_scores.items(), key=lambda x: -x[1])[:n_recommendations]
```

#### Phase 4: Bootstrap Learning

As users interact with books, update the transfer mapping:

```
Day 1: Pure transfer (100% source knowledge)
Day 7: 70% transfer + 30% target learning
Day 30: 30% transfer + 70% target learning
Day 90: 10% transfer + 90% target learning
```

**The Bootstrap Decay**:

$$\alpha_t = \alpha_0 \cdot e^{-\lambda t}$$

where $\alpha_t$ is the transfer weight at time $t$.

---

## Feature Mapping Strategies: What Transfers, What Doesn't

### Systematic Feature Analysis

| Feature Type | Transferability | Reason | Example |
|-------------|-----------------|--------|---------|
| **Genre** | HIGH | Abstract concept | Action -> Action |
| **Theme** | HIGH | Domain-agnostic | "Redemption" story |
| **Popularity** | MEDIUM | Context-dependent | Popular in one != popular in other |
| **Format** | LOW | Domain-specific | Hardcover vs streaming |
| **Duration** | LOW | Different scales | 300 pages vs 120 minutes |
| **Release date** | MEDIUM | Recency matters similarly | New releases popular |
| **Creator style** | MEDIUM | Partially transfers | Director style != author style |

### Feature Transfer Implementation

```python
class FeatureTransferAnalyzer:
    """Analyze which features transfer across domains."""

    def __init__(self, bridge_users, source_data, target_data):
        self.bridge_users = bridge_users
        self.source_data = source_data
        self.target_data = target_data

    def compute_feature_transferability(self, feature_name):
        """
        Measure how well a feature predicts cross-domain preferences.

        Returns: correlation coefficient
        """
        source_feature_values = []
        target_preferences = []

        for user in self.bridge_users:
            # Get user's average for this feature in source
            source_val = self.source_data.get_user_feature_preference(
                user, feature_name
            )

            # Get user's overall satisfaction in target
            target_val = self.target_data.get_user_satisfaction(user)

            source_feature_values.append(source_val)
            target_preferences.append(target_val)

        return np.corrcoef(source_feature_values, target_preferences)[0, 1]

    def rank_features_by_transferability(self, feature_list):
        """Rank features by how well they transfer."""
        scores = {}
        for feature in feature_list:
            scores[feature] = self.compute_feature_transferability(feature)

        return sorted(scores.items(), key=lambda x: -abs(x[1]))


# Example usage
analyzer = FeatureTransferAnalyzer(bridge_users, movies, books)

features = ['genre', 'theme', 'popularity', 'duration', 'release_year']
transferability = analyzer.rank_features_by_transferability(features)

print("Feature Transferability Ranking:")
for feature, score in transferability:
    transfer_quality = "HIGH" if abs(score) > 0.5 else "MEDIUM" if abs(score) > 0.3 else "LOW"
    print(f"  {feature}: {score:.3f} ({transfer_quality})")

# Output:
# Feature Transferability Ranking:
#   genre: 0.72 (HIGH)
#   theme: 0.65 (HIGH)
#   release_year: 0.41 (MEDIUM)
#   popularity: 0.38 (MEDIUM)
#   duration: 0.12 (LOW)
```

---

## What Can Go Wrong: Failure Modes

*"Professor, when does cross-domain transfer actually HURT performance?"*

Excellent question. Transfer learning has significant risks.

### Failure Mode 1: Negative Transfer

**Definition**: When source domain knowledge degrades target performance.

**When it happens**:
- Domains are too different
- Source model learns domain-specific quirks
- Preference correlation is negative or zero

**Example**:
```
Source: Professional tool recommendations (B2B)
Target: Consumer gadget recommendations (B2C)

Problem: B2B users want reliability, durability
         B2C users want novelty, aesthetics

Result: Transferring B2B preferences recommends "boring" products to consumers
        -> NEGATIVE TRANSFER
```

**Detection**:
```python
def detect_negative_transfer(baseline_model, transfer_model, test_data):
    """Check if transfer hurts performance."""
    baseline_ndcg = evaluate_ndcg(baseline_model, test_data)
    transfer_ndcg = evaluate_ndcg(transfer_model, test_data)

    if transfer_ndcg < baseline_ndcg:
        print(f"WARNING: Negative transfer detected!")
        print(f"Baseline NDCG: {baseline_ndcg:.4f}")
        print(f"Transfer NDCG: {transfer_ndcg:.4f}")
        print(f"Performance drop: {(baseline_ndcg - transfer_ndcg) / baseline_ndcg * 100:.1f}%")
        return True
    return False
```

### Failure Mode 2: Domain Mismatch

**The Distribution Shift Problem**:

Source distribution: $P_S(x, y)$
Target distribution: $P_T(x, y)$

If $P_S \neq P_T$, learned patterns may not apply.

**Types of Mismatch**:

1. **Covariate Shift**: $P_S(x) \neq P_T(x)$ but $P(y|x)$ same
   - Example: Different user demographics across domains

2. **Label Shift**: $P_S(y) \neq P_T(y)$
   - Example: Different rating distributions (movies avg 3.5, books avg 4.0)

3. **Concept Shift**: $P_S(y|x) \neq P_T(y|x)$
   - Example: "Action" means different things for games vs movies

**Mitigation**:
```python
def align_distributions(source_data, target_data):
    """Align source and target distributions before transfer."""

    # 1. Normalize ratings to same scale
    source_data['rating'] = (source_data['rating'] - source_data['rating'].mean()) / source_data['rating'].std()
    target_data['rating'] = (target_data['rating'] - target_data['rating'].mean()) / target_data['rating'].std()

    # 2. Reweight source samples to match target distribution
    source_weights = compute_importance_weights(source_data, target_data)

    return source_data, target_data, source_weights
```

### Failure Mode 3: User Preference Shift

**The Temporal Problem**: Users' preferences change over time and across contexts.

**Example**:
- Alice's movie preferences (2020): Action, Thriller
- Alice's book preferences (2024): Self-help, Biography

**Has Alice changed, or are the domains different?**

**Detection**:
```python
def detect_preference_shift(user_id, source_history, target_history):
    """Detect if user preferences have shifted."""

    # Get user's genre distribution in source (historical)
    source_genres = get_genre_distribution(user_id, source_history)

    # Get user's genre distribution in target (recent)
    target_genres = get_genre_distribution(user_id, target_history)

    # Compute distribution divergence
    kl_divergence = compute_kl_divergence(source_genres, target_genres)

    if kl_divergence > SHIFT_THRESHOLD:
        print(f"User {user_id}: Preference shift detected (KL={kl_divergence:.3f})")
        return True
    return False
```

### Failure Mode 4: Overfitting to Bridge Users

**The Problem**: Bridge users may not be representative.

**Example**:
- Bridge users: Power users active in both domains (10%)
- Target users: Casual users only in target (90%)

Power users have different preferences than casual users!

**Mitigation**: Weight bridge users to match target population.

---

## Shared User Representations

### Assumption

**Users have consistent preferences across domains.**

**Example**:
- User likes Sci-Fi movies -> likely to like Sci-Fi books
- User dislikes Romance movies -> likely to dislikes Romance books

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

**Transfer Requirements**:
- Bridge user coverage > 5%
- Semantic overlap between domains
- Positive preference correlation

**What Transfers vs. What Doesn't**:
- **Transfers**: Genre preferences, themes, emotional tone
- **Doesn't Transfer**: Consumption patterns, format preferences, timing

**When to use**:
- Target domain has limited data
- Shared users/items across domains
- Preferences correlate across domains

**Best practices**:
- Start with simple shared-user MF
- Use adversarial training for distribution shift
- Evaluate on held-out target domain data
- Monitor for negative transfer

---

## Practice Problems

**Problem 1**: Implement cross-domain matrix factorization for MovieLens (source) -> Amazon Books (target). Measure improvement vs. target-only baseline.

**Problem 2**: Design an experiment to test when cross-domain transfer helps vs. hurts. What characteristics make transfer effective?

**Problem 3**: Implement CCA-based embedding alignment. Compare with linear mapping (single matrix).

**Problem 4**: Use adversarial domain adaptation to transfer from music recommendations to podcast recommendations. Evaluate on shared users.

**Problem 5** (New): Given the numerical example above, compute what happens when you try to transfer a user with embedding [0.1, 0.9] (romance lover). Does the mapping make semantic sense?

**Problem 6** (New): Analyze the transferability of features between video games and board games. Which features would transfer well? Design an experiment to test your hypotheses.

---

## References

1. **Pan, W., et al. (2010)**. "Transfer Learning in Collaborative Filtering for Sparsity Reduction". *AAAI*.

2. **Hu, L., et al. (2018)**. "CoNet: Collaborative Cross Networks for Cross-Domain Recommendation". *CIKM*.

3. **Ganin, Y., et al. (2016)**. "Domain-Adversarial Training of Neural Networks". *JMLR*.

4. **Zhu, F., et al. (2021)**. "Cross-Domain Recommendation: Challenges, Progress, and Prospects". *IJCAI*.

5. **Man, T., et al. (2017)**. "Cross-Domain Recommendation: An Embedding and Mapping Approach". *IJCAI*.

6. **Zhao, C., et al. (2020)**. "CATN: Cross-Domain Recommendation for Cold-Start Users via Aspect Transfer Network". *SIGIR*.
