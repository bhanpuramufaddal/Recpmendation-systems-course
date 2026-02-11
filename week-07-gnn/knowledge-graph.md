# Week 7: Knowledge Graph Integration

## The Problem: Why Pure Collaborative Filtering Isn't Enough

*Before we dive into knowledge graphs, let me ask you a question.*

**Imagine you're building a movie recommendation system.** A new user joins and hasn't rated anything yet. What do you recommend?

With pure collaborative filtering, you're stuck. No ratings = no similar users = no recommendations.

But wait - you *know* things about this user:
- They searched for "Christopher Nolan movies"
- They clicked on "Inception" (but haven't rated it)
- They live in a city with lots of sci-fi fans

And you *know* things about movies:
- "Inception" was directed by Christopher Nolan
- Christopher Nolan also directed "The Dark Knight" and "Interstellar"
- "Inception" is in the "Mind-bending Sci-Fi" genre

**The key insight**: This external knowledge - relationships between entities - can bridge the gap when interaction data is sparse!

---

## Overview

**Knowledge Graphs (KG)**: Structured representations of entities and relationships.

**In RecSys**: Leverage external knowledge (item attributes, user context, relationships) to enhance recommendations.

**Example**: Movie KG
```
(Matrix, directed_by, Wachowskis)
(Matrix, genre, Sci-Fi)
(Matrix, actor, Keanu Reeves)
(Keanu Reeves, acted_in, John Wick)
```

**Benefits**:
- Alleviate sparsity (use item attributes even without interactions)
- Cold start (new items have KG relations)
- Interpretability (explain via knowledge paths)

---

## Learning Objectives

By the end of this section, you will:
- Understand knowledge graph fundamentals and triple notation
- **Derive step-by-step why TransE uses h + r ≈ t**
- **Explain the DistMult symmetry problem with concrete examples**
- Master KGAT attention mechanisms with numerical walkthrough
- **Generate and evaluate explanation paths**
- Recognize common failure modes in KG-based recommendations

---

## Knowledge Graph Embeddings

### TransE (Translating Embeddings)

**Idea**: Relationships as translations in embedding space.

#### The TransE Derivation: Why h + r ≈ t?

*Let me walk you through the intuition step by step.*

**Step 1: What are we trying to capture?**

Consider the triple: ("Paris", "capital_of", "France")

We want embeddings such that:
- Similar entities are close together
- Relationships *transform* one entity into another

**Step 2: The translation intuition**

*Think about what "capital_of" means geometrically.*

If you're at "Paris" and want to get to "France", what direction should you go?

The relationship "capital_of" should act like a *translation vector* - a direction and magnitude that takes you from the head entity to the tail entity.

**Numerical Example**:
```
Let's use 2D embeddings for visualization:

Paris  = [1.0, 2.0]    (a city in France)
France = [3.0, 1.0]    (a country)

The relationship "capital_of" should be:
r = France - Paris = [3.0 - 1.0, 1.0 - 2.0] = [2.0, -1.0]

Verification:
Paris + capital_of = [1.0, 2.0] + [2.0, -1.0] = [3.0, 1.0] = France ✓
```

**Step 3: Why translation makes sense**

*Here's the beautiful part.* If the same relationship holds for multiple entities, they should use the *same* translation vector:

```
Berlin + capital_of ≈ Germany
Tokyo  + capital_of ≈ Japan
London + capital_of ≈ UK

All these use the SAME r vector!
```

This means:
- Cities that are capitals form a "cluster" in embedding space
- Countries form another "cluster"
- The "capital_of" relationship is the vector from the city cluster to the country cluster

**Step 4: The formal TransE formula**

$$h + r \approx t$$

where:
- $h$ = head entity embedding (e.g., "Paris")
- $r$ = relation embedding (e.g., "capital_of")
- $t$ = tail entity embedding (e.g., "France")

**Training objective**: Minimize distance for true triples, maximize for false ones

$$\mathcal{L} = \sum_{(h,r,t) \in \mathcal{T}} \sum_{(h',r,t') \in \mathcal{T}'} [\gamma + d(h+r, t) - d(h'+r, t')]_+$$

where:
- $\mathcal{T}$ = positive triples (true facts from KG)
- $\mathcal{T}'$ = negative triples (corrupted - either h or t replaced with random entity)
- $\gamma$ = margin (how much better should true triples score?)
- $d(\cdot, \cdot)$ = distance function (L1 or L2)
- $[\cdot]_+$ = max(0, ·) - only penalize when margin is violated

**Complete Numerical Walkthrough**:

```
Training example:
  Positive: ("Paris", "capital_of", "France")
  Negative: ("Paris", "capital_of", "Germany")  ← corrupted tail

Current embeddings:
  Paris   = [1.0, 2.0]
  France  = [3.0, 1.0]
  Germany = [4.0, 2.5]
  capital_of = [2.0, -1.0]

Compute distances:
  Positive: Paris + capital_of = [3.0, 1.0]
            d(prediction, France) = ||[3.0,1.0] - [3.0,1.0]|| = 0.0 ✓

  Negative: Paris + capital_of = [3.0, 1.0]
            d(prediction, Germany) = ||[3.0,1.0] - [4.0,2.5]||
                                   = sqrt(1 + 2.25) = 1.8

Margin loss (γ = 1.0):
  loss = max(0, 1.0 + 0.0 - 1.8) = max(0, -0.8) = 0

No loss! The model correctly ranks the true triple higher.
```

---

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransE(nn.Module):
    def __init__(self, n_entities, n_relations, embedding_dim=100, margin=1.0):
        """
        TransE model for knowledge graph embeddings.

        Args:
            n_entities: Number of entities in KG
            n_relations: Number of relation types
            embedding_dim: Dimensionality of embeddings
            margin: Margin for ranking loss
        """
        super().__init__()

        # Entity embeddings
        self.entity_embeddings = nn.Embedding(n_entities, embedding_dim)

        # Relation embeddings
        self.relation_embeddings = nn.Embedding(n_relations, embedding_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # Normalize entity embeddings
        self.entity_embeddings.weight.data = F.normalize(
            self.entity_embeddings.weight.data, p=2, dim=1
        )

        self.margin = margin

    def forward(self, heads, relations, tails):
        """
        Compute scores for triples.

        Args:
            heads: [batch_size] head entity IDs
            relations: [batch_size] relation IDs
            tails: [batch_size] tail entity IDs

        Returns:
            scores: [batch_size] negative distances (higher = more plausible)
        """
        # Get embeddings
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)

        # Compute h + r - t
        score = h + r - t

        # L2 distance
        distance = torch.norm(score, p=2, dim=1)

        # Return negative distance (higher = better)
        return -distance

    def loss(self, positive_triples, negative_triples):
        """
        Compute margin ranking loss.

        Args:
            positive_triples: [batch_size, 3] (head, relation, tail)
            negative_triples: [batch_size, 3] corrupted triples

        Returns:
            loss: Scalar loss value
        """
        # Positive scores
        pos_heads, pos_rels, pos_tails = positive_triples.t()
        pos_scores = self.forward(pos_heads, pos_rels, pos_tails)

        # Negative scores
        neg_heads, neg_rels, neg_tails = negative_triples.t()
        neg_scores = self.forward(neg_heads, neg_rels, neg_tails)

        # Margin ranking loss: max(0, margin + neg_score - pos_score)
        loss = F.relu(self.margin + neg_scores - pos_scores).mean()

        return loss


# Training
model = TransE(n_entities=10000, n_relations=50, embedding_dim=100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for positive_batch, negative_batch in kg_loader:
        # positive_batch: [batch_size, 3] triples from KG
        # negative_batch: [batch_size, 3] corrupted triples

        loss = model.loss(positive_batch, negative_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Re-normalize entity embeddings
        model.entity_embeddings.weight.data = F.normalize(
            model.entity_embeddings.weight.data, p=2, dim=1
        )

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")


# Use embeddings for recommendation
def get_entity_embedding(entity_id):
    """Get embedding for entity (user or item)."""
    return model.entity_embeddings(torch.tensor([entity_id])).detach()
```

---

### DistMult (Multiplicative Interactions)

**Idea**: Model relations as diagonal matrices.

**Score function**:
$$f(h, r, t) = h^T \text{diag}(r) t = \sum_i h_i \cdot r_i \cdot t_i$$

**Benefits**: Simpler than TransE, more efficient.

#### The DistMult Symmetry Problem: A Critical Limitation

*Let me show you a fundamental flaw in DistMult that you must understand.*

**Step 1: What is the DistMult score?**

$$f(h, r, t) = \sum_i h_i \cdot r_i \cdot t_i$$

*Notice anything about this formula?*

**Step 2: Let's compute the reverse triple**

What if we swap head and tail?

$$f(t, r, h) = \sum_i t_i \cdot r_i \cdot h_i$$

*Look carefully...* Multiplication is commutative! So:

$$f(h, r, t) = f(t, r, h)$$

**This means DistMult gives the SAME score to (h, r, t) and (t, r, h)!**

**Step 3: Why is this a problem?**

*Let's use a concrete example to see why this matters.*

```
Triple 1: ("Paris", "capital_of", "France")  ✓ TRUE
Triple 2: ("France", "capital_of", "Paris")  ✗ FALSE!

With DistMult:
  f(Paris, capital_of, France) = f(France, capital_of, Paris)

The model gives the SAME score to both!
It cannot distinguish the direction of the relationship.
```

**Numerical Example**:

```
Paris = [0.5, 0.8, -0.3]
France = [0.7, 0.2, 0.9]
capital_of = [1.0, 0.5, 0.5]

f(Paris, capital_of, France):
  = (0.5 × 1.0 × 0.7) + (0.8 × 0.5 × 0.2) + (-0.3 × 0.5 × 0.9)
  = 0.35 + 0.08 - 0.135
  = 0.295

f(France, capital_of, Paris):
  = (0.7 × 1.0 × 0.5) + (0.2 × 0.5 × 0.8) + (0.9 × 0.5 × -0.3)
  = 0.35 + 0.08 - 0.135
  = 0.295

Same score! DistMult cannot tell them apart.
```

**Step 4: Which relations are affected?**

| Relation Type | Direction Matters? | DistMult Works? |
|---------------|-------------------|-----------------|
| "married_to" | No (symmetric) | Yes ✓ |
| "sibling_of" | No (symmetric) | Yes ✓ |
| "capital_of" | Yes (asymmetric) | No ✗ |
| "directed_by" | Yes (asymmetric) | No ✗ |
| "works_for" | Yes (asymmetric) | No ✗ |

**Key insight**: DistMult only works for symmetric relations where (h, r, t) and (t, r, h) are both true or both false.

**Solutions**:
- Use ComplEx (complex-valued embeddings)
- Use RotatE (rotation in complex space)
- Use TransE for asymmetric relations

---

### Implementation

```python
class DistMult(nn.Module):
    def __init__(self, n_entities, n_relations, embedding_dim=100):
        super().__init__()

        self.entity_embeddings = nn.Embedding(n_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(n_relations, embedding_dim)

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, heads, relations, tails):
        """
        Compute scores for triples.

        Returns:
            scores: [batch_size] triple plausibility scores
        """
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)

        # Element-wise multiplication then sum
        score = (h * r * t).sum(dim=1)

        return score

    def loss(self, positive_triples, negative_triples):
        """Binary cross-entropy loss."""
        pos_heads, pos_rels, pos_tails = positive_triples.t()
        pos_scores = self.forward(pos_heads, pos_rels, pos_tails)

        neg_heads, neg_rels, neg_tails = negative_triples.t()
        neg_scores = self.forward(neg_heads, neg_rels, neg_tails)

        # Positive triples should have high scores, negatives low
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )

        return pos_loss + neg_loss
```

---

### ComplEx (Complex Embeddings)

**Idea**: Use complex-valued embeddings to model asymmetric relations.

**Score function**:
$$f(h, r, t) = \text{Re}(\langle h, r, \bar{t} \rangle)$$

where $\bar{t}$ = complex conjugate of $t$.

*Why does this work for asymmetric relations?*

The conjugate operation on $t$ breaks the symmetry! Now $f(h, r, t) \neq f(t, r, h)$ in general.

**Benefits**: Models both symmetric and asymmetric relations.

---

## KGAT (Knowledge Graph Attention Network)

### Architecture

**Idea**: Propagate user/item embeddings along KG using attention.

*But wait - why do we need attention?*

### KGAT Attention Step-by-Step

**The intuition**: Not all neighbors are equally important!

*Think about recommending movies to Alice:*
- Alice loved "Inception"
- "Inception" connects to: Nolan (director), DiCaprio (actor), Sci-Fi (genre), 2010 (year)

*Should all these neighbors influence the recommendation equally?*

No! The director connection might be more important for Alice (she loves Nolan films) than the year.

**Attention learns which connections matter most for each user/item.**

---

### Step-by-Step Numerical Example

*Let's walk through the attention computation for the movie "Inception" with 4 neighbors.*

**Setup**:
```
Entity: Inception
Neighbors with relations:
  1. Nolan (relation: directed_by)
  2. DiCaprio (relation: stars)
  3. Sci-Fi (relation: genre)
  4. 2010 (relation: year)

Embedding dimension: d = 4
```

**Initial embeddings** (random for illustration):
```
e_inception = [0.5, 0.3, -0.2, 0.8]
e_nolan     = [0.8, 0.2, 0.1, 0.7]
e_dicaprio  = [0.4, 0.6, -0.1, 0.3]
e_scifi     = [0.6, 0.1, 0.5, 0.4]
e_2010      = [0.2, 0.1, 0.0, 0.1]
```

**Step 1: Concatenate entity with each neighbor**

For each neighbor, we create: $[e_{entity} || e_{neighbor}]$

```
concat_1 = [e_inception || e_nolan]    = [0.5, 0.3, -0.2, 0.8, 0.8, 0.2, 0.1, 0.7]
concat_2 = [e_inception || e_dicaprio] = [0.5, 0.3, -0.2, 0.8, 0.4, 0.6, -0.1, 0.3]
concat_3 = [e_inception || e_scifi]    = [0.5, 0.3, -0.2, 0.8, 0.6, 0.1, 0.5, 0.4]
concat_4 = [e_inception || e_2010]     = [0.5, 0.3, -0.2, 0.8, 0.2, 0.1, 0.0, 0.1]
```

**Step 2: Apply relation-specific transformation**

Each relation has its own weight matrix $W_r$ that maps 2d → 1.

Let's say for relation "directed_by", $W_{directed\_by} = [0.5, 0.3, -0.1, 0.2, 0.4, 0.1, 0.2, 0.3]$

```
score_1 = W_directed_by · concat_1
        = 0.5×0.5 + 0.3×0.3 + (-0.1)×(-0.2) + 0.2×0.8
          + 0.4×0.8 + 0.1×0.2 + 0.2×0.1 + 0.3×0.7
        = 0.25 + 0.09 + 0.02 + 0.16 + 0.32 + 0.02 + 0.02 + 0.21
        = 1.09
```

Similarly compute for other neighbors (using their respective relation's W):
```
score_1 (Nolan, directed_by)    = 1.09
score_2 (DiCaprio, stars)       = 0.73
score_3 (Sci-Fi, genre)         = 0.91
score_4 (2010, year)            = 0.42
```

**Step 3: Apply LeakyReLU**

```
score_1 = LeakyReLU(1.09) = 1.09  (positive, unchanged)
score_2 = LeakyReLU(0.73) = 0.73
score_3 = LeakyReLU(0.91) = 0.91
score_4 = LeakyReLU(0.42) = 0.42
```

**Step 4: Softmax to get attention weights**

$$\alpha_i = \frac{\exp(\text{score}_i)}{\sum_j \exp(\text{score}_j)}$$

```
exp(1.09) = 2.97
exp(0.73) = 2.07
exp(0.91) = 2.48
exp(0.42) = 1.52

sum = 2.97 + 2.07 + 2.48 + 1.52 = 9.04

α_nolan    = 2.97 / 9.04 = 0.328  (32.8%)
α_dicaprio = 2.07 / 9.04 = 0.229  (22.9%)
α_scifi    = 2.48 / 9.04 = 0.274  (27.4%)
α_2010     = 1.52 / 9.04 = 0.168  (16.8%)
```

**Step 5: Weighted aggregation of neighbors**

$$e_{inception}^{(1)} = \sum_i \alpha_i \cdot e_{neighbor_i}$$

```
aggregated = 0.328 × [0.8, 0.2, 0.1, 0.7]    (Nolan)
           + 0.229 × [0.4, 0.6, -0.1, 0.3]   (DiCaprio)
           + 0.274 × [0.6, 0.1, 0.5, 0.4]    (Sci-Fi)
           + 0.168 × [0.2, 0.1, 0.0, 0.1]    (2010)

= [0.262, 0.066, 0.033, 0.230]   (Nolan contribution)
+ [0.092, 0.137, -0.023, 0.069]  (DiCaprio contribution)
+ [0.164, 0.027, 0.137, 0.110]   (Sci-Fi contribution)
+ [0.034, 0.017, 0.000, 0.017]   (2010 contribution)

= [0.552, 0.247, 0.147, 0.426]
```

**Interpretation**: The new embedding for "Inception" (after one KGAT layer) is most influenced by Nolan (32.8%) and Sci-Fi (27.4%), less by the year (16.8%).

*This makes sense!* The director and genre are more important for recommendations than when the movie was released.

---

### Attention mechanism formula

$$\alpha_{eh} = \frac{\exp(\text{LeakyReLU}(W_r [e || h]))}{\sum_{h' \in \mathcal{N}_e} \exp(\text{LeakyReLU}(W_r [e || h']))}$$

where:
- $e$ = entity embedding
- $h$ = neighbor entity
- $W_r$ = relation-specific weight matrix
- $||$ = concatenation

---

### Implementation

```python
class KGATLayer(nn.Module):
    def __init__(self, embedding_dim, n_relations):
        """
        KGAT aggregation layer.

        Args:
            embedding_dim: Dimensionality of entity embeddings
            n_relations: Number of relation types
        """
        super().__init__()

        # Relation-specific transformation matrices
        self.W_r = nn.ModuleList([
            nn.Linear(embedding_dim * 2, 1)
            for _ in range(n_relations)
        ])

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, entity_emb, neighbors, relations):
        """
        Aggregate neighbor information with attention.

        Args:
            entity_emb: [batch_size, embedding_dim]
            neighbors: [batch_size, n_neighbors, embedding_dim]
            relations: [batch_size, n_neighbors] relation types

        Returns:
            aggregated_emb: [batch_size, embedding_dim]
        """
        batch_size, n_neighbors, emb_dim = neighbors.size()

        # Expand entity embedding to match neighbors
        entity_expanded = entity_emb.unsqueeze(1).expand(batch_size, n_neighbors, emb_dim)

        # Concatenate entity and neighbor embeddings
        concat = torch.cat([entity_expanded, neighbors], dim=2)  # [batch, n_neighbors, 2*emb_dim]

        # Compute attention scores for each neighbor
        attention_scores = []

        for i in range(batch_size):
            scores = []
            for j in range(n_neighbors):
                relation_type = relations[i, j].item()

                # Apply relation-specific transformation
                score = self.W_r[relation_type](concat[i, j])
                scores.append(score)

            scores = torch.stack(scores)  # [n_neighbors, 1]
            attention_scores.append(scores)

        attention_scores = torch.stack(attention_scores).squeeze(-1)  # [batch, n_neighbors]

        # Apply LeakyReLU and softmax
        attention_scores = self.leaky_relu(attention_scores)
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, n_neighbors]

        # Weighted sum of neighbor embeddings
        attention_weights = attention_weights.unsqueeze(2)  # [batch, n_neighbors, 1]
        aggregated = (neighbors * attention_weights).sum(dim=1)  # [batch, emb_dim]

        return aggregated


class KGAT(nn.Module):
    def __init__(self, n_users, n_items, n_entities, n_relations, embedding_dim=64, n_layers=3):
        """
        KGAT model for knowledge-aware recommendations.

        Args:
            n_users: Number of users
            n_items: Number of items
            n_entities: Number of entities in KG (including users and items)
            n_relations: Number of relation types
            embedding_dim: Dimensionality of embeddings
            n_layers: Number of KGAT layers
        """
        super().__init__()

        # Entity embeddings (users, items, and KG entities)
        self.entity_embeddings = nn.Embedding(n_entities, embedding_dim)

        # KGAT layers
        self.layers = nn.ModuleList([
            KGATLayer(embedding_dim, n_relations)
            for _ in range(n_layers)
        ])

        # Prediction layer
        self.predict = nn.Linear(embedding_dim * (n_layers + 1), 1)

        nn.init.xavier_uniform_(self.entity_embeddings.weight)

    def forward(self, user_ids, item_ids, kg_graph):
        """
        Predict user-item preferences.

        Args:
            user_ids: [batch_size] user IDs
            item_ids: [batch_size] item IDs
            kg_graph: Knowledge graph structure (neighbors and relations)

        Returns:
            scores: [batch_size] predicted scores
        """
        # Get initial embeddings
        user_emb_0 = self.entity_embeddings(user_ids)
        item_emb_0 = self.entity_embeddings(item_ids)

        # Collect embeddings from each layer
        user_embs = [user_emb_0]
        item_embs = [item_emb_0]

        # Propagate through KGAT layers
        user_emb = user_emb_0
        item_emb = item_emb_0

        for layer in self.layers:
            # Get neighbors and relations from KG
            user_neighbors, user_relations = kg_graph.get_neighbors(user_ids)
            item_neighbors, item_relations = kg_graph.get_neighbors(item_ids)

            # Aggregate with attention
            user_emb = layer(user_emb, user_neighbors, user_relations)
            item_emb = layer(item_emb, item_neighbors, item_relations)

            user_embs.append(user_emb)
            item_embs.append(item_emb)

        # Concatenate embeddings from all layers
        user_final = torch.cat(user_embs, dim=1)
        item_final = torch.cat(item_embs, dim=1)

        # Element-wise product
        interaction = user_final * item_final

        # Prediction
        score = self.predict(interaction).squeeze()

        return score


# Training
model = KGAT(
    n_users=10000,
    n_items=5000,
    n_entities=20000,  # Users + items + KG entities
    n_relations=10,
    embedding_dim=64,
    n_layers=3
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(100):
    for user_ids, item_ids_pos, item_ids_neg, kg_graph in train_loader:
        # Positive samples
        pos_scores = model(user_ids, item_ids_pos, kg_graph)
        pos_labels = torch.ones_like(pos_scores)

        # Negative samples
        neg_scores = model(user_ids, item_ids_neg, kg_graph)
        neg_labels = torch.zeros_like(neg_scores)

        # Loss
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([pos_labels, neg_labels])
        loss = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

---

## KG-Enhanced Collaborative Filtering

### CKE (Collaborative Knowledge Base Embedding)

**Idea**: Jointly learn from user-item interactions and KG.

**Loss function**:
$$\mathcal{L} = \mathcal{L}_{CF} + \lambda_K \mathcal{L}_{KG}$$

where:
- $\mathcal{L}_{CF}$ = collaborative filtering loss (e.g., BPR)
- $\mathcal{L}_{KG}$ = knowledge graph embedding loss (e.g., TransE)
- $\lambda_K$ = weight balancing CF and KG

---

### Implementation

```python
class CKE(nn.Module):
    def __init__(self, n_users, n_items, n_entities, n_relations, embedding_dim=64):
        super().__init__()

        # User embeddings (CF component)
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)

        # Entity embeddings (includes items + KG entities)
        self.entity_embeddings = nn.Embedding(n_entities, embedding_dim)

        # Relation embeddings (KG component)
        self.relation_embeddings = nn.Embedding(n_relations, embedding_dim)

        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def cf_loss(self, user_ids, item_ids_pos, item_ids_neg):
        """
        Collaborative filtering loss (BPR).
        """
        user_emb = self.user_embeddings(user_ids)
        item_emb_pos = self.entity_embeddings(item_ids_pos)
        item_emb_neg = self.entity_embeddings(item_ids_neg)

        # Positive and negative scores
        pos_scores = (user_emb * item_emb_pos).sum(dim=1)
        neg_scores = (user_emb * item_emb_neg).sum(dim=1)

        # BPR loss
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        return loss

    def kg_loss(self, heads, relations, tails):
        """
        Knowledge graph loss (TransE).
        """
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)

        # TransE: h + r ≈ t
        distance = torch.norm(h + r - t, p=2, dim=1)

        # Minimize distance for positive triples
        loss = distance.mean()

        return loss

    def forward(self, user_ids, item_ids_pos, item_ids_neg, kg_triples, lambda_kg=0.1):
        """
        Joint training loss.
        """
        # CF loss
        loss_cf = self.cf_loss(user_ids, item_ids_pos, item_ids_neg)

        # KG loss
        heads, relations, tails = kg_triples.t()
        loss_kg = self.kg_loss(heads, relations, tails)

        # Total loss
        total_loss = loss_cf + lambda_kg * loss_kg

        return total_loss, loss_cf, loss_kg


# Training
model = CKE(n_users=10000, n_items=5000, n_entities=20000, n_relations=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for user_ids, item_pos, item_neg, kg_triples in train_loader:
        loss, loss_cf, loss_kg = model(user_ids, item_pos, item_neg, kg_triples, lambda_kg=0.1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Total={loss:.4f}, CF={loss_cf:.4f}, KG={loss_kg:.4f}")
```

---

## Explainable Recommendations with KG

### Path-Based Explanations: The Intuition

*Why do knowledge graphs make recommendations explainable?*

**The key insight**: Paths in a KG tell a *story* connecting the user to the item.

Consider this path:
```
User → liked → Movie1 → directed_by → Director → directed → Movie2
```

This path naturally translates to:
> "We recommend Movie2 because you liked Movie1, which was directed by Director, who also directed Movie2"

**This is something pure collaborative filtering cannot do!** CF just says "similar users liked Movie2" - no narrative, no reasoning.

---

### Path Quality and Relevance

*Not all paths are equally good explanations. Let me show you why.*

**Path Quality Factors**:

1. **Path Length**: Shorter = more relevant
   - Length 2: User → liked → MovieA → genre → Sci-Fi → genre → MovieB
   - This is a stretch! "Both are Sci-Fi" isn't compelling.

2. **Relation Types**: Some relations are more meaningful
   - Good: "directed_by", "stars", "sequel_to"
   - Weak: "year", "production_company"

3. **Entity Popularity**: Rare connections are more interesting
   - Weak: "Both have Tom Hanks" (he's in many movies)
   - Strong: "Both directed by Denis Villeneuve" (more distinctive)

**Example Path Ranking**:

```
User Alice liked "Inception"
Candidate: "Interstellar"

Path 1: Inception → directed_by → Nolan → directed → Interstellar
        Quality: HIGH (same director, length 2)

Path 2: Inception → genre → Sci-Fi → genre → Interstellar
        Quality: MEDIUM (same genre, but Sci-Fi is common)

Path 3: Inception → year → 2010 → year → Shutter Island
        Quality: LOW (same year is coincidental)

Path 4: Inception → studio → Warner Bros → studio → Interstellar
        Quality: LOW (many movies share studio)
```

---

### Example Explanation Generation

```
User: Alice
Recommended: "Interstellar"

Explanation:
"We recommend 'Interstellar' because:
 1. You loved 'Inception' (rated 5 stars)
 2. 'Inception' was directed by Christopher Nolan
 3. Christopher Nolan also directed 'Interstellar'
 4. Both films are in the 'Mind-bending Sci-Fi' genre you enjoy"

Supporting evidence:
- You've rated 4 other Nolan films (avg: 4.5 stars)
- 87% of users who liked 'Inception' also liked 'Interstellar'
```

---

### Implementation

```python
def find_explanation_path(user_id, item_id, kg_graph, max_length=3):
    """
    Find path between user and item in KG.

    Args:
        user_id: User entity ID
        item_id: Item entity ID
        kg_graph: Knowledge graph structure
        max_length: Maximum path length

    Returns:
        path: List of (entity, relation) tuples
    """
    from collections import deque

    # BFS to find shortest path
    queue = deque([(user_id, [])])
    visited = {user_id}

    while queue:
        current_entity, path = queue.popleft()

        if current_entity == item_id:
            return path

        if len(path) >= max_length:
            continue

        # Explore neighbors
        for neighbor, relation in kg_graph.get_neighbors(current_entity):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [(current_entity, relation, neighbor)]
                queue.append((neighbor, new_path))

    return None  # No path found


def generate_explanation(user_id, item_id, kg_graph, entity_names):
    """
    Generate natural language explanation.
    """
    path = find_explanation_path(user_id, item_id, kg_graph)

    if path is None:
        return "Recommended based on your preferences"

    explanation = "We recommend this because: "

    for i, (entity, relation, next_entity) in enumerate(path):
        if i == 0:
            explanation += f"you {relation} {entity_names[entity]}"
        else:
            explanation += f", which {relation} {entity_names[next_entity]}"

    return explanation


# Example
user_id = 1234
item_id = 5678
kg_graph = load_kg()
entity_names = load_entity_names()

explanation = generate_explanation(user_id, item_id, kg_graph, entity_names)
print(explanation)
# Output: "We recommend this because: you liked The Matrix, which was directed by Wachowskis, which also directed Inception"
```

---

## What Can Go Wrong?

*Let me warn you about common failure modes when using knowledge graphs for recommendations.*

### Failure Mode 1: Incomplete Knowledge Graph

**Problem**: Your KG is missing important entities or relations.

**Symptoms**:
- Cold start persists for items that should have KG data
- Paths end prematurely
- Explanations are generic or missing

**Example**:
```
Missing: (Inception, composer, Hans Zimmer)

Without this:
- Can't connect Inception to Interstellar via Hans Zimmer
- Lose a powerful signal for soundtrack-lovers
```

**Solutions**:
- Entity linking to external KGs (Wikidata, Freebase)
- Automatic relation extraction from text
- Crowdsourced knowledge completion
- Confidence scores for uncertain triples

---

### Failure Mode 2: Noisy or Incorrect Relations

**Problem**: KG contains errors or outdated information.

**Symptoms**:
- Wrong recommendations based on false facts
- Explanations that don't make sense
- User trust erodes

**Example**:
```
Incorrect: (Matrix, director, James Cameron)  ← Should be Wachowskis!

Result: Recommending "Titanic" to Matrix fans
        "Because the same director made both"
        (Wrong and confusing!)
```

**Solutions**:
- Confidence-weighted embedding learning
- Anomaly detection for suspicious triples
- User feedback to flag errors
- Regular validation against trusted sources

---

### Failure Mode 3: Popularity Bias in Graph Structure

**Problem**: Popular entities have many more connections, dominating recommendations.

**Symptoms**:
- Same popular items recommended via many paths
- Long-tail items get ignored despite good matches
- Diversity suffers

**Example**:
```
Tom Hanks appears in 50+ movies in KG
   → Any user who likes one Tom Hanks movie
   → Gets recommended all 50 via "stars" relation

Meanwhile: Indie actor with 3 films
   → Never gets recommended despite great fit
```

**Solutions**:
- Path diversity: Limit recommendations via same relation
- Inverse popularity weighting in attention
- Explicit diversity constraints
- Separate embeddings for high vs. low-degree entities

---

### Failure Mode 4: Relation Conflation

**Problem**: Different relation types treated as equivalent.

**Symptoms**:
- "wrote" and "acted_in" connections weighted equally
- Genre connections dominate over more specific ones
- Quality of explanations varies wildly

**Example**:
```
Good path: Inception → director → Nolan → director → Dunkirk
           (Strong signal: same director)

Bad path:  Inception → year → 2010 → year → Toy Story 3
           (Weak signal: same year is coincidence)

If model treats both equally, recommendations suffer.
```

**Solutions**:
- Relation-specific attention weights (KGAT does this!)
- Learn relation importance from user feedback
- Filter low-quality relation types
- Hierarchical relation modeling

---

### Failure Mode 5: Path Explosion

**Problem**: Exponential number of paths between entities.

**Symptoms**:
- Slow inference (exploring too many paths)
- Memory issues with large KGs
- Difficulty selecting "best" path for explanation

**Example**:
```
User connected to 100 items
Each item connected to 50 entities
Each entity connected to 20 more

Paths of length 3: 100 × 50 × 20 = 100,000 paths!
```

**Solutions**:
- Beam search (keep top-K paths)
- Reinforcement learning for path finding
- Pre-compute important meta-paths
- Graph sampling techniques

---

### Failure Mode 6: Cold Entities in KG

**Problem**: Entities with few connections don't get meaningful embeddings.

**Symptoms**:
- New entities have random-like embeddings
- Can't leverage KG for truly new items
- Falls back to pure CF (which also fails for new items)

**Example**:
```
New movie just added to KG:
- Only connection: (NewMovie, year, 2024)
- No director, actors, or genre yet

TransE embedding: Basically random
KGAT: Only propagates from "2024" (useless)
```

**Solutions**:
- Content features as fallback (title, description)
- Zero-shot entity embeddings from text
- Active knowledge acquisition
- Hybrid: CF + KG + Content

---

## Summary

**Key Takeaways**:
1. **TransE**: Relationships as translations (h + r ≈ t) - works best for 1-to-1 relations
2. **DistMult**: Multiplicative interactions - only works for symmetric relations!
3. **ComplEx**: Complex embeddings for asymmetric relations
4. **KGAT**: Attention-based propagation along KG - learns which connections matter
5. **CKE**: Joint training on CF + KG - best of both worlds
6. **Explainability**: Path-based explanations via KG - tell a story!

**Benefits of KG**:
- Alleviate sparsity (use attributes when interactions are sparse)
- Cold start for new items (leverage known relations)
- Explainable recommendations (path narratives)
- Incorporate domain knowledge

**What Can Go Wrong**:
- Incomplete KG (missing entities/relations)
- Noisy/incorrect relations
- Popularity bias in graph structure
- Relation conflation
- Path explosion
- Cold entities in KG

**Best practices**:
- Use TransE for simple, mostly 1-to-1 relations
- Use KGAT for complex, heterogeneous KGs
- Balance CF and KG losses ($\lambda_K = 0.01-0.1$)
- Generate explanations for transparency
- Validate KG quality regularly
- Handle cold entities with content fallback

---

## Practice Problems

**Problem 1**: Implement TransE on MovieLens + KG (genres, actors, directors). Evaluate link prediction accuracy.

**Problem 2**: Compare TransE vs. DistMult vs. ComplEx on the same KG. Which performs best? Analyze whether it correlates with relation symmetry.

**Problem 3**: Implement KGAT for movie recommendations. Does it improve over standard CF (no KG)?

**Problem 4**: Generate path-based explanations for top-10 recommendations. What fraction of recommendations have valid KG paths? How does path length affect explanation quality?

**Problem 5**: Analyze the DistMult symmetry problem on your KG. What fraction of relations are symmetric vs. asymmetric? How does this affect prediction quality?

---

## References

1. **Bordes, A., et al. (2013)**. "Translating Embeddings for Modeling Multi-relational Data". *NeurIPS* (TransE).

2. **Yang, B., et al. (2015)**. "Embedding Entities and Relations for Learning and Inference in Knowledge Bases". *ICLR* (DistMult).

3. **Trouillon, T., et al. (2016)**. "Complex Embeddings for Simple Link Prediction". *ICML* (ComplEx).

4. **Wang, X., et al. (2019)**. "KGAT: Knowledge Graph Attention Network for Recommendation". *KDD*.

5. **Zhang, F., et al. (2016)**. "Collaborative Knowledge Base Embedding for Recommender Systems". *KDD* (CKE).

6. **Ai, Q., et al. (2018)**. "Learning Heterogeneous Knowledge Base Embeddings for Explainable Recommendation". *Algorithms* (Explainable KG Recommendations).
