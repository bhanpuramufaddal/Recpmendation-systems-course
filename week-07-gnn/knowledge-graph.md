# Week 7: Knowledge Graph Integration

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

## Knowledge Graph Embeddings

### TransE (Translating Embeddings)

**Idea**: Relationships as translations in embedding space.

**Formula**:
$$h + r \approx t$$

where:
- $h$ = head entity embedding
- $r$ = relation embedding
- $t$ = tail entity embedding

**Training objective**: Minimize distance
$$\mathcal{L} = \sum_{(h,r,t) \in \mathcal{T}} \sum_{(h',r,t') \in \mathcal{T}'} [\gamma + d(h+r, t) - d(h'+r, t')]_+$$

where:
- $\mathcal{T}$ = positive triples
- $\mathcal{T}'$ = negative triples (corrupted)
- $\gamma$ = margin
- $d(\cdot, \cdot)$ = distance function (L1 or L2)

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

**Limitation**: Can only model symmetric relations.

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

**Benefits**: Models both symmetric and asymmetric relations.

---

## KGAT (Knowledge Graph Attention Network)

### Architecture

**Idea**: Propagate user/item embeddings along KG using attention.

**Components**:
1. **Embedding layer**: Initialize user/item embeddings
2. **Attention aggregation**: Aggregate neighbor information with attention
3. **Prediction layer**: Compute recommendation scores

**Attention mechanism**:
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

### Path-Based Explanations

**Idea**: Explain recommendations via paths in KG.

**Example**:
```
User -> liked -> The Matrix
The Matrix -> directed_by -> Wachowskis
Wachowskis -> directed -> Inception
→ Recommend Inception because: "You liked The Matrix, which was directed by the Wachowskis, who also directed Inception"
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

## Summary

**Key Takeaways**:
1. **TransE**: Relationships as translations (h + r ≈ t)
2. **DistMult**: Multiplicative interactions (symmetric relations)
3. **ComplEx**: Complex embeddings (asymmetric relations)
4. **KGAT**: Attention-based propagation along KG
5. **CKE**: Joint training on CF + KG
6. **Explainability**: Path-based explanations via KG

**Benefits of KG**:
- Alleviate sparsity
- Cold start for new items
- Explainable recommendations
- Incorporate domain knowledge

**Best practices**:
- Use TransE for simple KGs
- Use KGAT for complex, heterogeneous KGs
- Balance CF and KG losses (λ_K = 0.01-0.1)
- Generate explanations for transparency

---

## Practice Problems

**Problem 1**: Implement TransE on MovieLens + KG (genres, actors, directors). Evaluate link prediction accuracy.

**Problem 2**: Compare TransE vs. DistMult vs. ComplEx on the same KG. Which performs best?

**Problem 3**: Implement KGAT for movie recommendations. Does it improve over standard CF (no KG)?

**Problem 4**: Generate path-based explanations for top-10 recommendations. What fraction of recommendations have valid KG paths?

---

## References

1. **Bordes, A., et al. (2013)**. "Translating Embeddings for Modeling Multi-relational Data". *NeurIPS* (TransE).

2. **Yang, B., et al. (2015)**. "Embedding Entities and Relations for Learning and Inference in Knowledge Bases". *ICLR* (DistMult).

3. **Trouillon, T., et al. (2016)**. "Complex Embeddings for Simple Link Prediction". *ICML* (ComplEx).

4. **Wang, X., et al. (2019)**. "KGAT: Knowledge Graph Attention Network for Recommendation". *KDD*.

5. **Zhang, F., et al. (2016)**. "Collaborative Knowledge Base Embedding for Recommender Systems". *KDD* (CKE).
