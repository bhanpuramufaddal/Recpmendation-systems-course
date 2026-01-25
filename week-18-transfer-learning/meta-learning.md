# Week 18: Meta-Learning for Recommendations

## Overview

**Meta-learning**: "Learning to learn" - train model to adapt quickly to new tasks/users with limited data.

**Motivation**:
- **Cold-start users**: New users with few interactions
- **Personalization**: Each user is a task
- **Few-shot adaptation**: Learn from 5-10 interactions

**Key idea**: Learn initialization that enables fast adaptation.

---

## Problem Formulation

### Tasks in Recommendation

**Task**: Recommend items to a single user.

**Meta-learning setup**:
- **Training**: Learn from many users (tasks)
- **Adaptation**: Fine-tune on new user with few interactions
- **Goal**: Good recommendations after few-shot adaptation

---

### MAML (Model-Agnostic Meta-Learning)

**Intuition**: Find model parameters that are close to optimal for all tasks.

**Algorithm**:
1. Initialize model parameters $\theta$
2. For each task (user):
   - Sample support set (few interactions)
   - Compute task-specific parameters: $\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_i(\theta)$
   - Evaluate on query set
3. Update meta-parameters: $\theta \leftarrow \theta - \beta \nabla_\theta \sum_i \mathcal{L}_i(\theta_i')$

**Key**: Optimize for fast adaptation, not final performance.

---

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RecommenderModel(nn.Module):
    def __init__(self, n_items, embedding_dim=64):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, item_ids):
        item_emb = self.item_embedding(item_ids)
        x = torch.relu(self.fc1(item_emb))
        x = torch.relu(self.fc2(x))
        score = self.fc3(x)
        return score.squeeze()


class MAML:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, n_inner_steps=5):
        """
        Args:
            model: Recommender model
            inner_lr: Learning rate for task adaptation
            outer_lr: Learning rate for meta-update
            n_inner_steps: Number of gradient steps for adaptation
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.n_inner_steps = n_inner_steps

        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)

    def adapt(self, support_items, support_labels):
        """
        Adapt model to task using support set.

        Args:
            support_items: Items user interacted with
            support_labels: Ratings or binary labels

        Returns:
            adapted_params: Task-specific parameters
        """
        # Clone model parameters
        adapted_params = [p.clone() for p in self.model.parameters()]

        for step in range(self.n_inner_steps):
            # Forward pass
            predictions = self.model(support_items)

            # Compute loss
            loss = F.mse_loss(predictions, support_labels)

            # Compute gradients
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

            # Update parameters
            adapted_params = [p - self.inner_lr * g for p, g in zip(adapted_params, grads)]

            # Update model parameters for next iteration
            for param, adapted_param in zip(self.model.parameters(), adapted_params):
                param.data = adapted_param.data

        return adapted_params

    def meta_train(self, task_batch):
        """
        Meta-training step on batch of tasks.

        Args:
            task_batch: List of tasks, each with support and query sets
        """
        meta_loss = 0

        for task in task_batch:
            support_items, support_labels = task['support']
            query_items, query_labels = task['query']

            # Adapt to task
            adapted_params = self.adapt(support_items, support_labels)

            # Evaluate on query set with adapted parameters
            query_predictions = self.model(query_items)
            task_loss = F.mse_loss(query_predictions, query_labels)

            meta_loss += task_loss

        # Meta-update
        meta_loss /= len(task_batch)

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def fine_tune(self, user_items, user_labels, n_steps=10):
        """
        Fine-tune on new user.

        Args:
            user_items: Few items user interacted with
            user_labels: User's ratings
            n_steps: Number of fine-tuning steps
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)

        for step in range(n_steps):
            predictions = self.model(user_items)
            loss = F.mse_loss(predictions, user_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self.model


# Training example
model = RecommenderModel(n_items=10000, embedding_dim=64)
maml = MAML(model, inner_lr=0.01, outer_lr=0.001, n_inner_steps=5)

# Meta-training
for epoch in range(100):
    # Sample batch of tasks (users)
    task_batch = sample_tasks(train_users, n_tasks=32)

    meta_loss = maml.meta_train(task_batch)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Meta-loss = {meta_loss:.4f}")

# Few-shot adaptation for new user
new_user_items = torch.tensor([10, 25, 47, 103, 256])  # Items user liked
new_user_labels = torch.tensor([5.0, 4.5, 5.0, 4.0, 4.5])  # Ratings

adapted_model = maml.fine_tune(new_user_items, new_user_labels, n_steps=10)

# Generate recommendations
all_items = torch.arange(10000)
scores = adapted_model(all_items)
top_k_items = torch.topk(scores, k=10).indices

print(f"Top-10 recommendations for new user: {top_k_items}")
```

---

## Prototypical Networks

### Idea

**Learn metric space** where items close to user's liked items are recommended.

**Algorithm**:
1. For each user, compute prototype: $c = \frac{1}{|S|} \sum_{x \in S} f_\theta(x)$
2. Measure distance to prototype: $d(f_\theta(x), c)$
3. Recommend items with smallest distance

**Benefits**: Simple, interpretable, works well for few-shot.

---

### Implementation

```python
class PrototypicalRecommender(nn.Module):
    def __init__(self, n_items, embedding_dim=64):
        super().__init__()
        self.item_encoder = nn.Sequential(
            nn.Embedding(n_items, embedding_dim),
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, item_ids):
        """Encode items to embedding space."""
        return self.item_encoder(item_ids)

    def compute_prototype(self, support_items):
        """
        Compute prototype from support set.

        Args:
            support_items: Items user liked

        Returns:
            prototype: Mean embedding of support items
        """
        embeddings = self.forward(support_items)
        prototype = embeddings.mean(dim=0)
        return prototype

    def predict(self, query_items, prototype):
        """
        Predict scores based on distance to prototype.

        Args:
            query_items: Items to score
            prototype: User prototype

        Returns:
            scores: Negative distance to prototype (higher = better)
        """
        query_embeddings = self.forward(query_items)

        # Euclidean distance
        distances = torch.norm(query_embeddings - prototype, dim=1)

        # Negative distance (higher score = closer to prototype)
        scores = -distances

        return scores


# Training
model = PrototypicalRecommender(n_items=10000, embedding_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for user_id in train_users:
        # Get user's liked items
        liked_items = get_liked_items(user_id)

        # Split into support and query
        support_items = liked_items[:5]
        query_items_pos = liked_items[5:10]
        query_items_neg = sample_negative_items(n=5)

        # Compute prototype
        prototype = model.compute_prototype(support_items)

        # Predict scores
        scores_pos = model.predict(query_items_pos, prototype)
        scores_neg = model.predict(query_items_neg, prototype)

        # Contrastive loss (positive items should be closer than negative)
        loss = F.relu(1.0 + scores_neg.mean() - scores_pos.mean())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Few-shot recommendation for new user
new_user_liked = torch.tensor([10, 25, 47, 103, 256])
prototype = model.compute_prototype(new_user_liked)

# Score all items
all_items = torch.arange(10000)
scores = model.predict(all_items, prototype)

# Top-K
top_k_items = torch.topk(scores, k=10).indices
print(f"Recommendations: {top_k_items}")
```

---

## Matching Networks

### Architecture

**Idea**: Attend to support set when making predictions.

**Attention mechanism**:
$$\hat{y} = \sum_{(x_i, y_i) \in S} a(x, x_i) y_i$$

where $a(x, x_i)$ = attention weight (similarity between query $x$ and support $x_i$).

---

### Implementation

```python
class MatchingNetworkRecommender(nn.Module):
    def __init__(self, n_items, embedding_dim=64):
        super().__init__()
        self.item_encoder = nn.Sequential(
            nn.Embedding(n_items, embedding_dim),
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)

    def forward(self, query_items, support_items, support_labels):
        """
        Predict query labels using support set.

        Args:
            query_items: Items to score
            support_items: Items user liked
            support_labels: User's ratings for support items
        """
        # Encode items
        query_emb = self.item_encoder(query_items)  # [n_query, 64]
        support_emb = self.item_encoder(support_items)  # [n_support, 64]

        # Attention: query attends to support
        attn_output, attn_weights = self.attention(
            query_emb.unsqueeze(1),  # [n_query, 1, 64]
            support_emb.unsqueeze(1),  # [n_support, 1, 64]
            support_emb.unsqueeze(1)
        )

        # Weighted average of support labels
        attn_weights = attn_weights.squeeze()  # [n_query, n_support]
        predictions = torch.matmul(attn_weights, support_labels.unsqueeze(1)).squeeze()

        return predictions


# Training
model = MatchingNetworkRecommender(n_items=10000, embedding_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for user_id in train_users:
        # Get user's interactions
        liked_items = get_liked_items(user_id)
        ratings = get_ratings(user_id, liked_items)

        # Split into support and query
        support_items = liked_items[:5]
        support_labels = ratings[:5]

        query_items = liked_items[5:10]
        query_labels = ratings[5:10]

        # Predict
        predictions = model(query_items, support_items, support_labels)

        # Loss
        loss = F.mse_loss(predictions, query_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Few-shot recommendation
new_user_items = torch.tensor([10, 25, 47, 103, 256])
new_user_ratings = torch.tensor([5.0, 4.5, 5.0, 4.0, 4.5])

# Predict for all items
all_items = torch.arange(10000)
predictions = model(all_items, new_user_items, new_user_ratings)

# Top-K
top_k_items = torch.topk(predictions, k=10).indices
print(f"Recommendations: {top_k_items}")
```

---

## Personalized Initialization

### Task-Specific Initialization

**Idea**: Initialize each user with personalized parameters.

**Approach**:
1. Cluster users by behavior
2. Learn cluster-specific initialization
3. New user → assign to cluster → use cluster initialization

---

### Implementation

```python
from sklearn.cluster import KMeans
import numpy as np

class PersonalizedInitializer:
    def __init__(self, n_clusters=10):
        """
        Args:
            n_clusters: Number of user clusters
        """
        self.n_clusters = n_clusters
        self.cluster_params = {}
        self.kmeans = KMeans(n_clusters=n_clusters)

    def fit(self, users, user_behaviors):
        """
        Cluster users and learn cluster-specific parameters.

        Args:
            users: List of user IDs
            user_behaviors: User behavior vectors [n_users, n_features]
        """
        # Cluster users
        clusters = self.kmeans.fit_predict(user_behaviors)

        # Train separate model for each cluster
        for cluster_id in range(self.n_clusters):
            cluster_users = [u for u, c in zip(users, clusters) if c == cluster_id]

            # Train model on cluster users
            cluster_model = train_model_on_users(cluster_users)

            # Store parameters
            self.cluster_params[cluster_id] = cluster_model.state_dict()

    def get_initialization(self, new_user_behavior):
        """
        Get personalized initialization for new user.

        Args:
            new_user_behavior: Behavior vector for new user

        Returns:
            init_params: Model parameters for initialization
        """
        # Assign to cluster
        cluster_id = self.kmeans.predict([new_user_behavior])[0]

        # Return cluster parameters
        return self.cluster_params[cluster_id]

    def fine_tune(self, new_user_items, new_user_labels, init_params):
        """
        Fine-tune from personalized initialization.

        Args:
            new_user_items: Items user interacted with
            new_user_labels: User's ratings
            init_params: Initialization parameters
        """
        # Create model with initialization
        model = RecommenderModel(n_items=10000, embedding_dim=64)
        model.load_state_dict(init_params)

        # Fine-tune
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for step in range(10):
            predictions = model(new_user_items)
            loss = F.mse_loss(predictions, new_user_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model


# Usage
initializer = PersonalizedInitializer(n_clusters=10)

# Learn cluster-specific initializations
train_users = get_train_users()
train_behaviors = extract_user_behaviors(train_users)  # [n_users, n_features]

initializer.fit(train_users, train_behaviors)

# Adapt to new user
new_user_behavior = extract_user_behavior(new_user)
init_params = initializer.get_initialization(new_user_behavior)

new_user_items = torch.tensor([10, 25, 47])
new_user_labels = torch.tensor([5.0, 4.5, 5.0])

adapted_model = initializer.fine_tune(new_user_items, new_user_labels, init_params)
```

---

## Cold-Start Alleviation

### Comparison with Baselines

**Baselines**:
1. **Popularity**: Recommend popular items
2. **Content-based**: Use item features only
3. **Random**: Random recommendations

**Meta-learning advantage**: Leverages patterns from existing users.

---

### Experimental Results

```python
def evaluate_cold_start(model_type, n_shots=[1, 3, 5, 10]):
    """
    Evaluate cold-start performance with varying number of interactions.

    Args:
        model_type: 'maml', 'prototypical', 'matching', 'popularity', 'random'
        n_shots: Number of user interactions to adapt from
    """
    results = {}

    for n_shot in n_shots:
        ndcg_scores = []

        for user in test_users:
            # Get user's first n_shot interactions
            user_items = get_user_items(user)[:n_shot]
            user_labels = get_user_labels(user)[:n_shot]

            # Adapt model
            if model_type == 'maml':
                model = maml.fine_tune(user_items, user_labels, n_steps=5)
            elif model_type == 'prototypical':
                prototype = prototypical_model.compute_prototype(user_items)
            elif model_type == 'matching':
                # Use matching network
                pass
            elif model_type == 'popularity':
                # Use popularity baseline
                model = popularity_model
            elif model_type == 'random':
                # Random recommendations
                model = random_model

            # Evaluate on held-out items
            held_out_items = get_user_items(user)[n_shot:]
            held_out_labels = get_user_labels(user)[n_shot:]

            # Compute NDCG
            ndcg = compute_ndcg(model, held_out_items, held_out_labels, k=10)
            ndcg_scores.append(ndcg)

        results[n_shot] = np.mean(ndcg_scores)

    return results


# Evaluate all methods
methods = ['maml', 'prototypical', 'matching', 'popularity', 'random']

for method in methods:
    results = evaluate_cold_start(method, n_shots=[1, 3, 5, 10])

    print(f"\n{method.upper()} Results:")
    for n_shot, ndcg in results.items():
        print(f"  {n_shot}-shot: NDCG@10 = {ndcg:.4f}")

# Example output:
# MAML Results:
#   1-shot: NDCG@10 = 0.2145
#   3-shot: NDCG@10 = 0.2987
#   5-shot: NDCG@10 = 0.3456
#   10-shot: NDCG@10 = 0.3892
#
# POPULARITY Results:
#   1-shot: NDCG@10 = 0.1523
#   3-shot: NDCG@10 = 0.1523
#   5-shot: NDCG@10 = 0.1523
#   10-shot: NDCG@10 = 0.1523
```

---

## Summary

**Key Takeaways**:
1. **MAML**: Learn initialization for fast adaptation
2. **Prototypical networks**: Compute user prototype from liked items
3. **Matching networks**: Attend to support set for predictions
4. **Personalized initialization**: Cluster-specific starting points
5. **Cold-start**: Meta-learning significantly outperforms baselines

**When to use**:
- Cold-start users with few interactions
- Personalization with limited data
- Rapid adaptation required (online learning)

**Best practices**:
- Start with simple prototypical networks
- Use MAML for complex models (deep networks)
- Evaluate across different n-shot scenarios (1, 3, 5, 10)

---

## Practice Problems

**Problem 1**: Implement MAML for sequential recommendation (GRU-based model). How does it compare to standard GRU?

**Problem 2**: Design an experiment to test meta-learning on cold-start items (not just users). What modifications are needed?

**Problem 3**: Compare MAML vs. prototypical networks vs. matching networks on MovieLens cold-start task. Which performs best?

**Problem 4**: Implement personalized initialization with user clustering. How many clusters give best performance?

---

## References

1. **Finn, C., et al. (2017)**. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks". *ICML*.

2. **Snell, J., et al. (2017)**. "Prototypical Networks for Few-shot Learning". *NeurIPS*.

3. **Vinyals, O., et al. (2016)**. "Matching Networks for One Shot Learning". *NeurIPS*.

4. **Bharadhwaj, H., et al. (2019)**. "Meta-Learning for User Cold-Start Recommendation". *IJCNN*.

5. **Du, Y., et al. (2019)**. "Sequential Scenario-Specific Meta Learner for Online Recommendation". *KDD*.
