# Week 18: Federated Learning for Recommendations

## Overview

**Federated learning**: Train model collaboratively without centralizing user data.

**Motivation**:
1. **Privacy**: User data stays on device (GDPR, CCPA compliance)
2. **Data sensitivity**: Personal preferences, browsing history
3. **Bandwidth**: Avoid sending raw data to server
4. **Local personalization**: Adapt model on-device

**Key principle**: "Bring code to data, not data to code"

---

## Problem Formulation

### Standard vs. Federated Learning

**Standard (centralized)**:
- Server collects all user data
- Trains model on centralized dataset
- Deploys model to users

**Federated**:
- Users keep data locally
- Each user trains model on local data
- Users send model updates (not data) to server
- Server aggregates updates

---

### Notation

- $N$ = number of users (clients)
- $\mathcal{D}_i$ = local dataset for user $i$
- $\theta$ = global model parameters
- $\theta_i$ = local model parameters for user $i$

**Objective**:
$$\min_\theta F(\theta) = \sum_{i=1}^N \frac{|\mathcal{D}_i|}{|\mathcal{D}|} F_i(\theta)$$

where $F_i(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_i}[\ell(f_\theta(x), y)]$

---

## Federated Averaging (FedAvg)

### Algorithm

**Core idea**: Parallel SGD with periodic averaging.

**Steps**:
1. **Server** initializes global model $\theta$
2. **Server** selects subset of clients
3. **Each client** downloads $\theta$
4. **Each client** trains on local data: $\theta_i \leftarrow \theta - \eta \nabla F_i(\theta)$
5. **Each client** sends $\theta_i$ to server
6. **Server** aggregates: $\theta \leftarrow \sum_{i} \frac{|\mathcal{D}_i|}{|\mathcal{D}|} \theta_i$
7. Repeat 2-6

---

### Implementation

```python
import torch
import torch.nn as nn
import copy

class RecommenderModel(nn.Module):
    def __init__(self, n_items, embedding_dim=64):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, item_ids):
        item_emb = self.item_embedding(item_ids)
        x = torch.relu(self.fc1(item_emb))
        score = self.fc2(x)
        return score.squeeze()


class FederatedServer:
    def __init__(self, model, n_clients):
        """
        Args:
            model: Global model
            n_clients: Number of clients
        """
        self.global_model = model
        self.n_clients = n_clients

    def select_clients(self, fraction=0.1):
        """Select random fraction of clients for round."""
        import random
        n_selected = max(1, int(fraction * self.n_clients))
        return random.sample(range(self.n_clients), n_selected)

    def aggregate(self, client_models, client_data_sizes):
        """
        Aggregate client models using weighted average.

        Args:
            client_models: List of client model parameters
            client_data_sizes: List of client dataset sizes
        """
        total_data = sum(client_data_sizes)

        # Initialize aggregated parameters
        aggregated_state = {}

        for key in self.global_model.state_dict().keys():
            # Weighted average
            aggregated_state[key] = sum(
                client_model[key] * (client_size / total_data)
                for client_model, client_size in zip(client_models, client_data_sizes)
            )

        # Update global model
        self.global_model.load_state_dict(aggregated_state)

    def get_global_model(self):
        """Return copy of global model."""
        return copy.deepcopy(self.global_model)


class FederatedClient:
    def __init__(self, client_id, local_data, local_labels):
        """
        Args:
            client_id: Unique client identifier
            local_data: Client's local item interactions
            local_labels: Client's ratings/labels
        """
        self.client_id = client_id
        self.local_data = local_data
        self.local_labels = local_labels

    def train(self, global_model, n_epochs=5, lr=0.01):
        """
        Train global model on local data.

        Args:
            global_model: Model received from server
            n_epochs: Number of local training epochs
            lr: Learning rate

        Returns:
            updated_model: Trained model parameters
        """
        # Create local copy
        local_model = copy.deepcopy(global_model)

        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(n_epochs):
            # Forward pass
            predictions = local_model(self.local_data)

            # Loss
            loss = criterion(predictions, self.local_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return local_model.state_dict()


# Federated learning training
def federated_training(server, clients, n_rounds=100, client_fraction=0.1):
    """
    Federated training loop.

    Args:
        server: FederatedServer instance
        clients: List of FederatedClient instances
        n_rounds: Number of communication rounds
        client_fraction: Fraction of clients selected per round
    """
    for round_num in range(n_rounds):
        print(f"Round {round_num + 1}/{n_rounds}")

        # Select clients
        selected_client_ids = server.select_clients(fraction=client_fraction)
        selected_clients = [clients[i] for i in selected_client_ids]

        # Get global model
        global_model = server.get_global_model()

        # Clients train locally
        client_models = []
        client_data_sizes = []

        for client in selected_clients:
            updated_model = client.train(global_model, n_epochs=5, lr=0.01)
            client_models.append(updated_model)
            client_data_sizes.append(len(client.local_data))

        # Server aggregates
        server.aggregate(client_models, client_data_sizes)

        # Evaluate global model
        if (round_num + 1) % 10 == 0:
            global_model = server.get_global_model()
            accuracy = evaluate_model(global_model, test_data)
            print(f"  Global model accuracy: {accuracy:.4f}")


# Example usage
n_items = 10000
global_model = RecommenderModel(n_items=n_items, embedding_dim=64)

# Create server
server = FederatedServer(global_model, n_clients=1000)

# Create clients with local data
clients = []
for client_id in range(1000):
    local_data, local_labels = load_client_data(client_id)
    client = FederatedClient(client_id, local_data, local_labels)
    clients.append(client)

# Train
federated_training(server, clients, n_rounds=100, client_fraction=0.1)
```

---

## Privacy Considerations

### Differential Privacy

**Goal**: Protect individual user data from being inferred.

**Approach**: Add noise to model updates.

**Mechanism**:
$$\theta_i^{noisy} = \theta_i + \mathcal{N}(0, \sigma^2 C^2 I)$$

where:
- $C$ = clipping norm (bounds update magnitude)
- $\sigma$ = noise scale

**Privacy-utility trade-off**: More noise → more privacy, less accuracy.

---

### Implementation

```python
import torch

class DifferentiallyPrivateClient(FederatedClient):
    def __init__(self, client_id, local_data, local_labels, noise_scale=0.1, clip_norm=1.0):
        super().__init__(client_id, local_data, local_labels)
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm

    def train(self, global_model, n_epochs=5, lr=0.01):
        """Train with differential privacy."""
        # Train normally
        updated_state = super().train(global_model, n_epochs, lr)

        # Apply differential privacy
        dp_state = self._add_differential_privacy(updated_state, global_model.state_dict())

        return dp_state

    def _add_differential_privacy(self, updated_state, original_state):
        """
        Add noise to model updates for differential privacy.
        """
        dp_state = {}

        for key in updated_state.keys():
            # Compute update (difference from original)
            update = updated_state[key] - original_state[key]

            # Clip update norm
            update_norm = torch.norm(update)
            if update_norm > self.clip_norm:
                update = update * (self.clip_norm / update_norm)

            # Add Gaussian noise
            noise = torch.randn_like(update) * self.noise_scale * self.clip_norm

            # Apply noisy update
            dp_state[key] = original_state[key] + update + noise

        return dp_state


# Usage
client = DifferentiallyPrivateClient(
    client_id=0,
    local_data=data,
    local_labels=labels,
    noise_scale=0.1,
    clip_norm=1.0
)
```

---

## Communication Efficiency

### Challenge

**Problem**: Sending full model parameters is expensive.
- Deep models: millions of parameters
- Limited bandwidth (mobile devices)
- Frequent communication rounds

**Solution**: Reduce communication cost.

---

### Gradient Compression

**Idea**: Compress gradients before sending.

**Techniques**:
1. **Sparsification**: Send only top-K gradients
2. **Quantization**: Reduce precision (32-bit → 8-bit)
3. **Sketching**: Use random projections

---

### Implementation: Top-K Sparsification

```python
class CompressedClient(FederatedClient):
    def __init__(self, client_id, local_data, local_labels, compression_ratio=0.1):
        super().__init__(client_id, local_data, local_labels)
        self.compression_ratio = compression_ratio

    def train(self, global_model, n_epochs=5, lr=0.01):
        """Train and compress updates."""
        # Train normally
        updated_state = super().train(global_model, n_epochs, lr)

        # Compress updates
        compressed_state = self._compress_updates(updated_state, global_model.state_dict())

        return compressed_state

    def _compress_updates(self, updated_state, original_state):
        """
        Compress updates using top-K sparsification.
        """
        compressed_state = {}

        for key in updated_state.keys():
            # Compute update
            update = updated_state[key] - original_state[key]

            # Flatten update
            update_flat = update.flatten()

            # Select top-K by magnitude
            k = max(1, int(len(update_flat) * self.compression_ratio))
            top_k_values, top_k_indices = torch.topk(update_flat.abs(), k)

            # Create sparse update
            sparse_update = torch.zeros_like(update_flat)
            sparse_update[top_k_indices] = update_flat[top_k_indices]

            # Reshape and apply
            sparse_update = sparse_update.reshape(update.shape)
            compressed_state[key] = original_state[key] + sparse_update

        return compressed_state


# Usage
client = CompressedClient(
    client_id=0,
    local_data=data,
    local_labels=labels,
    compression_ratio=0.1  # Send only top 10% of gradients
)
```

---

## Handling Non-IID Data

### Challenge

**Problem**: User data is not identically distributed.
- Different users have different preferences
- Data heterogeneity hurts convergence

**Example**:
- User A: Likes action movies
- User B: Likes romance movies
- Non-IID: Each user's data is skewed

---

### Solutions

**1. Personalization layers**: User-specific layers, shared base model.

**2. FedProx**: Add proximal term to keep local updates close to global.

$$\min_{\theta_i} F_i(\theta_i) + \frac{\mu}{2} \|\theta_i - \theta\|^2$$

**3. Scaffold**: Use control variates to correct for data heterogeneity.

---

### Implementation: FedProx

```python
class FedProxClient(FederatedClient):
    def __init__(self, client_id, local_data, local_labels, mu=0.01):
        super().__init__(client_id, local_data, local_labels)
        self.mu = mu  # Proximal term coefficient

    def train(self, global_model, n_epochs=5, lr=0.01):
        """Train with FedProx proximal term."""
        local_model = copy.deepcopy(global_model)

        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Store global parameters for proximal term
        global_params = [p.clone().detach() for p in global_model.parameters()]

        for epoch in range(n_epochs):
            # Forward pass
            predictions = local_model(self.local_data)

            # Standard loss
            loss = criterion(predictions, self.local_labels)

            # Proximal term: keep local model close to global
            proximal_term = 0
            for local_param, global_param in zip(local_model.parameters(), global_params):
                proximal_term += torch.norm(local_param - global_param) ** 2

            # Total loss
            total_loss = loss + (self.mu / 2) * proximal_term

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        return local_model.state_dict()
```

---

## Personalized Federated Learning

### Local Fine-Tuning

**Idea**: Global model + local adaptation.

**Process**:
1. Train global model federally
2. Each user fine-tunes on local data
3. Use personalized model for recommendations

---

### Implementation

```python
class PersonalizedFederatedClient(FederatedClient):
    def __init__(self, client_id, local_data, local_labels):
        super().__init__(client_id, local_data, local_labels)
        self.personalized_model = None

    def train(self, global_model, n_epochs=5, lr=0.01):
        """Train global model (contribution to federated learning)."""
        return super().train(global_model, n_epochs, lr)

    def personalize(self, global_model, n_epochs=10, lr=0.001):
        """
        Fine-tune global model on local data for personalization.

        Args:
            global_model: Global model from server
            n_epochs: Number of fine-tuning epochs
            lr: Learning rate for fine-tuning
        """
        # Create personalized model from global
        self.personalized_model = copy.deepcopy(global_model)

        optimizer = torch.optim.Adam(self.personalized_model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(n_epochs):
            predictions = self.personalized_model(self.local_data)
            loss = criterion(predictions, self.local_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def recommend(self, candidate_items, k=10):
        """
        Generate recommendations using personalized model.
        """
        if self.personalized_model is None:
            raise ValueError("Must call personalize() first")

        scores = self.personalized_model(candidate_items)
        top_k_items = torch.topk(scores, k).indices

        return top_k_items


# Usage
client = PersonalizedFederatedClient(client_id=0, local_data=data, local_labels=labels)

# Participate in federated learning
global_model = server.get_global_model()
client_update = client.train(global_model, n_epochs=5)

# Personalize for recommendations
client.personalize(global_model, n_epochs=10)

# Generate recommendations
candidate_items = torch.arange(10000)
recommendations = client.recommend(candidate_items, k=10)
```

---

## Practical Considerations

### Device Heterogeneity

**Challenge**: Devices have different compute/battery/network.

**Solution**:
- Adaptive client selection (prioritize powerful devices)
- Asynchronous updates (don't wait for slow clients)
- Computation offloading (partial training on device)

---

### Dropout and Stragglers

**Challenge**: Clients may disconnect mid-training.

**Solution**:
- Server aggregates available updates only
- Weighted averaging accounts for partial participation
- Clients resume from last checkpoint

```python
class RobustFederatedServer(FederatedServer):
    def aggregate(self, client_models, client_data_sizes):
        """
        Aggregate client models, handling missing clients.
        """
        if len(client_models) == 0:
            print("Warning: No client updates received")
            return

        # Aggregate available updates
        super().aggregate(client_models, client_data_sizes)
```

---

## Summary

**Key Takeaways**:
1. **FedAvg**: Parallel SGD with periodic averaging
2. **Differential privacy**: Add noise to protect individual users
3. **Communication efficiency**: Compress gradients (top-K, quantization)
4. **Non-IID data**: FedProx, personalization layers
5. **Personalization**: Global model + local fine-tuning

**Benefits**:
- Privacy: Data stays on device
- Scalability: Distributed computation
- Personalization: User-specific models

**Challenges**:
- Communication cost
- Data heterogeneity
- Device constraints

---

## Practice Problems

**Problem 1**: Implement FedAvg for matrix factorization on MovieLens. Compare convergence with centralized training.

**Problem 2**: Add differential privacy to FedAvg. Measure privacy-utility trade-off (noise scale vs. accuracy).

**Problem 3**: Implement gradient compression with top-K sparsification. How much can you compress while maintaining accuracy?

**Problem 4**: Design a personalized federated learning system where each user has a private embedding layer. How do you aggregate?

---

## References

1. **McMahan, B., et al. (2017)**. "Communication-Efficient Learning of Deep Networks from Decentralized Data". *AISTATS*.

2. **Kairouz, P., et al. (2021)**. "Advances and Open Problems in Federated Learning". *Foundations and Trends in Machine Learning*.

3. **Li, T., et al. (2020)**. "Federated Optimization in Heterogeneous Networks". *MLSys*.

4. **Ammad-Ud-Din, M., et al. (2019)**. "Federated Collaborative Filtering for Privacy-Preserving Personalized Recommendation System". *arXiv*.

5. **Yang, Q., et al. (2019)**. "Federated Machine Learning: Concept and Applications". *ACM TIST*.
