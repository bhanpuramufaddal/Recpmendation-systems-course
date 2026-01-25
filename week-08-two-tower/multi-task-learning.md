# Week 8: Multi-Task Learning for Recommendations

## Overview

**Multi-Task Learning (MTL)**: Train single model to optimize multiple objectives simultaneously.

**In RecSys**: Predict multiple user behaviors:
- **Clicks**: Did user click?
- **Watch time**: How long did user engage?
- **Likes**: Did user like?
- **Shares**: Did user share?
- **Purchases**: Did user buy?

**Why MTL?**
- **Efficiency**: One model for all tasks
- **Knowledge transfer**: Tasks help each other learn
- **Better features**: Shared representations generalize

---

## Problem Formulation

### Single vs. Multi-Task

**Single-task**:
$$\min_\theta \mathcal{L}_1(\theta)$$

Train separate model for each task (clicks, likes, etc.).

**Multi-task**:
$$\min_\theta \sum_{t=1}^T w_t \mathcal{L}_t(\theta)$$

where:
- $T$ = number of tasks
- $w_t$ = task weight
- $\mathcal{L}_t$ = loss for task $t$

**Challenge**: How to balance tasks? (Some tasks may dominate)

---

## Hard Parameter Sharing

### Architecture

**Idea**: Share all hidden layers, task-specific output layers.

```
Input → Shared Layers → Task 1 Head → Click prediction
                      → Task 2 Head → Watch time prediction
                      → Task 3 Head → Like prediction
```

**Benefits**:
- Simple
- Reduces overfitting (regularization effect)
- Fast training

**Drawbacks**:
- Tasks may conflict (negative transfer)
- One-size-fits-all representation

---

### Implementation

```python
import torch
import torch.nn as nn

class HardParameterSharing(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], n_tasks=3):
        """
        Multi-task model with hard parameter sharing.

        Args:
            input_dim: Input feature dimensionality
            hidden_dims: List of hidden layer sizes
            n_tasks: Number of tasks
        """
        super().__init__()

        # Shared layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], 1)
            for _ in range(n_tasks)
        ])

        self.n_tasks = n_tasks

    def forward(self, x):
        """
        Forward pass for all tasks.

        Args:
            x: [batch_size, input_dim] input features

        Returns:
            outputs: List of [batch_size] predictions per task
        """
        # Shared representation
        shared_repr = self.shared_layers(x)

        # Task-specific predictions
        outputs = []
        for head in self.task_heads:
            task_output = head(shared_repr).squeeze()
            outputs.append(task_output)

        return outputs


# Training
model = HardParameterSharing(input_dim=100, hidden_dims=[256, 128], n_tasks=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Task losses
loss_fns = [
    nn.BCEWithLogitsLoss(),  # Task 1: Click (binary)
    nn.MSELoss(),            # Task 2: Watch time (regression)
    nn.BCEWithLogitsLoss()   # Task 3: Like (binary)
]

# Task weights
task_weights = [1.0, 0.5, 0.3]

for epoch in range(100):
    for features, labels_click, labels_watch, labels_like in train_loader:
        # Forward pass
        outputs = model(features)

        # Compute losses
        loss_click = loss_fns[0](outputs[0], labels_click.float())
        loss_watch = loss_fns[1](outputs[1], labels_watch.float())
        loss_like = loss_fns[2](outputs[2], labels_like.float())

        # Weighted sum
        total_loss = (
            task_weights[0] * loss_click +
            task_weights[1] * loss_watch +
            task_weights[2] * loss_like
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
```

---

## Soft Parameter Sharing

### Architecture

**Idea**: Each task has own layers, regularize to be similar.

```
Input → Task 1 Layers → Task 1 Output
      → Task 2 Layers → Task 2 Output
      → Task 3 Layers → Task 3 Output
```

**Regularization**:
$$\mathcal{L} = \sum_t \mathcal{L}_t + \lambda \sum_{i \neq j} \|\theta_i - \theta_j\|^2$$

Penalize differences between task parameters.

**Benefits**:
- More flexibility
- Less task interference

**Drawbacks**:
- More parameters
- Slower training

---

## MMOE (Multi-gate Mixture-of-Experts)

### Architecture

**Idea**: Multiple experts, task-specific gating.

**Components**:
1. **Experts**: $k$ neural networks (different specializations)
2. **Gating networks**: One per task (selects expert combination)
3. **Task towers**: Task-specific final layers

**Formula**:
$$h_t = \sum_{i=1}^k g_t^i \cdot f_i(x)$$

where:
- $h_t$ = task $t$ representation
- $g_t^i$ = gate weight for task $t$, expert $i$
- $f_i(x)$ = expert $i$ output

---

### Implementation

```python
class MMOE(nn.Module):
    def __init__(self, input_dim, expert_dim=128, n_experts=4, n_tasks=3):
        """
        Multi-gate Mixture-of-Experts.

        Args:
            input_dim: Input feature dimensionality
            expert_dim: Expert hidden dimension
            n_experts: Number of expert networks
            n_tasks: Number of tasks
        """
        super().__init__()

        self.n_experts = n_experts
        self.n_tasks = n_tasks

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, expert_dim),
                nn.ReLU()
            )
            for _ in range(n_experts)
        ])

        # Gating networks (one per task)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, n_experts),
                nn.Softmax(dim=1)
            )
            for _ in range(n_tasks)
        ])

        # Task towers
        self.task_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            for _ in range(n_tasks)
        ])

    def forward(self, x):
        """
        Forward pass for all tasks.

        Args:
            x: [batch_size, input_dim]

        Returns:
            outputs: List of [batch_size] predictions per task
        """
        # Expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)
            expert_outputs.append(expert_out)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, n_experts, expert_dim]

        # Task-specific outputs
        outputs = []

        for task_id in range(self.n_tasks):
            # Gating weights for this task
            gate_weights = self.gates[task_id](x)  # [batch, n_experts]

            # Weighted combination of experts
            gate_weights = gate_weights.unsqueeze(2)  # [batch, n_experts, 1]
            task_repr = (expert_outputs * gate_weights).sum(dim=1)  # [batch, expert_dim]

            # Task tower
            task_output = self.task_towers[task_id](task_repr).squeeze()
            outputs.append(task_output)

        return outputs


# Training
model = MMOE(input_dim=100, expert_dim=128, n_experts=4, n_tasks=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fns = [
    nn.BCEWithLogitsLoss(),  # Click
    nn.MSELoss(),            # Watch time
    nn.BCEWithLogitsLoss()   # Like
]

for epoch in range(100):
    for features, labels_click, labels_watch, labels_like in train_loader:
        outputs = model(features)

        # Compute losses
        loss_click = loss_fns[0](outputs[0], labels_click.float())
        loss_watch = loss_fns[1](outputs[1], labels_watch.float())
        loss_like = loss_fns[2](outputs[2], labels_like.float())

        # Total loss
        total_loss = loss_click + 0.5 * loss_watch + 0.3 * loss_like

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
```

---

## Task Weighting Strategies

### Fixed Weights

**Simplest**: Manually set task weights.

```python
task_weights = [1.0, 0.5, 0.3]  # Click, Watch time, Like
```

**Problem**: Requires tuning, not adaptive.

---

### Uncertainty Weighting

**Idea**: Weight by task uncertainty (Kendall & Gal, 2018).

**Loss**:
$$\mathcal{L} = \sum_t \frac{1}{2\sigma_t^2} \mathcal{L}_t + \log \sigma_t$$

where $\sigma_t$ = learned task uncertainty.

**Interpretation**: Tasks with high uncertainty get lower weight.

---

### Implementation

```python
class UncertaintyWeighting(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()

        # Log-variance for each task
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        """
        Compute uncertainty-weighted loss.

        Args:
            losses: List of task losses

        Returns:
            weighted_loss: Scalar loss
        """
        weighted_loss = 0

        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss += precision * loss + self.log_vars[i]

        return weighted_loss


# Training with uncertainty weighting
model = MMOE(input_dim=100, expert_dim=128, n_experts=4, n_tasks=3)
uncertainty = UncertaintyWeighting(n_tasks=3)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(uncertainty.parameters()),
    lr=0.001
)

for epoch in range(100):
    for features, labels_click, labels_watch, labels_like in train_loader:
        outputs = model(features)

        # Individual task losses
        loss_click = F.binary_cross_entropy_with_logits(outputs[0], labels_click.float())
        loss_watch = F.mse_loss(outputs[1], labels_watch.float())
        loss_like = F.binary_cross_entropy_with_logits(outputs[2], labels_like.float())

        losses = [loss_click, loss_watch, loss_like]

        # Uncertainty-weighted loss
        total_loss = uncertainty(losses)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
        print(f"  Task uncertainties: {torch.exp(uncertainty.log_vars).detach()}")
```

---

### GradNorm (Gradient Normalization)

**Idea**: Balance gradient magnitudes across tasks.

**Algorithm**:
1. Compute task gradients
2. Normalize so all tasks have similar gradient magnitudes
3. Adjust task weights dynamically

**Benefits**: Automatic balancing, no manual tuning.

---

## YouTube Case Study: Multi-Objective Ranking

### Objectives

**YouTube optimizes for**:
1. **Watch time** (primary): Total time user watches
2. **Engagement**: Likes, shares, comments
3. **Satisfaction**: User surveys, retention

**Challenge**: Watch time conflicts with satisfaction (clickbait maximizes watch time but hurts satisfaction).

---

### Weighted Combination

**Ranking score**:
$$\text{Score} = \text{Logit}_{\text{click}} + w_1 \cdot \text{Logit}_{\text{watch time}} + w_2 \cdot \text{Logit}_{\text{like}}$$

**Weights**: Tuned via A/B testing.

---

### Implementation

```python
class YouTubeRanking(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()

        # Shared base
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Task heads
        self.click_head = nn.Linear(hidden_dim, 1)
        self.watch_time_head = nn.Linear(hidden_dim, 1)
        self.like_head = nn.Linear(hidden_dim, 1)

        # Learned task weights
        self.task_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.3]))

    def forward(self, x, return_score=True):
        """
        Predict multi-objectives and ranking score.
        """
        shared_repr = self.shared(x)

        # Predictions
        click_logit = self.click_head(shared_repr).squeeze()
        watch_time_logit = self.watch_time_head(shared_repr).squeeze()
        like_logit = self.like_head(shared_repr).squeeze()

        if return_score:
            # Combined ranking score
            score = (
                click_logit +
                self.task_weights[0] * watch_time_logit +
                self.task_weights[1] * like_logit
            )
            return score, (click_logit, watch_time_logit, like_logit)
        else:
            return (click_logit, watch_time_logit, like_logit)


# Training
model = YouTubeRanking(input_dim=100, hidden_dim=256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for features, labels_click, labels_watch, labels_like in train_loader:
        click_logit, watch_logit, like_logit = model(features, return_score=False)

        # Individual losses
        loss_click = F.binary_cross_entropy_with_logits(click_logit, labels_click.float())
        loss_watch = F.mse_loss(watch_logit, labels_watch.float())
        loss_like = F.binary_cross_entropy_with_logits(like_logit, labels_like.float())

        # Total loss
        total_loss = loss_click + 0.5 * loss_watch + 0.3 * loss_like

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


# Ranking
def rank_videos(videos, user, model, k=10):
    """
    Rank videos by multi-objective score.
    """
    features = extract_features(user, videos)

    with torch.no_grad():
        scores, _ = model(features, return_score=True)

    # Top-K
    top_k_indices = torch.topk(scores, k).indices
    return [videos[i] for i in top_k_indices]
```

---

## Evaluation

### Metrics per Task

**For each task, track**:
- Task 1 (Click): AUC, Precision@K
- Task 2 (Watch time): MSE, R²
- Task 3 (Like): AUC, F1

**Overall**: Weighted average or Pareto efficiency.

---

### A/B Testing

**Real-world evaluation**: Deploy multi-task model, measure:
- Overall user engagement
- Time on platform
- User retention

**Compare**:
- Single-task models (baseline)
- Hard parameter sharing
- MMOE

**Example results**:
- MMOE: +5% watch time, +8% engagement vs. single-task
- Hard sharing: +3% watch time, +5% engagement

---

## Summary

**Key Takeaways**:
1. **Hard sharing**: Simple, effective, shared layers
2. **MMOE**: Multiple experts, task-specific gating
3. **Uncertainty weighting**: Automatic task balancing
4. **YouTube**: Multi-objective ranking (watch time, engagement, satisfaction)

**When to use**:
- Multiple related prediction tasks
- Want to reduce model complexity (one model vs. many)
- Tasks can benefit from shared representations

**Best practices**:
- Start with hard sharing (simplest)
- Use MMOE if tasks conflict
- Tune task weights via A/B testing
- Monitor per-task metrics separately

---

## Practice Problems

**Problem 1**: Implement hard parameter sharing for MovieLens (rating prediction, click prediction, watch time). Compare with single-task models.

**Problem 2**: Implement MMOE with 3 experts, 3 tasks. Visualize which experts each task uses (gating weights).

**Problem 3**: Implement uncertainty weighting. How do learned task uncertainties change during training?

**Problem 4**: Design multi-task model for e-commerce (click, purchase, add-to-cart, wishlist). Which tasks should share parameters?

---

## References

1. **Caruana, R. (1997)**. "Multitask Learning". *Machine Learning*.

2. **Ma, J., et al. (2018)**. "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts". *KDD* (MMOE).

3. **Kendall, A., Gal, Y., & Cipolla, R. (2018)**. "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics". *CVPR*.

4. **Chen, Z., et al. (2018)**. "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks". *ICML*.

5. **Covington, P., et al. (2016)**. "Deep Neural Networks for YouTube Recommendations". *RecSys*.
