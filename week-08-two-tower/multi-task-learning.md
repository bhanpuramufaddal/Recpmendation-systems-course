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

## Why Not Train Separate Models?

*Before diving into architectures, let's understand why we need multi-task learning at all.*

### The Naive Approach: One Model Per Task

Imagine you're building an e-commerce recommendation system. You need to predict:
- **Click**: Will user click this product?
- **Add-to-cart**: Will user add it to cart?
- **Purchase**: Will user buy it?

**Naive solution**: Train 3 separate models.

```
Model 1: features -> P(click)
Model 2: features -> P(add_to_cart)
Model 3: features -> P(purchase)
```

**Let's count the problems:**

**Problem 1: Data inefficiency**

```
Click data:     1,000,000 examples/day (easy to collect)
Add-to-cart:       50,000 examples/day (less common)
Purchase:          10,000 examples/day (rare!)
```

The purchase model only sees 1% of the data that the click model sees. It's starving for signal!

*But here's the key insight*: A user who purchases ALWAYS clicked and added-to-cart first. The click and cart models see related information that could help the purchase model.

**Problem 2: Compute cost**

```
3 models x 500 features x 3 layers = 3x compute
3 models x 100MB parameters = 300MB memory
3 models to maintain, monitor, and debug
```

**Problem 3: Inconsistency**

```
Model 1 predicts: P(click) = 0.9
Model 3 predicts: P(purchase) = 0.95

Wait... purchase probability > click probability?
That's logically impossible! (Must click before purchasing)
```

Separate models don't share knowledge and can produce inconsistent predictions.

---

### The Multi-Task Solution

**Core idea**: Share a learned representation, then predict all tasks from it.

```
features -> [Shared Layers] -> shared_repr -> P(click)
                                           -> P(add_to_cart)
                                           -> P(purchase)
```

**Benefits**:
1. **Data efficiency**: Purchase task benefits from click signal
2. **Compute efficiency**: One forward pass for all predictions
3. **Consistency**: Shared representation enforces coherent predictions
4. **Regularization**: Tasks regularize each other (prevent overfitting)

---

## Problem Formulation

### Single vs. Multi-Task Learning

**Single-task objective**:
$$\min_\theta \mathcal{L}_1(\theta)$$

Train one model for one task.

**Multi-task objective**:
$$\min_\theta \sum_{t=1}^T w_t \mathcal{L}_t(\theta)$$

where:
- $T$ = number of tasks
- $w_t$ = weight for task $t$
- $\mathcal{L}_t$ = loss for task $t$

*Can you see the challenge?* We need to balance tasks - if one task dominates the loss, others might be ignored.

**Numerical example**:

```
Task 1 (click):    loss = 0.5,    volume = 1,000,000 samples
Task 2 (purchase): loss = 0.3,    volume = 10,000 samples

If we use equal weights (w1 = w2 = 1):
Total loss dominated by click (100x more samples)
Purchase task barely influences gradients!

If we weight by inverse volume (w1 = 1, w2 = 100):
Total_loss = 1 * 0.5 + 100 * 0.3 = 30.5
Now purchase contributes: 30/30.5 = 98%!
```

Neither extreme is right. Finding the balance is one of MTL's core challenges.

---

## Architecture 1: Hard Parameter Sharing

### The Architecture

**Idea**: Share ALL hidden layers across tasks. Only the final output heads are task-specific.

```
                    Input Features
                          |
                          v
        +----------------------------------+
        |        Shared Hidden Layers      |
        |   [256] -> [128] -> [64]        |
        +----------------------------------+
                          |
            +-------------+-------------+
            |             |             |
            v             v             v
     +----------+   +----------+   +----------+
     |Click Head|   |Cart Head |   |Buy Head  |
     +----------+   +----------+   +----------+
            |             |             |
            v             v             v
       P(click)      P(cart)       P(buy)
```

**Mathematical formulation**:

$$h = f_{\text{shared}}(x; \theta_{\text{shared}})$$

$$\hat{y}_t = f_t(h; \theta_t) \quad \text{for task } t \in \{1, ..., T\}$$

where $\theta_{\text{shared}}$ is shared across all tasks, and $\theta_t$ is task-specific.

---

### When Hard Sharing Works Well

*Notice that hard sharing assumes all tasks want the SAME features.*

**Good for**:
- Highly correlated tasks (click, like, share - all positive engagement)
- Data-limited tasks that can borrow from data-rich tasks
- When you want strong regularization

**Numerical example - why sharing helps**:

```
Without sharing (separate models):
  Click model sees 1M examples -> learns feature: "red_button" matters
  Purchase model sees 10K examples -> hasn't learned "red_button" yet

With sharing:
  Shared layers see ALL 1.01M examples
  Both tasks benefit from "red_button" pattern
```

---

### When Hard Sharing Fails

*What happens if tasks want DIFFERENT representations?*

**Scenario**: Click vs. Watch-time

```
Click task wants: eye-catching features
  - Bright thumbnails
  - Sensational titles
  - Trending topics

Watch-time task wants: quality features
  - Content depth
  - Creator reputation
  - Relevance to user history
```

These features are **negatively correlated**! Clickbait has high click, low watch-time.

**The conflict**:
```
Gradient from click: "Increase weight for sensational_title"
Gradient from watch: "Decrease weight for sensational_title"

Net gradient: Cancellation! Neither task learns well.
```

This is called **negative transfer** - tasks hurt each other instead of helping.

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
            hidden_dims: List of hidden layer sizes for shared layers
            n_tasks: Number of output tasks
        """
        super().__init__()

        # Build shared layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # Task-specific output heads
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], 1)
            for _ in range(n_tasks)
        ])

        self.n_tasks = n_tasks

    def forward(self, x):
        """
        Forward pass for all tasks.

        Returns:
            outputs: List of predictions, one per task
        """
        # Shared representation (same for all tasks)
        shared_repr = self.shared_layers(x)

        # Task-specific predictions
        outputs = []
        for head in self.task_heads:
            task_output = head(shared_repr).squeeze()
            outputs.append(task_output)

        return outputs
```

---

## Architecture 2: Mixture-of-Experts (MMoE)

### The Key Insight

*What if different tasks need different representations?*

**Solution**: Instead of one shared representation, have MULTIPLE representations (experts) and let each task choose which to use.

```
                    Input Features
                          |
         +----------------+----------------+
         |                |                |
         v                v                v
    +--------+       +--------+       +--------+
    |Expert 1|       |Expert 2|       |Expert 3|
    | (MLP)  |       | (MLP)  |       | (MLP)  |
    +--------+       +--------+       +--------+
         |                |                |
         +-------+--------+-------+--------+
                 |                |
         +-------+        +-------+
         | Gate 1|        | Gate 2|
         |(click)|        | (buy) |
         +-------+        +-------+
              \              /
               \            /
                v          v
        Weighted combination for each task
                |          |
                v          v
           P(click)    P(buy)
```

### Why Experts Help

**Scenario**: Click vs. Purchase (conflicting tasks)

```
Expert 1 specializes in: Engagement features
  - Trending score, thumbnail appeal, title clickbaitiness

Expert 2 specializes in: Quality features
  - Product reviews, brand reputation, return rate

Expert 3 specializes in: User features
  - Purchase history, price sensitivity, category preferences
```

**Click task gate learns**: weight heavily toward Expert 1 (engagement)
```
Gate_click = [0.6, 0.2, 0.2]  # 60% engagement, 20% quality, 20% user
```

**Purchase task gate learns**: weight heavily toward Expert 2 + 3 (quality + user)
```
Gate_purchase = [0.1, 0.5, 0.4]  # 10% engagement, 50% quality, 40% user
```

*Notice: Each task gets a DIFFERENT combination of experts!* This resolves the conflict.

---

### MMoE Mathematical Formulation

**Expert outputs**:
$$e_i = f_i(x) \quad \text{for } i \in \{1, ..., K\}$$

where $K$ = number of experts, and $f_i$ is expert network $i$.

**Task-specific gating**:
$$g_t = \text{softmax}(W_t^g \cdot x) \in \mathbb{R}^K$$

where $g_t^i$ = weight for task $t$ on expert $i$.

**Task representation** (weighted sum of experts):
$$h_t = \sum_{i=1}^K g_t^i \cdot e_i$$

**Task prediction**:
$$\hat{y}_t = \text{tower}_t(h_t)$$

---

### Numerical Walkthrough: MMoE with 2 Tasks

Let's trace through a concrete example with actual numbers.

**Setup**:
- 2 tasks: Click (engagement), Purchase (conversion)
- 3 experts
- Input dimension: 4 (simplified)
- Expert output dimension: 3

**Input features** (one sample):
```
x = [0.8, 0.3, 0.9, 0.2]
    |     |     |     |
    price brand trend user_loyalty
```

**Expert computations** (each expert is a small MLP):

```
Expert 1 (focuses on engagement patterns):
  e_1 = ReLU(W_1 @ x + b_1)
  e_1 = ReLU([[-0.2, 0.9, 0.1, 0.3],   @ [0.8]   + [0.1])
              [0.1, 0.4, 0.8, 0.2],       [0.3]     [0.0]
              [0.0, 0.2, 0.7, 0.1]])      [0.9]     [-0.1])
                                          [0.2]

  e_1 = ReLU([0.38, 0.92, 0.67]) = [0.38, 0.92, 0.67]

Expert 2 (focuses on quality signals):
  e_2 = ReLU(W_2 @ x + b_2)
  e_2 = [0.75, 0.21, 0.44]

Expert 3 (focuses on user behavior):
  e_3 = ReLU(W_3 @ x + b_3)
  e_3 = [0.51, 0.83, 0.29]
```

**Gating computations**:

```
Click task gate:
  g_click = softmax(W_click @ x)
  g_click = softmax([1.2, 0.4, 0.5]) = [0.58, 0.26, 0.29]

  Interpretation: Click task uses 58% Expert 1, 26% Expert 2, 29% Expert 3
  (Expert 1 = engagement patterns, which makes sense for clicks!)

Purchase task gate:
  g_purchase = softmax(W_purchase @ x)
  g_purchase = softmax([0.3, 1.1, 0.9]) = [0.21, 0.47, 0.38]

  Interpretation: Purchase uses 21% Expert 1, 47% Expert 2, 38% Expert 3
  (Experts 2+3 = quality + user patterns, which makes sense for purchase!)
```

**Task-specific representations** (weighted combination):

```
Click representation:
  h_click = 0.58 * [0.38, 0.92, 0.67]
          + 0.26 * [0.75, 0.21, 0.44]
          + 0.29 * [0.51, 0.83, 0.29]

  h_click = [0.22, 0.53, 0.39]    # Step 1: 0.58 * e_1
          + [0.20, 0.05, 0.11]    # Step 2: + 0.26 * e_2
          + [0.15, 0.24, 0.08]    # Step 3: + 0.29 * e_3

  h_click = [0.57, 0.82, 0.58]

Purchase representation:
  h_purchase = 0.21 * [0.38, 0.92, 0.67]
             + 0.47 * [0.75, 0.21, 0.44]
             + 0.38 * [0.51, 0.83, 0.29]

  h_purchase = [0.08, 0.19, 0.14]    # 0.21 * e_1
             + [0.35, 0.10, 0.21]    # 0.47 * e_2
             + [0.19, 0.32, 0.11]    # 0.38 * e_3

  h_purchase = [0.62, 0.61, 0.46]
```

*Notice that h_click and h_purchase are DIFFERENT even though they came from the same input!* The gating mechanism allows task-specific representations.

**Final predictions** (through task towers):

```
Click tower: P(click) = sigmoid(w_click @ h_click + b_click)
           = sigmoid([0.5, 0.3, 0.2] @ [0.57, 0.82, 0.58] + 0.1)
           = sigmoid(0.285 + 0.246 + 0.116 + 0.1)
           = sigmoid(0.747)
           = 0.68

Purchase tower: P(purchase) = sigmoid(w_purchase @ h_purchase + b_purchase)
              = sigmoid([0.3, 0.6, 0.1] @ [0.62, 0.61, 0.46] + (-0.5))
              = sigmoid(0.186 + 0.366 + 0.046 - 0.5)
              = sigmoid(0.098)
              = 0.52
```

**Final output**: P(click) = 0.68, P(purchase) = 0.52

---

### Visualizing Expert Specialization

After training, we can examine which tasks use which experts:

```
            Expert 1   Expert 2   Expert 3
            (engage)   (quality)  (user)
Click:        0.58      0.26       0.16
Add-to-cart:  0.35      0.40       0.25
Purchase:     0.15      0.45       0.40
Share:        0.70      0.15       0.15
Watch-time:   0.25      0.50       0.25
```

*Can you see the pattern?*
- **Click and Share** (engagement tasks) weight toward Expert 1
- **Purchase** (conversion task) weights toward Experts 2 and 3
- **Watch-time** (quality task) weights toward Expert 2

This specialization emerges automatically during training!

---

### Implementation

```python
class MMOE(nn.Module):
    def __init__(self, input_dim, expert_dim=128, n_experts=4, n_tasks=3):
        """
        Multi-gate Mixture-of-Experts.

        Args:
            input_dim: Input feature dimensionality
            expert_dim: Hidden dimension of each expert
            n_experts: Number of expert networks
            n_tasks: Number of prediction tasks
        """
        super().__init__()

        self.n_experts = n_experts
        self.n_tasks = n_tasks

        # Expert networks (each is a small MLP)
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
        # Each gate outputs weights over experts
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, n_experts),
                nn.Softmax(dim=1)
            )
            for _ in range(n_tasks)
        ])

        # Task-specific towers
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
        Forward pass computing all task predictions.
        """
        # Step 1: Compute all expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)
            expert_outputs.append(expert_out)

        # Stack: [batch, n_experts, expert_dim]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # Step 2: For each task, compute gated combination
        outputs = []
        for task_id in range(self.n_tasks):
            # Get gate weights for this task: [batch, n_experts]
            gate_weights = self.gates[task_id](x)

            # Weighted combination of experts
            # [batch, n_experts, 1] * [batch, n_experts, expert_dim]
            gate_weights = gate_weights.unsqueeze(2)
            task_repr = (expert_outputs * gate_weights).sum(dim=1)  # [batch, expert_dim]

            # Task-specific tower
            task_output = self.task_towers[task_id](task_repr).squeeze()
            outputs.append(task_output)

        return outputs
```

---

## Shared-Bottom vs MMoE: When to Use Which

### Decision Framework

```
                     Tasks highly correlated?
                            |
              +-------------+-------------+
              |                           |
             YES                          NO
              |                           |
              v                           v
      Hard Parameter               Tasks may conflict?
         Sharing                         |
                              +----------+----------+
                              |                     |
                             YES                   NO
                              |                    |
                              v                    v
                            MMoE              Soft Sharing
                     (expert separation)    (regularization)
```

### Comparison Table

| Aspect | Hard Sharing | MMoE |
|--------|--------------|------|
| **Parameters** | Fewer (shared + small heads) | More (K experts + gates) |
| **Compute** | Less (one forward pass) | More (K expert forwards + gating) |
| **Task correlation needed** | High | Can handle conflicting tasks |
| **When to use** | Similar tasks, limited data | Different tasks, task conflicts |
| **Risk** | Negative transfer | Over-complexity if tasks are simple |

### Quantitative Guideline

**Use Hard Sharing when**:
- Task correlation > 0.7
- Data per task < 100K samples
- Compute budget is tight

**Use MMoE when**:
- Task correlation < 0.5 OR some task pairs are negative
- Data is abundant (> 1M samples)
- Tasks have shown negative transfer with hard sharing

---

## Task Weighting: Balancing the Losses

### The Core Problem

Different tasks have different:
1. **Data volumes**: Click >>> Purchase
2. **Loss scales**: MSE (watch time) vs. BCE (click)
3. **Business importance**: Purchase > Click (for revenue)

**Without proper weighting**:
```
Total loss = L_click + L_watch + L_purchase

If L_click contributes 90% of gradient (high volume):
  - Model becomes a click predictor
  - Purchase task is essentially ignored
```

---

### Strategy 1: Fixed Weights (Manual Tuning)

**Approach**: Set weights inversely proportional to task volume.

```python
task_weights = {
    'click': 1.0,        # 1M samples
    'watch': 5.0,        # 200K samples
    'purchase': 10.0     # 100K samples
}

total_loss = (task_weights['click'] * loss_click +
              task_weights['watch'] * loss_watch +
              task_weights['purchase'] * loss_purchase)
```

**Pros**: Simple, interpretable

**Cons**: Requires manual tuning, doesn't adapt during training

---

### Strategy 2: Uncertainty Weighting (Learned)

**Key insight** (Kendall et al., 2018): Weight tasks by their inherent uncertainty.

**Intuition**: If a task is noisy (high uncertainty), it should contribute less to the total loss. Otherwise, we're fitting noise.

**Mathematical formulation**:
$$\mathcal{L} = \sum_t \frac{1}{2\sigma_t^2} \mathcal{L}_t + \log \sigma_t$$

where $\sigma_t$ is a **learned parameter** representing task $t$'s uncertainty.

**Derivation** (why this formula?):

Assume each task's prediction has Gaussian noise:
$$y_t = f_t(x) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_t^2)$$

The negative log-likelihood is:
$$-\log p(y_t | f_t(x)) = \frac{(y_t - f_t(x))^2}{2\sigma_t^2} + \log \sigma_t + \text{const}$$

*Can you see it?*
- $\frac{1}{2\sigma_t^2}$: High uncertainty ($\sigma_t$ large) -> lower weight
- $\log \sigma_t$: Prevents $\sigma_t \to \infty$ (can't ignore any task completely)

**Numerical example**:

```
Initial: sigma_click = sigma_purchase = 1.0

After training:
  sigma_click = 0.3     (click is predictable)
  sigma_purchase = 1.5  (purchase is noisy)

Effective weights:
  w_click = 1/(2 * 0.3^2) = 5.56
  w_purchase = 1/(2 * 1.5^2) = 0.22

Click task gets 25x more weight than purchase!
(Because click is more predictable, we should fit it more precisely)
```

---

### Strategy 3: GradNorm (Gradient Balancing)

**Key insight**: Instead of balancing losses, balance gradients directly.

**Problem with loss weighting**:
```
Task 1: loss = 0.1, gradient magnitude = 100
Task 2: loss = 0.5, gradient magnitude = 0.01

Even though Task 2 has higher loss,
Task 1 dominates training (huge gradients)!
```

**GradNorm algorithm**:
1. Compute gradient norms per task: $G_t = \|\nabla_W L_t\|$
2. Compute target: $\bar{G} = \text{average}(G_1, ..., G_T)$
3. Adjust weights to equalize gradient norms

**Update rule**:
$$w_t \leftarrow w_t \times \left(\frac{G_t}{\bar{G}}\right)^{-\alpha}$$

where $\alpha$ controls balancing strength (typically 0.1-1.0).

---

### Implementation: Uncertainty Weighting

```python
class UncertaintyWeighting(nn.Module):
    def __init__(self, n_tasks):
        """
        Learn task uncertainties for automatic loss weighting.
        """
        super().__init__()
        # Log-variance for numerical stability
        # Initialize to 0 -> sigma = 1 for all tasks
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        """
        Compute uncertainty-weighted total loss.

        Args:
            losses: List of individual task losses
        Returns:
            weighted_loss: Scalar
        """
        weighted_loss = 0

        for i, loss in enumerate(losses):
            # precision = 1/sigma^2 = exp(-log_var)
            precision = torch.exp(-self.log_vars[i])
            # Weighted loss + regularization
            weighted_loss += precision * loss + self.log_vars[i]

        return weighted_loss


# Training with uncertainty weighting
model = MMOE(input_dim=100, expert_dim=128, n_experts=4, n_tasks=3)
uncertainty = UncertaintyWeighting(n_tasks=3)

# Both model params and uncertainty params are optimized
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(uncertainty.parameters()),
    lr=0.001
)

for epoch in range(100):
    for batch in train_loader:
        features, labels_click, labels_watch, labels_like = batch

        # Forward pass
        outputs = model(features)

        # Individual task losses
        loss_click = F.binary_cross_entropy_with_logits(
            outputs[0], labels_click.float()
        )
        loss_watch = F.mse_loss(outputs[1], labels_watch.float())
        loss_like = F.binary_cross_entropy_with_logits(
            outputs[2], labels_like.float()
        )

        losses = [loss_click, loss_watch, loss_like]

        # Automatic weighting
        total_loss = uncertainty(losses)

        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Monitor learned uncertainties
    if epoch % 10 == 0:
        sigmas = torch.exp(0.5 * uncertainty.log_vars)
        print(f"Epoch {epoch}: Task uncertainties = {sigmas.detach().numpy()}")
```

---

## What Can Go Wrong: Failure Modes in Multi-Task Learning

*Now let's discuss the things that can silently break your MTL system.*

### Failure Mode 1: Negative Transfer

**Symptom**: Multi-task model performs WORSE than single-task models on one or more tasks.

**What's happening**: Tasks are anti-correlated. Learning one hurts the other.

**Example**:
```
Task 1: Maximize clicks (engagement)
  - Wants: sensational titles, bright thumbnails

Task 2: Maximize satisfaction (long-term retention)
  - Wants: quality content, honest titles

Shared layer learns compromise:
  - Neither sensational nor quality
  - Mediocre at both tasks!
```

**Detection**:
```python
# Compare MTL vs single-task
single_click_auc = 0.85
single_purchase_auc = 0.72

mtl_click_auc = 0.81      # WORSE than single-task!
mtl_purchase_auc = 0.74   # slightly better

# Negative transfer detected for click task
```

**Solutions**:
1. **Switch to MMoE**: Let tasks have different representations
2. **Remove the conflicting task**: Not everything should be in one model
3. **Gradient surgery**: Project conflicting gradients to be orthogonal
4. **Task grouping**: Group related tasks, separate conflicting ones

---

### Failure Mode 2: Task Dominance

**Symptom**: One task's metrics are great, others stagnate or degrade.

**What's happening**: One task dominates the gradient flow.

**Common causes**:
1. **Volume imbalance**: 1M click samples vs 10K purchase samples
2. **Loss scale difference**: MSE (large values) vs BCE (0-1)
3. **Learning speed difference**: Click converges fast, purchase is still learning

**Numerical illustration**:
```
Epoch 10:
  Click loss: 0.3,     gradient norm: 10.0
  Purchase loss: 0.5,  gradient norm: 0.1

Total gradient dominated by click!
  - Click improves: 0.3 -> 0.25
  - Purchase stuck:  0.5 -> 0.5
```

**Detection**:
```python
# Track per-task metrics separately
for epoch in range(100):
    train()

    click_auc = evaluate_click()
    purchase_auc = evaluate_purchase()

    print(f"Click AUC: {click_auc:.3f}, Purchase AUC: {purchase_auc:.3f}")

# If you see:
# Epoch 10: Click 0.80, Purchase 0.55
# Epoch 50: Click 0.85, Purchase 0.56  <- Purchase barely improved!
# Epoch 100: Click 0.87, Purchase 0.57
```

**Solutions**:
1. **Uncertainty weighting**: Let model learn appropriate weights
2. **GradNorm**: Explicitly balance gradient magnitudes
3. **Task sampling**: Over-sample rare task data
4. **Stop gradient**: Periodically freeze dominant task, train others

---

### Failure Mode 3: Gradient Conflicts

**Symptom**: Training is unstable. Loss oscillates. Tasks alternate between improving and degrading.

**What's happening**: Tasks want to update shared parameters in opposite directions.

**Illustration**:
```
Shared parameter W_shared:

Gradient from click:    ∂L_click/∂W = +0.5
Gradient from purchase: ∂L_purchase/∂W = -0.4

Net gradient: +0.1
  - Too weak! Both tasks wanted large updates
  - Update might even hurt BOTH tasks (compromise direction)
```

**Detection**: Monitor gradient cosine similarity
```python
def compute_gradient_conflict(model, losses):
    grads = []
    for loss in losses:
        model.zero_grad()
        loss.backward(retain_graph=True)
        grad = torch.cat([p.grad.flatten() for p in model.shared_layers.parameters()])
        grads.append(grad)

    # Cosine similarity between task gradients
    cos_sim = F.cosine_similarity(grads[0], grads[1], dim=0)
    return cos_sim.item()

# cos_sim < 0 means conflict!
# cos_sim ~ -1 means severe conflict (opposite directions)
```

**Solutions**:

**a) PCGrad (Projected Conflicting Gradients)**:
When gradients conflict, project one onto the orthogonal space of the other.

```python
def pcgrad(grad_1, grad_2):
    """Project grad_1 to remove conflict with grad_2"""
    cos = F.cosine_similarity(grad_1, grad_2, dim=0)
    if cos < 0:  # Conflict detected
        # Project grad_1 onto plane orthogonal to grad_2
        grad_1 = grad_1 - (grad_1 @ grad_2) / (grad_2 @ grad_2) * grad_2
    return grad_1
```

**b) Gradient Vaccine**: Add noise to gradients to escape local conflicts

**c) Task-specific learning rates**: Slow down dominant task

---

### Failure Mode 4: Seesaw Effect

**Symptom**: Improving one task consistently hurts another. Like a seesaw - one goes up, other goes down.

**What's happening**: Tasks are fundamentally competing for model capacity.

**Example**:
```
Epoch 20: Click AUC = 0.80, Purchase AUC = 0.65
Epoch 40: Click AUC = 0.82, Purchase AUC = 0.63  <- Seesaw!
Epoch 60: Click AUC = 0.84, Purchase AUC = 0.61
Epoch 80: Click AUC = 0.85, Purchase AUC = 0.60
```

Every time click improves, purchase degrades.

**Root cause**: The tasks genuinely need different representations, but model capacity is limited.

**Solutions**:
1. **Increase capacity**: More experts, wider layers
2. **Separate models**: Sometimes separation is the right answer
3. **Pareto optimization**: Accept the tradeoff, find best balance point
4. **Hierarchical MTL**: Cluster related tasks, separate conflicting clusters

---

### Failure Mode 5: Cold Task Problem

**Symptom**: A new task added to existing MTL system never catches up.

**What's happening**: Shared layers have already converged for existing tasks. New task can't influence them.

**Scenario**:
```
Phase 1: Train on Click + Watch (converges)
Phase 2: Add Purchase task

Shared layers are "frozen" for click/watch patterns.
Purchase can only learn through its small tower.
```

**Detection**:
```
Task added at epoch 100:
Epoch 100: New task AUC = 0.50 (random)
Epoch 200: New task AUC = 0.52
Epoch 300: New task AUC = 0.53  <- barely learning!

Existing tasks: stable at 0.85
```

**Solutions**:
1. **Retrain from scratch**: Include new task from the start
2. **Learning rate warmup**: Higher LR for shared layers when adding task
3. **Expert allocation**: Reserve unused experts for new tasks (MMoE)
4. **Progressive training**: Gradually increase new task's weight

---

## YouTube Case Study: Multi-Objective Ranking

### YouTube's Objectives

**YouTube optimizes for multiple objectives simultaneously**:

1. **Click probability**: P(user clicks this video)
2. **Expected watch time**: E[watch time | click]
3. **Engagement**: P(like), P(share), P(comment)
4. **Long-term satisfaction**: User returns tomorrow?

*Can you see the conflicts?*
- Clickbait maximizes clicks but hurts satisfaction
- Long videos maximize watch time but may bore users
- Controversial content maximizes engagement but harms platform health

---

### YouTube's Ranking Formula

**Final ranking score** (simplified):
$$\text{Score} = w_{\text{click}} \cdot \sigma(\text{logit}_{\text{click}}) \cdot w_{\text{watch}} \cdot \text{E}[\text{watch time}] + w_{\text{like}} \cdot \sigma(\text{logit}_{\text{like}})$$

Or equivalently using logits:
$$\text{Score} = \text{logit}_{\text{click}} + w_1 \cdot \text{logit}_{\text{watch}} + w_2 \cdot \text{logit}_{\text{like}}$$

**Weights are tuned via A/B testing**, not learned during training.

*Why separate training and ranking weights?*
- Training weights optimize for prediction accuracy
- Ranking weights optimize for business metrics
- These can be different! (e.g., we want accurate click prediction, but may rank by watch time)

---

### Implementation

```python
class YouTubeMultiTaskRanking(nn.Module):
    def __init__(self, input_dim, expert_dim=256, n_experts=8):
        """
        YouTube-style multi-task ranking model.
        """
        super().__init__()

        # MMoE backbone
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(expert_dim, expert_dim),
                nn.ReLU()
            )
            for _ in range(n_experts)
        ])

        # Gates for each task
        self.gate_click = nn.Linear(input_dim, n_experts)
        self.gate_watch = nn.Linear(input_dim, n_experts)
        self.gate_like = nn.Linear(input_dim, n_experts)

        # Task towers
        self.tower_click = nn.Sequential(
            nn.Linear(expert_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.tower_watch = nn.Sequential(
            nn.Linear(expert_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.tower_like = nn.Sequential(
            nn.Linear(expert_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Ranking weights (NOT learned, set via A/B testing)
        self.register_buffer('ranking_weights', torch.tensor([1.0, 0.5, 0.3]))

        self.n_experts = n_experts
        self.expert_dim = expert_dim

    def _compute_task_repr(self, x, expert_outputs, gate):
        """Compute gated expert combination for one task."""
        gate_weights = F.softmax(gate(x), dim=1)  # [batch, n_experts]
        gate_weights = gate_weights.unsqueeze(2)   # [batch, n_experts, 1]
        task_repr = (expert_outputs * gate_weights).sum(dim=1)  # [batch, expert_dim]
        return task_repr

    def forward(self, x, return_ranking_score=True):
        # Compute all expert outputs
        expert_outputs = torch.stack([exp(x) for exp in self.experts], dim=1)

        # Task-specific representations
        repr_click = self._compute_task_repr(x, expert_outputs, self.gate_click)
        repr_watch = self._compute_task_repr(x, expert_outputs, self.gate_watch)
        repr_like = self._compute_task_repr(x, expert_outputs, self.gate_like)

        # Task predictions
        logit_click = self.tower_click(repr_click).squeeze()
        logit_watch = self.tower_watch(repr_watch).squeeze()
        logit_like = self.tower_like(repr_like).squeeze()

        if return_ranking_score:
            # Combined ranking score
            score = (logit_click +
                     self.ranking_weights[0] * logit_watch +
                     self.ranking_weights[1] * logit_like)
            return score, (logit_click, logit_watch, logit_like)

        return (logit_click, logit_watch, logit_like)


# Training
model = YouTubeMultiTaskRanking(input_dim=500, expert_dim=256, n_experts=8)
uncertainty = UncertaintyWeighting(n_tasks=3)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(uncertainty.parameters()),
    lr=0.001
)

for epoch in range(100):
    for batch in train_loader:
        features, y_click, y_watch, y_like = batch

        # Forward (training mode, no ranking score needed)
        logit_click, logit_watch, logit_like = model(features, return_ranking_score=False)

        # Task losses
        loss_click = F.binary_cross_entropy_with_logits(logit_click, y_click.float())
        loss_watch = F.mse_loss(logit_watch, y_watch.float())
        loss_like = F.binary_cross_entropy_with_logits(logit_like, y_like.float())

        # Uncertainty-weighted total loss
        total_loss = uncertainty([loss_click, loss_watch, loss_like])

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


# Inference (ranking videos)
def rank_videos(user_features, candidate_videos, model, top_k=20):
    """Rank candidate videos for a user."""
    features = create_features(user_features, candidate_videos)

    with torch.no_grad():
        scores, _ = model(features, return_ranking_score=True)

    top_indices = torch.topk(scores, top_k).indices
    return [candidate_videos[i] for i in top_indices]
```

---

## Evaluation

### Per-Task Metrics

**Always monitor EACH task separately**:

| Task | Metrics |
|------|---------|
| Click | AUC, Precision@K, Recall@K |
| Watch time | RMSE, R-squared, MAE |
| Like | AUC, F1, Precision-Recall curve |
| Purchase | AUC, Revenue impact |

**Don't**: Only track a single combined metric.

**Do**: Dashboard showing all task metrics over time.

---

### A/B Testing for Multi-Task Models

**What to test**:
1. **Architecture**: Hard sharing vs. MMoE vs. PLE
2. **Number of experts**: 4 vs. 8 vs. 16
3. **Task weighting strategy**: Fixed vs. uncertainty vs. GradNorm
4. **Ranking weight combinations**: Different business tradeoffs

**Metrics to track**:
- Per-task prediction accuracy (offline)
- User engagement metrics (online)
- Business metrics (revenue, retention)

**Example results** (illustrative):
```
Experiment: Hard Sharing vs. MMoE (8 experts)

                    Hard Sharing    MMoE
Click AUC:              0.82        0.83  (+1.2%)
Watch time R²:          0.45        0.51  (+13%)
Like AUC:               0.78        0.79  (+1.3%)
Total watch time:       baseline    +4.2%
User retention:         baseline    +1.8%

Winner: MMoE (statistically significant)
```

---

## Summary

**Key Takeaways**:

1. **Hard parameter sharing**: Simple, effective when tasks are correlated
2. **MMoE**: Multiple experts with task-specific gating, handles task conflicts
3. **Task weighting**: Critical for balancing tasks (uncertainty weighting, GradNorm)
4. **Failure modes**: Negative transfer, task dominance, gradient conflicts - monitor and address

**Architecture Decision Tree**:
```
Tasks correlated? --> YES --> Hard Parameter Sharing
                 |
                 --> NO --> Tasks conflict? --> YES --> MMoE
                                          |
                                          --> NO --> Soft Parameter Sharing
```

**Best Practices**:
1. Start with hard sharing (simplest baseline)
2. Monitor per-task metrics separately
3. If one task degrades, try MMoE
4. Use uncertainty weighting for automatic task balancing
5. A/B test architecture choices

---

## Practice Problems

**Problem 1**: Implement hard parameter sharing for MovieLens (predict: rating, will-click, watch-time). Compare with single-task models. When does MTL help? When does it hurt?

**Problem 2**: Implement MMoE with 4 experts and 3 tasks. After training, visualize the gate weights - which tasks share experts? Which have distinct patterns?

**Problem 3**: Implement uncertainty weighting. Plot how learned uncertainties ($\sigma_t$) change during training. Do they stabilize? What do the final values tell you about task difficulty?

**Problem 4**: Design a multi-task model for e-commerce with tasks: click, add-to-cart, purchase, wishlist. Which tasks should share experts? Propose a task grouping strategy.

---

## References

1. **Caruana, R. (1997)**. "Multitask Learning". *Machine Learning*.
   - Foundational paper on multi-task learning

2. **Ma, J., et al. (2018)**. "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts". *KDD*.
   - Original MMoE paper

3. **Kendall, A., Gal, Y., & Cipolla, R. (2018)**. "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics". *CVPR*.
   - Uncertainty weighting method

4. **Chen, Z., et al. (2018)**. "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks". *ICML*.
   - Gradient-based task balancing

5. **Yu, T., et al. (2020)**. "Gradient Surgery for Multi-Task Learning". *NeurIPS*.
   - PCGrad method for handling gradient conflicts

6. **Tang, H., et al. (2020)**. "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations". *RecSys*.
   - Advanced MTL architecture from Tencent
