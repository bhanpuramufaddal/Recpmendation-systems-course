# Week 5: From Matrix Factorization to Neural Networks

## Opening: Why Does Matrix Factorization's Linearity Fail?

*Before we add neural network complexity, let me show you exactly why we need it.*

### The Geometric Limitation of Dot Products

**The dot product creates a hyperplane decision boundary.** Let me show you what this means.

**Setup**: Consider a 2D latent space (2 factors).

**User Alice's embedding**: $\mathbf{u}_{Alice} = [0.8, 0.6]$

**MF prediction formula**:
$$\hat{r}_{Alice, i} = \mathbf{u}_{Alice}^T \mathbf{v}_i = 0.8 \cdot v_{i1} + 0.6 \cdot v_{i2}$$

**The "like" threshold**: Say Alice likes items with predicted rating > 0.5.

This means: $0.8 \cdot v_{i1} + 0.6 \cdot v_{i2} > 0.5$

*Notice that* this is a **linear equation** in the item space! It defines a straight line:

```
v_{i2}
  ↑
  │
  │  Alice DISLIKES     /
  │  (below line)      /
  │                   /  Alice LIKES
  │                  /   (above line)
  │                 /
  │                /
  │───────────────/────────→ v_{i1}
             (0.625, 0)
```

**The line**: $v_{i2} = \frac{0.5 - 0.8 \cdot v_{i1}}{0.6} = 0.833 - 1.33 \cdot v_{i1}$

**Problem**: What if Alice's true preference is **non-linear**?

### A Concrete Pattern MF Cannot Capture

**The "Purity" Preference**: Alice likes movies that are EITHER action OR romance, but NOT action-romance hybrids.

**Ground truth** (what Alice actually likes):

| Movie | Action Score | Romance Score | Alice Likes? |
|-------|-------------|---------------|--------------|
| Die Hard | 0.9 | 0.1 | Yes |
| The Notebook | 0.1 | 0.9 | Yes |
| Mr. & Mrs. Smith | 0.7 | 0.7 | **No** (hybrid!) |
| Boring Documentary | 0.2 | 0.2 | No |

**What MF predicts** (with $\mathbf{u}_{Alice} = [0.8, 0.6]$):

- Die Hard: $0.8 \times 0.9 + 0.6 \times 0.1 = 0.72 + 0.06 = 0.78$ (correct: high)
- The Notebook: $0.8 \times 0.1 + 0.6 \times 0.9 = 0.08 + 0.54 = 0.62$ (correct: high)
- Mr. & Mrs. Smith: $0.8 \times 0.7 + 0.6 \times 0.7 = 0.56 + 0.42 = **0.98**$ (WRONG: highest!)
- Boring Documentary: $0.8 \times 0.2 + 0.6 \times 0.2 = 0.16 + 0.12 = 0.28$ (correct: low)

**The failure**: MF predicts Alice will love Mr. & Mrs. Smith the MOST, but she actually dislikes it!

*Can you see the geometric problem?* Alice's true preference region looks like an "L" shape (corners of the space), but MF can only draw a straight line. No matter how we adjust Alice's embedding, we cannot separate "pure action" and "pure romance" from "hybrid."

**This is why we need neural networks**: They can draw curved, non-linear decision boundaries.

---

## Learning Objectives

By the end of this section, you will:
- Understand MF as a shallow neural network (single layer, no activation)
- Recognize the limitations of linear interactions geometrically
- See the step-by-step bridge: MF → +layer → +activation → deep network
- Know when neural complexity helps vs. when it's overkill
- Be prepared for NCF and other advanced models

---

## Matrix Factorization: A Neural Network Perspective

### MF as a One-Layer Network

*Let me reframe MF in neural network language.*

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
2. **Embedding layer**: Lookup tables (nn.Embedding in PyTorch)
3. **Interaction function**: Dot product (element-wise multiply + sum)
4. **Output**: Predicted rating

**Key insight**: MF is a **one-layer neural network** with:
- **No hidden layers**
- **No non-linear activation**
- **Fixed interaction function** (dot product)

*Notice that* this is the simplest possible neural architecture for recommendation!

---

### The MF Formula as Neural Computation

**Standard MF**:
$$\hat{r}_{ui} = \mathbf{u}_u^T \mathbf{v}_i = \sum_{f=1}^k u_{uf} \cdot v_{fi}$$

**Neural network equivalent**:
$$\hat{r}_{ui} = \text{sum}(\text{EmbedUser}(u) \odot \text{EmbedItem}(i))$$

where $\odot$ is element-wise multiplication.

**PyTorch implementation**:

```python
import torch
import torch.nn as nn

class MatrixFactorizationNN(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        # These ARE the P and Q matrices from classical MF!
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)  # (batch, k)
        item_emb = self.item_embedding(item_ids)  # (batch, k)

        # Dot product = element-wise multiply + sum
        prediction = (user_emb * item_emb).sum(dim=1)  # (batch,)

        return prediction

# This is IDENTICAL to classical MF!
model = MatrixFactorizationNN(n_users=1000, n_items=5000, n_factors=50)
```

*Can you see* that this code does exactly what SVD-based MF does? The embedding matrices are P and Q, and the forward pass computes $\mathbf{p}_u^T \mathbf{q}_i$.

---

## The Bridge: From MF to Neural Networks

*Now let me show you the step-by-step path from MF to deep learning.*

### Step 0: Baseline MF

**Architecture**:
```
[user_emb, item_emb] → dot product → prediction
```

**Formula**: $\hat{r} = \mathbf{u}^T \mathbf{v}$

**Capacity**: Linear combinations only.

**Parameters**: $k \times (|U| + |I|)$ (just embeddings)

---

### Step 1: Add a Learnable Weight Vector

**Architecture**:
```
[user_emb, item_emb] → element-wise product → weighted sum → prediction
```

**Formula**: $\hat{r} = \mathbf{h}^T (\mathbf{u} \odot \mathbf{v})$ where $\mathbf{h}$ is learnable.

**What changes?** Instead of all dimensions contributing equally (implicit $\mathbf{h} = [1,1,...,1]$), the model learns which dimensions matter.

**Numerical example**:
- $\mathbf{u} = [0.8, 0.3, 0.5]$, $\mathbf{v} = [0.4, 0.7, 0.2]$
- Element-wise: $[0.32, 0.21, 0.10]$
- MF (equal weights): $0.32 + 0.21 + 0.10 = 0.63$
- Weighted ($\mathbf{h} = [2.0, 0.5, 0.1]$): $2.0(0.32) + 0.5(0.21) + 0.1(0.10) = 0.64 + 0.105 + 0.01 = 0.755$

*Notice that* $\mathbf{h}$ amplifies dimension 1 and suppresses dimension 3.

**This is GMF** (Generalized Matrix Factorization).

---

### Step 2: Concatenate Instead of Multiply

**Architecture**:
```
[user_emb, item_emb] → concatenate → linear layer → prediction
```

**Formula**: $\hat{r} = \mathbf{w}^T [\mathbf{u}; \mathbf{v}] + b$

**What changes?** Now we can learn **any linear function** of the concatenated embeddings, not just weighted element-wise products.

**Numerical example**:
- $\mathbf{u} = [0.8, 0.3]$, $\mathbf{v} = [0.4, 0.7]$
- Concatenate: $[0.8, 0.3, 0.4, 0.7]$
- Linear layer ($\mathbf{w} = [0.5, 0.2, 0.3, 0.4]$, $b = 0.1$):
  - $\hat{r} = 0.5(0.8) + 0.2(0.3) + 0.3(0.4) + 0.4(0.7) + 0.1$
  - $= 0.4 + 0.06 + 0.12 + 0.28 + 0.1 = 0.96$

**Still linear!** Adding layers without activation doesn't add expressiveness.

---

### Step 3: Add Non-Linear Activation

**Architecture**:
```
[user_emb, item_emb] → concatenate → linear → ReLU → linear → prediction
```

**Formula**: $\hat{r} = \mathbf{w}_2^T \text{ReLU}(\mathbf{W}_1 [\mathbf{u}; \mathbf{v}] + \mathbf{b}_1) + b_2$

**This is the key transition!** ReLU introduces non-linearity:
$$\text{ReLU}(x) = \max(0, x)$$

**Numerical example** (continuing from above):
- Input: $[0.8, 0.3, 0.4, 0.7]$
- Layer 1 ($\mathbf{W}_1 \in \mathbb{R}^{2 \times 4}$):
  - $\mathbf{W}_1 = \begin{bmatrix} 0.5 & 0.2 & -0.3 & 0.1 \\ -0.2 & 0.4 & 0.6 & -0.5 \end{bmatrix}$
  - $\mathbf{b}_1 = [0.1, -0.2]$
  - Pre-activation: $[0.5(0.8) + 0.2(0.3) - 0.3(0.4) + 0.1(0.7) + 0.1, ...]$
    - $= [0.4 + 0.06 - 0.12 + 0.07 + 0.1, -0.16 + 0.12 + 0.24 - 0.35 - 0.2]$
    - $= [0.51, -0.35]$
  - After ReLU: $[\max(0, 0.51), \max(0, -0.35)] = [0.51, 0]$

*Notice that* the second neuron "died" (output 0). This is ReLU creating a **piecewise linear** function!

- Layer 2 ($\mathbf{w}_2 = [0.6, 0.4]$, $b_2 = 0.2$):
  - $\hat{r} = 0.6(0.51) + 0.4(0) + 0.2 = 0.306 + 0.2 = 0.506$

**Capacity**: Now we can approximate any continuous function! (Universal Approximation Theorem)

---

### Step 4: Add More Layers (Go Deep)

**Architecture**:
```
[user_emb, item_emb] → concat → Linear → ReLU → Linear → ReLU → ... → prediction
```

**Formula**:
$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 [\mathbf{u}; \mathbf{v}] + \mathbf{b}_1)$$
$$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)$$
$$\vdots$$
$$\hat{r} = \mathbf{w}_L^T \mathbf{h}_{L-1} + b_L$$

**Why go deep?**
- Each layer learns more abstract features
- Layer 1: Raw interaction patterns ("user likes action")
- Layer 2: Combined patterns ("user likes action from specific era")
- Layer 3: Complex patterns ("user likes Nolan's action movies but not Bay's")

**This is the MLP component of NCF.**

---

## Complete Numerical Walkthrough: MF vs. Neural at Each Step

*Let's trace the same user-item pair through increasingly complex models.*

### Setup

**User Alice**: ID = 0
**Item "Inception"**: ID = 42
**Embedding dimension**: k = 4

**Learned embeddings** (same for all models):
- $\mathbf{u}_{Alice} = [0.8, -0.3, 0.5, 0.2]$
- $\mathbf{v}_{Inception} = [0.6, 0.4, -0.2, 0.7]$

### Model 1: Plain MF (Dot Product)

$$\hat{r} = \mathbf{u}^T \mathbf{v} = 0.8(0.6) + (-0.3)(0.4) + 0.5(-0.2) + 0.2(0.7)$$
$$= 0.48 - 0.12 - 0.10 + 0.14 = \boxed{0.40}$$

**Interpretation**: Moderate predicted rating.

---

### Model 2: GMF (Weighted Element-wise)

**Learned weights**: $\mathbf{h} = [1.5, 0.5, 2.0, 0.8]$

**Step 1** - Element-wise product:
$$\mathbf{u} \odot \mathbf{v} = [0.48, -0.12, -0.10, 0.14]$$

**Step 2** - Weighted sum:
$$\hat{r} = 1.5(0.48) + 0.5(-0.12) + 2.0(-0.10) + 0.8(0.14)$$
$$= 0.72 - 0.06 - 0.20 + 0.112 = \boxed{0.572}$$

**Interpretation**: Higher than MF because dimension 1 (where both are positive) is upweighted.

---

### Model 3: One Hidden Layer (ReLU)

**Learned parameters**:
- $\mathbf{W}_1 \in \mathbb{R}^{3 \times 8}$ (8 = concat of two 4-dim embeddings)
- $\mathbf{b}_1 \in \mathbb{R}^3$
- $\mathbf{w}_2 \in \mathbb{R}^3$, $b_2 \in \mathbb{R}$

**Step 1** - Concatenate:
$$\mathbf{z}_0 = [\mathbf{u}; \mathbf{v}] = [0.8, -0.3, 0.5, 0.2, 0.6, 0.4, -0.2, 0.7]$$

**Step 2** - First linear layer (simplified weights for illustration):
$$\mathbf{W}_1 = \begin{bmatrix} 0.3 & 0.5 & -0.2 & 0.1 & 0.4 & -0.3 & 0.2 & 0.1 \\ -0.4 & 0.2 & 0.6 & -0.1 & 0.3 & 0.5 & -0.4 & 0.2 \\ 0.1 & -0.3 & 0.4 & 0.5 & -0.2 & 0.1 & 0.6 & -0.3 \end{bmatrix}$$
$$\mathbf{b}_1 = [0.1, -0.1, 0.05]$$

Computing $\mathbf{W}_1 \mathbf{z}_0 + \mathbf{b}_1$:
- Neuron 1: $0.3(0.8) + 0.5(-0.3) + (-0.2)(0.5) + 0.1(0.2) + 0.4(0.6) + (-0.3)(0.4) + 0.2(-0.2) + 0.1(0.7) + 0.1$
  - $= 0.24 - 0.15 - 0.10 + 0.02 + 0.24 - 0.12 - 0.04 + 0.07 + 0.1 = 0.26$
- Neuron 2: $-0.4(0.8) + 0.2(-0.3) + 0.6(0.5) + (-0.1)(0.2) + 0.3(0.6) + 0.5(0.4) + (-0.4)(-0.2) + 0.2(0.7) - 0.1$
  - $= -0.32 - 0.06 + 0.30 - 0.02 + 0.18 + 0.20 + 0.08 + 0.14 - 0.1 = 0.40$
- Neuron 3: $0.1(0.8) + (-0.3)(-0.3) + 0.4(0.5) + 0.5(0.2) + (-0.2)(0.6) + 0.1(0.4) + 0.6(-0.2) + (-0.3)(0.7) + 0.05$
  - $= 0.08 + 0.09 + 0.20 + 0.10 - 0.12 + 0.04 - 0.12 - 0.21 + 0.05 = 0.11$

Pre-ReLU: $[0.26, 0.40, 0.11]$

**Step 3** - ReLU:
$$\mathbf{h}_1 = \text{ReLU}([0.26, 0.40, 0.11]) = [0.26, 0.40, 0.11]$$ (all positive, no change)

**Step 4** - Output layer ($\mathbf{w}_2 = [0.5, 0.3, 0.4]$, $b_2 = 0.15$):
$$\hat{r} = 0.5(0.26) + 0.3(0.40) + 0.4(0.11) + 0.15$$
$$= 0.13 + 0.12 + 0.044 + 0.15 = \boxed{0.444}$$

---

### Model 4: Two Hidden Layers (Deep)

**Add another layer before output**:

From Model 3, we had $\mathbf{h}_1 = [0.26, 0.40, 0.11]$

**Step 4b** - Second hidden layer ($\mathbf{W}_2 \in \mathbb{R}^{2 \times 3}$):
$$\mathbf{W}_2 = \begin{bmatrix} 0.6 & -0.4 & 0.3 \\ 0.2 & 0.5 & -0.2 \end{bmatrix}, \quad \mathbf{b}_2 = [0.05, -0.05]$$

$$\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2 = \begin{bmatrix} 0.6(0.26) - 0.4(0.40) + 0.3(0.11) + 0.05 \\ 0.2(0.26) + 0.5(0.40) - 0.2(0.11) - 0.05 \end{bmatrix}$$
$$= \begin{bmatrix} 0.156 - 0.16 + 0.033 + 0.05 \\ 0.052 + 0.20 - 0.022 - 0.05 \end{bmatrix} = \begin{bmatrix} 0.079 \\ 0.18 \end{bmatrix}$$

After ReLU: $\mathbf{h}_2 = [0.079, 0.18]$

**Step 5** - Output ($\mathbf{w}_3 = [0.7, 0.6]$, $b_3 = 0.1$):
$$\hat{r} = 0.7(0.079) + 0.6(0.18) + 0.1 = 0.055 + 0.108 + 0.1 = \boxed{0.263}$$

---

### Summary: Predictions Across Models

| Model | Architecture | Prediction | Notes |
|-------|-------------|------------|-------|
| MF | dot product | 0.40 | Baseline linear |
| GMF | weighted element-wise | 0.572 | Learned importance |
| 1-layer NN | concat → ReLU → linear | 0.444 | Basic non-linear |
| 2-layer NN | concat → ReLU → ReLU → linear | 0.263 | Deep non-linear |

*Can you see why* the predictions differ? Each model has a different "decision surface." The neural networks can capture patterns that MF cannot, but they might also fit noise if not regularized.

---

## When MF Is Sufficient (And Neural Networks Are Overkill)

*This is important: more complex is not always better.*

### The Dirty Secret: MF Often Wins

**A 2019 study** (Dacrema et al., "Are We Really Making Much Progress?") found:
- Many neural recommendation papers compared against **poorly tuned MF baselines**
- When MF is properly tuned, the gap shrinks dramatically
- On some datasets, **well-tuned MF beats untuned deep learning**

### When MF Works Well

1. **Simple patterns**: User preferences are approximately linear
2. **Dense data**: Many ratings per user/item (MF has enough signal)
3. **Latency-critical**: MF is 10-100x faster (just dot product)
4. **Interpretability needed**: MF factors are somewhat explainable
5. **Small datasets**: Neural networks overfit with <100K interactions

### Decision Table

| Situation | Choose MF | Choose Neural |
|-----------|-----------|---------------|
| Dataset size | < 1M interactions | > 10M interactions |
| Latency requirement | < 5ms | < 50ms acceptable |
| Team ML expertise | Limited | Strong |
| Need explainability | Yes | No (black box OK) |
| Available compute | CPU only | GPU available |
| Side features | Not available | Text, images, etc. available |
| Patterns in data | Mostly linear | Complex, non-linear |

### The Rule of Thumb

**Start with MF. Only move to neural if:**
1. You have lots of data (>10M interactions)
2. You have GPU infrastructure
3. You've tuned MF well and hit a ceiling
4. You have evidence of complex patterns MF can't capture

---

## What Can Go Wrong: Failure Modes of Neural Approaches

### Failure Mode 1: Neural Doesn't Help (Sparse Data)

**Symptoms**:
- Neural network performs same as or worse than MF
- Training loss similar to MF
- Adding layers doesn't improve validation metrics

**Concrete example**:
```
Dataset: 50K interactions, 10K users, 5K items
Average interactions per user: 5

MF (k=64):       NDCG@10 = 0.312
1-layer NN:      NDCG@10 = 0.308  ← Worse!
2-layer NN:      NDCG@10 = 0.295  ← Even worse!
3-layer NN:      NDCG@10 = 0.271  ← Terrible!
```

**Why this happens**:
- Neural networks have more parameters → need more data
- Sparse data: most user-item pairs are unobserved
- Neural networks learn spurious patterns from few examples
- MF's simplicity acts as implicit regularization

**Solutions**:
1. **Use MF** — it's the right tool for this data size
2. **If you must use neural**: heavy regularization (dropout 0.5, weight decay 1e-3)
3. **Collect more data** before switching to neural
4. **Use pre-trained embeddings** from MF to initialize neural network

---

### Failure Mode 2: Computational Overhead Without Benefit

**Symptoms**:
- Training takes 10-100x longer than MF
- Inference latency too high for production
- GPU costs exceed business value
- Similar final metrics to well-tuned MF

**Concrete example**:
```
MF training:     2 hours on CPU,  NDCG@10 = 0.415
Neural training: 48 hours on GPU, NDCG@10 = 0.423  ← Only 2% better!

Cost analysis:
- MF: $0 (CPU time is free)
- Neural: $200 (GPU rental for 48 hours)
- Improvement: 2% in NDCG

Is 2% worth $200 and 46 extra hours? Probably not.
```

**Why this happens**:
- Neural networks need many epochs to converge
- Each epoch processes all data through multiple layers
- Backpropagation through deep networks is expensive
- Marginal returns diminish with complexity

**Solutions**:
1. **Cost-benefit analysis** before choosing neural
2. **Early stopping** — don't train to convergence if unnecessary
3. **Smaller networks** — [64, 32] often works as well as [256, 128, 64]
4. **Knowledge distillation** — train small model to mimic large one

---

### Failure Mode 3: Overfitting Despite Regularization

**Symptoms**:
- Training loss: 0.01 (very low)
- Validation loss: 0.35 (much higher)
- Gap keeps growing with more training
- Dropout and weight decay don't fully fix it

**Concrete example**:
```
Architecture: [512, 256, 128, 64, 32]
Dropout: 0.5
Weight decay: 1e-4

Epoch 5:  Train=0.20, Val=0.25  ← Small gap, OK
Epoch 10: Train=0.08, Val=0.28  ← Gap growing
Epoch 20: Train=0.02, Val=0.38  ← Severe overfitting
Epoch 30: Train=0.01, Val=0.45  ← Model memorized training data
```

**Why this happens**:
- Network too expressive for data
- Each user-item pair is unique → easy to memorize
- Regularization can't overcome fundamental capacity mismatch

**Solutions**:
1. **Reduce network size dramatically**: Try [64, 32] first
2. **Use embedding regularization**:
```python
reg_loss = 0.01 * (user_emb.norm() + item_emb.norm())
total_loss = prediction_loss + reg_loss
```
3. **Aggressive early stopping**: Stop at first validation increase
4. **Consider if neural is appropriate** — MF might be better for your data size

---

### Failure Mode 4: Vanishing/Exploding Gradients

**Symptoms**:
- Training loss doesn't decrease (stuck at initial value)
- Gradients are all 0 or NaN
- Weights become very large or very small
- Model predicts same value for all inputs

**Concrete example**:
```
Epoch 1: Loss = 0.693 (random chance for binary)
Epoch 2: Loss = 0.693
Epoch 3: Loss = 0.693
...
Epoch 20: Loss = 0.693  ← Nothing learned!

Checking gradients: all ~1e-12 (vanishing)
```

**Why this happens**:
- Deep networks with poor initialization
- Wrong activation functions (sigmoid/tanh for deep nets)
- Learning rate too high (exploding) or too low (slow/vanishing)

**Solutions**:
1. **Use ReLU** (not sigmoid/tanh) for hidden layers
2. **Xavier/He initialization**:
```python
nn.init.xavier_uniform_(layer.weight)  # For tanh
nn.init.kaiming_uniform_(layer.weight)  # For ReLU
```
3. **Batch normalization** between layers:
```python
self.bn1 = nn.BatchNorm1d(hidden_size)
h = self.bn1(torch.relu(self.fc1(x)))
```
4. **Gradient clipping**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## The Hybrid Solution: Best of Both Worlds

### NeuMF: MF + Neural Combined

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

**Why this works**:
- **GMF path**: Captures linear patterns (fast, interpretable)
- **MLP path**: Captures non-linear patterns (expressive)
- **Fusion**: Model learns to weight each contribution
- **Result**: Best of both worlds

**Formula**:
$$\hat{r}_{ui} = \sigma(\mathbf{h}^T [\mathbf{u}_u^{GMF} \odot \mathbf{v}_i^{GMF}; \phi_{MLP}(\mathbf{u}_u^{MLP}, \mathbf{v}_i^{MLP})])$$

---

### Implementation Sketch

```python
class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=64, layers=[64, 32, 16, 8]):
        super().__init__()

        # GMF embeddings (for linear path)
        self.user_embedding_gmf = nn.Embedding(n_users, n_factors)
        self.item_embedding_gmf = nn.Embedding(n_items, n_factors)

        # MLP embeddings (for non-linear path)
        self.user_embedding_mlp = nn.Embedding(n_users, layers[0] // 2)
        self.item_embedding_mlp = nn.Embedding(n_items, layers[0] // 2)

        # MLP layers
        mlp_modules = []
        for i in range(len(layers) - 1):
            mlp_modules.append(nn.Linear(layers[i], layers[i+1]))
            mlp_modules.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_modules)

        # Final prediction (GMF output + MLP output → 1 score)
        self.output = nn.Linear(n_factors + layers[-1], 1)

    def forward(self, user_ids, item_ids):
        # GMF path: element-wise product
        user_gmf = self.user_embedding_gmf(user_ids)
        item_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_gmf * item_gmf

        # MLP path: concatenate → deep layers
        user_mlp = self.user_embedding_mlp(user_ids)
        item_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=1)
        mlp_output = self.mlp(mlp_input)

        # Fuse and predict
        concat = torch.cat([gmf_output, mlp_output], dim=1)
        prediction = torch.sigmoid(self.output(concat)).squeeze()

        return prediction
```

---

## Comparison Summary

| Aspect | Matrix Factorization | Neural Networks |
|--------|---------------------|-----------------|
| **Interaction** | Dot product (fixed) | Learned (flexible) |
| **Expressiveness** | Linear combinations | Arbitrary non-linear |
| **Parameters** | $O(k \cdot (|U| + |I|))$ | $O(k \cdot (|U| + |I|) + L \cdot d^2)$ |
| **Training speed** | Fast | Slower |
| **Inference speed** | Very fast | Fast (with GPU) |
| **Overfitting risk** | Lower | Higher |
| **Data requirements** | Moderate | Large |
| **Interpretability** | Somewhat interpretable | Black box |
| **Best for** | Dense ratings, simple patterns | Sparse data with features, complex patterns |

---

## Summary

**Key Takeaways**:

1. **MF is a one-layer neural network** with dot product interaction
2. **The bridge**: MF → +weights → +layers → +activation → deep network
3. **Non-linearity is the key**: ReLU enables complex patterns
4. **More complex is not always better**: MF often wins with proper tuning
5. **Hybrid models** (NeuMF) combine linear + non-linear advantages

**The fundamental insight**: Recommendation is about learning a user-item similarity function.
- MF learns: $\text{sim}(u, i) = \mathbf{u}^T \mathbf{v}$ (linear, fixed form)
- Neural learns: $\text{sim}(u, i) = f_\theta(\mathbf{u}, \mathbf{v})$ (non-linear, learned form)

**When the extra complexity pays off**:
- Large datasets (>10M interactions)
- Complex, non-linear user preferences
- Rich side information available
- Computational resources available

**When to stick with MF**:
- Small/medium datasets
- Latency-critical applications
- Need interpretability
- Limited compute budget

**Next Steps**: See **ncf.md** for full NCF implementation and training details.

---

## References

1. **He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017)**. "Neural Collaborative Filtering". *WWW*.
   - Foundational NCF paper

2. **Dacrema, M. F., et al. (2019)**. "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches". *RecSys*.
   - Critical analysis of neural vs. traditional methods

3. **Cheng, H. T., et al. (2016)**. "Wide & Deep Learning for Recommender Systems". *DLRS Workshop*.
   - Google's hybrid approach

4. **Covington, P., Adams, J., & Sargin, E. (2016)**. "Deep Neural Networks for YouTube Recommendations". *RecSys*.
   - Industrial deep learning at scale
