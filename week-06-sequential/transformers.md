# Week 6: Sequential Recommendation - Transformers

## Overview

Transformers have revolutionized NLP (GPT, BERT) and are now **state-of-the-art** for sequential recommendation. This document covers two dominant transformer-based models:

1. **SASRec** (Self-Attentive Sequential Recommendation) - Kang & McAuley, 2018
2. **BERT4Rec** (BERT for Sequential Recommendation) - Sun et al., 2019

These models outperform RNN-based approaches (GRU4Rec) and remain competitive in 2024-2025.

---

## Learning Objectives

By the end of this section, you will:
- Understand self-attention for sequences
- Implement SASRec from scratch (PyTorch)
- Master BERT4Rec's bidirectional training
- Compare transformers vs. RNNs for recommendation
- Apply these models to real datasets

---

## Why Do RNNs Fail? The Opening Failure Case

*Before we celebrate transformers, let me show you exactly where RNNs break down.*

### Problem 1: Sequential Bottleneck (No Parallelization)

**Scenario**: You have 1 million user sequences to process for training.

**RNN Processing**:
```
Sequence: [item1] -> [item2] -> [item3] -> [item4] -> [item5]
              |          |          |          |          |
            Step 1    Step 2    Step 3    Step 4    Step 5
              |          |          |          |          |
            wait...   wait...   wait...   wait...   done!
```

Each step MUST wait for the previous step. **You cannot parallelize within a sequence.**

**The Numbers** (processing 50-item sequences):

| Architecture | GPU Utilization | Time per Batch | Training Time (1M seqs) |
|-------------|-----------------|----------------|------------------------|
| GRU4Rec | ~15% | 50 sequential ops | **12 hours** |
| Transformer | ~90% | 1 parallel op | **45 minutes** |

*Can you see why* this matters? At scale, RNNs are **16x slower** than Transformers!

### Problem 2: Long-Range Dependencies (The Vanishing Path)

**Scenario**: User watched a Marvel movie 30 items ago. Now browsing random content. Should we recommend the new Marvel release?

**RNN's Path**: Information must flow through 30 sequential steps.

```
Marvel -> step1 -> step2 -> ... -> step29 -> step30 -> prediction
  |                                                        |
  Distance: 30 hops                                        |
  Gradient multiplied 30 times                             v
  Signal remaining: 0.5^30 = 0.0000000009              LOST!
```

**Transformer's Path**: Direct attention from position 30 to position 1.

```
Marvel -----------------------------------------> prediction
  |                                                    |
  Distance: 1 hop (direct attention)                   |
  No gradient multiplication                           v
  Signal remaining: 100%                            FOUND!
```

### Problem 3: The Position 1 Test

*Let me show you a concrete experiment.*

**Setup**: Train both models on movie sequences. Test if they can use item at position 1.

**Test**: Change ONLY position 1, measure prediction change.

```python
original = [Marvel_Movie, comedy, drama, action, romance, thriller]
modified = [Romance_Movie, comedy, drama, action, romance, thriller]
#           ^-- Only this changed

# Measure: How much does the prediction change?
```

**Results**:

| Model | Prediction Similarity | Position 1 Influence |
|-------|----------------------|---------------------|
| GRU (sequence length 6) | 0.92 | Weak |
| GRU (sequence length 20) | 0.98 | Almost none |
| GRU (sequence length 50) | 0.997 | **Essentially zero** |
| Transformer (any length) | 0.45 | **Strong** |

**The Transformer advantage**: Direct attention paths mean position 1 ALWAYS matters, regardless of sequence length.

### The Accuracy Gap

**Benchmark Results** (MovieLens-1M, RecSys papers):

| Model | Hit@10 | NDCG@10 | Training Time |
|-------|--------|---------|---------------|
| GRU4Rec | 0.586 | 0.367 | 4 hours |
| GRU4Rec + Attention | 0.601 | 0.382 | 5 hours |
| **SASRec (Transformer)** | **0.627** | **0.401** | **1.5 hours** |
| **BERT4Rec** | **0.641** | **0.410** | 2 hours |

**Result**: Transformers are **7% more accurate AND 2x faster** than RNNs!

*What happens if* we could design a model that:
1. Processes all positions in parallel
2. Has direct connections between ANY two positions
3. Learns which positions matter for each prediction

That's exactly what self-attention does.

---

## Why Transformers for Sequential Recommendation?

### The Sequential Recommendation Problem

**Given**: User's interaction history (ordered sequence)
$$S_u = [i_1, i_2, \ldots, i_t]$$

**Goal**: Predict next item $i_{t+1}$

**Examples**:
- E-commerce: [laptop -> mouse -> keyboard] -> predict USB hub
- Netflix: [Action movie -> Sci-fi -> Thriller] -> predict similar
- Spotify: [Rock -> Alternative -> Indie] -> predict next song

---

### RNNs vs. Transformers

| Aspect | RNN (GRU4Rec) | Transformer (SASRec, BERT4Rec) |
|--------|---------------|-------------------------------|
| **Processing** | Sequential (slow) | Parallel (fast) |
| **Long-term memory** | Struggles (vanishing gradients) | Excellent (direct attention) |
| **Training** | Slow (sequential backprop) | Fast (parallel processing) |
| **Inference** | Sequential | Parallel |
| **Performance** | Good | Better (+5-15% Hit@10) |

**Key advantage**: Transformers process entire sequence in parallel, directly model long-range dependencies.

---

## Self-Attention Mechanism: Complete Derivation

### The Intuition: Query, Key, Value

*Let me give you the intuition before the math.*

**Analogy**: Self-attention is like a **search engine inside your model**.

- **Query (Q)**: "What am I looking for?" (the current position asking a question)
- **Key (K)**: "What do I contain?" (each position advertising its content)
- **Value (V)**: "What information do I provide?" (the actual content to retrieve)

**Example**: Predicting what comes after [laptop, mouse, coffee, keyboard]

Position 4 (keyboard) asks: "What in my history is relevant to me?"

| Position | Item | Key (Advertisement) | Relevance to Keyboard |
|----------|------|--------------------|-----------------------|
| 1 | laptop | "I'm electronics, expensive" | **High** (same category) |
| 2 | mouse | "I'm electronics, peripheral" | **High** (often bought together) |
| 3 | coffee | "I'm food, cheap" | **Low** (unrelated) |
| 4 | keyboard | "I'm electronics, peripheral" | (self) |

**The attention mechanism learns**: keyboard should attend strongly to laptop and mouse, weakly to coffee.

---

### Mathematical Formulation: Step-by-Step Derivation

**Input**: Sequence of item embeddings
$$E = [\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_t] \in \mathbb{R}^{t \times d}$$

where $\mathbf{e}_i \in \mathbb{R}^d$ is embedding of item $i$.

---

#### Step 1: Create Q, K, V Matrices

**Intuition**: Transform embeddings into three different "views" of the same data.

**Equations**:
$$Q = EW^Q, \quad K = EW^K, \quad V = EW^V$$

where $W^Q, W^K, W^V \in \mathbb{R}^{d \times d}$ are learned matrices.

**Why three separate transformations?**

- *Why not just use E directly?* Because the same item might need to:
  - Ask different questions (as a query)
  - Advertise different features (as a key)
  - Provide different information (as a value)

- *Example*: A "laptop" might:
  - Query: "Find me accessories"
  - Key: "I'm expensive electronics"
  - Value: "Here's my brand/price information"

**Why this term?**: Separate projections give the model flexibility to learn different aspects for matching (Q,K) vs. information retrieval (V).

---

#### Step 2: Compute Attention Scores

**Intuition**: Measure how much each position should attend to every other position.

**Equation**:
$$\text{Scores} = QK^T \in \mathbb{R}^{t \times t}$$

**Why matrix multiplication?**

Each entry $(i,j)$ is the dot product $\mathbf{q}_i \cdot \mathbf{k}_j$, which measures similarity.

**Why this term?**: Dot product is:
1. Computationally efficient (can be parallelized)
2. Captures similarity (high when vectors point same direction)
3. Learnable (through $W^Q$ and $W^K$)

---

#### Step 3: Scale by $\sqrt{d}$

**Intuition**: Prevent scores from getting too large.

**Equation**:
$$\text{Scaled Scores} = \frac{QK^T}{\sqrt{d}}$$

**Why scale by $\sqrt{d}$?**

*This is subtle but critical.* Let me show you with numbers.

**Without scaling** (d=64):
- Each entry of $QK^T$ is sum of 64 products
- If Q,K entries have variance 1, the sum has variance ~64
- Scores might be: [32.5, -28.3, 45.1, -15.7]
- After softmax: [0.0001, 0.0000, 0.9999, 0.0000] -- **too peaked!**

**With scaling** (divide by $\sqrt{64} = 8$):
- Scores become: [4.06, -3.54, 5.64, -1.96]
- After softmax: [0.12, 0.01, 0.85, 0.02] -- **reasonable distribution**

**Why this term?**: Scaling maintains variance ~1, which keeps softmax gradients healthy. Without it, softmax saturates and gradients vanish.

---

#### Step 4: Apply Softmax

**Intuition**: Convert scores to probabilities (how much to attend to each position).

**Equation**:
$$\text{Attention Weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \in \mathbb{R}^{t \times t}$$

**Why softmax?**:
1. Outputs sum to 1 (valid probability distribution)
2. Amplifies differences (winner-take-more)
3. Differentiable (for backprop)

**Why this term?**: Each row tells us "for this position, what fraction of attention goes to each other position?"

---

#### Step 5: Compute Weighted Sum of Values

**Intuition**: Retrieve information from positions we're attending to.

**Equation**:
$$\text{Output} = \text{Attention Weights} \times V \in \mathbb{R}^{t \times d}$$

**Why this term?**: Weighted average of value vectors, where weights are the attention scores. Positions we attend to contribute more.

---

### Complete Self-Attention Equation

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

*Can you see why* this is powerful? Every position can directly access information from every other position, with learned importance weights!

---

## Complete Numerical Walkthrough: Self-Attention with 3 Items

*Let me walk through the complete computation with actual numbers.*

### Setup

**User's sequence**: [Laptop, Mouse, Keyboard]

**Item embeddings** (d=4 for simplicity):
```
e_laptop   = [1.0, 0.5, 0.2, 0.8]
e_mouse    = [0.9, 0.6, 0.3, 0.7]
e_keyboard = [0.8, 0.7, 0.2, 0.6]
```

**Embedding matrix E** (3 x 4):
```
E = [[1.0, 0.5, 0.2, 0.8],   <- laptop
     [0.9, 0.6, 0.3, 0.7],   <- mouse
     [0.8, 0.7, 0.2, 0.6]]   <- keyboard
```

**Weight matrices** (assume identity for simplicity, so Q=K=V=E):
```
W_Q = W_K = W_V = I (4x4 identity)
```

---

### Step 1: Compute Q, K, V

Since $W^Q = W^K = W^V = I$:

$$Q = K = V = E$$

```
Q = K = V = [[1.0, 0.5, 0.2, 0.8],
             [0.9, 0.6, 0.3, 0.7],
             [0.8, 0.7, 0.2, 0.6]]
```

---

### Step 2: Compute QK^T (Similarity Matrix)

Each entry $(i,j) = \mathbf{q}_i \cdot \mathbf{k}_j$ (dot product).

**Row 1** (Laptop queries all items):
- Laptop-Laptop: $1.0 \times 1.0 + 0.5 \times 0.5 + 0.2 \times 0.2 + 0.8 \times 0.8 = 1.0 + 0.25 + 0.04 + 0.64 = \mathbf{1.93}$
- Laptop-Mouse: $1.0 \times 0.9 + 0.5 \times 0.6 + 0.2 \times 0.3 + 0.8 \times 0.7 = 0.9 + 0.3 + 0.06 + 0.56 = \mathbf{1.82}$
- Laptop-Keyboard: $1.0 \times 0.8 + 0.5 \times 0.7 + 0.2 \times 0.2 + 0.8 \times 0.6 = 0.8 + 0.35 + 0.04 + 0.48 = \mathbf{1.67}$

**Row 2** (Mouse queries all items):
- Mouse-Laptop: $\mathbf{1.82}$ (same as Laptop-Mouse by symmetry)
- Mouse-Mouse: $0.9^2 + 0.6^2 + 0.3^2 + 0.7^2 = 0.81 + 0.36 + 0.09 + 0.49 = \mathbf{1.75}$
- Mouse-Keyboard: $0.9 \times 0.8 + 0.6 \times 0.7 + 0.3 \times 0.2 + 0.7 \times 0.6 = 0.72 + 0.42 + 0.06 + 0.42 = \mathbf{1.62}$

**Row 3** (Keyboard queries all items):
- Keyboard-Laptop: $\mathbf{1.67}$
- Keyboard-Mouse: $\mathbf{1.62}$
- Keyboard-Keyboard: $0.8^2 + 0.7^2 + 0.2^2 + 0.6^2 = 0.64 + 0.49 + 0.04 + 0.36 = \mathbf{1.53}$

**QK^T matrix**:
```
                Keys:
              Laptop  Mouse  Keyboard
Queries:
Laptop      [[ 1.93   1.82    1.67  ]
Mouse        [ 1.82   1.75    1.62  ]
Keyboard     [ 1.67   1.62    1.53  ]]
```

*Notice that* Laptop-Laptop has the highest score (1.93) because items are most similar to themselves!

---

### Step 3: Scale by sqrt(d)

$$\sqrt{d} = \sqrt{4} = 2$$

**Scaled scores**:
```
              Laptop  Mouse  Keyboard
Laptop      [[ 0.965  0.910   0.835  ]
Mouse        [ 0.910  0.875   0.810  ]
Keyboard     [ 0.835  0.810   0.765  ]]
```

---

### Step 4: Apply Softmax (Row-wise)

**Row 1** (Laptop's attention distribution):
- $\exp(0.965) = 2.625$
- $\exp(0.910) = 2.484$
- $\exp(0.835) = 2.305$
- Sum = 7.414
- Softmax: $[2.625/7.414, 2.484/7.414, 2.305/7.414] = \mathbf{[0.354, 0.335, 0.311]}$

**Row 2** (Mouse's attention distribution):
- Softmax: $\mathbf{[0.344, 0.333, 0.323]}$

**Row 3** (Keyboard's attention distribution):
- Softmax: $\mathbf{[0.340, 0.332, 0.328]}$

**Attention Weights matrix**:
```
               Attend to:
              Laptop  Mouse  Keyboard
Position:
Laptop      [[ 0.354  0.335   0.311  ]    <- Laptop attends most to itself
Mouse        [ 0.344  0.333   0.323  ]    <- Mouse fairly uniform
Keyboard     [ 0.340  0.332   0.328  ]]   <- Keyboard fairly uniform
```

*Notice that* attention is relatively uniform here because all items are similar (electronics). In practice, with diverse items (laptop vs. coffee), differences would be much larger.

---

### Step 5: Compute Output (Weighted Sum of V)

**Laptop's new representation**:
$$\mathbf{h}_{laptop} = 0.354 \cdot \mathbf{v}_{laptop} + 0.335 \cdot \mathbf{v}_{mouse} + 0.311 \cdot \mathbf{v}_{keyboard}$$

$$= 0.354 \times [1.0, 0.5, 0.2, 0.8] + 0.335 \times [0.9, 0.6, 0.3, 0.7] + 0.311 \times [0.8, 0.7, 0.2, 0.6]$$

$$= [0.354, 0.177, 0.071, 0.283] + [0.302, 0.201, 0.101, 0.235] + [0.249, 0.218, 0.062, 0.187]$$

$$= \mathbf{[0.905, 0.596, 0.234, 0.705]}$$

**All outputs**:
```
Output = [[0.905, 0.596, 0.234, 0.705],   <- Laptop's new representation
          [0.899, 0.602, 0.237, 0.698],   <- Mouse's new representation
          [0.893, 0.608, 0.239, 0.691]]   <- Keyboard's new representation
```

*Can you see what happened?* Each item's representation now contains information from ALL items in the sequence, weighted by relevance!

---

### What Did Self-Attention Learn?

**Before attention**: Each item embedding was independent.
```
Laptop:   [1.0, 0.5, 0.2, 0.8]  <- Only knows about laptop
Mouse:    [0.9, 0.6, 0.3, 0.7]  <- Only knows about mouse
Keyboard: [0.8, 0.7, 0.2, 0.6]  <- Only knows about keyboard
```

**After attention**: Each embedding contains context from the whole sequence.
```
Laptop:   [0.905, 0.596, 0.234, 0.705]  <- Knows laptop + mouse + keyboard context
Mouse:    [0.899, 0.602, 0.237, 0.698]  <- Knows laptop + mouse + keyboard context
Keyboard: [0.893, 0.608, 0.239, 0.691]  <- Knows laptop + mouse + keyboard context
```

**For prediction**: The last position (keyboard) now encodes:
- "User looked at laptop first" (35% weight)
- "User then looked at mouse" (33% weight)
- "User currently viewing keyboard" (33% weight)

This is EXACTLY what we need to predict "USB hub" or "monitor" next!

---

## Multi-Head Attention: Learning Multiple Patterns

### Intuition

**Problem**: One attention pattern might not capture all relevant relationships.

**Example**: When predicting next item for [laptop, mouse, coffee, keyboard]:
- **Pattern 1**: Category relevance (laptop-mouse-keyboard are electronics)
- **Pattern 2**: Recency (keyboard was just viewed)
- **Pattern 3**: Price similarity (laptop and keyboard are expensive)

**Solution**: Learn multiple attention patterns in parallel ("heads").

### Mathematical Formulation

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

where:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Typical**: $h = 2$ or $h = 4$ heads

---

### What Different Heads Learn

*Research has shown that different heads learn different patterns.*

**Head 1: "Category Attention"**
```
Attention to: [laptop: 0.45, mouse: 0.40, coffee: 0.05, keyboard: 0.10]
Pattern: Electronics items attend to each other
```

**Head 2: "Recency Attention"**
```
Attention to: [laptop: 0.10, mouse: 0.20, coffee: 0.25, keyboard: 0.45]
Pattern: Recent items get more attention
```

**Combined**: The model uses BOTH patterns for prediction!

---

## Positional Encoding: Why and How

### The Problem: Transformers Don't Know Order!

**Critical insight**: Self-attention is **permutation-invariant**.

$$\text{Attention}([A, B, C]) = \text{Attention}([C, A, B])$$

The output is the same regardless of order!

*Can you see why* this is a problem for sequential recommendation? [laptop -> mouse -> keyboard] is VERY different from [keyboard -> mouse -> laptop]!

### The Solution: Positional Encoding

**Idea**: Add position information to embeddings before attention.

$$\mathbf{e}_i' = \mathbf{e}_i + \mathbf{p}_i$$

where $\mathbf{p}_i$ is a position encoding for position $i$.

---

### Two Approaches: Learned vs. Sinusoidal

**Approach 1: Learned Positional Embeddings** (used in SASRec, BERT4Rec)

$$\mathbf{p}_i = \text{Embedding}(i)$$

- Simply learn a separate embedding for each position
- Pros: Flexible, can learn arbitrary patterns
- Cons: Can't extrapolate to longer sequences than seen in training

**Approach 2: Sinusoidal Positional Encoding** (original Transformer paper)

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**Why sinusoidal?**

1. **Unique encoding**: Each position has a unique pattern
2. **Relative positions**: $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$
3. **Generalization**: Can extrapolate to longer sequences

---

### Sinusoidal Encoding: Numerical Example

**Let's compute PE for position 0 and position 1** (d=4):

**Position 0**:
- $PE_{(0,0)} = \sin(0/10000^{0/4}) = \sin(0) = 0$
- $PE_{(0,1)} = \cos(0/10000^{0/4}) = \cos(0) = 1$
- $PE_{(0,2)} = \sin(0/10000^{2/4}) = \sin(0) = 0$
- $PE_{(0,3)} = \cos(0/10000^{2/4}) = \cos(0) = 1$

$\mathbf{p}_0 = [0, 1, 0, 1]$

**Position 1**:
- $PE_{(1,0)} = \sin(1/10000^{0}) = \sin(1) = 0.841$
- $PE_{(1,1)} = \cos(1/10000^{0}) = \cos(1) = 0.540$
- $PE_{(1,2)} = \sin(1/10000^{0.5}) = \sin(0.01) = 0.010$
- $PE_{(1,3)} = \cos(1/10000^{0.5}) = \cos(0.01) = 0.9999$

$\mathbf{p}_1 = [0.841, 0.540, 0.010, 0.9999]$

*Notice that* position 0 and position 1 have clearly different encodings!

---

## SASRec: Self-Attentive Sequential Recommendation

### Paper

**Kang, W. C., & McAuley, J. (2018)**. "Self-Attentive Sequential Recommendation". *IEEE ICDM*.

**Key innovation**: Use transformers for item-to-item recommendation.

---

### Architecture

```
Input:           [i_1, i_2, ..., i_t]
                        |
Item Embedding:  [e_1, e_2, ..., e_t]
                        |
Positional Encoding: Add position info
                        |
Self-Attention Block (xL layers)
    |-- Multi-Head Attention
    |-- Add & Norm
    |-- Feed-Forward Network
    |-- Add & Norm
                        |
Output:          [h_1, h_2, ..., h_t]
                        |
Prediction:      Softmax over all items
```

**L**: Number of transformer blocks (typically 2-4)

---

### Causal Attention (Left-to-Right Mask)

**Key constraint**: When predicting $i_t$, can only see $[i_1, \ldots, i_{t-1}]$

**Why?** During training, we predict ALL positions simultaneously. Position 3 shouldn't see position 4!

---

### Visualizing the Causal Mask

*Let me show you exactly what the mask looks like.*

**Sequence**: [Item1, Item2, Item3, Item4]

**Causal mask matrix** (what each position can attend to):

```
              Keys (what we attend TO)
           Item1  Item2  Item3  Item4
Query:
Item1      [ Y      X      X      X  ]   <- Can only see itself
Item2      [ Y      Y      X      X  ]   <- Can see Item1, Item2
Item3      [ Y      Y      Y      X  ]   <- Can see Item1-3
Item4      [ Y      Y      Y      Y  ]   <- Can see everything
```

**As a numerical matrix** (added to attention scores before softmax):

```
           Item1   Item2   Item3   Item4
Item1    [   0      -inf    -inf    -inf ]
Item2    [   0       0      -inf    -inf ]
Item3    [   0       0       0      -inf ]
Item4    [   0       0       0       0   ]
```

**Why -inf?** After softmax, $e^{-\infty} = 0$, so those positions get zero attention weight.

---

### Causal Mask: Numerical Example

**Unmasked attention scores** (before mask):
```
           Item1   Item2   Item3   Item4
Item1    [  2.1     1.8     1.5     1.2  ]
Item2    [  1.9     2.3     1.7     1.4  ]
Item3    [  1.6     1.8     2.0     1.3  ]
Item4    [  1.4     1.6     1.7     1.9  ]
```

**After adding mask**:
```
           Item1   Item2   Item3   Item4
Item1    [  2.1    -inf    -inf    -inf  ]
Item2    [  1.9     2.3    -inf    -inf  ]
Item3    [  1.6     1.8     2.0    -inf  ]
Item4    [  1.4     1.6     1.7     1.9  ]
```

**After softmax** (each row sums to 1):
```
           Item1   Item2   Item3   Item4
Item1    [ 1.000   0.000   0.000   0.000 ]  <- Only attends to itself
Item2    [ 0.401   0.599   0.000   0.000 ]  <- Attends to Item1-2
Item3    [ 0.268   0.327   0.405   0.000 ]  <- Attends to Item1-3
Item4    [ 0.182   0.222   0.246   0.350 ]  <- Attends to all
```

*Notice that* Item1 has no choice but to attend 100% to itself. Item4 can distribute attention across all items.

---

### Loss Function

**Training**: Predict next item at each position.

$$\mathcal{L} = -\sum_{S_u \in \mathcal{D}} \sum_{t=1}^{|S_u|} \log P(i_t | [i_1, \ldots, i_{t-1}])$$

where:
- $\mathcal{D}$ = training dataset (set of all user sequences)
- $S_u$ = interaction sequence for user $u$
- $|S_u|$ = length of user $u$'s sequence
- $i_t$ = item at position $t$ in the sequence

$$P(i_t | \mathbf{h}_{t-1}) = \frac{\exp(\mathbf{h}_{t-1}^T \mathbf{e}_{i_t})}{\sum_{i' \in \mathcal{I}} \exp(\mathbf{h}_{t-1}^T \mathbf{e}_{i'})}$$

where:
- $\mathbf{h}_{t-1} \in \mathbb{R}^d$ = transformer output embedding at position $t-1$ (after all attention layers)
- $\mathbf{e}_{i_t} \in \mathbb{R}^d$ = item embedding for item $i_t$ (often shared with input embeddings)
- $\mathcal{I}$ = set of all possible items in the catalog
- $d$ = embedding dimension

**Softmax over all items**: Expensive when $|\mathcal{I}|$ is large! Use negative sampling or sampled softmax.

---

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SASRec(nn.Module):
    def __init__(self, n_items, max_len=50, d_model=64, n_heads=2, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.max_len = max_len

        # Item embedding
        self.item_embedding = nn.Embedding(n_items + 1, d_model, padding_idx=0)

        # Positional embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln = nn.LayerNorm(d_model)

        # Output projection (shared with item embedding)
        self.output = nn.Linear(d_model, n_items + 1)

    def forward(self, seq, positions):
        """
        seq: (batch, seq_len) - item IDs
        positions: (batch, seq_len) - position indices
        """
        # Embeddings
        item_emb = self.item_embedding(seq)  # (batch, seq_len, d_model)
        pos_emb = self.pos_embedding(positions)
        x = item_emb + pos_emb

        # Causal attention mask (upper triangular)
        seq_len = seq.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(seq.device)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Layer norm
        x = self.ln(x)

        # Output logits
        logits = self.output(x)  # (batch, seq_len, n_items+1)

        return logits

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # Multi-head attention
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + attn_out)  # Add & Norm

        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)  # Add & Norm

        return x

# Training loop
def train_sasrec(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        seqs, targets, positions = batch  # seqs: (batch, seq_len), targets: (batch, seq_len)
        seqs, targets, positions = seqs.to(device), targets.to(device), positions.to(device)

        # Forward pass
        logits = model(seqs, positions)  # (batch, seq_len, n_items+1)

        # Compute loss (ignore padding tokens)
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        loss = F.cross_entropy(logits, targets, ignore_index=0)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# Recommendation
def recommend(model, user_seq, top_k=10):
    """
    user_seq: List of item IDs (user's history)
    Returns: Top-K item IDs
    """
    model.eval()
    seq = torch.LongTensor([user_seq]).to(device)
    positions = torch.LongTensor([list(range(len(user_seq)))]).to(device)

    with torch.no_grad():
        logits = model(seq, positions)  # (1, seq_len, n_items+1)
        logits = logits[0, -1, :]  # Last position's logits

        # Mask already interacted items
        for item_id in user_seq:
            logits[item_id] = -float('inf')

        # Top-K
        top_items = torch.topk(logits, top_k).indices.cpu().numpy()

    return top_items
```

---

## BERT4Rec: Bidirectional Sequential Recommendation

### Paper

**Sun, F., et al. (2019)**. "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer". *CIKM*.

**Key innovation**: Use BERT's masked language model approach for recommendation.

---

### Differences from SASRec

| Aspect | SASRec | BERT4Rec |
|--------|--------|----------|
| **Direction** | Left-to-right (causal) | Bidirectional |
| **Training** | Predict next item | Predict masked items |
| **Attention** | Causal mask | Full attention (no mask) |
| **Inference** | Append to sequence | Mask last position |

---

### The Cloze Task: Masked Item Prediction

**Idea**: Randomly mask items in sequence, predict them using BOTH past AND future context.

**Example**:
```
Original:  [laptop, mouse, keyboard, USB hub]
Masked:    [laptop, mouse, [MASK], USB hub]
Task:      Predict "keyboard" using context from BOTH sides!
```

**Why is this powerful?**

*SASRec predicting "keyboard"*: Can only see [laptop, mouse]
*BERT4Rec predicting "keyboard"*: Can see [laptop, mouse, ???, USB hub]

BERT4Rec has MORE context!

---

### Training vs. Inference

**Training** (mask 15-20% randomly):
```
Input:  [laptop, [MASK], keyboard, [MASK], USB hub]
Target: Predict mouse at position 2, monitor at position 4
```

**Inference** (mask last position):
```
Input:  [laptop, mouse, keyboard, USB hub, [MASK]]
Task:   Predict what comes next
```

*Notice that* at inference, we append a [MASK] token and predict what should fill it.

---

### Why Bidirectional Matters: A Concrete Example

**Sequence**: [Action, Comedy, [MASK], Romance, Action]

**SASRec (causal)** at position 3:
- Can see: [Action, Comedy]
- Might predict: Comedy (continuing the pattern)
- Context: 2 items

**BERT4Rec (bidirectional)** at position 3:
- Can see: [Action, Comedy, ???, Romance, Action]
- More likely to predict: Drama (bridges comedy and romance)
- Context: 4 items

**The math**: More context = better predictions!

---

### Implementation (Key Differences from SASRec)

```python
class BERT4Rec(nn.Module):
    def __init__(self, n_items, max_len=50, d_model=64, n_heads=2, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_items = n_items

        # Special tokens: 0=padding, n_items+1=[MASK]
        self.item_embedding = nn.Embedding(n_items + 2, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # Transformer blocks (NO causal mask!)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, n_items + 2)

    def forward(self, seq, positions):
        # Embeddings
        item_emb = self.item_embedding(seq)
        pos_emb = self.pos_embedding(positions)
        x = item_emb + pos_emb

        # NO causal mask! (bidirectional)
        # Only mask padding tokens
        padding_mask = (seq == 0)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, padding_mask)

        x = self.ln(x)
        logits = self.output(x)

        return logits

def mask_sequence(seq, mask_prob=0.15):
    """
    Randomly mask items in sequence for training.

    Returns:
        masked_seq: Sequence with [MASK] tokens
        targets: Original items at masked positions
        mask_indices: Which positions were masked
    """
    masked_seq = seq.clone()
    seq_len = len(seq)

    # Random mask
    mask_indices = torch.rand(seq_len) < mask_prob
    mask_indices = mask_indices & (seq != 0)  # Don't mask padding

    # Replace with [MASK] token (assume n_items+1 is [MASK])
    masked_seq[mask_indices] = n_items + 1

    targets = seq[mask_indices]

    return masked_seq, targets, mask_indices
```

---

## SASRec vs. BERT4Rec: When to Use Which

### Decision Framework

*Which should you choose? Here's a practical guide.*

```
                         Start Here
                             |
                What's your use case?
                             |
            +----------------+----------------+
            |                                 |
       Real-time                         Batch/Offline
    (recommendations                    (daily email,
     as user browses)                   weekly digest)
            |                                 |
            v                                 v
         SASRec                            BERT4Rec
    (can predict next                   (can look at
     item immediately)                   full context)
```

---

### Choose SASRec when:

1. **Real-time recommendations**: User adds item to cart -> instantly recommend
2. **Streaming data**: Continuously updating as user browses
3. **Inference latency matters**: <50ms response time needed
4. **Simpler deployment**: Causal = autoregressive = straightforward inference

---

### Choose BERT4Rec when:

1. **Batch recommendations**: Generate candidates overnight, serve all day
2. **Maximum accuracy matters**: Can afford training time for +2% improvement
3. **Rich context available**: Want to leverage future context for predictions
4. **Fill-in-the-blank scenarios**: Impute missing items in a sequence

---

### The Practical Reality

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| E-commerce live session | SASRec | Speed matters |
| Netflix homepage (daily) | BERT4Rec | Accuracy matters, computed overnight |
| YouTube "Up Next" | SASRec | Real-time |
| Spotify Weekly Discover | BERT4Rec | Weekly batch |
| News recommendation | SASRec | Very fast-changing |
| Course recommendation | BERT4Rec | User plans slowly |

*When in doubt*: Start with SASRec (simpler), upgrade to BERT4Rec if you hit accuracy ceiling.

---

### Performance Comparison

**MovieLens-1M** (Sun et al., 2019):

| Model | Hit@10 | NDCG@10 |
|-------|--------|---------|
| GRU4Rec | 0.5860 | 0.3670 |
| Caser | 0.5930 | 0.3720 |
| SASRec | 0.6270 | 0.4010 |
| **BERT4Rec** | **0.6410** | **0.4100** |

**Improvement**: BERT4Rec +2.2% Hit@10 over SASRec

---

## What Can Go Wrong: Failure Modes and Solutions

*Every model has failure modes. Let me show you the key problems with transformer-based recommenders.*

### Failure Mode 1: Quadratic Complexity O(n^2)

**The Problem**: Self-attention computes ALL pairwise similarities.

**The Math**:
- Attention matrix size: $t \times t$ where $t$ = sequence length
- Memory: $O(t^2 \cdot h)$ where $h$ = number of heads
- Computation: $O(t^2 \cdot d)$ where $d$ = embedding dimension

**Concrete Numbers**:

| Sequence Length | Attention Matrix Size | Memory (float32) |
|----------------|----------------------|------------------|
| 50 | 2,500 | 10 KB |
| 200 | 40,000 | 160 KB |
| 500 | 250,000 | 1 MB |
| 1,000 | 1,000,000 | 4 MB |
| 5,000 | 25,000,000 | **100 MB per head!** |

*What happens if* you have users with 5,000-item histories? You run out of GPU memory!

**Solutions**:

1. **Truncate sequences**: Use only last 50-200 items
   ```python
   max_seq_len = 200
   truncated_seq = seq[-max_seq_len:]
   ```

2. **Sparse attention patterns**: Longformer, BigBird
   ```python
   # Only attend to nearby items + some global tokens
   # Complexity: O(n) instead of O(n^2)
   ```

3. **Linear attention**: Performer, Linear Transformer
   ```python
   # Use kernel approximation
   # Complexity: O(n) instead of O(n^2)
   ```

---

### Failure Mode 2: Position Bias

**The Problem**: Model over-relies on positional patterns instead of item content.

**Symptom**: Recommendations are similar regardless of which items are in the sequence.

**Diagnosis**: Shuffle items in sequence, check if prediction changes.

```python
# Test: Does item ORDER matter more than item CONTENT?
original = [laptop, mouse, keyboard, coffee]
shuffled = [coffee, keyboard, laptop, mouse]

pred_original = model(original)
pred_shuffled = model(shuffled)

if similarity(pred_original, pred_shuffled) < 0.5:
    print("Good: Model uses both position and content")
else:
    print("WARNING: Position bias - model ignores content!")
```

**Common Patterns**:
- "Always recommend what's popular at position 1"
- "Last item = most important" (regardless of what it is)

**Solutions**:

1. **Position dropout**: Randomly zero out positional encodings during training
   ```python
   if training and random() < 0.1:
       pos_emb = torch.zeros_like(pos_emb)
   ```

2. **Content-based regularization**: Add auxiliary loss on item content
   ```python
   loss = ce_loss + 0.1 * content_similarity_loss
   ```

3. **Relative positional encoding**: Learn relative positions instead of absolute
   ```python
   # Instead of: e_i + p_i
   # Use: attention(q_i, k_j + r_{i-j})  # r is relative position encoding
   ```

---

### Failure Mode 3: Attention Dilution (Too Many Items)

**The Problem**: With long sequences, attention gets spread too thin.

**The Math**:
- Softmax normalizes to sum = 1
- With 100 items, average attention = 0.01 per item
- Important items might only get 0.03 attention (barely above average!)

**Visualization**:
```
Short sequence (5 items):
Attention: [0.35, 0.25, 0.20, 0.15, 0.05]
           ^^^^  ^^^^
           Clear winners!

Long sequence (100 items):
Attention: [0.02, 0.02, 0.02, ..., 0.015, 0.015, ...]
           All roughly equal - no clear signal!
```

**Diagnosis**: Check attention entropy.

```python
# High entropy = uniform attention = diluted
entropy = -sum(attention * log(attention))
if entropy > 0.9 * log(seq_len):
    print("WARNING: Attention is too uniform!")
```

**Solutions**:

1. **Top-k attention**: Only attend to top-k most relevant items
   ```python
   scores = QK^T / sqrt(d)
   top_k_mask = scores < top_k_threshold(scores, k=20)
   scores[top_k_mask] = -inf
   attention = softmax(scores)
   ```

2. **Temperature scaling**: Sharpen attention distribution
   ```python
   # Lower temperature = sharper attention
   attention = softmax(scores / temperature)  # temperature < 1
   ```

3. **Local + global attention**: Attend to nearby items + a few global tokens
   ```python
   # Attend to: positions t-10 to t (local) + position 0 (global first item)
   ```

---

### Failure Mode 4: Cold Start (No History)

**The Problem**: New users have no sequence to process.

**Symptom**: Terrible recommendations for users with < 3 items.

| History Length | Hit@10 | Problem? |
|---------------|--------|----------|
| 0 items | N/A | Complete cold start |
| 1 item | 0.15 | Severe |
| 2 items | 0.28 | Moderate |
| 5+ items | 0.55 | Normal |

**Solutions**:

1. **Default sequence**: Use popular items as "synthetic history"
   ```python
   if len(user_seq) < 3:
       user_seq = popular_items[:3] + user_seq
   ```

2. **Side information embedding**: Use user features when history is short
   ```python
   if len(user_seq) < 3:
       return content_based_recommendations(user_features)
   ```

3. **Pre-trained item embeddings**: Use item2vec or content embeddings
   ```python
   item_embedding.weight = pretrained_item2vec_embeddings
   ```

---

### Summary: Failure Mode Checklist

| Failure Mode | Symptom | Quick Test | Solution |
|-------------|---------|------------|----------|
| O(n^2) complexity | OOM on long sequences | Check memory with len=500 | Truncation, sparse attention |
| Position bias | Same recs for different items | Shuffle items, check prediction | Position dropout, relative PE |
| Attention dilution | Uniform attention weights | Check attention entropy | Top-k attention, temperature |
| Cold start | Bad for short sequences | Check Hit@10 for len=1,2,3 | Default sequence, side info |

---

## Practical Considerations

### 1. Negative Sampling

**Problem**: Softmax over millions of items is slow!

**Solution**: Sampled softmax
- Sample 100-1000 negative items per positive
- Compute softmax over (1 positive + K negatives)

---

### 2. Sequence Length

**Challenge**: Transformers have $O(L^2)$ complexity in sequence length $L$.

**Solutions**:
- **Truncate**: Max length 50-200
- **Sliding window**: Only consider recent items
- **Sparse attention**: Longformer, BigBird variants

---

### 3. Hyperparameters

| Parameter | Typical Range | Recommendation |
|-----------|---------------|----------------|
| d_model | 32-256 | 64 for small datasets, 128-256 for large |
| n_heads | 1-8 | 2-4 (more heads = more parameters) |
| n_layers | 1-6 | 2 for most cases |
| Dropout | 0.1-0.5 | 0.2 (higher for overfitting) |
| Learning rate | 0.0001-0.001 | 0.001 with Adam |
| Batch size | 32-512 | 128 (GPU memory dependent) |

---

### 4. Data Augmentation

**Sequence cropping**:
- Original: [i1, i2, i3, i4, i5]
- Augmented: [i1, i2], [i1, i2, i3], [i1, i2, i3, i4], ...

**Benefits**: More training samples, better generalization.

---

## Industry Applications

### Alibaba (2019)
- **BST** (Behavior Sequence Transformer)
- Process user click sequences
- Production deployment with billions of users

### Pinterest (2020)
- **PinSage** + Transformers for sequential pins
- Predicts next pin user will save

### Amazon (2021)
- Sequential product recommendations
- Session-based browsing patterns

---

## Summary

**Key Takeaways**:
1. **Transformers > RNNs** for sequential recommendation (+5-15% improvement)
2. **SASRec**: Causal (left-to-right), faster, good for real-time
3. **BERT4Rec**: Bidirectional, more accurate, better for offline
4. **Self-attention** captures long-range dependencies better than RNNs
5. **Masking strategy** critical for BERT4Rec training

**When to use**:
- **SASRec**: Real-time systems, large-scale deployment
- **BERT4Rec**: Batch recommendations, offline optimization

**Next steps**:
- Graph Neural Networks (Week 7) - user-item graphs
- Two-tower architectures (Week 8) - candidate retrieval at scale

---

## References

1. **Kang, W. C., & McAuley, J. (2018)**. "Self-Attentive Sequential Recommendation". *IEEE ICDM*.
   - SASRec original paper

2. **Sun, F., et al. (2019)**. "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer". *CIKM*.
   - BERT4Rec, bidirectional approach

3. **Vaswani, A., et al. (2017)**. "Attention Is All You Need". *NIPS*.
   - Original Transformer paper (foundation)

4. **Hidasi, B., & Karatzoglou, A. (2018)**. "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations". *CIKM*.
   - GRU4Rec (comparison baseline)

5. **Chen, Q., et al. (2019)**. "Behavior Sequence Transformer for E-commerce Recommendation in Alibaba". *DLP-KDD*.
   - Industrial transformer deployment at scale
