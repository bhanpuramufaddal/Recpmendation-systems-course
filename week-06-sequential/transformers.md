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

## Why Transformers for Sequential Recommendation?

### The Sequential Recommendation Problem

**Given**: User's interaction history (ordered sequence)
$$S_u = [i_1, i_2, \ldots, i_t]$$

**Goal**: Predict next item $i_{t+1}$

**Examples**:
- E-commerce: [laptop → mouse → keyboard] → predict USB hub
- Netflix: [Action movie → Sci-fi → Thriller] → predict similar
- Spotify: [Rock → Alternative → Indie] → predict next song

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

## Self-Attention Mechanism

### Intuition

**Question**: When predicting next item, which past items are relevant?

**Example**:
```
History: [breakfast cereal, milk, coffee, laptop, mouse, keyboard]
Predict: ?
```

**Attention**:
- High weight on [laptop, mouse, keyboard] → predict USB hub
- Low weight on [breakfast cereal, milk, coffee] → unrelated

**Self-attention** learns these weights automatically!

---

### Mathematical Formulation

**Input**: Sequence of item embeddings
$$E = [\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_t] \in \mathbb{R}^{t \times d}$$

where $\mathbf{e}_i \in \mathbb{R}^d$ is embedding of item $i$.

**Self-Attention**:

1. **Compute queries, keys, values**:
   $$Q = EW^Q, \quad K = EW^K, \quad V = EW^V$$
   where $W^Q, W^K, W^V \in \mathbb{R}^{d \times d}$ are learned matrices.

2. **Compute attention scores**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

**Breakdown**:
- $QK^T \in \mathbb{R}^{t \times t}$: Similarity between all pairs of items
- $\text{softmax}$: Normalize to probabilities
- Multiply by $V$: Weighted combination of values

---

### Multi-Head Attention

**Idea**: Learn multiple attention patterns (different "heads")

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

where:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Benefits**:
- Head 1: Focus on recent items
- Head 2: Focus on similar categories
- Head 3: Focus on temporal patterns

**Typical**: $h = 2$ or $h = 4$ heads

---

## SASRec: Self-Attentive Sequential Recommendation

### Paper

**Kang, W. C., & McAuley, J. (2018)**. "Self-Attentive Sequential Recommendation". *IEEE ICDM*.

**Key innovation**: Use transformers for item-to-item recommendation.

---

### Architecture

```
Input:           [i_1, i_2, ..., i_t]
                        ↓
Item Embedding:  [e_1, e_2, ..., e_t]
                        ↓
Positional Encoding: Add position info
                        ↓
Self-Attention Block (×L layers)
    ├─ Multi-Head Attention
    ├─ Add & Norm
    ├─ Feed-Forward Network
    └─ Add & Norm
                        ↓
Output:          [h_1, h_2, ..., h_t]
                        ↓
Prediction:      Softmax over all items
```

**L**: Number of transformer blocks (typically 2-4)

---

### Positional Encoding

**Problem**: Self-attention is permutation-invariant (doesn't know order!)

**Solution**: Add positional encodings to item embeddings.

**Learnable Positional Embedding**:
$$\mathbf{e}_i' = \mathbf{e}_i + \mathbf{p}_i$$

where $\mathbf{p}_i \in \mathbb{R}^d$ is learned position embedding for position $i$.

---

### Causal Attention (Left-to-Right)

**Key constraint**: When predicting $i_t$, can only see $[i_1, \ldots, i_{t-1}]$

**Implementation**: Use attention mask
$$\text{mask}[i, j] = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

This prevents "cheating" by looking at future items.

---

### Loss Function

**Training**: Predict next item at each position.

$$\mathcal{L} = -\sum_{S_u \in \mathcal{D}} \sum_{t=1}^{|S_u|} \log P(i_t | [i_1, \ldots, i_{t-1}])$$

where:
$$P(i_t | \mathbf{h}_{t-1}) = \frac{\exp(\mathbf{h}_{t-1}^T \mathbf{e}_{i_t})}{\sum_{i' \in \mathcal{I}} \exp(\mathbf{h}_{t-1}^T \mathbf{e}_{i'})}$$

**Softmax over all items**: Expensive! Use negative sampling or sampled softmax.

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

### Cloze Task (Masked Item Prediction)

**Idea**: Randomly mask items in sequence, predict them.

**Example**:
```
Original:  [laptop, mouse, [MASK], USB hub]
Predict:   keyboard (at masked position)
```

**Training objective**:
$$\mathcal{L} = -\sum_{S_u} \sum_{i_m \in \text{masked}} \log P(i_m | S_u^{\text{masked}})$$

**Mask strategy**:
- Mask 15-20% of items randomly
- Replace with special [MASK] token

---

### Architecture

```
Input:         [i_1, i_2, [MASK], i_4, ...]
                      ↓
Embedding:     [e_1, e_2, e_[M], e_4, ...]
                      ↓
Positional Encoding
                      ↓
Bidirectional Transformer Blocks (×L)
    ├─ Multi-Head Attention (NO causal mask!)
    ├─ Add & Norm
    ├─ Feed-Forward
    └─ Add & Norm
                      ↓
Output:        [h_1, h_2, h_[M], h_4, ...]
                      ↓
Predict masked: Softmax(h_[M])
```

**Key**: Bidirectional attention can see both past and future!

---

### Why Bidirectional?

**SASRec** (causal):
- Predicting $i_3$ can only see $[i_1, i_2]$

**BERT4Rec** (bidirectional):
- Predicting $i_3$ can see $[i_1, i_2, i_4, i_5, \ldots]$ (except $i_3$ itself)

**Advantage**: More context → better predictions.

**Example**:
```
Sequence: [action movie, [MASK], thriller, action movie]
```
- SASRec: Only sees "action movie" before mask → might predict comedy
- BERT4Rec: Sees "action movie" before AND "thriller, action movie" after → correctly predicts action/thriller

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

## Comparison: SASRec vs. BERT4Rec

### Performance

**MovieLens-1M** (Sun et al., 2019):

| Model | Hit@10 | NDCG@10 |
|-------|--------|---------|
| GRU4Rec | 0.5860 | 0.3670 |
| Caser | 0.5930 | 0.3720 |
| SASRec | 0.6270 | 0.4010 |
| **BERT4Rec** | **0.6410** | **0.4100** |

**Improvement**: BERT4Rec +2.2% Hit@10 over SASRec

---

### Trade-offs

| Aspect | SASRec | BERT4Rec |
|--------|--------|----------|
| **Training speed** | Faster | Slower (bidirectional) |
| **Accuracy** | Good | Better |
| **Inference** | Direct | Needs masking strategy |
| **Use case** | Real-time, large-scale | Offline batch recommendation |

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
