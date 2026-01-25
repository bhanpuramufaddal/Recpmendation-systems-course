# Week 6: Sequential Recommendations - RNNs for RecSys

## Overview

**Recurrent Neural Networks (RNNs)** can model long-term dependencies in user sequences, overcoming the limitations of Markov chains.

**Key advantages**:
- **Long-term memory**: Remember items from distant past
- **Personalization**: Learn user-specific patterns
- **Flexibility**: Handle variable-length sequences

**Breakthrough paper**: Hidasi et al., "Session-based Recommendations with Recurrent Neural Networks" (ICLR 2016) - **GRU4Rec**

This document covers RNN-based architectures for sequential recommendation.

---

## Learning Objectives

By the end of this section, you will:
- Understand RNN architectures (GRU, LSTM) for recommendation
- Implement GRU4Rec for session-based recommendation
- Master attention mechanisms for sequences
- Apply RNNs to real-world sequential recommendation problems
- Compare RNNs with Markov chains and transformers

---

## From Markov Chains to RNNs

### Limitations of Markov Chains (Recap)

**First-order Markov**:
$$P(i_t | i_1, \ldots, i_{t-1}) = P(i_t | i_{t-1})$$

**Problems**:
1. **Short memory**: Only last item
2. **Fixed patterns**: Can't adapt to new sequences
3. **No personalization**: Same for all users

---

### RNN Solution

**RNN approach**:
$$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$$
$$\hat{y}_t = g(\mathbf{h}_t)$$

where:
- $\mathbf{h}_t$ = hidden state at time $t$ (encodes full history)
- $\mathbf{x}_t$ = input at time $t$ (item embedding)
- $f$ = recurrent function (GRU or LSTM)
- $g$ = output function

**Advantage**: $\mathbf{h}_t$ encodes **entire history** $[i_1, i_2, \ldots, i_t]$, not just last item.

---

## RNN Architectures

### 1. Vanilla RNN

**Recurrence**:
$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$

**Problem**: **Vanishing gradients** (can't learn long-term dependencies).

**Solution**: Use LSTM or GRU.

---

### 2. LSTM (Long Short-Term Memory)

**Components**:
- **Forget gate**: Decide what to forget from memory
- **Input gate**: Decide what new information to store
- **Output gate**: Decide what to output

**Equations**:
$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

**Benefit**: Can remember long-term dependencies.

---

### 3. GRU (Gated Recurrent Unit)

**Simpler than LSTM** (fewer parameters).

**Gates**:
- **Reset gate**: Decide how much past to forget
- **Update gate**: Decide how much to update hidden state

**Equations**:
$$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t])$$
$$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t])$$
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

**Trade-off**: GRU is faster (fewer params), LSTM may be more accurate for very long sequences.

**Recommendation systems**: GRU is more popular (GRU4Rec).

---

## GRU4Rec: Session-Based RNN

### Paper

**Hidasi et al., "Session-based Recommendations with Recurrent Neural Networks" (ICLR 2016)**

**Use case**: Recommend next item in user's session.

**Key innovation**: Use GRU to model session sequences.

---

### Architecture

```
Input: Sequence of items [i1, i2, i3, ..., it]
         ↓
    Item Embeddings
         ↓
    GRU Layers (stacked)
         ↓
    Hidden State ht
         ↓
    Output Layer (scores for all items)
         ↓
    Softmax
         ↓
    Probabilities for next item
```

---

### Implementation

```python
import torch
import torch.nn as nn

class GRU4Rec(nn.Module):
    def __init__(self, n_items, embedding_dim=50, hidden_dim=100, num_layers=1, dropout=0.2):
        super().__init__()

        # Item embedding
        self.item_embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)

        # GRU layers
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer (predict next item)
        self.fc = nn.Linear(hidden_dim, n_items)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.item_embedding.weight, -0.1, 0.1)
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, item_seq, hidden=None):
        """
        item_seq: (batch, seq_len) - sequence of item IDs
        hidden: (num_layers, batch, hidden_dim) - initial hidden state (optional)
        """
        # Embed items
        embedded = self.item_embedding(item_seq)  # (batch, seq_len, embedding_dim)

        # GRU
        output, hidden = self.gru(embedded, hidden)  # output: (batch, seq_len, hidden_dim)

        # Predict next item using last hidden state
        last_hidden = output[:, -1, :]  # (batch, hidden_dim)
        scores = self.fc(last_hidden)  # (batch, n_items)

        return scores, hidden


# Example usage
n_items = 1000
batch_size = 32
seq_len = 10

model = GRU4Rec(n_items=n_items, embedding_dim=50, hidden_dim=100, num_layers=1)

# Sample session: batch of sequences
item_seq = torch.randint(1, n_items, (batch_size, seq_len))  # (32, 10)

# Forward pass
scores, hidden = model(item_seq)
print(f"Scores shape: {scores.shape}")  # (32, 1000) - scores for all items

# Top-5 recommendations
top5_items = torch.topk(scores, k=5, dim=1).indices
print(f"Top-5 recommendations shape: {top5_items.shape}")  # (32, 5)
```

---

### Training GRU4Rec

**Loss function**: Cross-entropy (next-item prediction)

$$\mathcal{L} = -\sum_{t=1}^T \log P(i_{t+1} | i_1, \ldots, i_t)$$

**Training strategy**:
1. For each session, create subsequences:
   - Input: $[i_1]$, target: $i_2$
   - Input: $[i_1, i_2]$, target: $i_3$
   - ...
2. Train to predict next item at each step

---

### Training Implementation

```python
import torch.optim as optim
import torch.nn.functional as F

def train_gru4rec(model, sessions, n_epochs=10, lr=0.001, batch_size=32):
    """
    Train GRU4Rec model.

    sessions: list of item sequences, e.g., [[1,5,3,7], [2,4,8], ...]
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0

        # Create mini-batches
        for i in range(0, len(sessions), batch_size):
            batch_sessions = sessions[i:i+batch_size]

            # Pad sessions to same length
            max_len = max(len(s) for s in batch_sessions)
            padded_sessions = [s + [0]*(max_len - len(s)) for s in batch_sessions]

            # Convert to tensors
            session_tensor = torch.tensor(padded_sessions)  # (batch, max_len)

            # Training: predict each item from previous items
            for t in range(1, max_len):
                # Input: items up to time t-1
                input_seq = session_tensor[:, :t]

                # Target: item at time t
                target = session_tensor[:, t]

                # Forward
                scores, _ = model(input_seq)

                # Loss (only for non-padded items)
                mask = (target != 0)  # Ignore padding
                if mask.sum() == 0:
                    continue

                loss = criterion(scores[mask], target[mask])

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    return model


# Example
# sessions = [[1,5,3,7,9], [2,4,8,6], [5,3,9,1,4], ...]
# trained_model = train_gru4rec(model, sessions, n_epochs=10, lr=0.001, batch_size=32)
```

---

## LSTM for Sequential Recommendation

### LSTM Architecture for RecSys

Similar to GRU4Rec, but using LSTM cells.

```python
class LSTM4Rec(nn.Module):
    def __init__(self, n_items, embedding_dim=50, hidden_dim=100, num_layers=1, dropout=0.2):
        super().__init__()

        self.item_embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, n_items)

    def forward(self, item_seq, hidden=None):
        embedded = self.item_embedding(item_seq)
        output, (hidden, cell) = self.lstm(embedded, hidden)

        last_hidden = output[:, -1, :]
        scores = self.fc(last_hidden)

        return scores, (hidden, cell)
```

**When to use LSTM vs. GRU**:
- **GRU**: Faster, fewer parameters, good for most cases
- **LSTM**: Better for very long sequences (>100 items)

---

## Attention Mechanisms

### Why Attention?

**Problem with basic RNN**: All items in sequence contribute equally to prediction.

**Reality**: Some items are more relevant than others.

**Example**:
```
Session: [laptop, laptop_bag, mouse, charger, water_bottle, ...]

To predict next item:
  - laptop, laptop_bag, mouse, charger → highly relevant (computer accessories)
  - water_bottle → less relevant (unrelated)
```

**Attention**: Assign different weights to different items.

---

### Attention Mechanism

**Idea**: Compute weighted sum of hidden states.

$$\mathbf{c} = \sum_{t=1}^T \alpha_t \mathbf{h}_t$$

where $\alpha_t$ = attention weight for item at position $t$.

**Attention weights**:
$$\alpha_t = \frac{\exp(e_t)}{\sum_{t'=1}^T \exp(e_{t'})}$$

where $e_t = \text{score}(\mathbf{h}_t, \mathbf{h}_T)$ (similarity between current hidden state and final hidden state).

---

### Implementation

```python
class GRU4RecWithAttention(nn.Module):
    def __init__(self, n_items, embedding_dim=50, hidden_dim=100):
        super().__init__()

        self.item_embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        # Attention
        self.attention = nn.Linear(hidden_dim, 1)

        # Output
        self.fc = nn.Linear(hidden_dim, n_items)

    def forward(self, item_seq):
        embedded = self.item_embedding(item_seq)  # (batch, seq_len, emb_dim)
        output, _ = self.gru(embedded)  # (batch, seq_len, hidden_dim)

        # Attention scores
        attn_scores = self.attention(output)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

        # Weighted sum
        context = torch.sum(attn_weights * output, dim=1)  # (batch, hidden_dim)

        # Predict
        scores = self.fc(context)  # (batch, n_items)

        return scores


# Example
model_attn = GRU4RecWithAttention(n_items=1000, embedding_dim=50, hidden_dim=100)
scores = model_attn(item_seq)
print(f"Scores with attention: {scores.shape}")
```

---

## Self-Attention (Preview)

### Transformer-Based Models

**Limitation of RNNs**: Sequential processing (slow for long sequences).

**Self-Attention** (Transformers): Process all items in parallel.

**Key models**:
1. **SASRec** (Self-Attentive Sequential Recommendation, 2018)
2. **BERT4Rec** (2019)

**Coverage**: Detailed in Week 6's transformers.md file.

**Preview**: Self-attention allows each item to attend to all other items in sequence.

---

## Comparison: Markov vs. RNN vs. Transformer

| Aspect | Markov Chain | RNN (GRU/LSTM) | Transformer |
|--------|--------------|----------------|-------------|
| **Memory** | Last 1-2 items | Full sequence | Full sequence |
| **Computation** | O(1) per step | O(T) sequential | O(T²) parallel |
| **Long-term deps** | Poor | Good | Excellent |
| **Parallelization** | Yes | No | Yes |
| **Training speed** | Fast | Medium | Fast (parallel) |
| **Inference speed** | Fast | Medium | Medium |
| **Accuracy** | Low | Medium-High | Highest |

**Recommendation**:
- **Small data, fast inference**: Markov chain
- **Medium data, good accuracy**: GRU4Rec
- **Large data, best accuracy**: Transformer (SASRec, BERT4Rec)

---

## Real-World Applications

### 1. Music Streaming (Spotify)

**Use case**: Generate personalized playlists.

**Approach**:
- RNN to model listening sequences
- Predict next songs based on current session
- Combine with audio features for cold-start songs

**Result**: "Discover Weekly" playlist uses sequential models.

---

### 2. E-Commerce (eBay)

**Use case**: "You may also like" recommendations.

**Approach**:
- GRU4Rec to model browsing sessions
- Predict next product based on current session
- Real-time recommendations

**Result**: Improved CTR by 10-15% (A/B test).

---

### 3. Video Streaming (YouTube)

**Use case**: "Up Next" video recommendations.

**Approach**:
- LSTM to model watch history
- Predict next video based on current session + user history
- Consider watch time, engagement signals

**Result**: Drives 70%+ of watch time.

---

## Summary

**Key Takeaways**:
1. **RNNs > Markov chains**: Can model long-term dependencies
2. **GRU4Rec**: Standard RNN approach for session-based recommendation
3. **LSTM**: Better for very long sequences
4. **Attention**: Focus on relevant items in sequence
5. **Transformers**: Best accuracy, but more complex (next section)

**Best Practices**:
- Start with GRU4Rec (good balance)
- Use attention for long sequences
- Batch normalization + dropout for regularization
- Learning rate 0.001, hidden dim 100-200

**When to use**:
- **Session-based**: GRU4Rec (e-commerce, news)
- **Long-term history**: LSTM with attention
- **Large-scale, state-of-the-art**: Transformers (SASRec, BERT4Rec)

**Next**: Transformers for sequential recommendation (SASRec, BERT4Rec).

---

## References

1. **Hidasi, B., et al. (2016)**. "Session-based Recommendations with Recurrent Neural Networks". *ICLR*.
   - **GRU4Rec** paper

2. **Hidasi, B., & Karatzoglou, A. (2018)**. "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations". *CIKM*.
   - **Improved GRU4Rec** with ranking loss

3. **Quadrana, M., Karatzoglou, A., Hidasi, B., & Cremonesi, P. (2017)**. "Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks". *RecSys*.
   - **Hierarchical RNN** for personalization

4. **Li, J., et al. (2017)**. "Neural Attentive Session-based Recommendation". *CIKM*.
   - **Attention mechanisms** for session-based RecSys

5. **Kang, W.-C., & McAuley, J. (2018)**. "Self-Attentive Sequential Recommendation". *ICDM*.
   - **SASRec** (preview of Transformer-based models)

---

## Practice Problems

### Problem 1: GRU Hidden State Size

**Given**:
```
Embedding dim: 50
Hidden dim: 100
Batch size: 32
Sequence length: 15
Num layers: 2
```

**Compute**: Shape of GRU output and hidden state.

**Solution**:
```python
# GRU output: (batch, seq_len, hidden_dim)
output_shape = (32, 15, 100)

# Hidden state: (num_layers, batch, hidden_dim)
hidden_shape = (2, 32, 100)
```

---

### Problem 2: Attention Weights

**Given**:
```
Attention scores for 4 items: [0.5, 1.0, 0.3, 0.8]
```

**Compute**: Attention weights (after softmax).

**Solution**:
```python
import numpy as np

scores = np.array([0.5, 1.0, 0.3, 0.8])
weights = np.exp(scores) / np.sum(np.exp(scores))
print(weights)
# [0.224, 0.369, 0.184, 0.303]
```

---

### Problem 3: Training Data Preparation

**Given session**:
```
[item_1, item_2, item_3, item_4]
```

**Create**: Training pairs (input sequence, target).

**Solution**:
```
([item_1], item_2)
([item_1, item_2], item_3)
([item_1, item_2, item_3], item_4)
```
