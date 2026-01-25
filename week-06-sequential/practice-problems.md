# Week 6: Sequential and Session-Based Recommendations - Practice Problems

## Overview
These problems test your understanding of sequential patterns in user behavior, RNN/LSTM/Transformer architectures for recommendations, and session-based models. Focus on temporal modeling, attention mechanisms, and next-item prediction.

---

## Problem 1: Markov Chain Basics
**Difficulty:** Easy
**Topics:** Markov chains, transition probabilities, sequential patterns

Given user session data:
- Session 1: A → B → C
- Session 2: A → C → D
- Session 3: A → B → D
- Session 4: B → C → D

**Tasks:**
1. Compute transition probabilities P(B|A), P(C|A), P(C|B), P(D|C)
2. Using first-order Markov chain, predict the next item after sequence "A → B"
3. What is the limitation of first-order Markov chains?

**Hints:**
- P(j|i) = count(i→j) / count(i→*)
- First-order: next item depends only on current item
- Cannot capture long-range dependencies

**Learning Outcomes:**
- Understand Markov chain fundamentals
- Compute transition probabilities
- Recognize limitations for complex sequences

---

## Problem 2: RNN vs. LSTM for Sequences
**Difficulty:** Medium
**Topics:** RNN, LSTM, vanishing gradients

**Compare RNN and LSTM:**

**RNN:** $h_t = \tanh(W_h h_{t-1} + W_x x_t)$

**LSTM:** Uses forget, input, and output gates to control information flow

**Questions:**
1. Why do standard RNNs struggle with long sequences?
2. How does LSTM solve the vanishing gradient problem?
3. For a user session of 50 items, would you use RNN or LSTM? Why?
4. What is the computational cost difference between RNN and LSTM?

**Learning Outcomes:**
- Understand recurrent architectures
- Recognize vanishing gradient problem
- Choose appropriate sequence models

---

## Problem 3: GRU4Rec Architecture
**Difficulty:** Medium
**Topics:** GRU4Rec, session-based recommendations

GRU4Rec uses GRU (Gated Recurrent Unit) for session-based recommendations:

```
Input: item sequence in session [i1, i2, ..., it]
Embedding: each item → vector
GRU: processes sequence → hidden states
Output: predict next item it+1
```

**Given session:**
- Items: [laptop, mouse, keyboard]
- Goal: Predict next item (monitor?)

**Questions:**
1. How does GRU4Rec handle variable-length sessions?
2. What loss function should you use?
3. How do you generate negative samples for training?
4. Compare GRU4Rec with item-to-item CF. When is GRU4Rec better?

**Learning Outcomes:**
- Understand session-based models
- Work with variable-length sequences
- Apply RNNs to recommendations

---

## Problem 4: Self-Attention Mechanism
**Difficulty:** Hard
**Topics:** Attention, transformers, SASRec

Self-attention computes:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where:
- Q = queries (items in sequence)
- K = keys (items in sequence)
- V = values (items in sequence)

**Given sequence:** [i1, i2, i3]
**Embeddings:** i1=[1,0], i2=[0,1], i3=[1,1]

**Tasks:**
1. Compute attention scores between all pairs
2. Why divide by $\sqrt{d_k}$?
3. What does the attention matrix tell you?
4. How is this different from RNN?

**Hints:**
- Attention allows each item to attend to all previous items
- Scaling prevents softmax saturation
- RNN processes sequentially, attention is parallel

**Learning Outcomes:**
- Understand self-attention mechanism
- Compute attention scores
- Compare with RNNs

---

## Problem 5: BERT4Rec Masked Prediction
**Difficulty:** Hard
**Topics:** BERT4Rec, bidirectional models, masked language modeling

BERT4Rec uses masked prediction:

**Training:**
- Input: [i1, i2, [MASK], i4, i5]
- Goal: Predict i3 using bidirectional context

**Questions:**
1. Why is bidirectional better than left-to-right for recommendations?
2. How do you select which items to mask during training?
3. What is the "Cloze task"?
4. Compare BERT4Rec with SASRec (unidirectional). When is each better?

**Learning Outcomes:**
- Understand masked prediction
- Work with bidirectional transformers
- Apply BERT concepts to recommendations

---

## Problem 6: Positional Encoding
**Difficulty:** Medium
**Topics:** Positional encoding, transformers

Transformers don't inherently understand order, so we add positional encodings:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

**Questions:**
1. Why do transformers need positional encoding but RNNs don't?
2. Calculate PE for position=0, position=1 with d=4
3. Why use sine/cosine instead of learned embeddings?
4. For recommendations, would you use absolute or relative positions?

**Learning Outcomes:**
- Understand positional encoding
- Recognize why order matters
- Choose encoding strategies

---

## Problem 7: Session-Based vs. User-Based Modeling
**Difficulty:** Medium
**Topics:** Session modeling, user modeling, architecture choice

**Scenario A:** E-commerce (anonymous users, short sessions, diverse interests)
**Scenario B:** Music streaming (logged-in users, long history, consistent taste)

For each:
1. Would you use session-based (GRU4Rec) or user-based (BERT4Rec) modeling?
2. What is the input to your model?
3. How do you handle cold start?
4. What evaluation metrics are appropriate?

**Learning Outcomes:**
- Choose between session and user models
- Adapt to different use cases
- Design appropriate architectures

---

## Problem 8: Temporal Feature Engineering
**Difficulty:** Medium
**Topics:** Temporal features, time gaps, recency

Beyond sequences, temporal features matter:
- **Time gap:** Time between consecutive actions
- **Recency:** Time since last action
- **Periodicity:** Daily/weekly patterns

**Design features for:**
- User watched movies at: [Monday 8pm, Tuesday 8pm, Friday 10pm, Saturday 2pm]

**Extract:**
1. Time-of-day pattern
2. Day-of-week pattern
3. Inter-event gaps
4. How would you encode these features?

**Learning Outcomes:**
- Engineer temporal features
- Capture periodic patterns
- Incorporate time into models

---

## Programming Exercises

### Exercise 1: Implement Markov Chain Recommender
**Dataset:** MovieLens with timestamps (ordered sequences)
**Task:** Build first-order and second-order Markov chains

**Implementation:**
```python
from collections import defaultdict, Counter

class MarkovChainRecommender:
    def __init__(self, order=1):
        self.order = order
        self.transitions = defaultdict(Counter)

    def fit(self, sequences):
        for seq in sequences:
            for i in range(len(seq) - self.order):
                state = tuple(seq[i:i+self.order])
                next_item = seq[i+self.order]
                self.transitions[state][next_item] += 1

    def predict_next(self, state, top_n=10):
        if isinstance(state, int):
            state = (state,)
        elif isinstance(state, list):
            state = tuple(state[-self.order:])

        if state not in self.transitions:
            return []

        candidates = self.transitions[state]
        top_items = candidates.most_common(top_n)
        return [item for item, count in top_items]
```

**Evaluation:**
- Precision@10: % of recommendations that user actually clicked next
- Recall@10: % of next items that were recommended

---

### Exercise 2: Implement GRU4Rec in PyTorch
**Dataset:** E-commerce session data (or MovieLens sessions)
**Task:** Build session-based recommender with GRU

**Architecture:**
```python
import torch
import torch.nn as nn

class GRU4Rec(nn.Module):
    def __init__(self, n_items, embedding_dim=100, hidden_dim=100, n_layers=1):
        super(GRU4Rec, self).__init__()
        self.embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, n_items)

    def forward(self, input_seq, hidden=None):
        # input_seq: [batch_size, seq_len]
        embedded = self.embedding(input_seq)
        output, hidden = self.gru(embedded, hidden)
        # Use last hidden state
        logits = self.output(output[:, -1, :])
        return logits, hidden
```

**Training:**
- Session-parallel mini-batches
- Negative sampling: sample from popular items
- Loss: Cross-entropy or BPR
- Optimizer: Adam

**Evaluation:** Recall@20, MRR@20 on test sessions

---

### Exercise 3: Implement SASRec (Self-Attentive Sequential Recommendation)
**Dataset:** MovieLens 1M with timestamps
**Task:** Build transformer-based sequential recommender

**Key Components:**
```python
class SASRec(nn.Module):
    def __init__(self, n_items, max_len=200, d_model=64, n_heads=2, n_layers=2):
        super(SASRec, self).__init__()
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, n_layers)

        self.output = nn.Linear(d_model, n_items)

    def forward(self, seq):
        # seq: [batch_size, seq_len]
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)

        x = self.item_emb(seq) + self.pos_emb(positions)

        # Causal mask (prevent attending to future items)
        mask = torch.triu(torch.ones(seq.size(1), seq.size(1)), diagonal=1).bool()
        x = self.transformer(x.transpose(0, 1), mask=mask).transpose(0, 1)

        logits = self.output(x[:, -1, :])  # Predict next item
        return logits
```

**Training:**
- Sample subsequences from user histories
- Predict last item given previous items
- Negative sampling

**Comparison:** GRU4Rec vs. SASRec on NDCG@10

---

### Exercise 4: Implement BERT4Rec
**Dataset:** MovieLens 1M
**Task:** Build masked item prediction model

**Training Strategy:**
```python
class BERT4Rec(nn.Module):
    def __init__(self, n_items, max_len=200, d_model=64, n_heads=2, n_layers=2):
        super(BERT4Rec, self).__init__()
        # Similar to SASRec but bidirectional (no causal mask)
        # ... embeddings, transformer ...

    def forward(self, seq, masked_positions):
        # seq: [batch_size, seq_len] with some items masked
        # masked_positions: indices of masked items
        # ... transformer encoding ...
        # Predict masked items
        pass

def create_masked_sequence(seq, mask_prob=0.15):
    masked_seq = seq.copy()
    masked_positions = []

    for i in range(len(seq)):
        if random.random() < mask_prob:
            masked_seq[i] = MASK_TOKEN
            masked_positions.append(i)

    return masked_seq, masked_positions, [seq[i] for i in masked_positions]
```

**Evaluation:**
- Mask last item in sequence
- Predict it using bidirectional context
- Measure Hit Rate, NDCG

---

### Exercise 5: Time-Aware Recommendations
**Dataset:** MovieLens with timestamps
**Task:** Incorporate time gaps and recency

**Features:**
```python
def extract_temporal_features(user_history):
    items, timestamps = zip(*user_history)

    # Time gaps (seconds between consecutive items)
    time_gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

    # Recency (hours since last action)
    recency = [(timestamps[-1] - t) / 3600 for t in timestamps]

    # Time of day (0-23)
    time_of_day = [datetime.fromtimestamp(t).hour for t in timestamps]

    # Day of week (0-6)
    day_of_week = [datetime.fromtimestamp(t).weekday() for t in timestamps]

    return {
        'items': items,
        'time_gaps': time_gaps,
        'recency': recency,
        'time_of_day': time_of_day,
        'day_of_week': day_of_week
    }
```

**Model:**
- Extend GRU4Rec to include temporal features
- Concatenate temporal features with item embeddings

**Evaluation:** Does adding time improve next-item prediction?

---

### Exercise 6: Session Recommendation with Contextual Bandits
**Dataset:** E-commerce click logs
**Task:** Balance exploration and exploitation in sessions

**Approach:**
- For each session position, choose item to display
- Observe click (reward = 1) or no click (reward = 0)
- Update model to improve future selections

**Simple Implementation (ε-greedy):**
```python
def recommend_next_item(session, epsilon=0.1):
    if random.random() < epsilon:
        # Explore: random item
        return random.choice(all_items)
    else:
        # Exploit: use GRU4Rec prediction
        return gru4rec.predict(session)[0]
```

**Evaluation:** Online A/B test simulation

---

### Exercise 7: Multi-Task Sequential Model
**Dataset:** MovieLens
**Task:** Jointly predict next item and rating

**Architecture:**
```python
class MultiTaskSequential(nn.Module):
    def __init__(self, n_items, d_model=64):
        super(MultiTaskSequential, self).__init__()
        self.item_emb = nn.Embedding(n_items, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

        # Task 1: Next item prediction
        self.item_head = nn.Linear(d_model, n_items)

        # Task 2: Rating prediction
        self.rating_head = nn.Linear(d_model, 1)

    def forward(self, seq):
        embedded = self.item_emb(seq)
        output, hidden = self.gru(embedded)
        last_hidden = output[:, -1, :]

        next_item_logits = self.item_head(last_hidden)
        rating_pred = self.rating_head(last_hidden)

        return next_item_logits, rating_pred
```

**Loss:** $L = L_{item} + \lambda L_{rating}$

**Evaluation:** Does multi-task learning improve both tasks?

---

## Discussion Questions

1. **Sequence Length:** What is the optimal sequence length for training? Too short misses context, too long includes irrelevant history.

2. **Cold Start:** How do session-based models handle new users? New items? Compare with traditional CF.

3. **Real-time Inference:** Transformers are slow for long sequences. How would you optimize SASRec/BERT4Rec for production?

4. **Diversity:** Sequential models may get stuck in loops (recommend similar items). How do you increase diversity?

5. **Context Switching:** Users change interests within a session. How do you detect and handle context switches?

6. **Evaluation Metrics:** Is NDCG appropriate for sequential recommendations? What about time-based metrics?

7. **Transfer Learning:** Can you pre-train a sequential model on one domain and fine-tune on another?

8. **Interpretability:** Attention weights can show which past items influenced the prediction. How would you visualize this for users?

---

## Challenge Problem: Hierarchical Sequential Model

**Difficulty:** Very Hard
**Topics:** Hierarchical modeling, long-term + short-term interests

**Task:** Model both long-term user preferences and short-term session dynamics

**Architecture:**
```
User-level: Encoder for all user history → long-term embedding
Session-level: Encoder for current session → short-term embedding
Fusion: Combine long-term and short-term → prediction
```

**Implementation:**
```python
class HierarchicalSeqRec(nn.Module):
    def __init__(self, n_items, d_model=64):
        super(HierarchicalSeqRec, self).__init__()

        # Long-term user encoder (processes all user history)
        self.user_encoder = nn.GRU(d_model, d_model, batch_first=True)

        # Short-term session encoder (processes current session)
        self.session_encoder = nn.GRU(d_model, d_model, batch_first=True)

        # Fusion layer
        self.fusion = nn.Linear(d_model * 2, d_model)

        # Output
        self.output = nn.Linear(d_model, n_items)

    def forward(self, user_history, current_session):
        # Long-term
        user_emb = self.item_emb(user_history)
        _, user_hidden = self.user_encoder(user_emb)
        user_rep = user_hidden[-1]

        # Short-term
        session_emb = self.item_emb(current_session)
        _, session_hidden = self.session_encoder(session_emb)
        session_rep = session_hidden[-1]

        # Fusion
        combined = torch.cat([user_rep, session_rep], dim=-1)
        fused = torch.relu(self.fusion(combined))

        logits = self.output(fused)
        return logits
```

**Evaluation:**
- Compare with session-only and user-only models
- Analyze when long-term vs. short-term matters more

---

## References

### Papers
1. Hidasi, B., et al. (2016). "Session-based recommendations with recurrent neural networks". ICLR.
2. Kang, W. C., & McAuley, J. (2018). "Self-attentive sequential recommendation". ICDM.
3. Sun, F., et al. (2019). "BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer". CIKM.
4. Tang, J., & Wang, K. (2018). "Personalized top-n sequential recommendation via convolutional sequence embedding". WSDM.

### Libraries
- PyTorch: https://pytorch.org/
- Transformers (Hugging Face): https://huggingface.co/transformers/

### Datasets
- MovieLens with timestamps: https://grouplens.org/datasets/movielens/
- RecSys Challenge datasets: http://www.recsyschallenge.com/
- YOOCHOOSE (e-commerce): https://www.kaggle.com/chadgostopp/recsys-challenge-2015

---

*Return to [Week 6 Main Page](README.md)*
