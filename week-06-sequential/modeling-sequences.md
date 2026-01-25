# Week 6: Sequential Recommendations - Modeling Sequences

## Overview

**Traditional collaborative filtering** assumes user preferences are static. But in reality, **user preferences evolve over time** and are **context-dependent**.

**Examples**:
- **E-commerce**: User buys laptop → likely to buy laptop bag, mouse, charger (sequential pattern)
- **Music**: User plays Rock song → more likely to play another Rock song in same session
- **Videos**: User watches Ep 1 of a show → will watch Ep 2 next

**Sequential recommendation** models the temporal dynamics of user behavior.

This document covers the foundations of sequential recommendation systems.

---

## Learning Objectives

By the end of this section, you will:
- Understand why sequences matter in recommendations
- Master Markov chains and session-based models
- Learn item-to-item sequential patterns
- Implement transition-based recommendation
- Apply sequential methods to real-world problems

---

## Why Sequences Matter

### Static vs. Sequential Preferences

**Static CF assumption**:
$$\hat{r}_{ui} = f(u, i)$$

User preferences are time-invariant.

**Sequential reality**:
$$\hat{r}_{ui,t} = f(u, i, s_t)$$

where $s_t$ = user's interaction sequence up to time $t$.

---

### Examples Where Sequences Matter

**1. E-Commerce (Amazon)**
```
User's sequence:
  1. Views laptop
  2. Adds laptop to cart
  3. Buys laptop
  4. Next recommendation: laptop bag, mouse (complements)
```

**Static CF**: Might recommend another laptop (substitute)
**Sequential**: Recommends complements based on recent purchase

---

**2. Music (Spotify)**
```
Session:
  1. "Bohemian Rhapsody" (Rock, Queen)
  2. "Stairway to Heaven" (Rock, Led Zeppelin)
  3. "Hotel California" (Rock, Eagles)

Next: Likely another classic rock song
```

**Static CF**: Might recommend pop music (user listened to pop last week)
**Sequential**: Recommends based on current session's mood

---

**3. Video Streaming (Netflix)**
```
User behavior:
  - Morning: News, documentaries
  - Evening: Action movies, dramas

Next (if evening): Action/drama
```

**Static CF**: Averages all preferences
**Sequential**: Adapts to time of day, context

---

## Markov Chains for Sequential Recommendation

### First-Order Markov Chain

**Assumption**: Next item depends only on current item.

$$P(i_t | i_1, i_2, \ldots, i_{t-1}) = P(i_t | i_{t-1})$$

**Transition matrix**:
$$\mathbf{M}_{ij} = P(\text{next item} = j | \text{current item} = i)$$

**Estimation** (from data):
$$\hat{M}_{ij} = \frac{\text{count}(i \to j)}{\sum_{k} \text{count}(i \to k)}$$

---

### Example: Music Recommendations

**Data** (user sessions):
```
Session 1: [A, B, C]
Session 2: [A, C, D]
Session 3: [B, C, A]
```

**Transitions**:
```
A → B: 1
A → C: 1
B → C: 2
C → D: 1
C → A: 1
```

**Transition matrix**:
```
     A    B    C    D
A  [0.0, 0.5, 0.5, 0.0]
B  [0.0, 0.0, 1.0, 0.0]
C  [0.5, 0.0, 0.0, 0.5]
D  [0.0, 0.0, 0.0, 0.0]
```

**Recommendation**: If user just played song B, next recommendation is C (100% probability).

---

### Implementation

```python
import numpy as np
from collections import defaultdict

class FirstOrderMarkovChain:
    def __init__(self, n_items):
        self.n_items = n_items
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_matrix = None

    def fit(self, sessions):
        """
        Learn transition probabilities from sessions.

        sessions: list of lists, e.g., [[1,2,3], [1,3,4], ...]
        """
        # Count transitions
        for session in sessions:
            for i in range(len(session) - 1):
                current_item = session[i]
                next_item = session[i + 1]
                self.transition_counts[current_item][next_item] += 1

        # Build transition matrix
        self.transition_matrix = np.zeros((self.n_items, self.n_items))

        for current_item in self.transition_counts:
            total = sum(self.transition_counts[current_item].values())
            for next_item, count in self.transition_counts[current_item].items():
                self.transition_matrix[current_item, next_item] = count / total

    def recommend(self, current_item, top_k=5):
        """Recommend next items given current item."""
        if self.transition_matrix is None:
            raise ValueError("Model not fitted yet!")

        # Get transition probabilities
        probs = self.transition_matrix[current_item]

        # Top-K items
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_probs = probs[top_indices]

        return list(zip(top_indices, top_probs))


# Example
sessions = [
    [0, 1, 2],
    [0, 2, 3],
    [1, 2, 0]
]

model = FirstOrderMarkovChain(n_items=4)
model.fit(sessions)

# Recommend next item after item 0
recommendations = model.recommend(current_item=0, top_k=3)
print(f"After item 0, recommend: {recommendations}")
# Output: [(1, 0.5), (2, 0.5), (3, 0.0)]
```

---

### Limitations of First-Order Markov Chains

**1. Short memory**: Only considers last item (ignores earlier history).

**Example**:
```
User sequence: [Laptop, Laptop bag, ...]

With only "Laptop bag" as context, model might recommend another bag.
But full history suggests user bought laptop → should recommend mouse, charger.
```

**2. Data sparsity**: Many item pairs never co-occur → zero probabilities.

**3. No personalization**: Same recommendations for all users who viewed item $i$.

---

## Higher-Order Markov Chains

### Second-Order Markov Chain

**Assumption**: Next item depends on last **two** items.

$$P(i_t | i_1, \ldots, i_{t-1}) = P(i_t | i_{t-2}, i_{t-1})$$

**Transition tensor**:
$$\mathbf{M}_{ijk} = P(\text{next} = k | \text{prev} = i, \text{current} = j)$$

**Challenge**: Exponentially more parameters (sparsity increases).

---

### Variable-Order Markov Chains

**Idea**: Use higher-order context when available, fall back to lower-order when sparse.

**Example**:
```
If (laptop, laptop bag) → mouse: Use 2nd-order
If (laptop bag) → ?: Fall back to 1st-order
```

---

## Session-Based Recommendation

### Definition

**Session**: Sequence of user interactions within a short time window (e.g., 30 minutes).

**Goal**: Recommend next item based on current session.

**Difference from Markov chains**: Can consider entire session (not just last item).

---

### Item-to-Item Similarity (Amazon Approach)

**Paper**: Linden et al., "Amazon.com Recommendations: Item-to-Item Collaborative Filtering" (2003)

**Idea**: "Customers who bought X also bought Y"

**Process**:
1. For each item pair $(i, j)$, count co-occurrences in sessions
2. Compute similarity: $\text{sim}(i, j) = \frac{\text{co-occurrence}(i, j)}{\sqrt{\text{count}(i) \times \text{count}(j)}}$
3. Recommend items most similar to items in current session

---

### Implementation

```python
class ItemToItemSessionBased:
    def __init__(self):
        self.item_counts = defaultdict(int)
        self.cooccurrence = defaultdict(lambda: defaultdict(int))
        self.similarity = {}

    def fit(self, sessions):
        """
        Learn item-to-item similarities from sessions.
        """
        # Count item occurrences and co-occurrences
        for session in sessions:
            unique_items = set(session)

            for item in unique_items:
                self.item_counts[item] += 1

            # Co-occurrences
            for i in unique_items:
                for j in unique_items:
                    if i != j:
                        self.cooccurrence[i][j] += 1

        # Compute similarities
        for i in self.cooccurrence:
            for j in self.cooccurrence[i]:
                numerator = self.cooccurrence[i][j]
                denominator = np.sqrt(self.item_counts[i] * self.item_counts[j])
                self.similarity[(i, j)] = numerator / denominator

    def recommend(self, session_items, top_k=5):
        """
        Recommend items based on current session.
        """
        scores = defaultdict(float)

        # Aggregate scores from all items in session
        for item in session_items:
            for other_item in self.cooccurrence.get(item, {}):
                if other_item not in session_items:  # Don't recommend already-seen items
                    scores[other_item] += self.similarity.get((item, other_item), 0)

        # Sort by score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]


# Example
sessions = [
    [1, 2, 3],
    [1, 3, 4],
    [2, 3, 5],
    [1, 2, 4]
]

model_i2i = ItemToItemSessionBased()
model_i2i.fit(sessions)

# Current session: user viewed items 1, 2
current_session = [1, 2]
recommendations = model_i2i.recommend(current_session, top_k=3)
print(f"Session {current_session}, recommend: {recommendations}")
# Output: [(3, ...), (4, ...), (5, ...)]
```

---

## Sliding Window Approach

### Idea

Instead of modeling entire session, use a **sliding window** of recent items.

**Example**:
```
Full sequence: [1, 2, 3, 4, 5, 6, 7]
Window size: 3

Windows:
  [1, 2, 3] → 4
  [2, 3, 4] → 5
  [3, 4, 5] → 6
  [4, 5, 6] → 7
```

**Benefit**: More training samples, focuses on local context.

---

### Implementation

```python
def create_sliding_windows(sequence, window_size=3):
    """
    Create sliding windows from sequence.

    Returns: list of (window, next_item) tuples
    """
    windows = []
    for i in range(len(sequence) - window_size):
        window = sequence[i:i+window_size]
        next_item = sequence[i+window_size]
        windows.append((window, next_item))
    return windows


# Example
sequence = [1, 2, 3, 4, 5, 6, 7]
windows = create_sliding_windows(sequence, window_size=3)

for window, next_item in windows:
    print(f"Window: {window} → Next: {next_item}")

# Output:
# Window: [1, 2, 3] → Next: 4
# Window: [2, 3, 4] → Next: 5
# Window: [3, 4, 5] → Next: 6
# Window: [4, 5, 6] → Next: 7
```

---

## Temporal Dynamics

### Time Decay

**Observation**: Recent items more relevant than old items.

**Approach**: Weight items by recency.

$$\text{score}(j | \text{session}) = \sum_{i \in \text{session}} w(t_i) \cdot \text{sim}(i, j)$$

where $w(t_i) = e^{-\lambda (t_{\text{now}} - t_i)}$ = weight for item seen at time $t_i$.

---

### Implementation

```python
import time

class TimeDecaySessionBased:
    def __init__(self, decay_rate=0.1):
        self.decay_rate = decay_rate
        self.item_similarity = {}  # Pre-computed similarities

    def recommend(self, session_with_timestamps, current_time, top_k=5):
        """
        Recommend with time decay.

        session_with_timestamps: list of (item, timestamp) tuples
        """
        scores = defaultdict(float)

        for item, timestamp in session_with_timestamps:
            # Time decay weight
            time_diff = current_time - timestamp
            weight = np.exp(-self.decay_rate * time_diff)

            # Add weighted similarity
            for other_item, sim in self.item_similarity.get(item, {}).items():
                scores[other_item] += weight * sim

        # Sort
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]
```

---

## Evaluation Metrics for Sequential Recommendation

### Next-Item Prediction

**Task**: Given sequence $[i_1, i_2, \ldots, i_t]$, predict $i_{t+1}$.

**Metrics**:

**1. Hit Rate@K**:
$$\text{HR@K} = \frac{1}{|T|} \sum_{s \in T} \mathbb{1}(\text{true next item} \in \text{top-K predictions})$$

**2. Mean Reciprocal Rank (MRR)**:
$$\text{MRR} = \frac{1}{|T|} \sum_{s \in T} \frac{1}{\text{rank of true item}}$$

---

### Implementation

```python
def evaluate_next_item(model, test_sessions, top_k=10):
    """
    Evaluate next-item prediction.
    """
    hits = 0
    mrr_sum = 0
    total = 0

    for session in test_sessions:
        if len(session) < 2:
            continue

        # Split session
        history = session[:-1]
        true_next = session[-1]

        # Predict
        recommendations = model.recommend(history, top_k=top_k)
        predicted_items = [item for item, score in recommendations]

        # Hit@K
        if true_next in predicted_items:
            hits += 1
            # MRR
            rank = predicted_items.index(true_next) + 1
            mrr_sum += 1 / rank

        total += 1

    hit_rate = hits / total
    mrr = mrr_sum / total

    return hit_rate, mrr


# Example
# test_sessions = [...]
# hr, mrr = evaluate_next_item(model, test_sessions, top_k=10)
# print(f"Hit Rate@10: {hr:.3f}, MRR: {mrr:.3f}")
```

---

## Real-World Applications

### 1. E-Commerce (Amazon)

**"Frequently Bought Together"**:
- Learn item co-occurrences from purchase sessions
- Recommend complements (laptop + bag + mouse)

**Implementation**: Item-to-item similarity

---

### 2. Music Streaming (Spotify)

**"Radio" feature**:
- Given seed song, generate playlist
- Use transition probabilities between songs

**Implementation**: Markov chain + audio similarity

---

### 3. Video Streaming (YouTube)

**"Up Next" recommendations**:
- Predict next video based on current watch session
- Consider watch time, likes, shares

**Implementation**: Session-based RNN (covered in next sections)

---

## Limitations of Traditional Sequential Methods

**1. Fixed patterns**: Markov chains assume stationary transitions (same patterns always)

**2. No long-term memory**: Sliding windows forget distant past

**3. No personalization**: Same recommendations for all users with same recent items

**4. Cold start**: New items have no transition data

**Solution**: Neural sequential models (RNNs, Transformers) – covered in next sections.

---

## Summary

**Key Takeaways**:
1. **Sequences matter**: User preferences are temporal and context-dependent
2. **Markov chains**: Model item-to-item transitions
3. **Session-based**: Recommend based on current session
4. **Item-to-item similarity**: "Customers who bought X also bought Y"
5. **Time decay**: Weight recent items more heavily
6. **Evaluation**: Hit Rate@K, MRR for next-item prediction

**When to use**:
- E-commerce: Complement recommendations (bought X, suggest Y)
- Music/Video: Session-based playlists
- News: Related articles based on reading history

**Limitations**:
- Fixed patterns (no learning from new data)
- Short memory (only recent items)
- No personalization

**Next**: RNNs for sequential recommendation (long-term memory, personalization).

---

## References

1. **Linden, G., Smith, B., & York, J. (2003)**. "Amazon.com Recommendations: Item-to-Item Collaborative Filtering". *IEEE Internet Computing*.
   - **Item-to-item** sequential recommendations

2. **Shani, G., Heckerman, D., & Brafman, R. I. (2005)**. "An MDP-Based Recommender System". *JMLR*.
   - **Markov Decision Processes** for sequential recommendation

3. **Rendle, S., Freudenthaler, C., & Schmidt-Thieme, L. (2010)**. "Factorizing Personalized Markov Chains for Next-Basket Recommendation". *WWW*.
   - **Factorized Markov chains** (personalized)

4. **Hidasi, B., et al. (2016)**. "Session-based Recommendations with Recurrent Neural Networks". *ICLR*.
   - **RNN-based** session recommendations (preview of next section)

5. **Davidson, J., et al. (2010)**. "The YouTube Video Recommendation System". *RecSys*.
   - **YouTube's approach** to sequential video recommendation

---

## Practice Problems

### Problem 1: Transition Matrix

**Given sessions**:
```
[A, B, C]
[A, C, B]
[B, C, A]
```

**Compute**: Transition matrix $M$ where $M_{ij} = P(next = j | current = i)$.

**Solution**:
```
Transitions:
  A → B: 1, A → C: 1
  B → C: 2
  C → B: 1, C → A: 1

Probabilities:
  P(B|A) = 1/2, P(C|A) = 1/2
  P(C|B) = 2/2 = 1
  P(B|C) = 1/2, P(A|C) = 1/2

Matrix:
     A    B    C
A  [0.0, 0.5, 0.5]
B  [0.0, 0.0, 1.0]
C  [0.5, 0.5, 0.0]
```

---

### Problem 2: Item-to-Item Similarity

**Given sessions**:
```
[1, 2, 3]
[1, 3, 4]
[2, 3]
```

**Compute**: Similarity between items 1 and 3.

**Solution**:
```
Co-occurrence(1, 3) = 2 (sessions 1 and 2)
Count(1) = 2 (sessions 1 and 2)
Count(3) = 3 (all sessions)

Similarity = 2 / sqrt(2 × 3) = 2 / sqrt(6) ≈ 0.816
```

---

### Problem 3: Next-Item Prediction

**Given**:
```
Session: [A, B, C]
True next: D

Model recommends: [D, E, F, G, H] (top-5)

Compute: Hit@5 and MRR.
```

**Solution**:
```
Hit@5: True next (D) is in top-5 → Hit = 1

MRR: Rank of D = 1 → MRR = 1/1 = 1.0
```
