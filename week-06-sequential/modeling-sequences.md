# Week 6: Sequential Recommendations - Modeling Sequences

## The Failure That Changed Everything: Why Static CF Breaks

*Before we dive into sequential models, let me show you a failure case that will make you never think about recommendations the same way again.*

**The Scenario**: Alice just bought a laptop on Amazon.

**What Static Collaborative Filtering Recommends**:
```
Based on your purchase history, you might also like:
1. Dell XPS 15 Laptop       ($1,299)
2. MacBook Pro 14"          ($1,999)
3. ThinkPad X1 Carbon       ($1,549)
```

**The Problem**: Alice just bought a laptop! She doesn't need ANOTHER laptop!

**What She Actually Needs**:
```
1. Laptop bag               ($49)
2. Wireless mouse           ($29)
3. USB-C hub                ($35)
4. Laptop stand             ($39)
```

*Can you see why this happens?* Static collaborative filtering computes:

$$\hat{r}_{ui} = f(u, i)$$

It treats Alice as a static entity: "Alice = person who likes laptops." But Alice's **context has changed**. The moment she bought the laptop, her needs shifted from "laptop" to "laptop accessories."

**The Mathematical Blindness**:

Static CF asks: "What do users like Alice typically buy?"
- Users who bought laptops also bought... other laptops! (because they were *comparing* before deciding)

Sequential models ask: "What do users buy **after** buying a laptop?"
- Users who bought laptops then bought... bags, mice, chargers!

**The Key Insight**: **User preferences evolve over time** and are **context-dependent**. The order of interactions contains crucial information that static models throw away.

---

## Learning Objectives

By the end of this section, you will:
- Understand why sequences matter in recommendations
- **Derive Markov chains from first principles** (why and how)
- **Build a complete transition matrix step by step** with real numbers
- Master session-based models and when to use them
- Learn item-to-item sequential patterns
- **Derive time decay from first principles** and work through numerical examples
- Recognize what can go wrong and how to fix it

---

## Why Sequences Matter

### Static vs. Sequential Preferences

*Let me build up the mathematical difference step by step.*

**Static CF assumption**:
$$\hat{r}_{ui} = f(u, i)$$

User preferences are time-invariant. If Alice liked "The Matrix" last year, she'll want similar movies forever.

**Sequential reality**:
$$\hat{r}_{ui,t} = f(u, i, s_t)$$

where $s_t$ = user's interaction sequence up to time $t$.

*What does $s_t$ capture that static models miss?*

1. **Recency**: What did Alice do *most recently*?
2. **Order**: Did she view A before B, or B before A?
3. **Context**: Is she in "browsing mode" or "buying mode"?
4. **Session coherence**: What's the theme of this browsing session?

---

### Three Real-World Examples

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

### Building Intuition: The Simplest Possible Sequence Model

*Let me ask you a question before showing the formula: What is the simplest model you could build to predict the next item?*

**Approach 1: Ignore all history**
- Predict the most popular item globally
- Problem: Completely ignores context

**Approach 2: Use the entire history**
$$P(i_t | i_1, i_2, \ldots, i_{t-1})$$
- Condition on all previous items
- Problem: Exponentially many combinations! With 1000 items and history of 10: $1000^{10}$ possible histories

**Approach 3: The Markov Assumption**

*What would happen if we made a simplifying assumption: the next item depends ONLY on the current item?*

This is the **Markov assumption**:
$$P(i_t | i_1, i_2, \ldots, i_{t-1}) = P(i_t | i_{t-1})$$

*Before I show you why this works, can you see the trade-off?*
- **Pro**: Only need to model $|I| \times |I|$ transitions (manageable!)
- **Con**: Ignores all history except the last item

---

### Deriving the First-Order Markov Chain

*Let's derive this step by step, so every piece makes sense.*

**Step 1: Define what we want to learn**

We want to predict: Given current item $i$, what's the probability of next item $j$?

$$P(\text{next item} = j | \text{current item} = i)$$

**Step 2: Represent as a matrix**

Define the **transition matrix** $\mathbf{M} \in \mathbb{R}^{|I| \times |I|}$:

$$\mathbf{M}_{ij} = P(\text{next item} = j | \text{current item} = i)$$

*What properties must this matrix have?*

1. **Non-negative**: All entries $\geq 0$ (probabilities can't be negative)
2. **Row stochastic**: Each row sums to 1 (we must go somewhere)

$$\sum_{j} \mathbf{M}_{ij} = 1 \quad \forall i$$

**Step 3: Estimate from data**

*How do we learn this matrix? The simplest approach: count and normalize.*

$$\hat{M}_{ij} = \frac{\text{count}(i \to j)}{\sum_{k} \text{count}(i \to k)}$$

where $\text{count}(i \to j)$ = number of times item $j$ immediately follows item $i$ in our data.

*Why does this work?* This is the **Maximum Likelihood Estimate**. If we observe many $i \to j$ transitions relative to other transitions from $i$, then $P(j|i)$ should be high.

---

### Complete Numerical Example: Building a Transition Matrix

*Now let's work through a complete example with actual numbers. I'll show you every step.*

**The Data**: 3 user sessions with 4 items (A, B, C, D)

```
Session 1: [A, B, C, D]
Session 2: [A, C, B]
Session 3: [B, C, A, C]
```

**Step 1: Extract all transitions**

*Go through each session and list every consecutive pair:*

**Session 1**: A→B, B→C, C→D
**Session 2**: A→C, C→B
**Session 3**: B→C, C→A, A→C

*Now count how many times each transition appears:*

| From \\ To | A | B | C | D |
|------------|---|---|---|---|
| A | 0 | 1 | 2 | 0 |
| B | 0 | 0 | 2 | 0 |
| C | 1 | 1 | 0 | 1 |
| D | 0 | 0 | 0 | 0 |

*Let's verify a few:*
- A→B: Appears once (Session 1)
- A→C: Appears twice (Session 2 and Session 3)
- B→C: Appears twice (Session 1 and Session 3)
- C→A: Appears once (Session 3)
- C→B: Appears once (Session 2)
- C→D: Appears once (Session 1)

**Step 2: Normalize each row to get probabilities**

*For each row, divide by the row sum:*

**Row A**: Total outgoing = 0+1+2+0 = 3
- P(A|A) = 0/3 = 0.00
- P(B|A) = 1/3 = 0.33
- P(C|A) = 2/3 = 0.67
- P(D|A) = 0/3 = 0.00

**Row B**: Total outgoing = 0+0+2+0 = 2
- P(A|B) = 0/2 = 0.00
- P(B|B) = 0/2 = 0.00
- P(C|B) = 2/2 = 1.00
- P(D|B) = 0/2 = 0.00

**Row C**: Total outgoing = 1+1+0+1 = 3
- P(A|C) = 1/3 = 0.33
- P(B|C) = 1/3 = 0.33
- P(C|C) = 0/3 = 0.00
- P(D|C) = 1/3 = 0.33

**Row D**: Total outgoing = 0 (D never transitions to anything)
- This is a problem! D is an "absorbing state" (sessions end there)

**Final Transition Matrix**:
```
      A      B      C      D
A  [0.00,  0.33,  0.67,  0.00]
B  [0.00,  0.00,  1.00,  0.00]
C  [0.33,  0.33,  0.00,  0.33]
D  [0.00,  0.00,  0.00,  0.00]  ← Absorbing state
```

**Step 3: Make predictions**

*Now let's use our model!*

**Scenario**: User just interacted with item B. What should we recommend?

Look at row B: `[0.00, 0.00, 1.00, 0.00]`

**Prediction**: Recommend C with 100% confidence!

*Does this make sense?* Looking at our data, every time B appeared, C came next. The model learned this perfectly.

**Scenario**: User just interacted with item C. What should we recommend?

Look at row C: `[0.33, 0.33, 0.00, 0.33]`

**Prediction**: Recommend A, B, or D (all equally likely at 33%)

*In practice, we'd recommend all three, perhaps ranked by some secondary criterion (popularity, recency, etc.)*

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


# Example with our data
sessions = [
    [0, 1, 2, 3],  # A=0, B=1, C=2, D=3
    [0, 2, 1],
    [1, 2, 0, 2]
]

model = FirstOrderMarkovChain(n_items=4)
model.fit(sessions)

# Recommend next item after item 1 (B)
recommendations = model.recommend(current_item=1, top_k=3)
print(f"After item B, recommend: {recommendations}")
# Output: [(2, 1.0), (0, 0.0), (1, 0.0)] → C with probability 1.0
```

---

### Limitations of First-Order Markov Chains

*Can you see what we're losing with the Markov assumption?*

**1. Short memory**: Only considers last item (ignores earlier history).

**Example**:
```
User sequence: [Laptop, Laptop bag, ...]

With only "Laptop bag" as context, model might recommend another bag.
But full history suggests user bought laptop → should recommend mouse, charger.
```

**2. Data sparsity**: Many item pairs never co-occur → zero probabilities.

*What happens if user is at item X, but X never appeared in training?*
- All transition probabilities are 0 (or undefined)
- Fall back to popularity-based recommendation

**3. No personalization**: Same recommendations for all users who viewed item $i$.

*Alice and Bob both viewed "The Matrix" → they get identical recommendations.*

---

## Higher-Order Markov Chains

### The Intuition: Remembering More

*What would happen if we conditioned on the last TWO items instead of just one?*

**Second-Order Markov Chain**:

$$P(i_t | i_1, \ldots, i_{t-1}) = P(i_t | i_{t-2}, i_{t-1})$$

**Example**:
- 1st-order: "User is at Laptop Bag" → Recommend bags, wallets, accessories
- 2nd-order: "User went Laptop → Laptop Bag" → Recommend mouse, charger (laptop accessories!)

The two-item context captures the **intent**: This is a laptop-purchasing journey, not a bag-shopping journey.

**Transition tensor**:
$$\mathbf{M}_{ijk} = P(\text{next} = k | \text{prev} = i, \text{current} = j)$$

**Challenge**: Exponentially more parameters!
- 1st-order: $|I|^2$ parameters
- 2nd-order: $|I|^3$ parameters
- k-th order: $|I|^{k+1}$ parameters

With 10,000 items: 1st-order = 100M, 2nd-order = 1 trillion!

---

### Variable-Order Markov Chains

*Here's a clever idea: What if we use higher-order when possible, but fall back to lower-order when data is sparse?*

**Algorithm**:
```
Given context [laptop, laptop bag]:
  1. Try to find P(next | laptop, laptop bag) using 2nd-order
  2. If sparse (few observations), fall back to P(next | laptop bag) using 1st-order
  3. If still sparse, use P(next) based on popularity
```

**Example**:
```
If (laptop, laptop bag) → mouse: Use 2nd-order (we have data!)
If (laptop bag) → ?: Fall back to 1st-order
If never seen context: Use popularity
```

---

## Session-Based vs. Sequence-Based: When to Use Which?

*This is a common source of confusion. Let me clarify.*

### Session-Based Recommendation

**Definition**: A **session** is a bounded interaction sequence within a short time window (e.g., 30 minutes).

**Characteristics**:
- Clear start and end
- Single intent/goal
- Items within session are highly related
- User identity may be unknown (anonymous browsing)

**Examples**:
- E-commerce browsing session: "Looking for running shoes"
- Music listening session: "Evening relaxation playlist"
- News reading session: "Catching up on tech news"

**Model approach**: Consider **all items in session** (not just last item)

---

### Sequence-Based Recommendation

**Definition**: A **sequence** is the full history of user interactions over time.

**Characteristics**:
- No clear boundaries
- Multiple intents/goals over time
- User identity is known
- Includes preference drift

**Examples**:
- Full Netflix watch history (years of viewing)
- Complete Amazon purchase history
- Spotify listening history

**Model approach**: Model **long-term preferences** with attention to recency

---

### Decision Guide: When to Use Which?

| Scenario | Session-Based | Sequence-Based |
|----------|--------------|----------------|
| Anonymous users | Yes | No (need user ID) |
| Short-term intent | Yes | Maybe |
| Long-term preferences | No | Yes |
| Known user identity | Either | Yes |
| Cold start users | Yes | No |
| E-commerce browsing | Yes | |
| Music playlist | Yes | |
| Purchase history | | Yes |
| Video watch history | | Yes |

**Rule of thumb**:
- **Session-based**: When user is in a focused task, especially anonymous
- **Sequence-based**: When modeling long-term evolution of preferences

---

## Item-to-Item Similarity (Amazon Approach)

### The Foundational Approach

**Paper**: Linden et al., "Amazon.com Recommendations: Item-to-Item Collaborative Filtering" (2003)

**Idea**: "Customers who bought X also bought Y"

*This is simpler than Markov chains but surprisingly effective!*

**Process**:
1. For each item pair $(i, j)$, count co-occurrences in sessions
2. Compute similarity: $\text{sim}(i, j) = \frac{\text{co-occurrence}(i, j)}{\sqrt{\text{count}(i) \times \text{count}(j)}}$
3. Recommend items most similar to items in current session

**Key difference from Markov chains**:
- Markov: Cares about **order** (A→B is different from B→A)
- Item-to-item: Only cares about **co-occurrence** (A and B appeared together)

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

## Temporal Dynamics: Time Decay

### The Intuition: Why Recent Items Matter More

*Before I show you the formula, let me ask: if a user viewed Item A 5 minutes ago and Item B 5 days ago, which should influence recommendations more?*

Obviously Item A! But by how much?

**The problem**: How do we mathematically express "recent items matter more"?

---

### Deriving Time Decay from First Principles

**Step 1: Define what we want**

We want a weight $w(t)$ such that:
- Recent items (small $t$ = time since interaction) get high weight
- Old items (large $t$) get low weight
- Weight decreases smoothly as $t$ increases

**Step 2: Consider candidate functions**

*What properties should $w(t)$ have?*

1. $w(0) = 1$ (just-viewed item has full weight)
2. $\lim_{t \to \infty} w(t) = 0$ (very old items contribute nothing)
3. $w'(t) < 0$ (weight decreases with time)
4. Smooth (no discontinuities)

**Step 3: The exponential decay function**

The simplest function satisfying all these properties:

$$w(t) = e^{-\lambda t}$$

where $\lambda > 0$ is the **decay rate**.

*Why exponential?*
- It's the unique function where **rate of decay is proportional to current value**
- $\frac{dw}{dt} = -\lambda w(t)$
- This is the same as radioactive decay, population decline, etc.

**Step 4: Understanding $\lambda$**

$\lambda$ controls how fast weights decay:

| $\lambda$ | Half-life* | Interpretation |
|-----------|-----------|----------------|
| 0.001 | 693 time units | Very slow decay (long memory) |
| 0.01 | 69 time units | Moderate decay |
| 0.1 | 6.9 time units | Fast decay (short memory) |
| 1.0 | 0.69 time units | Very fast decay |

*Half-life = time for weight to drop to 0.5 = $\frac{\ln 2}{\lambda}$

---

### Complete Numerical Example: Time-Weighted Recommendations

**Scenario**: User's session with timestamps (in minutes from now):

| Item | Time ago | Raw interaction |
|------|----------|-----------------|
| A | 5 min | Clicked |
| B | 30 min | Clicked |
| C | 120 min | Clicked |

**We have item similarities** (from co-occurrence):
- sim(A, X) = 0.8
- sim(B, X) = 0.6
- sim(C, X) = 0.7

**Question**: What's the weighted score for recommending item X?

**Step 1: Choose decay rate**

Let's use $\lambda = 0.02$ (half-life ≈ 35 minutes)

**Step 2: Compute time weights**

$$w_A = e^{-0.02 \times 5} = e^{-0.1} = 0.905$$
$$w_B = e^{-0.02 \times 30} = e^{-0.6} = 0.549$$
$$w_C = e^{-0.02 \times 120} = e^{-2.4} = 0.091$$

*Notice how the 2-hour-old interaction (C) has much lower weight than the 5-minute-old one (A)!*

**Step 3: Compute weighted score**

$$\text{score}(X) = w_A \cdot \text{sim}(A, X) + w_B \cdot \text{sim}(B, X) + w_C \cdot \text{sim}(C, X)$$
$$= 0.905 \times 0.8 + 0.549 \times 0.6 + 0.091 \times 0.7$$
$$= 0.724 + 0.329 + 0.064$$
$$= 1.117$$

**Comparison**: Without time decay (all weights = 1):
$$\text{score}(X) = 0.8 + 0.6 + 0.7 = 2.1$$

**Interpretation**: With time decay, the recent interaction with A (high similarity to X) dominates. Without decay, the old interaction with C contributes equally, which may not reflect current intent.

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

## Sliding Window Approach

### The Idea

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

**Benefits**:
- More training samples (one per window, not one per sequence)
- Focuses on local context
- Natural fit for neural sequence models

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

## Evaluation Metrics for Sequential Recommendation

### Next-Item Prediction

**Task**: Given sequence $[i_1, i_2, \ldots, i_t]$, predict $i_{t+1}$.

*This is the core evaluation task. Let me explain the metrics:*

**1. Hit Rate@K** (also called Recall@K)

$$\text{HR@K} = \frac{1}{|T|} \sum_{s \in T} \mathbb{1}(\text{true next item} \in \text{top-K predictions})$$

*Intuition*: What fraction of the time is the correct item in our top-K recommendations?

**Example**:
- Model predicts: [C, A, D, E, B] (top 5)
- True next item: D
- HR@5 = 1 (D is in top 5)
- HR@3 = 1 (D is in top 3)
- HR@1 = 0 (D is not the top prediction)

**2. Mean Reciprocal Rank (MRR)**

$$\text{MRR} = \frac{1}{|T|} \sum_{s \in T} \frac{1}{\text{rank of true item}}$$

*Intuition*: Rewards models that rank the correct item higher. MRR of 0.5 means the correct item is typically ranked 2nd.

**Example**:
- Prediction: [C, A, D, E, B], True: D, Rank: 3 → RR = 1/3
- Prediction: [D, A, C, E, B], True: D, Rank: 1 → RR = 1

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

## What Can Go Wrong?

*Let me warn you about common failure modes with sequential models.*

### Failure Mode 1: Sparse Transition Matrix

**Problem**: Many item pairs never appear consecutively.

**Symptoms**:
- Most entries in transition matrix are 0
- Model can't make predictions for many current items
- Falls back to popularity everywhere

**Example**:
- 10,000 items → 100 million possible pairs
- Only 1 million transitions observed → 99% zeros!

**Solutions**:
- Smoothing: Add small probability to all transitions
- Hybrid: Combine with popularity or content-based
- Embedding methods: Learn dense representations (covered in RNN section)

---

### Failure Mode 2: Cold Start (New Items)

**Problem**: New items have no transition data.

**Symptoms**:
- New items never get recommended
- Popular items dominate recommendations
- No exploration of catalog

**Solutions**:
- Content-based features for new items
- Exploration/exploitation (bandits)
- Pre-train embeddings on item attributes

---

### Failure Mode 3: Session Boundary Issues

**Problem**: Treating unrelated sequences as connected.

**Example**:
```
User logs:
  Session 1 (Monday): [A, B, C]
  Session 2 (Friday): [X, Y, Z]

If we concatenate: [A, B, C, X, Y, Z]
Model learns C → X transition, but this is noise!
```

**Solutions**:
- Explicit session detection (time gaps > 30 min = new session)
- Session-aware training (don't learn across sessions)
- Session boundary tokens in sequence models

---

### Failure Mode 4: Position Bias in Evaluation

**Problem**: Users only see top-ranked items, so we only have feedback on what we recommended.

**Symptoms**:
- Model learns to recommend what it already recommends
- Popularity bias amplified
- Can't evaluate "what if" scenarios

**Solutions**:
- Randomization in production (some % random items)
- Inverse propensity scoring
- Counterfactual evaluation

---

### Failure Mode 5: Ignoring Session Intent

**Problem**: First-order Markov treats all transitions equally, ignoring overall session goal.

**Example**:
```
Session 1: Laptop → Mouse → Keyboard (buying computer setup)
Session 2: Mouse → Cheese → Trap (searching for mouse trap!)

After "Mouse", first-order Markov can't distinguish these contexts.
```

**Solutions**:
- Higher-order Markov (context from multiple items)
- Session embeddings (learn session-level representation)
- Attention mechanisms (let model weigh relevant history)

---

## Real-World Applications

### 1. E-Commerce (Amazon)

**"Frequently Bought Together"**:
- Learn item co-occurrences from purchase sessions
- Recommend complements (laptop + bag + mouse)

**Implementation**: Item-to-item similarity

### 2. Music Streaming (Spotify)

**"Radio" feature**:
- Given seed song, generate playlist
- Use transition probabilities between songs

**Implementation**: Markov chain + audio similarity

### 3. Video Streaming (YouTube)

**"Up Next" recommendations**:
- Predict next video based on current watch session
- Consider watch time, likes, shares

**Implementation**: Session-based RNN (covered in next sections)

---

## Summary

**Key Takeaways**:
1. **Sequences matter**: User preferences are temporal and context-dependent
2. **Markov chains**: Model item-to-item transitions with simple counting
3. **Session-based**: Recommend based on current session (short-term)
4. **Sequence-based**: Model long-term history (requires user identity)
5. **Item-to-item similarity**: "Customers who bought X also bought Y"
6. **Time decay**: Weight recent items exponentially higher
7. **Evaluation**: Hit Rate@K, MRR for next-item prediction

**When to use**:
- E-commerce: Complement recommendations (bought X, suggest Y)
- Music/Video: Session-based playlists
- News: Related articles based on reading history

**What can go wrong**:
- Sparse transition matrices
- Cold start for new items
- Session boundary confusion
- Position bias in evaluation
- Ignoring session-level intent

**Limitations of traditional methods**:
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

Similarity = 2 / sqrt(2 * 3) = 2 / sqrt(6) = 0.816
```

---

### Problem 3: Time-Weighted Score

**Given**:
- Item A: viewed 10 minutes ago, sim(A, X) = 0.9
- Item B: viewed 60 minutes ago, sim(B, X) = 0.8
- Decay rate: $\lambda = 0.05$

**Compute**: Time-weighted score for item X.

**Solution**:
```
w_A = e^(-0.05 * 10) = e^(-0.5) = 0.607
w_B = e^(-0.05 * 60) = e^(-3.0) = 0.050

Score(X) = 0.607 * 0.9 + 0.050 * 0.8
         = 0.546 + 0.040
         = 0.586
```

---

### Problem 4: Next-Item Prediction

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

---

### Problem 5: Second-Order Markov

**Given sessions**:
```
[A, B, C]
[A, B, D]
[B, C, A]
```

**Compute**: P(next | A, B) and P(next | B, C)

**Solution**:
```
Context (A, B):
  Appears twice: A→B→C and A→B→D
  P(C | A, B) = 1/2
  P(D | A, B) = 1/2

Context (B, C):
  Appears once: B→C→A
  P(A | B, C) = 1/1 = 1
```
