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

## Why Do Markov Chains Fail? The Opening Failure Case

*Before we dive into RNNs, let me show you exactly why Markov chains are fundamentally broken for real recommendation scenarios.*

### The Problem: Long-Range Dependencies

**Scenario**: A user browsing an electronics store over 30 minutes.

```
Position 1:  [Gaming Laptop]     ← Primary intent established
Position 2:  [Laptop Bag]
Position 3:  [Wireless Mouse]
Position 4:  [Coffee]            ← Distraction
Position 5:  [Snacks]            ← Distraction
Position 6:  [USB-C Hub]
Position 7:  [Mechanical Keyboard]
Position 8:  [Monitor Stand]
Position 9:  [Water Bottle]      ← Distraction
Position 10: [???]               ← What should we recommend?
```

**The right answer**: Gaming headset, external GPU, gaming mouse pad (items related to the Gaming Laptop at position 1).

**What a first-order Markov chain sees**: Only "Water Bottle" at position 9.
**What it recommends**: Snacks, cups, kitchen items. *Completely wrong!*

### Let's Do the Math

**First-order Markov chain probability**:
$$P(i_{10} | i_1, i_2, \ldots, i_9) = P(i_{10} | i_9) = P(\text{item} | \text{Water Bottle})$$

The gaming laptop at position 1? **Completely invisible**. The model has *zero information* about the user's primary intent.

**Can you see why** this is catastrophic? The most important signal (position 1) is 9 steps away, but Markov chains only remember 1 step!

### What About Higher-Order Markov Chains?

"Can't we just use a second-order or third-order Markov chain?"

**Second-order**: $P(i_{10} | i_8, i_9) = P(\text{item} | \text{Monitor Stand}, \text{Water Bottle})$

Still doesn't see the gaming laptop!

**The exponential explosion problem**:

| Order | Memory Length | Parameters Needed |
|-------|---------------|-------------------|
| 1st   | 1 item        | $N^2$ (1 million items = 1 trillion params) |
| 2nd   | 2 items       | $N^3$ (1 million items = impossible) |
| 3rd   | 3 items       | $N^4$ (astronomically impossible) |

*Notice that* even a modest catalog of 10,000 items needs $10^{12}$ parameters for a 3rd-order Markov chain. That's a terabyte just for the transition matrix!

### Concrete Numbers: The Accuracy Gap

**Experiment** (RecSys benchmark):

| Model | Hit@10 | NDCG@10 | Can see position 1? |
|-------|--------|---------|---------------------|
| 1st-order Markov | 0.312 | 0.189 | No |
| 2nd-order Markov | 0.341 | 0.207 | No |
| GRU4Rec | **0.518** | **0.342** | **Yes** |

**Result**: GRU4Rec is **66% better** at Hit@10 because it can remember the gaming laptop!

*What happens if* we could design a model that remembers the ENTIRE sequence, not just the last 1-2 items? That's exactly what RNNs do.

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

---

### The Vanishing Gradient Problem: A Complete Numerical Walkthrough

*Why can't vanilla RNNs learn long-term patterns? Let me show you with actual numbers.*

**Scenario**: User's 20-item browsing history. First item was a laptop, last item is a laptop stand.

```
Position 1:  [laptop] ← We should remember this!
Position 2:  [laptop_case]
Position 3:  [wireless_mouse]
...
Position 18: [coffee]
Position 19: [snacks]
Position 20: [laptop_stand] ← Predict next item
```

**The gradient problem**:

To learn "laptop at position 1 predicts laptop accessories at position 20", gradients must flow back 19 steps.

**Gradient at position 1** = gradient at position 20 x (derivative at step 19) x (derivative at step 18) x ... x (derivative at step 1)

#### Step-by-Step Numerical Example

**Setup**: Let's trace the gradient flow with actual numbers.

**The tanh derivative**: For $\tanh(x)$, the derivative is $1 - \tanh^2(x)$.

| Hidden state value | $\tanh(h)$ | Derivative $1 - \tanh^2$ |
|-------------------|-----------|-------------------------|
| 0.0 | 0.0 | **1.0** (maximum) |
| 0.5 | 0.46 | 0.79 |
| 1.0 | 0.76 | **0.42** |
| 2.0 | 0.96 | **0.07** |

*Notice that* the derivative is almost always less than 1, and often much less!

**The chain rule multiplication**:

Assume average derivative = 0.5 (optimistic).

| Backprop Steps | Gradient Multiplier | Interpretation |
|----------------|---------------------|----------------|
| 1 step | $0.5^1 = 0.5$ | Half signal |
| 5 steps | $0.5^5 = 0.031$ | 3% signal |
| 10 steps | $0.5^{10} = 0.00098$ | 0.1% signal |
| 19 steps | $0.5^{19} = 0.0000019$ | **0.0002% signal** |

**Visual representation**:
```
Position:    1        5       10       15       20
Gradient:   1e-6     1e-4    0.001    0.03     1.0
             |        |        |        |        |
         Vanished!   Tiny    Small    Weak    Strong

Learning:  NOTHING  Almost   A bit   Some    Full
           learned   none   learned  learned learned
```

**The catastrophic result**: The network cannot learn that position 1 matters for position 20.

*Can you see why* this is a fundamental mathematical problem, not just an implementation issue?

**Solution**: Use LSTM or GRU with **gating mechanisms** that create "highways" for gradients.

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

---

## GRU Gate Equations: Complete Derivation with Intuition

*Let me walk you through each equation step by step. First the intuition, then the math, then why each term exists.*

### The Reset Gate: "Should I Forget the Past?"

**Intuition**: Imagine you're browsing Amazon. You were looking at laptops, then suddenly you search for "birthday gift for mom." The reset gate asks: "Is my laptop context still relevant?"

**Equation**:
$$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t])$$

**Derivation step-by-step**:

1. **Concatenate** previous hidden state and current input: $[\mathbf{h}_{t-1}, \mathbf{x}_t]$
   - *Why?* We need both what we remembered AND what we just saw to decide

2. **Linear transformation**: $\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t]$
   - *Why?* Learn which combinations of past+present signal a context switch

3. **Sigmoid activation**: $\sigma(\cdot)$
   - *Why?* Output must be between 0 and 1 (a "gate" that's open or closed)

**What each term decides**:
- $\mathbf{r}_t \approx 0$: "The past is irrelevant. Start fresh."
- $\mathbf{r}_t \approx 1$: "The past is important. Keep it all."

### The Update Gate: "How Much Should I Change?"

**Intuition**: You saw a coffee mug while shopping for laptops. The update gate asks: "Is this mug important enough to change my shopping intent?"

**Equation**:
$$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t])$$

**Why this term?**:
- Similar structure to reset gate, but different purpose
- Learned separately (different weights $\mathbf{W}_z$ vs $\mathbf{W}_r$)
- Controls the interpolation between old and new state

**What each term decides**:
- $\mathbf{z}_t \approx 0$: "Ignore this input. Keep my old memory."
- $\mathbf{z}_t \approx 1$: "This input is crucial. Update heavily."

### The Candidate Hidden State: "What Would the New Memory Look Like?"

**Intuition**: If we do decide to update, what should the new memory contain?

**Equation**:
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$$

**Derivation step-by-step**:

1. **Apply reset gate**: $\mathbf{r}_t \odot \mathbf{h}_{t-1}$
   - *Why?* If $r_t = 0$, we compute new memory without looking at the past
   - *Why element-wise multiply?* Different dimensions can forget different things

2. **Concatenate with input**: $[\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t]$
   - *Why?* Combine (possibly filtered) past with current input

3. **Linear + tanh**: $\tanh(\mathbf{W} \cdot)$
   - *Why tanh?* Squash to [-1, 1] for stable hidden state values

### The Final Update: "Interpolate Between Old and New"

**Intuition**: Now blend the old memory with the candidate new memory.

**Equation**:
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

**Derivation step-by-step**:

1. **Keep from past**: $(1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1}$
   - *Why $(1 - \mathbf{z}_t)$?* If update gate is 0, keep ALL of past
   - *Why element-wise?* Different dimensions have different importance

2. **Add from new**: $\mathbf{z}_t \odot \tilde{\mathbf{h}}_t$
   - *Why $\mathbf{z}_t$?* If update gate is 1, take ALL of new
   - Weights sum to 1: it's a weighted average!

**The beautiful property**: This is a **convex combination**. The hidden state smoothly interpolates between "keep everything" and "replace everything."

*Can you see why* this solves the vanishing gradient problem? If $\mathbf{z}_t \approx 0$, the gradient flows DIRECTLY from $\mathbf{h}_t$ to $\mathbf{h}_{t-1}$ without any multiplication by small numbers!

---

## Complete Numerical Walkthrough: GRU Processing a 4-Item Sequence

*Let me trace through exactly what happens inside a GRU with actual numbers.*

### Setup

**User's sequence**: [Laptop, Mouse, Coffee, Keyboard]

**Dimensions**: Embedding dim = 3, Hidden dim = 3 (small for illustration)

**Item embeddings** (3-dimensional):
```
e_laptop   = [0.8, 0.2, 0.1]   ← Electronics, high first dimension
e_mouse    = [0.7, 0.3, 0.1]   ← Electronics, similar to laptop
e_coffee   = [0.1, 0.1, 0.9]   ← Food/drink, different pattern!
e_keyboard = [0.6, 0.4, 0.1]   ← Electronics, back to pattern
```

**Weight matrices** (simplified, 6x3 for [h, x] -> gate):
```
W_r (reset):  [[0.5, 0.2, 0.1, 0.3, 0.1, 0.2],
               [0.2, 0.5, 0.1, 0.1, 0.3, 0.1],
               [0.1, 0.1, 0.5, 0.2, 0.1, 0.3]]

W_z (update): [[0.4, 0.3, 0.1, 0.2, 0.2, 0.1],
               [0.3, 0.4, 0.2, 0.1, 0.3, 0.1],
               [0.1, 0.2, 0.4, 0.1, 0.1, 0.3]]
```

**Initial hidden state**: $\mathbf{h}_0 = [0, 0, 0]$

---

### Step 1: Processing "Laptop"

**Input**: $\mathbf{x}_1 = [0.8, 0.2, 0.1]$

**Concatenate** $[\mathbf{h}_0, \mathbf{x}_1] = [0, 0, 0, 0.8, 0.2, 0.1]$

**Reset gate**:
$$\mathbf{r}_1 = \sigma(\mathbf{W}_r \cdot [0, 0, 0, 0.8, 0.2, 0.1])$$
$$= \sigma([0.26, 0.14, 0.21]) = [0.56, 0.53, 0.55]$$

*Interpretation*: Reset gate is ~0.5 (neutral), but $\mathbf{h}_0 = 0$ anyway.

**Update gate**:
$$\mathbf{z}_1 = \sigma(\mathbf{W}_z \cdot [0, 0, 0, 0.8, 0.2, 0.1])$$
$$= \sigma([0.21, 0.22, 0.14]) = [0.55, 0.55, 0.54]$$

*Interpretation*: Update gate ~0.55, will incorporate about half of new info.

**Candidate hidden state**:
$$\tilde{\mathbf{h}}_1 = \tanh(\mathbf{W} \cdot [\mathbf{r}_1 \odot \mathbf{h}_0, \mathbf{x}_1])$$
$$= \tanh(\mathbf{W} \cdot [0, 0, 0, 0.8, 0.2, 0.1]) = [0.42, 0.35, 0.28]$$

**Final hidden state**:
$$\mathbf{h}_1 = (1 - \mathbf{z}_1) \odot \mathbf{h}_0 + \mathbf{z}_1 \odot \tilde{\mathbf{h}}_1$$
$$= [0.45, 0.45, 0.46] \odot [0, 0, 0] + [0.55, 0.55, 0.54] \odot [0.42, 0.35, 0.28]$$
$$= [0, 0, 0] + [0.23, 0.19, 0.15] = \mathbf{[0.23, 0.19, 0.15]}$$

**Result**: Hidden state now encodes "laptop" - notice higher first dimension (electronics pattern).

---

### Step 2: Processing "Mouse"

**Input**: $\mathbf{x}_2 = [0.7, 0.3, 0.1]$ (similar to laptop!)

**Concatenate** $[\mathbf{h}_1, \mathbf{x}_2] = [0.23, 0.19, 0.15, 0.7, 0.3, 0.1]$

**Reset gate**:
$$\mathbf{r}_2 = \sigma([0.38, 0.29, 0.32]) = [0.59, 0.57, 0.58]$$

*Notice that* the reset gate stays high (~0.58) because mouse is related to laptop!

**Update gate**:
$$\mathbf{z}_2 = \sigma([0.35, 0.36, 0.23]) = [0.59, 0.59, 0.56]$$

*The model wants to update* because this is relevant new information.

**Final hidden state**:
$$\mathbf{h}_2 = \mathbf{[0.35, 0.31, 0.22]}$$

*Interpretation*: Electronics pattern strengthens (first dimension increases).

---

### Step 3: Processing "Coffee" (The Distraction!)

**Input**: $\mathbf{x}_3 = [0.1, 0.1, 0.9]$ (very different pattern!)

**Concatenate** $[\mathbf{h}_2, \mathbf{x}_3] = [0.35, 0.31, 0.22, 0.1, 0.1, 0.9]$

**Reset gate**:
$$\mathbf{r}_3 = \sigma([0.37, 0.34, 0.58]) = [0.59, 0.58, 0.64]$$

**Update gate** (THE KEY!):
$$\mathbf{z}_3 = \sigma([0.30, 0.31, 0.45]) = [0.57, 0.58, 0.61]$$

*Something interesting happens*: The third dimension of $\mathbf{z}_3$ is higher (0.61), suggesting the model recognizes this is a different category. But overall, the update gate stays moderate.

**Final hidden state**:
$$\mathbf{h}_3 = \mathbf{[0.28, 0.27, 0.35]}$$

*Notice that* the third dimension increased (coffee influence), but the first dimension (electronics) is still strong at 0.28. **The GRU didn't forget laptops!**

---

### Step 4: Processing "Keyboard" (Back to Electronics!)

**Input**: $\mathbf{x}_4 = [0.6, 0.4, 0.1]$

**Reset gate**:
$$\mathbf{r}_4 = [0.61, 0.60, 0.57]$$

*The reset gate is high* - keep the past because keyboard is related to electronics!

**Update gate**:
$$\mathbf{z}_4 = [0.60, 0.61, 0.55]$$

**Final hidden state**:
$$\mathbf{h}_4 = \mathbf{[0.38, 0.36, 0.24]}$$

*Can you see the pattern?* The first dimension (electronics) bounced back to 0.38! The coffee at step 3 didn't destroy the electronics context.

---

### Summary: What the GRU Learned

| Step | Item | h[0] (Electronics) | h[2] (Food) | Interpretation |
|------|------|-------------------|-------------|----------------|
| 1 | Laptop | 0.23 | 0.15 | Electronics established |
| 2 | Mouse | 0.35 | 0.22 | Electronics strengthened |
| 3 | Coffee | 0.28 | 0.35 | Some coffee influence, but electronics preserved! |
| 4 | Keyboard | **0.38** | 0.24 | Electronics recovered and dominates |

**Final prediction**: The hidden state $\mathbf{h}_4 = [0.38, 0.36, 0.24]$ will predict electronics accessories (high first dimension), not coffee-related items.

**The Markov chain would have**: Only seen "Keyboard" and missed the laptop-mouse-keyboard pattern.

---

## The Shopping Example: Gate Behavior in Action

*Let me make this concrete with a shopping example.*

**Scenario**: User's browsing history on Amazon:
```
[laptop] -> [laptop_bag] -> [mouse] -> [coffee_mug] -> [keyboard]
```

**At step 5** (after seeing "keyboard"):

**Reset gate $\mathbf{r}_t$**: "Should I forget some of the past?"

- If $r_t \approx 0$: "Forget everything, start fresh"
  - *Use case*: User was browsing electronics, now switched to kitchen items
  - The old "electronics" context is no longer relevant

- If $r_t \approx 1$: "Remember everything"
  - *Use case*: User is still in the same shopping session/intent
  - Past context is still valuable

**Update gate $\mathbf{z}_t$**: "How much should the new item change my memory?"

- If $z_t \approx 0$: "Ignore this item, keep my old state"
  - *Use case*: The coffee mug was a random browse, not representative

- If $z_t \approx 1$: "This item is important, update heavily"
  - *Use case*: The keyboard is clearly part of the electronics shopping intent

**The update equation explained**:

$$\mathbf{h}_t = \underbrace{(1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1}}_{\text{keep from past}} + \underbrace{\mathbf{z}_t \odot \tilde{\mathbf{h}}_t}_{\text{add from new}}$$

It's a **weighted average** between:
- The old memory ($\mathbf{h}_{t-1}$)
- The new candidate memory ($\tilde{\mathbf{h}}_t$)

**Trade-off**: GRU is faster (fewer params), LSTM may be more accurate for very long sequences.

**Recommendation systems**: GRU is more popular (GRU4Rec).

---

## Session-Based vs. Sequence-Based Recommendation

*When should you use which approach?*

### Session-Based Recommendation

**Definition**: Model short-term user sessions without persistent user IDs.

**Characteristics**:
- Session = single browsing session (30 min - 2 hours)
- No user history across sessions
- Cold-start for every session
- Typically 5-50 items per session

**Use Cases**:
- Anonymous e-commerce browsing
- News websites
- Travel booking sites
- First-time visitors

**Model Choice**: GRU4Rec (original paper specifically for sessions)

```python
# Session-based: Each session starts fresh
session_1 = [item_a, item_b, item_c]  # User 1 morning
session_2 = [item_x, item_y]          # User 1 afternoon (treated as NEW user!)
```

### Sequence-Based Recommendation

**Definition**: Model long-term user behavior with persistent user IDs.

**Characteristics**:
- Sequence = user's entire history (weeks/months)
- User ID links sessions together
- Rich historical context
- Hundreds to thousands of items

**Use Cases**:
- Logged-in users on Netflix/Spotify
- Amazon purchase history
- Long-term content platforms

**Model Choice**: LSTM or Transformer (better for very long sequences)

```python
# Sequence-based: Full user history
user_1_history = [item_jan1, item_jan2, ..., item_mar15, item_mar16]  # Months of data
```

### Decision Framework

| Factor | Session-Based | Sequence-Based |
|--------|--------------|----------------|
| **User ID available?** | No | Yes |
| **History length** | 5-50 items | 100-10,000 items |
| **Time span** | Minutes to hours | Days to months |
| **Cold-start** | Every session | Only new users |
| **Model** | GRU4Rec | Hierarchical RNN, Transformers |
| **Memory requirement** | Low | High |

*What happens if* you use session-based on long histories? You lose valuable long-term patterns.

*What happens if* you use sequence-based on short sessions? Overkill, and you might overfit.

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
         |
    Item Embeddings
         |
    GRU Layers (stacked)
         |
    Hidden State ht
         |
    Output Layer (scores for all items)
         |
    Softmax
         |
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
  - laptop, laptop_bag, mouse, charger -> highly relevant (computer accessories)
  - water_bottle -> less relevant (unrelated)
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

### Attention Weights Example: What the Model "Looks At"

*Let me show you what learned attention looks like.*

**User's session**: Browsing electronics on Amazon

```
Position 1: [MacBook Pro]
Position 2: [USB-C Hub]
Position 3: [Coffee Mug]      <- random browse
Position 4: [Laptop Stand]
Position 5: [Water Bottle]    <- random browse
Position 6: [Keyboard]
```

**Predicting next item after position 6**:

The attention mechanism computes weights:

| Position | Item | Attention Weight | Interpretation |
|----------|------|------------------|----------------|
| 1 | MacBook Pro | **0.35** | Very relevant (main item) |
| 2 | USB-C Hub | **0.20** | Relevant (accessory) |
| 3 | Coffee Mug | 0.05 | Ignored (unrelated) |
| 4 | Laptop Stand | **0.22** | Relevant (accessory) |
| 5 | Water Bottle | 0.03 | Ignored (unrelated) |
| 6 | Keyboard | **0.15** | Relevant (just viewed) |

**Visualization**:
```
MacBook   USB-C    Mug    Stand   Water   Keyboard
 Pro      Hub                     Bottle
  |        |        |       |       |        |
  v        v        v       v       v        v
[0.35]   [0.20]   [0.05] [0.22]  [0.03]   [0.15]
  ----     ---       -     ---      .       --
```

**What the model learned**: Focus on electronics, ignore random items.

**The prediction**: High probability for items like "Monitor", "Mouse", "USB-C Cable".

*Notice how* the model automatically discovered that coffee mugs and water bottles are noise, even though we never told it about product categories!

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

## What Can Go Wrong: Failure Modes and Solutions

*Every model has failure modes. Let me show you the three most common problems with RNN-based recommenders and how to fix them.*

### Failure Mode 1: Vanishing Gradients (Still!)

**Symptom**: Model performs well on recent items (positions T-3 to T) but ignores early items (positions 1-5).

**Diagnosis**: Check if recommendations change when you modify early items.

```python
# Test: Does changing position 1 affect prediction?
seq_original = [laptop, mouse, keyboard, coffee, snacks]
seq_modified = [dress, mouse, keyboard, coffee, snacks]  # Changed position 1

pred_original = model(seq_original)
pred_modified = model(seq_modified)

similarity = cosine_similarity(pred_original, pred_modified)
if similarity > 0.95:
    print("WARNING: Model ignoring early positions!")
```

**The Numbers**:

| Sequence Length | Gradient at Position 1 | Can Learn Long-Range? |
|----------------|------------------------|----------------------|
| 10 items | 0.1 | Marginal |
| 20 items | 0.01 | Barely |
| 50 items | 0.0001 | No |
| 100 items | 0.00000001 | Definitely not |

**Solutions**:

1. **Use gradient clipping**:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
   ```

2. **Add skip connections** (residual learning):
   ```python
   h_new = gru_output + input_embedding  # Skip connection
   ```

3. **Use attention mechanism** (focus on relevant early items)

4. **Switch to Transformers** for sequences > 50 items

---

### Failure Mode 2: Long Sequence Performance Degradation

**Symptom**: Model accuracy DECREASES as sequence length increases.

**Diagnosis**: Plot Hit@10 vs sequence length.

```
Hit@10
  |
0.6|    *
0.5|      *    *
0.4|              *    *
0.3|                      *    *
0.2|________________________________
   0   10   20   30   40   50   60
              Sequence Length
```

*What happens if* the curve slopes downward? Your model can't handle long sequences!

**Causes**:
1. Vanishing gradients (see above)
2. Hidden state "saturation" (all values near +1 or -1)
3. Catastrophic forgetting of early context

**The Hidden State Saturation Problem**:

After many steps, hidden states can become saturated:
```
Step 1:  h = [0.2, 0.3, 0.1]      <- Good range
Step 10: h = [0.7, 0.8, 0.6]      <- Getting compressed
Step 50: h = [0.99, 0.98, 0.99]   <- Saturated! All look the same!
```

**Solutions**:

1. **Truncate sequences**: Use only last 50-100 items
   ```python
   max_seq_len = 50
   truncated_seq = seq[-max_seq_len:]  # Keep most recent
   ```

2. **Hierarchical RNN**: Encode sessions, then encode session sequence
   ```
   User history: [Session1] -> [Session2] -> [Session3]
                     |             |             |
                   GRU_1         GRU_1         GRU_1   <- Session encoder
                     |             |             |
                     v             v             v
                 [h_s1]  --->  [h_s2]  --->  [h_s3]   <- User encoder (GRU_2)
   ```

3. **Use Layer Normalization**:
   ```python
   self.layer_norm = nn.LayerNorm(hidden_dim)
   h = self.layer_norm(gru_output)  # Prevents saturation
   ```

---

### Failure Mode 3: Cold-Start Sessions

**Symptom**: Terrible recommendations for sessions with < 3 items.

**Diagnosis**: Check Hit@10 by session length.

| Session Length | Hit@10 | Problem? |
|---------------|--------|----------|
| 1 item | 0.12 | Severe cold-start |
| 2 items | 0.25 | Cold-start |
| 3 items | 0.38 | Warming up |
| 5+ items | 0.52 | Normal |

**The Math**: With only 1 item, the hidden state is:
$$\mathbf{h}_1 = \text{GRU}(\mathbf{0}, \mathbf{e}_1) \approx \mathbf{z}_1 \odot \tanh(\mathbf{W} \mathbf{e}_1)$$

This is essentially just a transformed item embedding - almost no sequential learning!

**Solutions**:

1. **Fallback to popularity**:
   ```python
   if len(session) < 3:
       return popular_items  # Simple baseline
   else:
       return rnn_prediction
   ```

2. **Side information**:
   ```python
   # Use item features for cold-start
   cold_start_embedding = item_embedding + category_embedding + brand_embedding
   ```

3. **Pre-training on item co-occurrence**:
   ```python
   # Pre-train embeddings using item2vec
   item_embedding.weight = pretrained_item2vec_weights
   ```

4. **Hybrid with content-based**:
   ```python
   if len(session) < 3:
       # Use item content similarity
       similar_items = content_based_neighbors(session[-1])
   ```

---

### Failure Mode 4: Repetitive Recommendations

**Symptom**: Model keeps recommending items user already interacted with.

**Example**:
```
User history: [laptop, mouse, keyboard]
Model output: [laptop, mouse, laptop, keyboard, mouse]  <- All repeats!
```

**Cause**: The model learned that "if you liked X, you'll buy X again" which is often true for consumables but wrong for durables.

**Solutions**:

1. **Explicit filtering** (simple but effective):
   ```python
   scores[already_interacted_items] = -float('inf')
   ```

2. **Learn repeat behavior** (more sophisticated):
   ```python
   # Add "time since last interaction" feature
   repeat_probability = sigmoid(time_decay * time_since_interaction)
   adjusted_score = score * (1 - repeat_probability)
   ```

3. **Diversity penalty**:
   ```python
   # MMR-style re-ranking
   for i in range(k):
       best_item = argmax(score - lambda * similarity_to_already_recommended)
       recommended.append(best_item)
   ```

---

### Summary: Failure Mode Checklist

Before deploying your RNN recommender, check:

| Failure Mode | Symptom | Quick Test | Solution |
|-------------|---------|------------|----------|
| Vanishing gradients | Ignores early items | Change item 1, check if prediction changes | Attention, Transformers |
| Long sequence degradation | Accuracy drops with length | Plot Hit@10 vs length | Truncation, Hierarchical RNN |
| Cold-start sessions | Bad for short sessions | Check Hit@10 for length=1,2,3 | Fallback, side information |
| Repetitive recs | Recommends same items | Check diversity in top-10 | Filtering, diversity penalty |

---

## Comparison: Markov vs. RNN vs. Transformer

| Aspect | Markov Chain | RNN (GRU/LSTM) | Transformer |
|--------|--------------|----------------|-------------|
| **Memory** | Last 1-2 items | Full sequence | Full sequence |
| **Computation** | O(1) per step | O(T) sequential | O(T^2) parallel |
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
