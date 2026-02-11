# Week 10: Context-Aware Recommendation

## The Opening Question

*"Why does the same user want completely different things at different times?"*

Let me show you something that will challenge everything we've learned about collaborative filtering so far.

---

## The Failure That Started It All

### The Same User, Two Different Moments

Consider **User Alice** on a streaming platform. Here's her behavior data:

| Time | Day | What Alice Wants |
|------|-----|------------------|
| 7:00 AM | Monday | News briefing, coffee playlist, quick podcasts |
| 10:00 PM | Friday | Action movies, party music, comedy specials |
| 2:00 PM | Sunday | Cooking shows, relaxing jazz, family content |

**Now here's the problem**: Our beautiful collaborative filtering model learned:

$$\hat{r}_{Alice, item} = \mathbf{u}_{Alice}^T \mathbf{v}_{item}$$

This gives Alice **the exact same recommendations** whether it's Monday morning or Friday night!

### Let's See the Damage Numerically

**Setup**: Alice has rated items. Our CF model learned her embedding.

| Item | Alice's CF Score | Best Time | Wrong Time |
|------|------------------|-----------|------------|
| Morning News | 0.85 | 7 AM | 10 PM |
| Action Movie | 0.82 | 10 PM | 7 AM |
| Cooking Show | 0.78 | 2 PM Sun | 7 AM Mon |

**What CF Recommends** (any time): Morning News, Action Movie, Cooking Show (by score)

**What Actually Happens**:

| Context | CF Recommendation | User Actually Wants | Match? |
|---------|-------------------|---------------------|--------|
| 7 AM Monday | Morning News | Morning News | Yes |
| 10 PM Friday | Morning News | Action Movie | **NO** |
| 2 PM Sunday | Morning News | Cooking Show | **NO** |
| 7 AM Saturday | Morning News | Sleep playlist | **NO** |
| 6 PM Weekday | Morning News | Workout playlist | **NO** |

**Accuracy**: 1/5 = **20%** of recommendations are contextually appropriate!

In a real system with diverse contexts, ignoring context gives approximately **40% wrong recommendations** - items the user likes in general, but not right now.

### The Core Insight

> **Static user preferences don't exist.** What exists is: *User U likes Item I in Context C.*

This requires a fundamental shift in how we model recommendations.

---

## Learning Objectives

By the end of this lecture, you will:
- Understand why context creates a third dimension beyond user-item
- Derive tensor factorization from first principles
- Build and factor a 3D tensor step-by-step
- Master cyclical time encoding (and see what breaks without it)
- Implement Factorization Machines for sparse context features
- Recognize the pitfalls of context-aware systems

---

## Contextual Dimensions

Before we dive into math, let's catalog what "context" means.

### Temporal Context

**Time of day**:
- Morning (6am-12pm): News, podcasts, breakfast recipes
- Afternoon (12pm-6pm): Work playlists, productivity tools
- Evening (6pm-12am): Movies, dinner recipes, relaxation
- Night (12am-6am): Sleep aids, ambient music

**Day of week**:
- Weekdays: Professional content, quick meals
- Weekends: Entertainment, elaborate recipes, travel

**Seasonality**:
- Summer: Travel, outdoor activities, light reading
- Winter: Indoor entertainment, comfort food, holiday shopping

### Spatial Context

**Location**: Home, work, airport, restaurant, gym.

**Weather**: Sunny leads to outdoor activities; Rainy leads to indoor entertainment.

**Nearby POIs**: Near restaurant suggests food recommendations; Near mall suggests shopping.

**Example** (Netflix):
- Home (evening): Long movies (2+ hours)
- Commute (mobile): Short episodes (20-30 min)
- Hotel (travel): Binge-watching series

### Device Context

**Device type**: Mobile, desktop, tablet, TV, smart speaker.

**Screen size**: Affects content format (short-form vs. long-form).

**Input method**: Touch vs. keyboard affects UI preferences.

**Example** (YouTube):
- Mobile: Shorts (vertical, <60s)
- Desktop: Long videos (horizontal, tutorials)
- TV: High-quality, family-friendly

### Social Context

**Companions**: Alone, with partner, with family, with friends.

**Example** (Spotify):
- Alone: Personal favorites
- With partner: Romantic playlists
- Party: Upbeat, popular hits

---

## Tensor Factorization: The Mathematical Foundation

### From 2D to 3D: Why Matrices Aren't Enough

**Matrix factorization** gave us:

$$R \approx UV^T$$

where $R$ is users $\times$ items.

**Problem**: Where does context go? We need a **third dimension**.

**Solution**: Extend to a 3D tensor:

$$\mathcal{R} \in \mathbb{R}^{|U| \times |I| \times |C|}$$

where:
- $|U|$ = number of users
- $|I|$ = number of items
- $|C|$ = number of context types (time periods, locations, etc.)

### Step-by-Step Derivation: From Matrix to Tensor Factorization

Let me walk you through the mathematical progression.

**Step 1: Matrix Factorization Review**

For a user-item matrix $R$:

$$\hat{r}_{ui} = \sum_{k=1}^{K} U_{uk} \cdot V_{ik} = \mathbf{u}_u^T \mathbf{v}_i$$

This is a sum over $K$ latent factors. Each factor captures one "aspect" (genre preference, quality preference, etc.).

**Step 2: Adding Context as a Third Factor**

Now, we want each prediction to also depend on context $c$. The natural extension:

$$\hat{r}_{uic} = \sum_{k=1}^{K} U_{uk} \cdot V_{ik} \cdot W_{ck}$$

This is called **CP Decomposition** (CANDECOMP/PARAFAC).

**Step 3: Understanding What This Means**

$$\hat{r}_{uic} = \sum_{k=1}^{K} \underbrace{U_{uk}}_{\text{user } u \text{'s weight on factor } k} \cdot \underbrace{V_{ik}}_{\text{item } i \text{'s weight on factor } k} \cdot \underbrace{W_{ck}}_{\text{context } c \text{'s weight on factor } k}$$

**Interpretation**:
- Factor 1 might capture "action content preference"
- User Alice has $U_{Alice,1} = 0.8$ (likes action)
- Item "Die Hard" has $V_{DieHard,1} = 0.9$ (is action)
- Context "evening" has $W_{evening,1} = 0.7$ (action fits evening)
- Context "morning" has $W_{morning,1} = 0.2$ (action doesn't fit morning)

The product $0.8 \times 0.9 \times 0.7 = 0.504$ (evening) vs $0.8 \times 0.9 \times 0.2 = 0.144$ (morning)

**Same user, same item, different context = different prediction!**

### CP vs Tucker Decomposition

**CP Decomposition** (simpler):

$$\mathcal{R} \approx \sum_{k=1}^K \mathbf{u}_k \circ \mathbf{v}_k \circ \mathbf{w}_k$$

where $\circ$ = outer product.

**Tucker Decomposition** (more expressive):

$$\mathcal{R} \approx \mathcal{G} \times_1 U \times_2 V \times_3 W$$

where:
- $U \in \mathbb{R}^{|U| \times r_u}$ (user factors)
- $V \in \mathbb{R}^{|I| \times r_i}$ (item factors)
- $W \in \mathbb{R}^{|C| \times r_c}$ (context factors)
- $\mathcal{G} \in \mathbb{R}^{r_u \times r_i \times r_c}$ (core tensor - allows different rank per dimension)

**Prediction**:
$$\hat{r}_{uic} = \sum_{p,q,r} \mathcal{G}_{pqr} \cdot U_{up} \cdot V_{iq} \cdot W_{cr}$$

**Trade-off**: Tucker is more expressive but has more parameters. CP assumes the same number of factors for all dimensions.

---

## Complete Numerical Walkthrough: Building a 3D Tensor

Let's build this from scratch with concrete numbers.

### Setup

**2 Users**: Alice (U1), Bob (U2)

**4 Items**: News (I1), Action Movie (I2), Cooking Show (I3), Workout Video (I4)

**3 Contexts**: Morning (C1), Afternoon (C2), Evening (C3)

### Step 1: Collect Ratings (Observed Data)

| User | Item | Context | Rating |
|------|------|---------|--------|
| Alice | News | Morning | 5 |
| Alice | News | Evening | 2 |
| Alice | Action Movie | Evening | 5 |
| Alice | Action Movie | Morning | 1 |
| Alice | Cooking Show | Afternoon | 4 |
| Bob | Workout Video | Morning | 5 |
| Bob | Workout Video | Evening | 2 |
| Bob | News | Morning | 4 |
| Bob | Action Movie | Evening | 4 |

### Step 2: Organize as 3D Tensor

**Tensor $\mathcal{R}$** with shape (2 users $\times$ 4 items $\times$ 3 contexts):

**Slice for Morning (C1)**:
```
         News  Action  Cooking  Workout
Alice [   5      1       ?        ?    ]
Bob   [   4      ?       ?        5    ]
```

**Slice for Afternoon (C2)**:
```
         News  Action  Cooking  Workout
Alice [   ?      ?       4        ?    ]
Bob   [   ?      ?       ?        ?    ]
```

**Slice for Evening (C3)**:
```
         News  Action  Cooking  Workout
Alice [   2      5       ?        ?    ]
Bob   [   ?      4       ?        2    ]
```

### Step 3: Factor the Tensor (CP Decomposition, K=2 factors)

We want to find:
- $U \in \mathbb{R}^{2 \times 2}$ (2 users, 2 factors)
- $V \in \mathbb{R}^{4 \times 2}$ (4 items, 2 factors)
- $W \in \mathbb{R}^{3 \times 2}$ (3 contexts, 2 factors)

**After optimization** (I'll give you learned values):

**User Matrix U**:
```
         Factor1  Factor2
Alice [   0.9      0.3   ]
Bob   [   0.4      0.8   ]
```

**Item Matrix V**:
```
              Factor1  Factor2
News      [   0.8      0.2   ]
Action    [   0.7      0.1   ]
Cooking   [   0.3      0.6   ]
Workout   [   0.2      0.9   ]
```

**Context Matrix W**:
```
              Factor1  Factor2
Morning   [   0.6      0.7   ]
Afternoon [   0.4      0.5   ]
Evening   [   0.9      0.2   ]
```

### Step 4: Make Predictions

**Prediction formula**:
$$\hat{r}_{uic} = \sum_{k=1}^{2} U_{uk} \cdot V_{ik} \cdot W_{ck}$$

**Example 1: Alice + Action Movie + Morning**

$$\hat{r} = (0.9 \times 0.7 \times 0.6) + (0.3 \times 0.1 \times 0.7)$$
$$\hat{r} = 0.378 + 0.021 = 0.399$$

Scale to 1-5 rating: $0.399 \times 5 \approx 2.0$ (low - matches our intuition!)

**Example 2: Alice + Action Movie + Evening**

$$\hat{r} = (0.9 \times 0.7 \times 0.9) + (0.3 \times 0.1 \times 0.2)$$
$$\hat{r} = 0.567 + 0.006 = 0.573$$

Scale to 1-5 rating: $0.573 \times 5 \approx 2.9$ (higher than morning!)

**Example 3: Bob + Workout Video + Morning**

$$\hat{r} = (0.4 \times 0.2 \times 0.6) + (0.8 \times 0.9 \times 0.7)$$
$$\hat{r} = 0.048 + 0.504 = 0.552$$

Scale: $\approx 2.8$

**Example 4: Bob + Workout Video + Evening**

$$\hat{r} = (0.4 \times 0.2 \times 0.9) + (0.8 \times 0.9 \times 0.2)$$
$$\hat{r} = 0.072 + 0.144 = 0.216$$

Scale: $\approx 1.1$ (much lower - workout doesn't fit evening for Bob!)

### Step 5: Verify Context Matters

| Prediction | Without Context (2D MF) | With Context (3D Tensor) |
|------------|------------------------|--------------------------|
| Alice + Action (Morning) | 3.5 | 2.0 |
| Alice + Action (Evening) | 3.5 | 2.9 |
| Bob + Workout (Morning) | 3.0 | 2.8 |
| Bob + Workout (Evening) | 3.0 | 1.1 |

**The 2D model gives the same prediction regardless of context. The 3D model adapts!**

---

## Time Encoding: Why Sin/Cos for Cyclical Features

### The Problem with Linear Time

Suppose we encode hour as a number 0-23.

**Question**: How similar is hour 23 to hour 0?

With linear encoding:
- Distance(23, 0) = |23 - 0| = 23 (maximum distance!)

But in reality, 11 PM and midnight are adjacent! They should be similar.

### The Sin/Cos Solution

**Encode hour using circular coordinates**:

$$hour_{sin} = \sin\left(\frac{2\pi \times hour}{24}\right)$$
$$hour_{cos} = \cos\left(\frac{2\pi \times hour}{24}\right)$$

**Let's compute**:

| Hour | Linear | sin | cos |
|------|--------|-----|-----|
| 0 | 0 | 0.00 | 1.00 |
| 6 | 6 | 1.00 | 0.00 |
| 12 | 12 | 0.00 | -1.00 |
| 18 | 18 | -1.00 | 0.00 |
| 23 | 23 | -0.26 | 0.97 |

**Now check distance(23, 0)**:

Linear: $|23 - 0| = 23$

Sin/Cos: $\sqrt{(-0.26 - 0)^2 + (0.97 - 1)^2} = \sqrt{0.068 + 0.001} = 0.26$

**Hour 23 and Hour 0 are now close!**

### What Breaks Without Sin/Cos

**Experiment**: Train model to predict "morning content preference"

| Hour | Target | Linear Prediction | Sin/Cos Prediction |
|------|--------|-------------------|-------------------|
| 6 | High | High | High |
| 7 | High | High | High |
| 23 | Low | **High** (close to 24) | Low |
| 0 | Low | Low | Low |

The linear model thinks hour 23 should have morning preferences because 23 is "close to 24 which wraps to 0"!

### Implementation

```python
import numpy as np
from datetime import datetime

def extract_temporal_features(timestamp):
    """
    Extract temporal features from timestamp with proper cyclical encoding.
    """
    dt = datetime.fromtimestamp(timestamp)

    # Hour of day (0-23)
    hour = dt.hour

    # Cyclical encoding (sin/cos to capture periodicity)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # Day of week (0=Monday, 6=Sunday)
    day_of_week = dt.weekday()
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    is_weekend = 1 if day_of_week >= 5 else 0

    # Month (for seasonality)
    month = dt.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    return {
        'hour': hour,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_of_week': day_of_week,
        'dow_sin': dow_sin,
        'dow_cos': dow_cos,
        'is_weekend': is_weekend,
        'month': month,
        'month_sin': month_sin,
        'month_cos': month_cos
    }


# Example
timestamp = 1704067200  # 2024-01-01 00:00:00 UTC
features = extract_temporal_features(timestamp)
print(f"Temporal features: {features}")
```

---

## Factorization Machines: Handling Sparse Context Features

### The Challenge

Tensor factorization assumes discrete contexts (morning/afternoon/evening).

**But real context is often**:
- Continuous: Temperature = 72.3F
- High-cardinality: Location = one of 50,000 zip codes
- Sparse: Most user-item-context combinations never observed

We need something more flexible.

### Factorization Machines Derivation

**Goal**: Model **all pairwise interactions** between features efficiently.

**Input**: Feature vector $\mathbf{x} = [x_1, x_2, ..., x_n]$

This could be: [is_user_123, is_item_456, is_morning, is_mobile, temperature, ...]

**Naive approach**: Model all pairs explicitly.

$$\hat{y} = w_0 + \sum_i w_i x_i + \sum_i \sum_{j>i} w_{ij} x_i x_j$$

**Problem**: $O(n^2)$ parameters for pairwise weights $w_{ij}$. With 10,000 features, that's 50 million parameters!

**FM insight**: Factor the interaction weights.

Instead of learning $w_{ij}$ directly, assume:

$$w_{ij} = \langle \mathbf{v}_i, \mathbf{v}_j \rangle = \sum_{f=1}^k v_{if} \cdot v_{jf}$$

where $\mathbf{v}_i \in \mathbb{R}^k$ is a $k$-dimensional embedding for feature $i$.

**FM Model**:

$$\hat{y} = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

**Parameters**: $O(nk)$ instead of $O(n^2)$ - linear in features!

### The Computational Trick

Computing all pairs is still $O(n^2)$ at inference. But there's a beautiful identity:

$$\sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2} \left[ \left(\sum_{i=1}^n x_i \mathbf{v}_i\right)^2 - \sum_{i=1}^n x_i^2 \mathbf{v}_i^2 \right]$$

This can be computed in $O(nk)$ time!

**Proof sketch**:
$$\left(\sum_i x_i \mathbf{v}_i\right)^2 = \sum_i \sum_j x_i x_j \langle \mathbf{v}_i, \mathbf{v}_j \rangle$$

This includes $i=j$ terms (self-interactions), so we subtract them.

### Example: FM for Context-Aware Recommendations

**Feature vector for "User 5 watches Item 100 on Monday morning via mobile"**:

```
x = [0, 0, 0, 0, 0, 1, 0, ...]   # One-hot user ID (position 5)
  + [0, ..., 0, 1, 0, ...]       # One-hot item ID (position 100)
  + [1, 0, 0, 0, 0, 0, 0]        # Day of week (Monday)
  + [1, 0, 0, 0]                 # Time bucket (morning)
  + [0, 1]                       # Device (mobile)
```

**Interactions FM captures**:
- User 5 $\times$ Item 100 (collaborative signal)
- User 5 $\times$ Monday (user's Monday preference)
- User 5 $\times$ morning (user's morning preference)
- Item 100 $\times$ mobile (item's mobile suitability)
- morning $\times$ mobile (morning-mobile interaction)

All learned through embedding dot products!

### Implementation

```python
import torch
import torch.nn as nn

class FactorizationMachine(nn.Module):
    def __init__(self, num_features, embedding_dim=10):
        """
        Factorization Machine for context-aware recommendations.

        num_features: Total number of features (user + item + context)
        embedding_dim: Dimension of factor vectors (k)
        """
        super().__init__()

        # Linear weights w_i
        self.linear = nn.Embedding(num_features, 1)

        # Pairwise interaction embeddings v_i
        self.embeddings = nn.Embedding(num_features, embedding_dim)

        # Global bias w_0
        self.bias = nn.Parameter(torch.zeros(1))

        # Initialize
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.normal_(self.embeddings.weight, std=0.01)

    def forward(self, x):
        """
        x: (batch, num_active_features) - indices of active features

        For sparse features, x contains indices where x_i = 1
        """
        # Linear term: sum of w_i for active features
        linear_part = self.linear(x).sum(dim=1).squeeze()  # (batch,)

        # Pairwise interactions using the O(nk) trick
        emb = self.embeddings(x)  # (batch, num_active_features, dim)

        # sum_of_embeddings = sum_i (x_i * v_i)
        sum_of_emb = emb.sum(dim=1)  # (batch, dim)

        # sum_of_squares = sum_i (x_i^2 * v_i^2) = sum_i v_i^2 for binary features
        sum_of_square = (emb ** 2).sum(dim=1)  # (batch, dim)

        # FM interaction: 0.5 * (square_of_sum - sum_of_squares)
        interaction = 0.5 * ((sum_of_emb ** 2) - sum_of_square).sum(dim=1)  # (batch,)

        # Final prediction
        output = self.bias + linear_part + interaction

        return output


# Example usage
num_features = 10000  # user IDs + item IDs + contexts
model = FactorizationMachine(num_features, embedding_dim=16)

# Sample: user 5, item 100, context "morning" (ID 5000), device "mobile" (ID 6000)
x = torch.tensor([[5, 100, 5000, 6000]])  # (1, 4)

prediction = model(x)
print(f"Prediction: {prediction.item():.4f}")
```

---

## CP Tensor Factorization Implementation

```python
import torch
import torch.nn as nn

class CPTensorFactorization(nn.Module):
    def __init__(self, num_users, num_items, num_contexts, rank=20):
        """
        CP tensor factorization for context-aware recommendations.

        rank: Number of latent factors (K)
        """
        super().__init__()

        self.user_factors = nn.Embedding(num_users, rank)
        self.item_factors = nn.Embedding(num_items, rank)
        self.context_factors = nn.Embedding(num_contexts, rank)

        # Initialize
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
        nn.init.normal_(self.context_factors.weight, std=0.01)

    def forward(self, user_ids, item_ids, context_ids):
        """
        Predict ratings for (user, item, context) triples.

        user_ids: (batch,)
        item_ids: (batch,)
        context_ids: (batch,)
        """
        # Get factors
        u = self.user_factors(user_ids)  # (batch, rank)
        v = self.item_factors(item_ids)  # (batch, rank)
        w = self.context_factors(context_ids)  # (batch, rank)

        # CP decomposition: sum of element-wise products
        prediction = (u * v * w).sum(dim=1)  # (batch,)

        return prediction


# Example
num_users = 1000
num_items = 5000
num_contexts = 10  # e.g., 10 time slots

model = CPTensorFactorization(num_users, num_items, num_contexts, rank=20)

# Sample batch
batch_size = 64
user_ids = torch.randint(0, num_users, (batch_size,))
item_ids = torch.randint(0, num_items, (batch_size,))
context_ids = torch.randint(0, num_contexts, (batch_size,))

# Forward
predictions = model(user_ids, item_ids, context_ids)
print(f"Predictions shape: {predictions.shape}")  # (64,)
```

---

## Context-Aware Matrix Factorization Approaches

### Approach 1: Contextual Pre-filtering

**Idea**: Filter data by context first, then apply standard MF.

**Steps**:
1. Select interactions matching target context (e.g., weekend, evening)
2. Train MF on filtered data
3. Use model for that context only

**Pros**: Simple, interpretable.
**Cons**: Data sparsity (each context has less data), need separate model per context.

### Approach 2: Contextual Post-filtering

**Idea**: Train MF on all data, then adjust predictions based on context.

**Steps**:
1. Train standard MF: $\hat{r}_{ui} = \mathbf{u}^T \mathbf{v}$
2. Learn context adjustment: $\hat{r}_{uic} = \hat{r}_{ui} + \Delta_c$
3. $\Delta_c$ = context-specific bias

**Pros**: Use all data, simple adjustment.
**Cons**: Context only affects bias, not latent features.

### Approach 3: Contextual Modeling (Integrated)

**Idea**: Incorporate context directly into MF (Factorization Machines approach).

This is the most principled approach - context affects the latent feature interactions.

---

## Time-Aware Recommendations

### Temporal Dynamics

**User preferences drift**: User who liked action movies 5 years ago may now prefer documentaries.

**Item popularity changes**: Viral trends, seasonal products.

**Recency bias**: Recent interactions more important than old ones.

### Time-Decayed Weighting

**Idea**: Give more weight to recent interactions.

**Exponential decay**:
$$w(t) = e^{-\lambda (t_{\text{now}} - t)}$$

where $\lambda$ = decay rate (e.g., 0.01 per day).

**Implementation**:
```python
import numpy as np

def time_decay_weights(timestamps, decay_rate=0.01):
    """
    Compute time decay weights for interactions.

    timestamps: Array of Unix timestamps
    decay_rate: Decay per day
    """
    now = np.max(timestamps)
    days_ago = (now - timestamps) / 86400  # Convert seconds to days

    weights = np.exp(-decay_rate * days_ago)

    return weights


# Example
timestamps = np.array([
    1704067200,  # 30 days ago
    1706659200,  # Today
    1705363200,  # 15 days ago
])

weights = time_decay_weights(timestamps, decay_rate=0.05)
print(f"Time decay weights: {weights}")
# Recent interactions have higher weight
```

### TimeSVD++

**Extension of SVD++**: Add temporal biases.

**Model**:
$$\hat{r}_{ui}(t) = \mu + b_u(t) + b_i(t) + \mathbf{u}^T \mathbf{v}_i$$

where:
- $b_u(t) = b_u + \alpha_u \cdot \text{dev}_u(t)$ (user bias changes over time)
- $b_i(t) = b_i + \alpha_i \cdot \text{dev}_i(t)$ (item bias changes over time)

**dev$(t)$**: Deviation from average time (e.g., days since first interaction).

---

## Location-Aware Recommendations

### Spatial Influence

**Nearby items**: Restaurants, stores, events near user's location.

**Distance decay**: Farther items less likely to be relevant.

**Example** (Yelp):
- User in San Francisco: SF restaurants
- Traveling to NYC: NYC restaurants

### Geo-Spatial Factorization

**Idea**: Incorporate location into user/item embeddings.

**Model**:
$$\text{score}(u, i) = \mathbf{u}^T \mathbf{v}_i \cdot \exp(-\lambda \cdot d(u, i))$$

where:
- $d(u, i)$ = distance between user and item (km)
- $\lambda$ = distance decay rate

**Implementation**:
```python
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute distance between two lat/lon points (km).
    """
    from math import radians, sin, cos, sqrt, atan2

    R = 6371  # Earth radius in km

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    distance = R * c
    return distance


def location_aware_score(user_emb, item_emb, user_loc, item_loc, decay=0.1):
    """
    Score with distance decay.

    user_loc, item_loc: (lat, lon) tuples
    decay: Distance decay rate
    """
    # Base score (dot product)
    base_score = np.dot(user_emb, item_emb)

    # Distance
    distance = haversine_distance(user_loc[0], user_loc[1], item_loc[0], item_loc[1])

    # Distance penalty
    location_weight = np.exp(-decay * distance)

    final_score = base_score * location_weight

    return final_score


# Example
user_emb = np.array([0.5, 0.8, 0.3])
item_emb = np.array([0.6, 0.7, 0.4])
user_loc = (37.7749, -122.4194)  # San Francisco
item_loc = (37.7849, -122.4094)  # 1.5 km away

score = location_aware_score(user_emb, item_emb, user_loc, item_loc, decay=0.1)
print(f"Location-aware score: {score:.4f}")
```

---

## Neural Context-Aware Models

### Deep Context-Aware Network

**Architecture**:
```
User Features -----+
                   |
Item Features -----+--> Concatenate --> MLP --> Prediction
                   |
Context Features --+
```

**Implementation**:
```python
class DeepContextAwareModel(nn.Module):
    def __init__(self, num_users, num_items, num_contexts, embedding_dim=64, hidden_dims=[128, 64]):
        super().__init__()

        # Embeddings
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        # Context features (assume continuous)
        context_dim = 10  # e.g., time features, location features

        # MLP
        input_dim = embedding_dim * 2 + context_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids, context_features):
        """
        user_ids: (batch,)
        item_ids: (batch,)
        context_features: (batch, context_dim)
        """
        # Embeddings
        u = self.user_emb(user_ids)  # (batch, dim)
        v = self.item_emb(item_ids)  # (batch, dim)

        # Concatenate
        x = torch.cat([u, v, context_features], dim=1)

        # MLP
        output = self.mlp(x).squeeze()

        return output
```

---

## What Can Go Wrong: Context-Aware Pitfalls

### Pitfall 1: Context Explosion

**Problem**: Too many context combinations leads to extreme sparsity.

**Example**:
- 24 hours $\times$ 7 days $\times$ 4 seasons $\times$ 5 devices $\times$ 10 locations = 33,600 contexts

Most user-item pairs never observed in most contexts!

**Solutions**:
- Bucket contexts: 24 hours becomes 4 time-of-day buckets becomes 24 hours  4 buckets
- Use FM instead of tensor: Handles sparse features gracefully
- Hierarchical contexts: Model morning/afternoon first, then hour within period

### Pitfall 2: Cold Context Problem

**Problem**: New context has no data.

**Examples**:
- New device type launched (smart glasses)
- User travels to new city
- Platform expands to new country (different time zone culture)

**Solutions**:
- Context embeddings: Similar contexts should have similar embeddings
- Context features: Model context through features, not IDs
- Transfer learning: Pre-train on similar contexts

### Pitfall 3: Feature Engineering Overhead

**Problem**: Extracting and maintaining context features is expensive.

**What you need**:
- Real-time location (GPS, IP)
- Device detection
- Weather API integration
- Social graph for "who's with user"
- Activity detection (working, commuting, exercising)

**Each feature requires**:
- Data pipeline
- Privacy compliance
- Handling missing values
- Monitoring for drift

**Solution**: Start with easy wins (time of day, device type), add complexity incrementally based on measured impact.

### Pitfall 4: Privacy Concerns

**Problem**: Context often reveals sensitive information.

- Location = where user lives/works
- Time patterns = daily schedule
- Social context = relationships

**Solutions**:
- Aggregate contexts: "urban area" not exact GPS
- On-device processing
- Differential privacy for context models

---

## Summary

**Key Takeaways**:

1. **Context matters**: Same user wants different things at different times - ignoring context gives ~40% wrong recommendations
2. **Tensor factorization**: Extends MF to User $\times$ Item $\times$ Context
3. **Cyclical encoding**: Use sin/cos for time features (hour 23 should be close to hour 0)
4. **Factorization Machines**: Handle sparse, high-dimensional context features efficiently
5. **Pre/Post/Integrated**: Three approaches to incorporating context, with integrated (FM) being most principled

**The Hierarchy of Context Methods**:
```
Complexity    Method                    When to Use
---------    ------                    -----------
Low          Post-filtering            Quick win, limited data
Medium       Tensor factorization      Discrete contexts, dense data
High         Factorization Machines    Sparse features, many contexts
Highest      Neural context models     Large data, complex interactions
```

**Best Practices**:
- **Feature encoding**: Cyclical for time (sin/cos), distance decay for location
- **Start simple**: Time-of-day and device type often provide 80% of context value
- **Monitor context distribution**: Watch for cold contexts and explosion
- **Privacy-first**: Aggregate sensitive contexts

**Next**: Multi-armed bandits for exploration-exploitation in recommendations.

---

## References

1. **Koren, Y. (2009)**. "Collaborative Filtering with Temporal Dynamics". *KDD*.
   - **TimeSVD++**

2. **Rendle, S. (2010)**. "Factorization Machines". *ICDM*.
   - **Factorization Machines**

3. **Adomavicius, G., & Tuzhilin, A. (2011)**. "Context-Aware Recommender Systems". *Recommender Systems Handbook*.
   - **Survey of context-aware methods**

4. **Karatzoglou, A., et al. (2010)**. "Multiverse Recommendation: N-dimensional Tensor Factorization for Context-aware Collaborative Filtering". *RecSys*.
   - **Tensor factorization**

5. **Lian, D., et al. (2014)**. "GeoMF: Joint Geographical Modeling and Matrix Factorization for Point-of-Interest Recommendation". *KDD*.
   - **Location-aware recommendations**
