# Week 10: Context-Aware Recommendation

## Overview

**Context-aware recommendations** incorporate **contextual information** beyond user-item interactions to improve relevance.

**Context dimensions**:
- **Temporal**: Time of day, day of week, season, trends
- **Spatial**: Location, weather, nearby POIs
- **Device**: Mobile, desktop, TV, smart speaker
- **Social**: Who user is with, social trends
- **Activity**: Working, commuting, exercising, relaxing

**Example** (Music streaming):
```
Traditional: User U likes song S → Recommend S
Context-aware:
  - Morning commute → Energetic playlists
  - Evening relaxation → Calm acoustic
  - Gym workout → High-tempo electronic
```

**Key benefit**: Same user, different contexts → different preferences.

This document covers context-aware recommendation techniques.

---

## Learning Objectives

By the end of this section, you will:
- Understand contextual dimensions in RecSys
- Implement tensor factorization for context
- Apply context-aware matrix factorization
- Build time-aware and location-aware systems
- Handle context in neural models

---

## Contextual Dimensions

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

**Implementation**:
```python
import numpy as np
from datetime import datetime

def extract_temporal_features(timestamp):
    """
    Extract temporal features from timestamp.
    """
    dt = datetime.fromtimestamp(timestamp)

    # Hour of day (0-23)
    hour = dt.hour

    # Cyclical encoding (sin/cos to capture periodicity)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # Day of week (0=Monday, 6=Sunday)
    day_of_week = dt.weekday()
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

### Spatial Context

**Location**: Home, work, airport, restaurant, gym.

**Weather**: Sunny → outdoor activities; Rainy → indoor entertainment.

**Nearby POIs**: Near restaurant → food recommendations; Near mall → shopping.

**Example** (Netflix):
- Home (evening) → Long movies (2+ hours)
- Commute (mobile) → Short episodes (20-30 min)
- Hotel (travel) → Binge-watching series

```python
def extract_spatial_features(lat, lon, weather=None):
    """
    Extract spatial features.
    """
    features = {
        'latitude': lat,
        'longitude': lon
    }

    # Weather (if available)
    if weather:
        features['is_sunny'] = 1 if weather == 'sunny' else 0
        features['is_rainy'] = 1 if weather == 'rainy' else 0
        features['temperature'] = weather.get('temp', 20)  # Celsius

    return features
```

---

### Device Context

**Device type**: Mobile, desktop, tablet, TV, smart speaker.

**Screen size**: Affects content format (short-form vs. long-form).

**Input method**: Touch vs. keyboard → UI preferences.

**Example** (YouTube):
- Mobile → Shorts (vertical, <60s)
- Desktop → Long videos (horizontal, tutorials)
- TV → High-quality, family-friendly

---

### Social Context

**Companions**: Alone, with partner, with family, with friends.

**Example** (Spotify):
- Alone → Personal favorites
- With partner → Romantic playlists
- Party → Upbeat, popular hits

---

## Tensor Factorization

### Motivation

**Matrix factorization**: $R \approx UV^T$ (users × items).

**Problem**: Ignores context! Context adds a **third dimension**.

**Solution**: **Tensor factorization** - extend to 3D (or higher).

**Tensor**: $\mathcal{R} \in \mathbb{R}^{|U| \times |I| \times |C|}$

where $C$ = contexts (time periods, locations, etc.).

---

### Tucker Decomposition

**Idea**: Factorize 3D tensor into core tensor + factor matrices.

$$\mathcal{R} \approx \mathcal{G} \times_1 U \times_2 V \times_3 W$$

where:
- $U \in \mathbb{R}^{|U| \times r_u}$ (user factors)
- $V \in \mathbb{R}^{|I| \times r_i}$ (item factors)
- $W \in \mathbb{R}^{|C| \times r_c}$ (context factors)
- $\mathcal{G} \in \mathbb{R}^{r_u \times r_i \times r_c}$ (core tensor)

**Prediction**:
$$\hat{r}_{uic} = \sum_{p,q,r} \mathcal{G}_{pqr} \cdot U_{up} \cdot V_{iq} \cdot W_{cr}$$

---

### CP Decomposition (CANDECOMP/PARAFAC)

**Simpler than Tucker**: Assume core tensor is diagonal.

$$\mathcal{R} \approx \sum_{k=1}^K \mathbf{u}_k \circ \mathbf{v}_k \circ \mathbf{w}_k$$

where $\circ$ = outer product.

**Prediction**:
$$\hat{r}_{uic} = \sum_{k=1}^K U_{uk} \cdot V_{ik} \cdot W_{ck}$$

**Simpler, but less expressive than Tucker.**

---

### Implementation: CP Tensor Factorization

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

## Context-Aware Matrix Factorization (CAMF)

### Contextual Pre-filtering

**Idea**: Filter data by context, then apply standard MF.

**Steps**:
1. Select interactions matching target context (e.g., weekend, evening)
2. Train MF on filtered data
3. Use model for that context only

**Pros**: Simple, interpretable.
**Cons**: Data sparsity (each context has less data), need separate model per context.

---

### Contextual Post-filtering

**Idea**: Train MF on all data, then adjust predictions based on context.

**Steps**:
1. Train standard MF: $\hat{r}_{ui} = \mathbf{u}^T \mathbf{v}$
2. Learn context adjustment: $\hat{r}_{uic} = \hat{r}_{ui} + \Delta_c$
3. $\Delta_c$ = context-specific bias

**Pros**: Use all data, simple adjustment.
**Cons**: Context only affects bias, not latent features.

---

### Contextual Modeling (Integrated)

**Idea**: Incorporate context directly into MF.

**Factorization Machines (FM)**: Model all pairwise interactions.

$$\hat{y} = w_0 + \sum_i w_i x_i + \sum_i \sum_{j>i} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

where $\mathbf{x}$ = [user features, item features, context features].

**Example**:
```
x = [is_user_123, is_item_456, is_morning, is_mobile]
     1           1            1          1

Interactions:
- user × item
- user × morning
- item × mobile
- morning × mobile
```

---

### Factorization Machines Implementation

```python
class FactorizationMachine(nn.Module):
    def __init__(self, num_features, embedding_dim=10):
        """
        Factorization Machine for context-aware recommendations.

        num_features: Total number of features (user + item + context)
        """
        super().__init__()

        # Linear weights
        self.linear = nn.Embedding(num_features, 1)

        # Pairwise interaction embeddings
        self.embeddings = nn.Embedding(num_features, embedding_dim)

        # Global bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: (batch, num_active_features) - indices of active features
        """
        # Linear term
        linear_part = self.linear(x).sum(dim=1).squeeze()  # (batch,)

        # Pairwise interactions
        emb = self.embeddings(x)  # (batch, num_active_features, dim)

        # FM interaction: 0.5 * (sum_of_squares - squares_of_sum)
        sum_of_emb = emb.sum(dim=1)  # (batch, dim)
        sum_of_square = (emb ** 2).sum(dim=1)  # (batch, dim)

        interaction = 0.5 * ((sum_of_emb ** 2) - sum_of_square).sum(dim=1)  # (batch,)

        # Final prediction
        output = self.bias + linear_part + interaction

        return output


# Example
num_features = 10000  # user IDs + item IDs + contexts
model = FactorizationMachine(num_features, embedding_dim=16)

# Sample: user 5, item 100, context "morning" (ID 5000), device "mobile" (ID 6000)
x = torch.tensor([[5, 100, 5000, 6000]])  # (1, 4)

prediction = model(x)
print(f"Prediction: {prediction.item():.4f}")
```

---

## Time-Aware Recommendations

### Temporal Dynamics

**User preferences drift**: User who liked action movies 5 years ago may now prefer documentaries.

**Item popularity changes**: Viral trends, seasonal products.

**Recency bias**: Recent interactions more important than old ones.

---

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

---

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
- User in San Francisco → SF restaurants
- Traveling to NYC → NYC restaurants

---

### Geo-Spatial Factorization

**Idea**: Incorporate location into user/item embeddings.

**Model**:
$$\text{score}(u, i) = \mathbf{u}^T \mathbf{v}_i \cdot \exp(-\lambda \cdot d(u, i))$$

where:
- $d(u, i)$ = distance between user and item (km)
- $\lambda$ = distance decay rate

**Implementation**:
```python
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
User Features ──┐
                ├─→ Concatenate ─→ MLP ─→ Prediction
Item Features ──┤
                │
Context Features┘
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

## Summary

**Key Takeaways**:
1. **Context dimensions**: Time, location, device, social, activity
2. **Tensor factorization**: CP, Tucker for multi-dimensional data
3. **CAMF**: Pre-filtering, post-filtering, integrated (FM)
4. **Time-aware**: Time decay, temporal biases (TimeSVD++)
5. **Location-aware**: Distance decay, geo-spatial models

**Best Practices**:
- **Feature encoding**: Cyclical for time (sin/cos), distance for location
- **Recency**: Weight recent interactions higher
- **Sparse contexts**: Use FM for feature interactions
- **Neural models**: Concatenate context features with user/item embeddings

**When to use**:
- **Contextual variation**: User preferences change with context
- **Location-based**: Restaurants, events, local services (Yelp, Airbnb)
- **Temporal patterns**: Music (time of day), e-commerce (seasons)

**Next**: Multi-armed bandits for exploration-exploitation.

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
