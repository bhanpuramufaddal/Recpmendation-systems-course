# Week 3: Matrix Factorization - Advanced Variants

## Overview

This document covers advanced matrix factorization variants that extend the basic MF model to incorporate additional signals and improve performance. These methods dominated the Netflix Prize competition and remain influential in modern recommendation systems.

**Topics covered**:
1. **SVD++**: Incorporating implicit feedback (Koren, 2008)
2. **TimeSVD++**: Adding temporal dynamics (Koren, 2009)
3. **Factorization Machines**: General framework for feature interactions (Rendle, 2010)

These variants represent the **state-of-the-art** in classical collaborative filtering before the deep learning era.

---

## Learning Objectives

By the end of this section, you will:
- Understand how SVD++ integrates implicit feedback
- Model temporal dynamics with TimeSVD++
- Recognize Factorization Machines as a general framework
- Implement these advanced techniques
- Know when to use which variant

---

## SVD++: Implicit Feedback Integration

### Motivation

**Standard MF**:
$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i$$

**Limitation**: Only uses explicit ratings, ignores implicit signals.

**Observation**: Even without ratings, we know which items user has interacted with!

**Example**:
- User rated "The Matrix" 5 stars
- User also watched (but didn't rate): "Inception", "Interstellar", "The Prestige"
- These implicit signals reveal user's preference for sci-fi/Christopher Nolan films

---

### The SVD++ Model

**Paper**: Koren, Y. (2008). "Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model". *KDD*.

**Key idea**: Augment user latent factors with implicit item factors.

**Prediction formula**:

$$\hat{r}_{ui} = \mu + b_u + b_i + \left( \mathbf{u}_u + |N(u)|^{-1/2} \sum_{j \in N(u)} \mathbf{y}_j \right)^T \mathbf{v}_i$$

where:
- $\mathbf{u}_u$: Explicit user factors (from ratings)
- $\mathbf{v}_i$: Item factors
- $\mathbf{y}_j$: **Implicit item factors** (new!)
- $N(u)$: Set of items user $u$ has interacted with (implicit feedback)
- $|N(u)|^{-1/2}$: Normalization factor

**Components**:
1. **$\mathbf{u}_u$**: User's explicit preferences (learned from ratings)
2. **$\sum_{j \in N(u)} \mathbf{y}_j$**: User's implicit profile (learned from interactions)
3. **$|N(u)|^{-1/2}$**: Normalize by number of interactions

---

### Intuition

**User representation** is now:
$$\text{Effective user vector} = \mathbf{u}_u + \frac{1}{\sqrt{|N(u)|}} \sum_{j \in N(u)} \mathbf{y}_j$$

**Two sources of information**:
- **Explicit**: User gave 5 stars to "The Matrix" → Learn $\mathbf{u}_u$
- **Implicit**: User watched "Inception" (no rating) → Add $\mathbf{y}_{\text{Inception}}$ to user profile

**Why** $|N(u)|^{-1/2}$?
- Users with many interactions: Each item contributes less
- Users with few interactions: Each item contributes more
- Prevents bias towards active users

---

### Optimization

**Objective function**:

$$\min \sum_{(u,i,r) \in \text{train}} \left( r_{ui} - \hat{r}_{ui} \right)^2 + \lambda \left( \|U\|^2 + \|V\|^2 + \|Y\|^2 + \|b_u\|^2 + \|b_i\|^2 \right)$$

where $Y$ is the matrix of implicit item factors.

**SGD update** for each rating $(u, i, r_{ui})$:

1. **Compute prediction**:
   $$\hat{r}_{ui} = \mu + b_u + b_i + \left( \mathbf{u}_u + |N(u)|^{-1/2} \sum_{j \in N(u)} \mathbf{y}_j \right)^T \mathbf{v}_i$$

2. **Compute error**:
   $$e_{ui} = r_{ui} - \hat{r}_{ui}$$

3. **Update user factors**:
   $$\mathbf{u}_u \leftarrow \mathbf{u}_u + \gamma (e_{ui} \cdot \mathbf{v}_i - \lambda \mathbf{u}_u)$$

4. **Update item factors**:
   $$\mathbf{v}_i \leftarrow \mathbf{v}_i + \gamma \left( e_{ui} \cdot \left( \mathbf{u}_u + |N(u)|^{-1/2} \sum_{j \in N(u)} \mathbf{y}_j \right) - \lambda \mathbf{v}_i \right)$$

5. **Update implicit factors** (for each $j \in N(u)$):
   $$\mathbf{y}_j \leftarrow \mathbf{y}_j + \gamma \left( e_{ui} \cdot |N(u)|^{-1/2} \cdot \mathbf{v}_i - \lambda \mathbf{y}_j \right)$$

6. **Update biases**:
   $$b_u \leftarrow b_u + \gamma (e_{ui} - \lambda b_u)$$
   $$b_i \leftarrow b_i + \gamma (e_{ui} - \lambda b_i)$$

---

### Implementation

```python
import numpy as np
from collections import defaultdict

class SVDPlusPlus:
    def __init__(self, n_factors=20, learning_rate=0.005, reg=0.02, n_epochs=20):
        self.k = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs

    def fit(self, ratings, implicit_feedback):
        """
        ratings: list of (user, item, rating) tuples
        implicit_feedback: dict {user: set of items user interacted with}
        """
        # Build user and item sets
        users = set([u for u, i, r in ratings])
        items = set([i for u, i, r in ratings])

        # Add implicit items
        for user, item_set in implicit_feedback.items():
            items.update(item_set)

        self.n_users = max(users) + 1
        self.n_items = max(items) + 1
        self.implicit = implicit_feedback

        # Global mean
        self.mu = np.mean([r for u, i, r in ratings])

        # Initialize parameters
        self.U = np.random.normal(0, 0.1, (self.k, self.n_users))  # Explicit user factors
        self.V = np.random.normal(0, 0.1, (self.k, self.n_items))  # Item factors
        self.Y = np.random.normal(0, 0.1, (self.k, self.n_items))  # Implicit item factors

        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)

        # Precompute |N(u)|^{-0.5} for each user
        self.sqrt_N_u = {}
        for u in range(self.n_users):
            N_u = len(implicit_feedback.get(u, []))
            self.sqrt_N_u[u] = 1.0 / np.sqrt(N_u) if N_u > 0 else 0

        # Training loop
        for epoch in range(self.n_epochs):
            np.random.shuffle(ratings)

            for u, i, r in ratings:
                # Compute implicit sum
                N_u = implicit_feedback.get(u, set())
                if N_u:
                    implicit_sum = np.sum([self.Y[:, j] for j in N_u], axis=0)
                    implicit_sum *= self.sqrt_N_u[u]
                else:
                    implicit_sum = 0

                # Prediction
                pred = self.mu + self.b_u[u] + self.b_i[i] + \
                       np.dot(self.U[:, u] + implicit_sum, self.V[:, i])

                # Error
                err = r - pred

                # Update explicit user factors
                u_vec = self.U[:, u].copy()
                self.U[:, u] += self.lr * (err * self.V[:, i] - self.reg * u_vec)

                # Update item factors
                v_vec = self.V[:, i].copy()
                user_profile = u_vec + implicit_sum
                self.V[:, i] += self.lr * (err * user_profile - self.reg * v_vec)

                # Update implicit factors
                if N_u:
                    for j in N_u:
                        y_vec = self.Y[:, j].copy()
                        self.Y[:, j] += self.lr * (err * self.sqrt_N_u[u] * v_vec - self.reg * y_vec)

                # Update biases
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])

            # Evaluate
            rmse = self.evaluate(ratings)
            print(f"Epoch {epoch+1}/{self.n_epochs}: RMSE = {rmse:.4f}")

    def predict(self, u, i):
        """Predict rating for user u, item i"""
        if u >= self.n_users or i >= self.n_items:
            return self.mu

        # Implicit sum
        N_u = self.implicit.get(u, set())
        if N_u:
            implicit_sum = np.sum([self.Y[:, j] for j in N_u], axis=0)
            implicit_sum *= self.sqrt_N_u[u]
        else:
            implicit_sum = 0

        return self.mu + self.b_u[u] + self.b_i[i] + \
               np.dot(self.U[:, u] + implicit_sum, self.V[:, i])

    def evaluate(self, ratings):
        """Compute RMSE"""
        errors = [(r - self.predict(u, i))**2 for u, i, r in ratings]
        return np.sqrt(np.mean(errors))

# Example usage
if __name__ == "__main__":
    # Ratings (explicit feedback)
    train_ratings = [
        (0, 0, 5.0), (0, 1, 3.0),
        (1, 0, 4.0), (1, 2, 5.0),
        (2, 1, 2.0), (2, 3, 4.0)
    ]

    # Implicit feedback (items users interacted with, but didn't rate)
    implicit_feedback = {
        0: {0, 1, 2, 5},  # User 0 interacted with items 0,1,2,5
        1: {0, 2, 4},     # User 1 interacted with items 0,2,4
        2: {1, 3, 6, 7}   # User 2 interacted with items 1,3,6,7
    }

    model = SVDPlusPlus(n_factors=10, learning_rate=0.01, reg=0.1, n_epochs=30)
    model.fit(train_ratings, implicit_feedback)

    print(f"\nPrediction for user 0, item 3: {model.predict(0, 3):.2f}")
```

---

### Results

**Netflix Prize** (Koren, 2008):
- **Basic MF**: RMSE = 0.9129
- **SVD++**: RMSE = 0.8747
- **Improvement**: **4.2%** (significant!)

**Why it works**:
- Leverages ALL user interactions (rated + unrated)
- More data → better user representations
- Especially helpful for users with few ratings but many implicit signals

---

## TimeSVD++: Temporal Dynamics

### Motivation

**Observation**: User preferences and item perceptions change over time.

**Examples**:
- User's taste evolves (starts liking documentaries in their 30s)
- Item popularity changes (new movie gets hyped, then forgotten)
- User rating behavior changes (grade inflation: "5 stars for everything!")

**Standard MF assumption**: Static preferences (wrong!)

---

### The TimeSVD++ Model

**Paper**: Koren, Y. (2009). "Collaborative Filtering with Temporal Dynamics". *KDD*.

**Key idea**: Add time-dependent components to SVD++.

**Prediction formula**:

$$\hat{r}_{ui}(t) = \mu + b_i(t) + b_u(t) + \left( \mathbf{u}_u(t) + |N(u)|^{-1/2} \sum_{j \in N(u)} \mathbf{y}_j \right)^T \mathbf{v}_i$$

where $t$ is the time of rating.

**Time-varying components**:

1. **Item bias** $b_i(t)$:
   $$b_i(t) = b_i + b_{i,\text{bin}(t)}$$
   - $b_i$: Static item bias
   - $b_{i,\text{bin}(t)}$: Bias for time bin (e.g., week, month)

2. **User bias** $b_u(t)$:
   $$b_u(t) = b_u + \alpha_u \cdot \text{dev}_u(t) + b_{u,t}$$
   - $b_u$: Static user bias
   - $\text{dev}_u(t) = \text{sign}(t - t_u) \cdot |t - t_u|^{\beta}$: Deviation from user's mean time
   - $\alpha_u$: User-specific drift parameter
   - $b_{u,t}$: User bias at time $t$ (binned)

3. **User factors** $\mathbf{u}_u(t)$:
   $$\mathbf{u}_u(t) = \mathbf{u}_u + \alpha_u(t)$$
   - $\mathbf{u}_u$: Static user factors
   - $\alpha_u(t)$: Time-dependent drift

---

### Intuition

**Time bins**:
- Divide time into bins (e.g., weeks, months)
- Each bin has separate parameters
- Captures seasonal effects, trends

**User drift**:
- User's rating baseline changes over time
- Example: User becomes more critical → $b_u(t)$ decreases

**Item drift**:
- Item popularity changes
- Example: "The Dark Knight" (2008) was hot, now less so

---

### Complexity

**Challenge**: Many more parameters!

**Standard MF**: $O(k \cdot (|U| + |I|))$ parameters

**TimeSVD++**: $O(k \cdot (|U| + |I|) + T \cdot (|U| + |I|))$ parameters (where $T$ = number of time bins)

**Regularization**: Critical to prevent overfitting!

---

### Implementation (Simplified)

```python
import numpy as np
from datetime import datetime

class TimeSVDPlusPlus:
    def __init__(self, n_factors=20, n_bins=30, learning_rate=0.005, reg=0.02, n_epochs=20):
        """
        n_bins: Number of time bins (e.g., 30 months)
        """
        self.k = n_factors
        self.n_bins = n_bins
        self.lr = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs

    def fit(self, ratings_with_time):
        """
        ratings_with_time: list of (user, item, rating, timestamp) tuples
        """
        # Build user/item sets
        users = set([u for u, i, r, t in ratings_with_time])
        items = set([i for u, i, r, t in ratings_with_time])

        self.n_users = max(users) + 1
        self.n_items = max(items) + 1

        # Global mean
        self.mu = np.mean([r for u, i, r, t in ratings_with_time])

        # Time binning
        timestamps = [t for u, i, r, t in ratings_with_time]
        self.min_time = min(timestamps)
        self.max_time = max(timestamps)
        self.bin_size = (self.max_time - self.min_time) / self.n_bins

        # Initialize parameters
        self.U = np.random.normal(0, 0.1, (self.k, self.n_users))
        self.V = np.random.normal(0, 0.1, (self.k, self.n_items))

        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)

        # Time-dependent biases
        self.b_u_t = np.zeros((self.n_users, self.n_bins))  # User bias per time bin
        self.b_i_t = np.zeros((self.n_items, self.n_bins))  # Item bias per time bin

        # Training loop (simplified, without full TimeSVD++ complexity)
        for epoch in range(self.n_epochs):
            np.random.shuffle(ratings_with_time)

            for u, i, r, timestamp in ratings_with_time:
                # Get time bin
                bin_idx = self.get_time_bin(timestamp)

                # Prediction
                pred = self.mu + self.b_u[u] + self.b_i[i] + \
                       self.b_u_t[u, bin_idx] + self.b_i_t[i, bin_idx] + \
                       np.dot(self.U[:, u], self.V[:, i])

                # Error
                err = r - pred

                # Update static factors
                u_vec = self.U[:, u].copy()
                v_vec = self.V[:, i].copy()

                self.U[:, u] += self.lr * (err * v_vec - self.reg * u_vec)
                self.V[:, i] += self.lr * (err * u_vec - self.reg * v_vec)

                # Update static biases
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])

                # Update time-dependent biases
                self.b_u_t[u, bin_idx] += self.lr * (err - self.reg * self.b_u_t[u, bin_idx])
                self.b_i_t[i, bin_idx] += self.lr * (err - self.reg * self.b_i_t[i, bin_idx])

            rmse = self.evaluate(ratings_with_time)
            print(f"Epoch {epoch+1}/{self.n_epochs}: RMSE = {rmse:.4f}")

    def get_time_bin(self, timestamp):
        """Map timestamp to bin index"""
        bin_idx = int((timestamp - self.min_time) / self.bin_size)
        return min(bin_idx, self.n_bins - 1)

    def predict(self, u, i, timestamp):
        """Predict rating at given time"""
        if u >= self.n_users or i >= self.n_items:
            return self.mu

        bin_idx = self.get_time_bin(timestamp)

        return self.mu + self.b_u[u] + self.b_i[i] + \
               self.b_u_t[u, bin_idx] + self.b_i_t[i, bin_idx] + \
               np.dot(self.U[:, u], self.V[:, i])

    def evaluate(self, ratings_with_time):
        """Compute RMSE"""
        errors = [(r - self.predict(u, i, t))**2 for u, i, r, t in ratings_with_time]
        return np.sqrt(np.mean(errors))

# Example usage
if __name__ == "__main__":
    # Ratings with timestamps (user, item, rating, timestamp)
    train_ratings = [
        (0, 0, 5.0, 1609459200),  # 2021-01-01
        (0, 1, 3.0, 1612137600),  # 2021-02-01
        (1, 0, 4.0, 1614556800),  # 2021-03-01
        (1, 2, 5.0, 1617235200),  # 2021-04-01
        (2, 1, 2.0, 1619827200),  # 2021-05-01
        (2, 3, 4.0, 1622505600)   # 2021-06-01
    ]

    model = TimeSVDPlusPlus(n_factors=10, n_bins=12, learning_rate=0.01, reg=0.1, n_epochs=30)
    model.fit(train_ratings)

    print(f"\nPrediction for user 0, item 2 at timestamp 1625097600: {model.predict(0, 2, 1625097600):.2f}")
```

---

### Results

**Netflix Prize** (Koren, 2009):
- **SVD++**: RMSE = 0.8747
- **TimeSVD++**: RMSE = 0.8799 → **0.8567** (with full model)
- **Improvement**: ~2% over SVD++

**BellKor's Pragmatic Chaos** (Netflix Prize winners):
- Used ensemble of TimeSVD++ and other models
- Final RMSE: **0.8567** (10.06% improvement over Cinematch)

---

## Factorization Machines (FM)

### The Problem: Why Do We Need Feature Interactions?

*Before we dive into the math, let me show you why standard approaches fail.*

**Scenario**: You're building a movie recommendation system. You have:
- User features: age, gender, location
- Movie features: genre, director, year
- Context: time of day, device

**The naive approach** - just use linear regression:

$$\hat{y} = w_0 + w_{age} \cdot \text{age} + w_{male} \cdot \text{male} + w_{action} \cdot \text{action} + \ldots$$

**What goes wrong?** Consider this pattern in your data:
- Young men love action movies (age < 25 AND male AND action → high rating)
- Older women love drama (age > 50 AND female AND drama → high rating)

The linear model sees these as *independent* effects:
- $w_{male}$ captures average effect of being male
- $w_{action}$ captures average effect of action genre

*But the magic happens in the combination!* A young male watching an action movie isn't just "young effect + male effect + action effect" — it's *multiplicatively* more predictive.

**Can you see why** a linear model would underestimate how much young men like action movies?

---

### The Naive Solution: Polynomial Features

**Idea**: Explicitly model all pairs:

$$\hat{y} = w_0 + \sum_i w_i x_i + \sum_i \sum_{j>i} w_{ij} x_i x_j$$

**The problem**: With $n$ features, you need $\binom{n}{2} = \frac{n(n-1)}{2}$ interaction parameters!

**Example**:
- 1000 users + 5000 items + 50 features = 6050 total features
- Interactions: $\frac{6050 \times 6049}{2} \approx 18$ million parameters!
- Most feature pairs never appear together in training data
- Result: **severe overfitting** on sparse data

*What would happen if* User #7 never watched any Action movies in training? The weight $w_{user7, action}$ would be random noise.

---

### Motivation

**Problem**: How to incorporate side information (features) into MF?

**Examples**:
- User demographics (age, gender, location)
- Item attributes (genre, director, year)
- Context (time of day, device, location)

**Challenge**: Feature interactions are critical!
- "Age × Genre": Young users like action, older users like drama
- "Gender × Director": Women prefer certain directors

**The FM insight**: Don't learn separate $w_{ij}$ for each pair. Instead, give each feature a **latent vector** $\mathbf{v}_i$, and model interactions as **dot products** $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$.

*Why does this help?* If User #7 likes Sci-Fi and Thriller, their vector $\mathbf{v}_{user7}$ will be similar to both genre vectors. When we encounter User #7 + Action (never seen!), we can still make a reasonable prediction because $\mathbf{v}_{action}$ is similar to $\mathbf{v}_{scifi}$.

---

### The Factorization Machines Framework

**Paper**: Rendle, S. (2010). "Factorization Machines". *IEEE ICDM*.

**Key idea**: Model all pairwise feature interactions with latent factors.

**Prediction formula**:

$$\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

where:
- $\mathbf{x} \in \mathbb{R}^n$: Feature vector (user, item, context)
- $w_0$: Global bias (average rating across all data)
- $w_i$: Weight for feature $i$ (main effect)
- $\mathbf{v}_i \in \mathbb{R}^k$: Latent factor for feature $i$ (k-dimensional "personality" vector)
- $\langle \mathbf{v}_i, \mathbf{v}_j \rangle = \sum_{f=1}^k v_{if} \cdot v_{jf}$: Interaction between features $i$ and $j$

**Components**:
1. **Global bias**: $w_0$ — "What's the average rating?"
2. **Linear term**: $\sum w_i x_i$ — "How does each feature affect rating independently?"
3. **Pairwise interactions**: $\sum_{i < j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$ — "How do features interact?"

---

### What Do Latent Factors Mean? (The Intuition)

*Before I show you the formula, what do you think* $\mathbf{v}_i$ *should represent?*

Each feature gets a k-dimensional vector that captures its **"interaction profile"**:

**Example with k=3 dimensions** (conceptually):
- Dimension 1: "Intensity" (calm ↔ exciting)
- Dimension 2: "Emotional" (cerebral ↔ emotional)
- Dimension 3: "Social" (solo experience ↔ social experience)

| Feature | $\mathbf{v}$ vector | Interpretation |
|---------|---------------------|----------------|
| User: Alice | [0.8, -0.3, 0.5] | Likes exciting, cerebral, social |
| User: Bob | [0.2, 0.7, -0.4] | Likes calm, emotional, solo |
| Genre: Action | [0.9, -0.2, 0.3] | Exciting, cerebral, somewhat social |
| Genre: Drama | [0.1, 0.8, 0.2] | Calm, emotional, social |

**The interaction** $\langle \mathbf{v}_{Alice}, \mathbf{v}_{Action} \rangle$:
$$= 0.8 \times 0.9 + (-0.3) \times (-0.2) + 0.5 \times 0.3 = 0.72 + 0.06 + 0.15 = 0.93$$

High positive value → Alice and Action movies are compatible!

**The interaction** $\langle \mathbf{v}_{Bob}, \mathbf{v}_{Action} \rangle$:
$$= 0.2 \times 0.9 + 0.7 \times (-0.2) + (-0.4) \times 0.3 = 0.18 - 0.14 - 0.12 = -0.08$$

Near zero/negative → Bob and Action movies are not a great match.

*Can you see why* this is more powerful than just having "Bob likes action = +0.3"? The latent factors capture *why* Bob might not like action (he prefers calm, emotional experiences).

---

### Efficient Computation: The Key Mathematical Insight

**Naive complexity**: $O(kn^2)$ for pairwise interactions (expensive!)

*Let me walk you through* why the naive approach is slow and how we fix it.

**Step 1: The naive formula**

$$\text{Interactions} = \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

For each of the $\binom{n}{2}$ pairs, we compute a k-dimensional dot product. That's $O(kn^2)$ operations.

**Step 2: The key insight — Expand the square**

*What if we could compute all interactions in one shot?* Consider:

$$\left( \sum_{i=1}^n v_{if} x_i \right)^2 = \sum_{i=1}^n \sum_{j=1}^n v_{if} v_{jf} x_i x_j$$

This includes **all pairs** (including $i=j$), but we only want $i < j$.

**Step 3: Separate diagonal from off-diagonal**

$$\sum_{i=1}^n \sum_{j=1}^n v_{if} v_{jf} x_i x_j = \underbrace{\sum_{i=1}^n v_{if}^2 x_i^2}_{\text{diagonal } (i=j)} + \underbrace{2 \sum_{i < j} v_{if} v_{jf} x_i x_j}_{\text{off-diagonal } (i \neq j)}$$

Solving for the off-diagonal (what we want):

$$\sum_{i < j} v_{if} v_{jf} x_i x_j = \frac{1}{2} \left[ \left( \sum_{i=1}^n v_{if} x_i \right)^2 - \sum_{i=1}^n v_{if}^2 x_i^2 \right]$$

**Step 4: Sum over all latent dimensions**

$$\sum_{i < j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2} \sum_{f=1}^k \left[ \left( \sum_{i=1}^n v_{if} x_i \right)^2 - \sum_{i=1}^n v_{if}^2 x_i^2 \right]$$

**Reduced complexity**: $O(kn)$ — linear in number of features!

**Why this works computationally**:
1. Compute $\sum_{i=1}^n v_{if} x_i$ for each $f$: $O(kn)$
2. Square it: $O(k)$
3. Compute $\sum_{i=1}^n v_{if}^2 x_i^2$ for each $f$: $O(kn)$
4. Subtract and sum: $O(k)$

**Total**: $O(kn)$ instead of $O(kn^2)$!

---

### Numerical Walkthrough: A Complete Example

*Let's work through a concrete prediction step by step.*

**Setup**: Predict rating for User 7 watching Movie 42 (an Action film).

**Feature vector** (one-hot encoded + binary features):
$$\mathbf{x} = [\underbrace{0,\ldots,0,1,0,\ldots,0}_{\text{user ID (position 7)}}, \underbrace{0,\ldots,0,1,0,\ldots,0}_{\text{movie ID (position 42)}}, \underbrace{1}_{\text{Genre=Action}}]$$

For simplicity, let's say:
- Feature 7 is User 7 (one-hot): $x_7 = 1$
- Feature 50 is Movie 42 (one-hot): $x_{50} = 1$
- Feature 100 is Genre=Action (binary): $x_{100} = 1$
- All other $x_i = 0$

**Model parameters** (k=2 latent factors):

| Parameter | Value |
|-----------|-------|
| $w_0$ | 3.5 |
| $w_7$ (User 7 bias) | +0.3 |
| $w_{50}$ (Movie 42 bias) | +0.5 |
| $w_{100}$ (Action bias) | +0.2 |
| $\mathbf{v}_7$ (User 7 vector) | [0.5, 0.3] |
| $\mathbf{v}_{50}$ (Movie 42 vector) | [0.6, -0.2] |
| $\mathbf{v}_{100}$ (Action vector) | [0.4, 0.5] |

**Step 1: Global bias**
$$w_0 = 3.5$$

**Step 2: Linear terms** (only non-zero features contribute)
$$\sum_i w_i x_i = w_7 \cdot 1 + w_{50} \cdot 1 + w_{100} \cdot 1 = 0.3 + 0.5 + 0.2 = 1.0$$

**Step 3: Pairwise interactions** (3 pairs with non-zero product)

1. User 7 × Movie 42:
   $$\langle \mathbf{v}_7, \mathbf{v}_{50} \rangle = 0.5 \times 0.6 + 0.3 \times (-0.2) = 0.30 - 0.06 = 0.24$$

2. User 7 × Action:
   $$\langle \mathbf{v}_7, \mathbf{v}_{100} \rangle = 0.5 \times 0.4 + 0.3 \times 0.5 = 0.20 + 0.15 = 0.35$$

3. Movie 42 × Action:
   $$\langle \mathbf{v}_{50}, \mathbf{v}_{100} \rangle = 0.6 \times 0.4 + (-0.2) \times 0.5 = 0.24 - 0.10 = 0.14$$

Total interactions: $0.24 + 0.35 + 0.14 = 0.73$

**Final prediction**:
$$\hat{y} = 3.5 + 1.0 + 0.73 = 5.23$$

After clipping to [1, 5]: **Predicted rating = 5.0 stars**

*What does this tell us?* User 7 has positive interactions with both the movie and the genre — they're predicted to love this movie!

---

### Gradient Derivations: How Does FM Learn?

*Now that you understand the prediction, let's derive how FM learns from errors.*

**Loss function** (squared error for one sample):

$$L = \frac{1}{2}(y - \hat{y})^2$$

where $e = y - \hat{y}$ is the prediction error.

**We need gradients for three types of parameters:**

---

#### Gradient for Global Bias $w_0$

$$\frac{\partial L}{\partial w_0} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_0} = -e \cdot 1 = -e$$

**Intuition**: If we underpredict ($e > 0$), increase $w_0$. If we overpredict ($e < 0$), decrease $w_0$.

**Update rule** (with learning rate $\eta$):
$$w_0 \leftarrow w_0 + \eta \cdot e$$

---

#### Gradient for Feature Weights $w_i$

$$\frac{\partial L}{\partial w_i} = -e \cdot \frac{\partial \hat{y}}{\partial w_i} = -e \cdot x_i$$

**Intuition**: The gradient is proportional to both the error AND the feature value. If feature $i$ is "on" ($x_i = 1$) and we underpredict, increase $w_i$.

**Update rule**:
$$w_i \leftarrow w_i + \eta \cdot e \cdot x_i$$

---

#### Gradient for Latent Factors $v_{if}$ (The Tricky One!)

*This is where it gets interesting.* We need:

$$\frac{\partial \hat{y}}{\partial v_{if}}$$

Recall the efficient form:
$$\text{Interactions} = \frac{1}{2} \sum_{f=1}^k \left[ \left( \sum_{j=1}^n v_{jf} x_j \right)^2 - \sum_{j=1}^n v_{jf}^2 x_j^2 \right]$$

Let $S_f = \sum_{j=1}^n v_{jf} x_j$ (sum for latent dimension $f$).

**Step 1**: Differentiate the squared term
$$\frac{\partial}{\partial v_{if}} \left( S_f \right)^2 = 2 S_f \cdot \frac{\partial S_f}{\partial v_{if}} = 2 S_f \cdot x_i$$

**Step 2**: Differentiate the diagonal correction term
$$\frac{\partial}{\partial v_{if}} \left( v_{if}^2 x_i^2 \right) = 2 v_{if} x_i^2$$

**Step 3**: Combine (with the 1/2 factor)
$$\frac{\partial \hat{y}}{\partial v_{if}} = \frac{1}{2} \left[ 2 S_f \cdot x_i - 2 v_{if} x_i^2 \right] = x_i \left( S_f - v_{if} x_i \right)$$

Substituting $S_f = \sum_j v_{jf} x_j$:

$$\frac{\partial \hat{y}}{\partial v_{if}} = x_i \left( \sum_{j=1}^n v_{jf} x_j - v_{if} x_i \right)$$

**Final gradient**:
$$\frac{\partial L}{\partial v_{if}} = -e \cdot x_i \left( \sum_{j=1}^n v_{jf} x_j - v_{if} x_i \right)$$

**Update rule**:
$$v_{if} \leftarrow v_{if} + \eta \cdot e \cdot x_i \left( \sum_{j \neq i} v_{jf} x_j \right)$$

**Intuition**: The latent factor $v_{if}$ is updated based on:
1. The error $e$ (how wrong were we?)
2. The feature value $x_i$ (is this feature active?)
3. The "context" $\sum_{j \neq i} v_{jf} x_j$ (what other features are present and what are their latent factors?)

*Can you see why* this makes sense? If User 7 watches an Action movie and we underpredict, we should move $\mathbf{v}_{user7}$ *towards* $\mathbf{v}_{action}$ (increase their dot product).

---

#### Numerical Gradient Example

*Let's verify our gradient with the previous example.*

**Setup** (same as before):
- Actual rating: $y = 5$
- Predicted: $\hat{y} = 5.23$ (clipped to 5.0, but use 5.23 for gradients)
- Error: $e = 5 - 5.23 = -0.23$ (we overpredicted)

**Compute $S_f$ for each latent dimension** (f=1 and f=2):

For $f=1$ (first latent dimension):
$$S_1 = v_{7,1} \cdot x_7 + v_{50,1} \cdot x_{50} + v_{100,1} \cdot x_{100} = 0.5 \cdot 1 + 0.6 \cdot 1 + 0.4 \cdot 1 = 1.5$$

For $f=2$:
$$S_2 = 0.3 \cdot 1 + (-0.2) \cdot 1 + 0.5 \cdot 1 = 0.6$$

**Gradient for $v_{7,1}$** (User 7's first latent factor):
$$\frac{\partial L}{\partial v_{7,1}} = -e \cdot x_7 \cdot (S_1 - v_{7,1} \cdot x_7) = -(-0.23) \cdot 1 \cdot (1.5 - 0.5 \cdot 1) = 0.23 \cdot 1.0 = 0.23$$

**Update** (with $\eta = 0.01$):
$$v_{7,1}^{new} = 0.5 + 0.01 \cdot (-0.23) = 0.5 - 0.0023 = 0.4977$$

*The gradient is positive but the error is negative, so we decrease $v_{7,1}$.*

This makes sense: we overpredicted, so we're slightly decreasing the interaction strength between User 7 and the other active features.

---

### Connection to Matrix Factorization

**Standard MF**:
$$\hat{r}_{ui} = \mathbf{u}_u^T \mathbf{v}_i$$

**As FM**: Define feature vector
$$\mathbf{x} = [\underbrace{0, \ldots, 0, 1, 0, \ldots, 0}_{\text{user one-hot}}, \underbrace{0, \ldots, 0, 1, 0, \ldots, 0}_{\text{item one-hot}}]$$

Then:
$$\hat{r}_{ui} = \langle \mathbf{v}_u, \mathbf{v}_i \rangle$$

**FM generalizes MF** by allowing arbitrary features!

---

### What Can Go Wrong? Edge Cases and Failure Modes

*Before implementing, let's understand when FM might fail.*

**1. Feature Sparsity: When Latent Factors Can't Learn**

**Problem**: If a feature appears only once in training, its latent vector $\mathbf{v}_i$ can't learn meaningful interactions.

**Example**:
- User 12345 rated only 1 movie
- $\mathbf{v}_{user12345}$ is trained on only 1 gradient update
- Result: Random noise, not useful for predictions

**Solution**: Regularization ($\lambda \|\mathbf{V}\|^2$) pulls rare features toward zero. Also consider minimum frequency thresholds.

---

**2. The Cold Start Problem**

**Problem**: New features (users, items) have no interaction data.

**Scenario**: Movie released yesterday. $\mathbf{v}_{new\_movie} = $ random initialization.

**Partial solution**: FM with side features! If the new movie has genre=Action, director=Nolan, the linear terms ($w_{action}$, $w_{nolan}$) and their learned interactions still contribute.

*This is precisely why FM is better than pure MF for cold start.*

---

**3. Choosing k (Number of Latent Factors)**

*What would happen if* k is too small?
- Can't capture complex interaction patterns
- Underfitting

*What would happen if* k is too large?
- Too many parameters for sparse data
- Overfitting
- Slower training

**Rule of thumb**: Start with $k = 8$ to $k = 64$. Validate on held-out data.

---

**4. Feature Scale Issues**

**Problem**: Non-binary features with different scales.

**Example**:
- $x_{age} = 35$ (continuous)
- $x_{is\_action} = 1$ (binary)

The interaction $\langle \mathbf{v}_{age}, \mathbf{v}_{action} \rangle \cdot 35 \cdot 1$ dominates!

**Solution**: Normalize continuous features to [0, 1] or standardize to mean=0, std=1.

---

### Implementation (Annotated with Example Values)

```python
import numpy as np

class FactorizationMachine:
    def __init__(self, n_factors=10, learning_rate=0.01, reg=0.01, n_epochs=20):
        self.k = n_factors        # Latent dimension (e.g., k=8 means 8-dimensional vectors)
        self.lr = learning_rate   # Step size for SGD (typical: 0.01-0.1)
        self.reg = reg            # L2 regularization strength (prevents overfitting)
        self.n_epochs = n_epochs  # Number of passes through training data

    def fit(self, X, y):
        """
        X: (n_samples, n_features) feature matrix
           Example: 1000 samples, 6050 features (1000 users + 5000 items + 50 context)
        y: (n_samples,) target vector (ratings)
           Example: [4.5, 3.0, 5.0, ...]
        """
        n_samples, self.n_features = X.shape

        # === INITIALIZATION ===
        # Global bias: start at 0, will converge to mean rating
        self.w0 = 0.0  # After training: ~3.5 for 1-5 rating scale

        # Linear weights: one per feature
        self.w = np.zeros(self.n_features)  # Shape: (6050,)

        # Latent factors: each feature gets a k-dimensional vector
        # Initialize small random to break symmetry
        self.V = np.random.normal(0, 0.01, (self.n_features, self.k))  # Shape: (6050, 10)

        # === TRAINING LOOP ===
        for epoch in range(self.n_epochs):
            for idx in range(n_samples):
                x = X[idx]       # Shape: (6050,) - mostly zeros (sparse!)
                target = y[idx]  # Scalar: actual rating (e.g., 4.0)

                # === FORWARD PASS ===
                pred = self.predict_instance(x)  # e.g., pred = 3.7

                # Error: how wrong are we?
                err = target - pred  # e.g., err = 4.0 - 3.7 = +0.3 (underpredicted)

                # === BACKWARD PASS (SGD Updates) ===

                # Update global bias: w0 += lr * err
                # If err > 0, we increase w0 (predictions were too low)
                self.w0 += self.lr * err  # w0 = 0 + 0.01 * 0.3 = 0.003

                # Update linear weights: w_i += lr * (err * x_i - reg * w_i)
                # Only non-zero x_i contribute; regularization shrinks weights
                self.w += self.lr * (err * x - self.reg * self.w)

                # Update latent factors (the key FM innovation)
                # Precompute S_f = sum_j(v_jf * x_j) for each factor f
                sum_vx = np.dot(x, self.V)  # Shape: (k,) = (10,)
                # Example: sum_vx = [1.5, 0.6, ...] for k=10

                for i in range(self.n_features):
                    if x[i] != 0:  # Only update active features (sparse optimization)
                        for f in range(self.k):
                            # Gradient: err * x_i * (S_f - v_if * x_i) - reg * v_if
                            # The (S_f - v_if * x_i) term is "sum over OTHER features"
                            grad = err * (x[i] * sum_vx[f] - self.V[i, f] * x[i]**2) - self.reg * self.V[i, f]
                            self.V[i, f] += self.lr * grad

            # === EPOCH EVALUATION ===
            predictions = np.array([self.predict_instance(X[i]) for i in range(n_samples)])
            rmse = np.sqrt(np.mean((y - predictions)**2))
            print(f"Epoch {epoch+1}/{self.n_epochs}: RMSE = {rmse:.4f}")
            # Example output: Epoch 1: RMSE = 1.2345, Epoch 50: RMSE = 0.8123

    def predict_instance(self, x):
        """
        Predict rating for a single feature vector.

        x: Shape (n_features,), sparse (mostly zeros)
        Returns: Scalar prediction (e.g., 4.23)
        """
        # === LINEAR COMPONENT ===
        # w0 + sum_i(w_i * x_i)
        linear = self.w0 + np.dot(x, self.w)
        # Example: 3.5 + (0.3*1 + 0.5*1 + 0.2*1) = 3.5 + 1.0 = 4.5

        # === INTERACTION COMPONENT (Efficient O(kn) formula) ===
        interaction = 0
        for f in range(self.k):
            # S_f = sum_i(v_if * x_i)
            sum_vx = np.dot(x, self.V[:, f])  # Scalar, e.g., 1.5

            # Diagonal correction: sum_i(v_if^2 * x_i^2)
            sum_v2x2 = np.dot(x**2, self.V[:, f]**2)  # Scalar, e.g., 0.35

            # Contribution from this latent dimension
            interaction += 0.5 * (sum_vx**2 - sum_v2x2)
            # Example: 0.5 * (1.5^2 - 0.35) = 0.5 * (2.25 - 0.35) = 0.5 * 1.9 = 0.95

        return linear + interaction  # 4.5 + 0.95 = 5.45

    def predict(self, X):
        """Predict for multiple instances (batch prediction)"""
        return np.array([self.predict_instance(X[i]) for i in range(X.shape[0])])

# === EXAMPLE USAGE ===
if __name__ == "__main__":
    # Toy dataset: 3 users, 3 items
    # Feature vector: [user_0, user_1, user_2, item_0, item_1, item_2]
    #                  <------ users ------>  <------ items ------>

    X = np.array([
        [1, 0, 0, 1, 0, 0],  # User 0 rates Item 0 → rating 5.0
        [1, 0, 0, 0, 1, 0],  # User 0 rates Item 1 → rating 3.0
        [0, 1, 0, 1, 0, 0],  # User 1 rates Item 0 → rating 4.0
        [0, 1, 0, 0, 0, 1],  # User 1 rates Item 2 → rating 5.0
        [0, 0, 1, 0, 1, 0],  # User 2 rates Item 1 → rating 2.0
        [0, 0, 1, 0, 0, 1]   # User 2 rates Item 2 → rating 4.0
    ], dtype=np.float32)

    y = np.array([5.0, 3.0, 4.0, 5.0, 2.0, 4.0])

    # Train FM with k=5 latent factors
    fm = FactorizationMachine(n_factors=5, learning_rate=0.01, reg=0.01, n_epochs=50)
    fm.fit(X, y)

    # Predict: What would User 0 rate Item 2?
    # (This pair was NOT in training data - true test of generalization!)
    test_x = np.array([1, 0, 0, 0, 0, 1], dtype=np.float32)  # User 0, Item 2
    print(f"\nPrediction for User 0, Item 2: {fm.predict_instance(test_x):.2f}")
    # Expected: ~4.5 (User 0 likes things, Item 2 is well-liked)
```

---

### Connection to Next Topic: Why FM Leads to Deep Learning

*Now that you understand FM, you're ready to see its evolution.*

**FM's key insight**: Learn feature interactions via latent factors.

**Limitation**: FM only models **pairwise** interactions. What about:
- User × Item × Time of day?
- Genre × Director × Decade?

**Deep Learning solution**: Stack multiple layers to capture **higher-order** interactions.

**Preview of Week 5**:
- **DeepFM** = FM + deep neural network
- **Neural CF** = Replace dot product with learned neural network
- **Wide & Deep** = Memorization (linear) + Generalization (deep)

*Can you see how* FM is a stepping stone to these models? The latent factor idea persists, but with neural networks providing more expressive interaction functions.

---

## Comparison of Variants

| Variant | Key Feature | Complexity | Use Case |
|---------|-------------|------------|----------|
| **Basic MF** | User/item factors only | $O(k \cdot |\mathcal{K}|)$ | Simple rating prediction |
| **SVD++** | + Implicit feedback | $O(k \cdot |\mathcal{K}| \cdot |N(u)|)$ | Explicit + implicit signals |
| **TimeSVD++** | + Temporal dynamics | $O(k \cdot |\mathcal{K}| \cdot T)$ | Time-varying preferences |
| **FM** | + Arbitrary features | $O(kn \cdot |\mathcal{K}|)$ | Feature-rich contexts |

---

## When to Use What?

### Basic MF
- Simple rating prediction
- Small datasets
- Baseline model

### SVD++
- Explicit + implicit feedback available
- User interaction history is rich
- Netflix-style systems

### TimeSVD++
- Long-term user data (years)
- Seasonal effects matter
- User/item popularity changes over time

### Factorization Machines
- Rich side information (demographics, context)
- Cold start (new users/items with features)
- Click-through rate prediction (ads, search)

---

## Summary

**Key takeaways**:
1. **SVD++** integrates implicit feedback → 4% improvement
2. **TimeSVD++** models temporal dynamics → 2% additional improvement
3. **Factorization Machines** generalize MF to arbitrary features
4. These methods won the Netflix Prize (ensemble of variants)
5. FM is the foundation for modern deep learning models (DeepFM, xDeepFM)

**Historical impact**:
- Dominated RecSys (2008-2015)
- Netflix Prize winners used ensembles of these models
- Paved the way for deep learning approaches

**Modern relevance**:
- Still used in production (faster than deep learning)
- Components integrated into neural models
- Baseline for comparing new methods

---

## References

1. **Koren, Y. (2008)**. "Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model". *KDD*.
   - SVD++ original paper

2. **Koren, Y. (2009)**. "Collaborative Filtering with Temporal Dynamics". *KDD*.
   - TimeSVD++ and Netflix Prize insights

3. **Rendle, S. (2010)**. "Factorization Machines". *IEEE ICDM*.
   - FM framework

4. **Koren, Y. (2009)**. "The BellKor Solution to the Netflix Grand Prize". *Netflix Prize documentation*.
   - Complete description of winning ensemble

5. **Rendle, S. (2012)**. "Factorization Machines with libFM". *ACM TIST*.
   - Practical FM implementation
