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

### Motivation

**Problem**: How to incorporate side information (features) into MF?

**Examples**:
- User demographics (age, gender, location)
- Item attributes (genre, director, year)
- Context (time of day, device, location)

**Challenge**: Feature interactions are critical!
- "Age × Genre": Young users like action, older users like drama
- "Gender × Director": Women prefer certain directors

---

### The Factorization Machines Framework

**Paper**: Rendle, S. (2010). "Factorization Machines". *IEEE ICDM*.

**Key idea**: Model all pairwise feature interactions with latent factors.

**Prediction formula**:

$$\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

where:
- $\mathbf{x} \in \mathbb{R}^n$: Feature vector (user, item, context)
- $w_0$: Global bias
- $w_i$: Weight for feature $i$
- $\mathbf{v}_i \in \mathbb{R}^k$: Latent factor for feature $i$
- $\langle \mathbf{v}_i, \mathbf{v}_j \rangle = \sum_{f=1}^k v_{if} \cdot v_{jf}$: Interaction between features $i$ and $j$

**Components**:
1. **Linear term**: $\sum w_i x_i$
2. **Pairwise interactions**: $\sum_{i < j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$

---

### Efficient Computation

**Naive complexity**: $O(kn^2)$ for pairwise interactions (expensive!)

**Clever reformulation**:

$$\sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2} \sum_{f=1}^k \left[ \left( \sum_{i=1}^n v_{if} x_i \right)^2 - \sum_{i=1}^n v_{if}^2 x_i^2 \right]$$

**Reduced complexity**: $O(kn)$ (linear in number of features!)

**Proof** (expand and rearrange):
$$\sum_{i < j} v_{if} v_{jf} x_i x_j = \frac{1}{2} \left[ \left( \sum_i v_{if} x_i \right)^2 - \sum_i v_{if}^2 x_i^2 \right]$$

This allows efficient computation even with millions of features!

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

### Implementation

```python
import numpy as np

class FactorizationMachine:
    def __init__(self, n_factors=10, learning_rate=0.01, reg=0.01, n_epochs=20):
        self.k = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs

    def fit(self, X, y):
        """
        X: (n_samples, n_features) feature matrix
        y: (n_samples,) target vector
        """
        n_samples, self.n_features = X.shape

        # Initialize parameters
        self.w0 = 0.0
        self.w = np.zeros(self.n_features)
        self.V = np.random.normal(0, 0.01, (self.n_features, self.k))

        # Training loop
        for epoch in range(self.n_epochs):
            for idx in range(n_samples):
                x = X[idx]
                target = y[idx]

                # Prediction
                pred = self.predict_instance(x)

                # Error
                err = target - pred

                # Update global bias
                self.w0 += self.lr * err

                # Update linear weights
                self.w += self.lr * (err * x - self.reg * self.w)

                # Update factors (using efficient formula)
                # Precompute sum for each factor
                sum_vx = np.dot(x, self.V)  # (k,)

                for i in range(self.n_features):
                    if x[i] != 0:
                        for f in range(self.k):
                            grad = err * (x[i] * sum_vx[f] - self.V[i, f] * x[i]**2) - self.reg * self.V[i, f]
                            self.V[i, f] += self.lr * grad

            # Evaluate
            predictions = np.array([self.predict_instance(X[i]) for i in range(n_samples)])
            rmse = np.sqrt(np.mean((y - predictions)**2))
            print(f"Epoch {epoch+1}/{self.n_epochs}: RMSE = {rmse:.4f}")

    def predict_instance(self, x):
        """Predict for a single instance"""
        # Linear term
        linear = self.w0 + np.dot(x, self.w)

        # Interaction term (efficient formula)
        interaction = 0
        for f in range(self.k):
            sum_vx = np.dot(x, self.V[:, f])
            sum_v2x2 = np.dot(x**2, self.V[:, f]**2)
            interaction += 0.5 * (sum_vx**2 - sum_v2x2)

        return linear + interaction

    def predict(self, X):
        """Predict for multiple instances"""
        return np.array([self.predict_instance(X[i]) for i in range(X.shape[0])])

# Example usage
if __name__ == "__main__":
    # Toy dataset: users (one-hot) + items (one-hot) + ratings
    # 3 users, 3 items
    # Feature vector: [user_0, user_1, user_2, item_0, item_1, item_2]

    X = np.array([
        [1, 0, 0, 1, 0, 0],  # User 0, Item 0
        [1, 0, 0, 0, 1, 0],  # User 0, Item 1
        [0, 1, 0, 1, 0, 0],  # User 1, Item 0
        [0, 1, 0, 0, 0, 1],  # User 1, Item 2
        [0, 0, 1, 0, 1, 0],  # User 2, Item 1
        [0, 0, 1, 0, 0, 1]   # User 2, Item 2
    ])

    y = np.array([5.0, 3.0, 4.0, 5.0, 2.0, 4.0])

    fm = FactorizationMachine(n_factors=5, learning_rate=0.01, reg=0.01, n_epochs=50)
    fm.fit(X, y)

    # Predict
    test_x = np.array([1, 0, 0, 0, 0, 1])  # User 0, Item 2
    print(f"\nPrediction for user 0, item 2: {fm.predict_instance(test_x):.2f}")
```

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
