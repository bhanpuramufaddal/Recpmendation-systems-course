# Week 4: Content-Based Filtering - Profile Learning

## Overview

**User profile learning** is the process of constructing a representation of user preferences from their past interactions with items.

**Goal**: Build a profile vector $\mathbf{p}_u$ that captures user $u$'s preferences such that items with features similar to $\mathbf{p}_u$ will be liked by the user.

This document covers techniques for learning accurate user profiles from **explicit feedback** (ratings), **implicit feedback** (clicks, views), and handling **temporal dynamics** (changing preferences).

---

## Learning Objectives

By the end of this section, you will:
- Master user profile construction from explicit and implicit feedback
- Implement weighted averaging and learned models
- Handle temporal dynamics and preference drift
- Solve cold start problems for new users
- Apply active learning for profile refinement
- Optimize profiles for production systems

---

## Profile Learning Fundamentals

### The Profile Learning Problem

**Given**:
- User $u$ has interacted with items $I_u = \{i_1, i_2, \ldots, i_k\}$
- Each item $i$ has feature vector $\mathbf{f}_i \in \mathbb{R}^d$
- User's feedback: ratings $r_{ui}$, clicks, views, etc.

**Goal**: Learn profile $\mathbf{p}_u \in \mathbb{R}^d$ such that:
$$\text{score}(u, i) = \mathbf{p}_u^T \mathbf{f}_i$$

predicts user's interest in item $i$.

---

### Types of Feedback

**1. Explicit Feedback**:
- **Ratings**: 1-5 stars, thumbs up/down
- **Reviews**: Text feedback
- **Likes/dislikes**: Binary preference

**Pros**: Clear signal of preference
**Cons**: Sparse (users rarely rate), biased (selection bias)

---

**2. Implicit Feedback**:
- **Clicks**: User clicked on item
- **Views**: User viewed item
- **Watch time**: How long user watched video
- **Purchases**: User bought product
- **Saves/bookmarks**: User saved item

**Pros**: Abundant, no effort required
**Cons**: Ambiguous (view ≠ like), noisy (accidental clicks)

---

## Profile Construction Methods

### 1. Simple Averaging

**Approach**: Average feature vectors of liked items.

$$\mathbf{p}_u = \frac{1}{|I_u^+|} \sum_{i \in I_u^+} \mathbf{f}_i$$

where $I_u^+$ = items user liked (e.g., rated ≥ 4 stars).

**Example**:
```python
import numpy as np

# User liked items 0, 2, 5
liked_items = [0, 2, 5]

# Item features (6 items, 4 features)
item_features = np.array([
    [1.0, 0.0, 0.5, 0.5],  # Item 0
    [0.0, 1.0, 0.0, 0.3],  # Item 1
    [1.0, 0.0, 0.6, 0.4],  # Item 2
    [0.5, 0.5, 0.0, 0.0],  # Item 3
    [0.0, 0.0, 1.0, 0.0],  # Item 4
    [0.9, 0.1, 0.5, 0.6],  # Item 5
])

# Compute user profile
user_profile = item_features[liked_items].mean(axis=0)
print(f"User profile: {user_profile}")
# Output: [0.967 0.033 0.533 0.5  ]
```

**Interpretation**: User prefers feature 0 (97%), feature 2 (53%), feature 3 (50%), not feature 1 (3%).

**Limitation**: All liked items weighted equally (doesn't account for ratings).

---

### 2. Weighted Averaging

**Approach**: Weight by rating or importance.

$$\mathbf{p}_u = \frac{\sum_{i \in I_u} r_{ui} \cdot \mathbf{f}_i}{\sum_{i \in I_u} r_{ui}}$$

where $r_{ui}$ = user's rating for item $i$.

**Example**:
```python
# User's ratings (5-star scale)
ratings = {0: 5, 2: 3, 5: 4}  # Item: Rating

# Weighted average
numerator = sum(ratings[i] * item_features[i] for i in ratings)
denominator = sum(ratings.values())
weighted_profile = numerator / denominator

print(f"Weighted profile: {weighted_profile}")
# Output: [0.917 0.025 0.525 0.508]
```

**Difference**: Item 0 (rating 5) has more influence than Item 2 (rating 3).

---

### 3. Rocchio Algorithm

**Classic from information retrieval**.

$$\mathbf{p}_u = \alpha \frac{1}{|I_u^+|} \sum_{i \in I_u^+} \mathbf{f}_i - \beta \frac{1}{|I_u^-|} \sum_{j \in I_u^-} \mathbf{f}_j$$

where:
- $I_u^+$ = liked items
- $I_u^-$ = disliked items
- $\alpha, \beta$ = weights (typically $\alpha=1$, $\beta=0.25$)

**Intuition**: Move profile toward liked items, away from disliked items.

**Example**:
```python
# User liked items 0, 5
liked = [0, 5]

# User disliked items 1, 3
disliked = [1, 3]

# Rocchio
alpha, beta = 1.0, 0.25
pos_centroid = item_features[liked].mean(axis=0)
neg_centroid = item_features[disliked].mean(axis=0)

rocchio_profile = alpha * pos_centroid - beta * neg_centroid
print(f"Rocchio profile: {rocchio_profile}")
# Output: [0.825  -0.188  0.5    0.55 ]
```

**Note**: Negative values indicate "anti-preferences" (avoid these features).

---

### 4. Logistic Regression

**Supervised learning approach**: Predict user's rating from item features.

**Model**:
$$P(y_{ui} = 1 | \mathbf{f}_i) = \sigma(\mathbf{w}_u^T \mathbf{f}_i + b_u)$$

where $\mathbf{w}_u$ is user's learned weight vector (= profile).

**Training**:
$$\min_{\mathbf{w}_u, b_u} \sum_{i \in I_u} \mathcal{L}(y_{ui}, \sigma(\mathbf{w}_u^T \mathbf{f}_i + b_u)) + \lambda \|\mathbf{w}_u\|^2$$

where $\mathcal{L}$ = cross-entropy loss, $\lambda$ = regularization.

**Implementation**:
```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Training data
X_train = []  # Item features
y_train = []  # User's feedback (1 = liked, 0 = disliked)

# User's history
user_interactions = {
    0: 1,  # Liked item 0
    1: 0,  # Disliked item 1
    2: 1,  # Liked item 2
    3: 0,  # Disliked item 3
    5: 1,  # Liked item 5
}

for item_id, label in user_interactions.items():
    X_train.append(item_features[item_id])
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Train logistic regression
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# User profile = learned weights
user_profile_lr = model.coef_[0]
print(f"Learned profile: {user_profile_lr}")
print(f"Intercept: {model.intercept_[0]}")

# Predict for new item
new_item = item_features[4]
prob_like = model.predict_proba([new_item])[0][1]
print(f"Probability user likes item 4: {prob_like:.3f}")
```

**Advantage**: Can handle non-linear relationships (with kernel methods or neural networks).

---

### 5. Neural Network Profile Learning

**Deep learning approach**: Learn user embedding.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class UserProfileLearner(nn.Module):
    def __init__(self, feature_dim, profile_dim=64):
        super().__init__()
        self.profile_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, profile_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(profile_dim + feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user_history, item_features):
        """
        user_history: (batch, n_items, feature_dim) - items user interacted with
        item_features: (batch, feature_dim) - candidate item
        """
        # Aggregate user history (mean pooling)
        user_profile = user_history.mean(dim=1)  # (batch, feature_dim)

        # Learn profile embedding
        profile_emb = self.profile_net(user_profile)  # (batch, profile_dim)

        # Combine profile and item
        combined = torch.cat([profile_emb, item_features], dim=1)

        # Predict
        score = self.predictor(combined)  # (batch, 1)
        return score.squeeze()

# Example usage
model = UserProfileLearner(feature_dim=4, profile_dim=16)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop
for epoch in range(100):
    # Sample user history and target item
    user_history = torch.tensor([item_features[liked]]).float()  # (1, 3, 4)
    target_item = torch.tensor([item_features[4]]).float()  # (1, 4)
    label = torch.tensor([0.0])  # Didn't like

    # Forward
    pred = model(user_history, target_item)
    loss = criterion(pred, label)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training complete!")
```

**Advantage**: Can learn complex, non-linear user preferences.

---

## Handling Implicit Feedback

### Interpreting Implicit Signals

**Problem**: Click ≠ Like. User may have clicked accidentally, or disliked after viewing.

**Solution**: Use multiple signals and weight by confidence.

---

### Confidence Weighting

**Idea**: Weight interactions by confidence of preference.

**Example signals**:
- **Click**: Low confidence (user might not like)
- **View duration**: Medium confidence (longer view = more interest)
- **Purchase**: High confidence (user definitely interested)
- **Repeat visits**: Very high confidence

**Formula**:
$$\mathbf{p}_u = \frac{\sum_{i \in I_u} c_{ui} \cdot \mathbf{f}_i}{\sum_{i \in I_u} c_{ui}}$$

where $c_{ui}$ = confidence score for interaction with item $i$.

**Confidence function** (for watch time):
$$c_{ui} = \begin{cases}
1 + \alpha \log(1 + t_{ui}) & \text{if } t_{ui} > 0 \\
0 & \text{otherwise}
\end{cases}$$

where $t_{ui}$ = watch time in seconds, $\alpha$ = scaling factor.

**Example**:
```python
# User interactions with confidence
interactions = [
    {"item": 0, "watch_time": 120},   # Watched 2 minutes
    {"item": 2, "watch_time": 30},    # Watched 30 seconds
    {"item": 5, "watch_time": 300},   # Watched 5 minutes (high interest)
]

# Confidence function
alpha = 0.5
def confidence(watch_time):
    return 1 + alpha * np.log(1 + watch_time)

# Weighted profile
numerator = np.zeros(4)
denominator = 0

for interaction in interactions:
    item_id = interaction["item"]
    c = confidence(interaction["watch_time"])
    numerator += c * item_features[item_id]
    denominator += c

implicit_profile = numerator / denominator
print(f"Implicit feedback profile: {implicit_profile}")
```

---

### Negative Sampling for Implicit Feedback

**Problem**: Only positive signals (clicks, views). No explicit negatives.

**Assumption**: Items user didn't interact with → likely not interested (weak negative signal).

**Approach**: Sample unobserved items as negatives.

```python
# Positive items (user clicked)
positive_items = [0, 2, 5]

# Negative samples (items user didn't click, randomly sampled)
all_items = set(range(len(item_features)))
negative_candidates = list(all_items - set(positive_items))
negative_items = np.random.choice(negative_candidates, size=5, replace=False)

print(f"Positive: {positive_items}")
print(f"Negative samples: {negative_items}")

# Train model with positive (label=1) and negative (label=0) samples
X_train = np.vstack([
    item_features[positive_items],
    item_features[negative_items]
])
y_train = np.array([1]*len(positive_items) + [0]*len(negative_items))

# Logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

user_profile_implicit = model.coef_[0]
print(f"Profile from implicit feedback: {user_profile_implicit}")
```

---

## Temporal Dynamics

### Preference Drift

**Observation**: User preferences change over time.

**Examples**:
- User watched action movies in 2020, now prefers documentaries
- User's music taste evolved from pop to jazz
- User's shopping preferences change with seasons

**Challenge**: Recent interactions more relevant than old interactions.

---

### Time Decay

**Solution**: Weight recent interactions more heavily.

$$\mathbf{p}_u(t) = \frac{\sum_{i \in I_u} w(t - t_i) \cdot r_{ui} \cdot \mathbf{f}_i}{\sum_{i \in I_u} w(t - t_i) \cdot r_{ui}}$$

where:
- $t$ = current time
- $t_i$ = time of interaction with item $i$
- $w(\Delta t)$ = decay function

**Exponential decay**:
$$w(\Delta t) = e^{-\lambda \Delta t}$$

**Example**:
```python
import datetime

# User interactions with timestamps
interactions = [
    {"item": 0, "rating": 5, "date": "2024-01-01"},
    {"item": 2, "rating": 4, "date": "2024-06-01"},
    {"item": 5, "rating": 5, "date": "2024-12-01"},
]

# Current date
current_date = datetime.datetime.strptime("2025-01-01", "%Y-%m-%d")

# Time decay function
lambda_decay = 0.01  # Decay rate (per day)

def time_weight(interaction_date, current_date, lambda_decay):
    delta_days = (current_date - datetime.datetime.strptime(interaction_date, "%Y-%m-%d")).days
    return np.exp(-lambda_decay * delta_days)

# Time-weighted profile
numerator = np.zeros(4)
denominator = 0

for interaction in interactions:
    item_id = interaction["item"]
    rating = interaction["rating"]
    weight = time_weight(interaction["date"], current_date, lambda_decay)

    numerator += weight * rating * item_features[item_id]
    denominator += weight * rating

temporal_profile = numerator / denominator
print(f"Time-weighted profile: {temporal_profile}")
```

**Effect**: Recent interaction (Dec 2024) has weight ≈ 0.97, old interaction (Jan 2024) has weight ≈ 0.05.

---

### Session-Based Profiles

**Observation**: User's intent varies within a session.

**Example**:
- Morning session: Read news articles
- Evening session: Watch entertainment videos

**Solution**: Maintain **short-term profile** (current session) + **long-term profile** (all history).

**Combined profile**:
$$\mathbf{p}_u = (1 - \alpha) \mathbf{p}_{\text{long-term}} + \alpha \mathbf{p}_{\text{session}}$$

where $\alpha$ controls short-term vs. long-term balance (e.g., $\alpha = 0.3$).

---

## Cold Start: New Users

### The New User Problem

**Challenge**: No interaction history → cannot build profile.

**Solutions**:

---

### 1. Onboarding Questionnaire

**Ask user to select preferences** during signup.

**Example**:
```
"Select 3 genres you like:"
☐ Action
☐ Comedy
☐ Drama
☑ Sci-Fi
☐ Romance
...

"Select directors you enjoy:"
☑ Christopher Nolan
☐ Quentin Tarantino
☑ Denis Villeneuve
...
```

**Build initial profile** from selections:
```python
# User selected Sci-Fi and Action genres, and specific directors
selected_genres = ["Sci-Fi", "Action"]
selected_directors = ["Nolan", "Villeneuve"]

# Find items matching selections
matching_items = [i for i in range(len(items))
                  if items[i]["genre"] in selected_genres
                  or items[i]["director"] in selected_directors]

# Average features of matching items
initial_profile = item_features[matching_items].mean(axis=0)
```

---

### 2. Demographic Defaults

**Use age, gender, location** to predict initial preferences.

**Example**:
```python
# Demographic clusters (learned from existing users)
demographic_profiles = {
    ("18-24", "male", "US"): [0.8, 0.2, 0.6, 0.4],    # Action, Sci-Fi
    ("18-24", "female", "US"): [0.3, 0.7, 0.5, 0.6],  # Comedy, Drama
    ("45-54", "male", "US"): [0.5, 0.3, 0.2, 0.8],    # Drama, Documentary
}

# New user
new_user_demo = ("18-24", "male", "US")
initial_profile = demographic_profiles.get(new_user_demo, default_profile)
```

---

### 3. Popular Items

**Start with globally popular items** (safe bet for new users).

**Gradually personalize** as user interacts.

```python
# Show popular items initially
popular_items = [2, 5, 7, 10]  # Items with most interactions

# After 5+ interactions, switch to personalized
if len(user_interactions) >= 5:
    recommendations = personalized_recommend(user_profile, item_features)
else:
    recommendations = popular_items
```

---

### 4. Active Learning

**Strategically ask for feedback** on diverse items to learn profile quickly.

**Goal**: Maximize information gain from each question.

**Approach**: Select items that maximally reduce uncertainty about user's preferences.

**Example** (simplified):
```python
# Select diverse items spanning different feature clusters
from sklearn.cluster import KMeans

# Cluster items
kmeans = KMeans(n_clusters=5)
item_clusters = kmeans.fit_predict(item_features)

# Select one item from each cluster
diverse_items = []
for cluster_id in range(5):
    cluster_items = np.where(item_clusters == cluster_id)[0]
    diverse_items.append(np.random.choice(cluster_items))

# Ask user to rate these items
print(f"Please rate these items: {diverse_items}")
```

**Benefit**: Learn user's profile with fewer interactions.

---

## Profile Updates

### Incremental Updates

**Problem**: Recomputing profile from scratch after each interaction is expensive.

**Solution**: Incremental update.

**Moving average**:
$$\mathbf{p}_u^{(t+1)} = \frac{n \cdot \mathbf{p}_u^{(t)} + r_{ui} \cdot \mathbf{f}_i}{n + 1}$$

where $n$ = number of interactions so far.

**Example**:
```python
class IncrementalProfile:
    def __init__(self, feature_dim):
        self.profile = np.zeros(feature_dim)
        self.n_interactions = 0

    def update(self, item_features, rating=1.0):
        """Incrementally update profile."""
        self.profile = (self.n_interactions * self.profile + rating * item_features) / (self.n_interactions + 1)
        self.n_interactions += 1

    def get_profile(self):
        return self.profile

# Usage
profile_tracker = IncrementalProfile(feature_dim=4)

# User likes item 0
profile_tracker.update(item_features[0], rating=5)
print(f"Profile after 1 interaction: {profile_tracker.get_profile()}")

# User likes item 5
profile_tracker.update(item_features[5], rating=4)
print(f"Profile after 2 interactions: {profile_tracker.get_profile()}")
```

---

## Multi-Level Profiles

### Hierarchical Preferences

**Observation**: Users have preferences at different granularities.

**Example** (movies):
- **Genre level**: Likes Sci-Fi, dislikes Romance
- **Director level**: Likes Nolan films
- **Actor level**: Likes Tom Hanks movies

**Approach**: Maintain multiple profiles.

```python
user_profiles = {
    "genre": [0.9, 0.1, 0.5, 0.3],      # Sci-Fi, Action, Drama, Comedy
    "director": [0.8, 0.2, 0.1, 0.0],   # Nolan, Spielberg, Tarantino, etc.
    "actor": [0.6, 0.4, 0.3, 0.2],      # Hanks, DiCaprio, etc.
}

# Score item by combining profiles
def hierarchical_score(item, user_profiles, weights):
    score = 0
    for level, profile in user_profiles.items():
        similarity = cosine_similarity(profile, item[level])
        score += weights[level] * similarity
    return score

weights = {"genre": 0.5, "director": 0.3, "actor": 0.2}
```

---

## Summary

**Key Takeaways**:
1. **Simple methods**: Average, weighted average, Rocchio (fast, interpretable)
2. **Learned methods**: Logistic regression, neural networks (more accurate, complex)
3. **Implicit feedback**: Confidence weighting, negative sampling
4. **Temporal dynamics**: Time decay, session-based profiles
5. **Cold start**: Onboarding, demographics, popular items, active learning
6. **Incremental updates**: Efficient online learning
7. **Multi-level profiles**: Hierarchical preferences

**Best Practices**:
- Start simple (weighted average)
- Add complexity if needed (learned models)
- Handle temporal changes (decay old interactions)
- Bootstrap new users (onboarding, demographics)
- Update incrementally (online learning)

**Next**: Hybrid strategies (combining content-based with collaborative filtering).

---

## References

1. **Pazzani, M. J., & Billsus, D. (2007)**. "Content-based recommendation systems". *The Adaptive Web*.
   - User profile learning techniques

2. **Hu, Y., Koren, Y., & Volinsky, C. (2008)**. "Collaborative Filtering for Implicit Feedback Datasets". *ICDM*.
   - **Confidence weighting** for implicit feedback

3. **Ding, Y., & Li, X. (2005)**. "Time weight collaborative filtering". *CIKM*.
   - **Temporal dynamics** in user profiles

4. **Elahi, M., Ricci, F., & Rubens, N. (2016)**. "A survey of active learning in collaborative filtering recommender systems". *Computer Science Review*.
   - **Active learning** for cold start

5. **Rendle, S., et al. (2009)**. "BPR: Bayesian Personalized Ranking from Implicit Feedback". *UAI*.
   - **Negative sampling** for implicit feedback

---

## Practice Problems

### Problem 1: Weighted Profile

**Given**:
```
User ratings:
  Item 0: 5 stars, features [1.0, 0.0, 0.5]
  Item 2: 3 stars, features [0.8, 0.2, 0.6]
  Item 5: 4 stars, features [0.9, 0.1, 0.4]

Compute weighted user profile.
```

**Solution**:
```python
ratings = {0: 5, 2: 3, 5: 4}
features = {
    0: np.array([1.0, 0.0, 0.5]),
    2: np.array([0.8, 0.2, 0.6]),
    5: np.array([0.9, 0.1, 0.4])
}

numerator = sum(ratings[i] * features[i] for i in ratings)
denominator = sum(ratings.values())
profile = numerator / denominator

print(profile)  # [0.925 0.05  0.475]
```

---

### Problem 2: Time Decay

**Given**:
```
Interactions:
  Item 0: 100 days ago, rating 5
  Item 5: 10 days ago, rating 4

Decay rate: λ = 0.01 per day

Compute time-weighted profile.
```

**Solution**:
```python
lambda_decay = 0.01
weight_0 = np.exp(-lambda_decay * 100)  # 0.368
weight_5 = np.exp(-lambda_decay * 10)   # 0.905

numerator = weight_0 * 5 * features[0] + weight_5 * 4 * features[5]
denominator = weight_0 * 5 + weight_5 * 4

profile = numerator / denominator
print(profile)
```

---

### Problem 3: Cold Start

**Task**: New user, age 22, male, located in US. Assign initial profile using demographic defaults.

**Solution**:
```python
demographic_profiles = {
    ("18-24", "male", "US"): [0.8, 0.2, 0.6],
    ("25-34", "male", "US"): [0.6, 0.4, 0.5],
    ("18-24", "female", "US"): [0.4, 0.6, 0.7],
}

new_user_demo = ("18-24", "male", "US")
initial_profile = demographic_profiles[new_user_demo]
print(f"Initial profile: {initial_profile}")
# [0.8, 0.2, 0.6]
```
