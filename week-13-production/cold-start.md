# Week 13: Cold Start Problem

## Overview

**Cold start**: New users/items with no interaction history.

**Types**:
1. **User cold start**: New user, no preferences known
2. **Item cold start**: New item, no interactions yet
3. **System cold start**: New platform, no data

---

## User Cold Start

### Problem

**New user**: No ratings, clicks, or purchases → can't personalize.

**Naive solution**: Show popular items (not personalized).

---

### Solutions

**1. Onboarding Survey**

Ask preferences upfront.

```python
def onboarding_recommendations(selected_genres, selected_items):
    """Recommend based on initial preferences"""
    candidates = []

    # Items from selected genres
    for genre in selected_genres:
        candidates.extend(get_items_by_genre(genre, k=20))

    # Similar to selected items
    for item in selected_items:
        similar = get_similar_items(item, k=10)
        candidates.extend(similar)

    return rank_by_popularity(candidates)[:50]
```

**2. Demographic Matching**

Use demographics (age, location) to find similar users.

```python
def demographic_cold_start(age, location, gender):
    """Find similar users by demographics"""
    similar_users = find_users(age_range=(age-5, age+5),
                                location=location,
                                gender=gender)

    # Aggregate their preferences
    popular_items = get_popular_items_for_users(similar_users)
    return popular_items
```

**3. Hybrid: Content + CF**

Start with content-based, transition to CF as data accumulates.

**4. Active Learning**

Strategically ask for ratings on diverse items.

---

## Item Cold Start

### Problem

**New item**: Just released, no clicks/ratings.

**Challenge**: Can't use CF (no user-item interactions).

---

### Solutions

**1. Content-Based**

Use item features (title, category, description).

```python
def cold_item_recommendations(new_item, user_history):
    """Recommend new item based on content similarity"""
    new_item_features = extract_features(new_item)

    # Find similar items user liked
    scores = []
    for hist_item in user_history:
        hist_features = extract_features(hist_item)
        similarity = cosine_similarity(new_item_features, hist_features)
        scores.append(similarity)

    # If user liked similar items, recommend new item
    avg_similarity = np.mean(scores)
    return avg_similarity > 0.7
```

**2. Exploration**

Show new items to random sample of users (A/B test).

**3. Transfer Learning**

Use pre-trained embeddings (e.g., BERT for text, CLIP for images).

---

## Meta-Learning

### Concept

**Learn to learn**: Model adapts quickly to new users/items with few examples.

**MAML** (Model-Agnostic Meta-Learning):

**Idea**: Train model initialization that adapts fast to new tasks.

**Process**:
1. **Meta-train**: Learn good initialization across many users
2. **Adapt**: Fine-tune on new user's few interactions

---

### Implementation (Simplified)

```python
class MetaRecommender:
    def __init__(self, base_model):
        self.meta_model = base_model  # Meta-learned initialization

    def adapt_to_user(self, user_interactions, n_steps=5):
        """Quickly adapt to new user"""
        user_model = copy.deepcopy(self.meta_model)
        optimizer = torch.optim.SGD(user_model.parameters(), lr=0.01)

        for _ in range(n_steps):
            for item, rating in user_interactions:
                pred = user_model(item)
                loss = (pred - rating) ** 2
                loss.backward()
                optimizer.step()

        return user_model
```

---

## Bandit Approaches

### Exploration for Cold Start

**Multi-armed bandits**: Explore new items, exploit known good ones.

**Thompson Sampling**: Naturally explores new items (high uncertainty).

```python
# New item has uniform prior Beta(1, 1)
# Gets explored more than well-known items
```

---

## Summary

**Key Takeaways**:
1. **User cold start**: Onboarding, demographics, hybrid
2. **Item cold start**: Content-based, exploration, transfer learning
3. **Meta-learning**: MAML for fast adaptation
4. **Bandits**: Natural exploration of new items

**Best Strategy**: Combine multiple approaches
- **Day 1**: Demographics + onboarding
- **Week 1**: Hybrid (content + emerging CF signal)
- **Month 1+**: Full CF

**Next**: Model management and MLOps.

---

## References

1. **Vartak, M., et al. (2017)**. "Meta-Learning for User Cold-Start Recommendation". *IJCNN*.
2. **Finn, C., et al. (2017)**. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks". *ICML*.
3. **Schein, A. I., et al. (2002)**. "Methods and Metrics for Cold-Start Recommendations". *SIGIR*.
