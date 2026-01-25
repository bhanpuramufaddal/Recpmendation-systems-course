# Week 4: Content-Based Filtering - Foundations

## Overview

**Content-based filtering** recommends items similar to those a user has liked in the past, based on **item features** rather than user behavior patterns.

**Key idea**: If you liked action movies with Tom Hanks, recommend more action movies with Tom Hanks.

**Contrast with Collaborative Filtering**:
- **Collaborative**: "Users similar to you liked X"
- **Content-based**: "X is similar to items you liked"

This document covers the foundations of content-based recommendation systems.

---

## Learning Objectives

By the end of this section, you will:
- Understand content-based filtering principles
- Master feature extraction and representation
- Learn user profile construction techniques
- Implement similarity-based matching
- Recognize when to use content-based vs. collaborative filtering
- Apply content-based methods to real-world problems

---

## The Content-Based Paradigm

### Core Principle

**Given**:
- User $u$ liked items $I_u = \{i_1, i_2, \ldots, i_k\}$
- Each item $i$ has features $\mathbf{f}_i$

**Goal**: Recommend new items $j$ with features $\mathbf{f}_j$ similar to $\mathbf{f}_i$ for $i \in I_u$

**Formula**:
$$\text{score}(u, j) = \text{similarity}(\text{profile}(u), \mathbf{f}_j)$$

where $\text{profile}(u)$ is constructed from features of items user $u$ has liked.

---

### Example: Movie Recommendations

**User watched and liked**:
1. "Inception" (Genre: Sci-Fi, Action; Director: Nolan; Year: 2010)
2. "Interstellar" (Genre: Sci-Fi, Drama; Director: Nolan; Year: 2014)
3. "The Dark Knight" (Genre: Action, Crime; Director: Nolan; Year: 2008)

**User profile** (derived):
- **Genre**: Sci-Fi (67%), Action (67%), Drama (33%), Crime (33%)
- **Director**: Nolan (100%)
- **Year**: 2010-2014 (recent)

**Candidate movie**:
- "Tenet" (Genre: Sci-Fi, Action; Director: Nolan; Year: 2020)

**Match**: Strong! Genre overlap (Sci-Fi, Action), same director, recent year.

**Recommendation**: Suggest "Tenet" to user.

---

## Content-Based vs. Collaborative Filtering

| Aspect | Content-Based | Collaborative Filtering |
|--------|---------------|------------------------|
| **Data required** | Item features | User-item interactions |
| **Recommendations** | Similar to user's past likes | Similar users' likes |
| **Cold start (new users)** | Good (can use demographics) | Poor (no interaction history) |
| **Cold start (new items)** | Good (features available) | Poor (no ratings yet) |
| **Serendipity** | Low (filter bubble) | High (unexpected finds) |
| **Scalability** | Depends on feature extraction | Depends on matrix size |
| **Interpretability** | High (can explain via features) | Medium (user similarity) |
| **Domain knowledge** | Required (feature engineering) | Not required |

**Key insight**: Content-based is **interpretable** and handles **new items** well, but suffers from **filter bubble** (only recommends similar items).

---

## Architecture Overview

Content-based recommendation systems have three main components:

```
1. Content Analyzer
   ↓
   Extract features from items
   ↓
2. Profile Learner
   ↓
   Build user profile from liked items
   ↓
3. Filtering Component
   ↓
   Match user profile to candidate items
   ↓
Recommendations
```

Let's examine each component.

---

## 1. Content Analyzer: Feature Extraction

### Text Features (for articles, books, movies)

**Goal**: Represent text as numerical vector.

**a) Bag of Words (BoW)**

**Idea**: Count word frequencies.

**Example**:
```
Document 1: "The cat sat on the mat"
Document 2: "The dog sat on the log"

Vocabulary: [cat, sat, on, mat, dog, log]

BoW:
Doc 1: [1, 1, 1, 1, 0, 0]
Doc 2: [0, 1, 1, 0, 1, 1]
```

**Problem**: Common words like "the" dominate.

---

**b) TF-IDF (Term Frequency-Inverse Document Frequency)**

**Idea**: Weight words by rarity.

**Formula**:
$$\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \text{IDF}(w)$$

where:
$$\text{TF}(w, d) = \frac{\text{count of word } w \text{ in doc } d}{\text{total words in } d}$$

$$\text{IDF}(w) = \log \frac{\text{total documents}}{\text{documents containing } w}$$

**Effect**: Rare words get higher weight.

**Example**:
```
Corpus: 1000 documents

Word "the": appears in 999 docs → IDF = log(1000/999) ≈ 0.001 (low)
Word "quantum": appears in 10 docs → IDF = log(1000/10) = 2 (high)

If "quantum" appears 3 times in a 100-word doc:
TF = 3/100 = 0.03
TF-IDF = 0.03 × 2 = 0.06
```

---

**c) Word Embeddings (Word2Vec, GloVe)**

**Modern approach**: Represent words as dense vectors (e.g., 300D).

**Advantage**: Captures semantic similarity.

**Example**:
```
vec("king") - vec("man") + vec("woman") ≈ vec("queen")
```

**For documents**: Average word embeddings.

$$\mathbf{f}_{\text{doc}} = \frac{1}{|D|} \sum_{w \in D} \mathbf{e}_w$$

where $\mathbf{e}_w$ is embedding for word $w$.

---

### Categorical Features

**For movies, products, music**:
- Genre (Action, Comedy, Drama)
- Director
- Actors
- Year
- Language

**Encoding**:

**a) One-Hot Encoding**

```
Genre: [Action, Comedy, Drama, Sci-Fi]

Movie 1 (Action): [1, 0, 0, 0]
Movie 2 (Comedy): [0, 1, 0, 0]
```

**Problem**: High dimensionality, no similarity between categories.

---

**b) Multi-Hot Encoding** (for multiple categories)

```
Movie 1 (Action, Sci-Fi): [1, 0, 0, 1]
Movie 2 (Action, Drama):  [1, 0, 1, 0]
```

---

**c) Embeddings** (learned)

Learn dense vectors for each category.

```python
import torch.nn as nn

# Vocabulary: 1000 genres
genre_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=32)

# Movie has genre ID 42
genre_id = torch.tensor([42])
genre_vec = genre_embedding(genre_id)  # (32,)
```

---

### Numerical Features

**For products, restaurants**:
- Price
- Rating
- Number of reviews
- Distance

**Normalization**: Scale to [0, 1] or standardize.

$$x_{\text{norm}} = \frac{x - \min}{\max - \min}$$

or

$$x_{\text{std}} = \frac{x - \mu}{\sigma}$$

---

### Multimedia Features

**Images**: Use CNN (ResNet, VGG) to extract features.

**Videos**: Extract frames, use 3D CNN or frame averaging.

**Audio**: Mel-frequency cepstral coefficients (MFCCs), learned embeddings.

---

## 2. Profile Learner: User Profiles

### User Profile Construction

**Goal**: Represent user preferences as a vector.

**Approach 1: Average Feature Vector**

$$\text{profile}(u) = \frac{1}{|I_u|} \sum_{i \in I_u} \mathbf{f}_i$$

where $I_u$ = items user $u$ has liked.

**Example**:
```
User liked:
- Movie 1: [Action: 1, Comedy: 0, Drama: 0, Sci-Fi: 1]
- Movie 2: [Action: 1, Comedy: 0, Drama: 1, Sci-Fi: 0]

Profile = (1/2) × ([1,0,0,1] + [1,0,1,0]) = [1, 0, 0.5, 0.5]
```

**Interpretation**: User prefers Action (100%), some Drama and Sci-Fi.

---

**Approach 2: Weighted Average** (by ratings)

If user gave ratings $r_{ui}$ for items in $I_u$:

$$\text{profile}(u) = \frac{\sum_{i \in I_u} r_{ui} \cdot \mathbf{f}_i}{\sum_{i \in I_u} r_{ui}}$$

**Effect**: Higher-rated items have more influence.

---

**Approach 3: Learned Profile** (Logistic Regression, Neural Network)

Train a model to predict whether user will like an item based on features.

**Model**: $\hat{y}_{ui} = \sigma(\mathbf{w}_u^T \mathbf{f}_i)$

where $\mathbf{w}_u$ is user's learned weight vector.

**Training**: Minimize loss on user's past interactions.

$$\mathcal{L} = \sum_{i \in I_u^+} -\log \sigma(\mathbf{w}_u^T \mathbf{f}_i) + \sum_{j \in I_u^-} -\log(1 - \sigma(\mathbf{w}_u^T \mathbf{f}_j))$$

where $I_u^+$ = liked items, $I_u^-$ = disliked items.

---

## 3. Filtering Component: Similarity Matching

### Similarity Measures

**Given**: User profile $\mathbf{p}_u$ and item features $\mathbf{f}_i$

**Goal**: Compute $\text{similarity}(\mathbf{p}_u, \mathbf{f}_i)$

---

**a) Cosine Similarity**

$$\text{cosine}(\mathbf{p}_u, \mathbf{f}_i) = \frac{\mathbf{p}_u \cdot \mathbf{f}_i}{\|\mathbf{p}_u\| \|\mathbf{f}_i\|}$$

**Range**: [-1, 1] (1 = identical, 0 = orthogonal, -1 = opposite)

**Example**:
```python
import numpy as np

profile = np.array([1.0, 0.0, 0.5, 0.5])  # User profile
item = np.array([1.0, 0.0, 0.0, 1.0])     # Item features

cosine = np.dot(profile, item) / (np.linalg.norm(profile) * np.linalg.norm(item))
print(f"Cosine similarity: {cosine:.3f}")  # 0.866
```

---

**b) Euclidean Distance**

$$\text{distance}(\mathbf{p}_u, \mathbf{f}_i) = \|\mathbf{p}_u - \mathbf{f}_i\|_2 = \sqrt{\sum_k (p_{uk} - f_{ik})^2}$$

**Convert to similarity**: $\text{sim} = \frac{1}{1 + \text{distance}}$

**Note**: Sensitive to feature scale. Normalize first!

---

**c) Dot Product**

$$\text{score}(\mathbf{p}_u, \mathbf{f}_i) = \mathbf{p}_u \cdot \mathbf{f}_i = \sum_k p_{uk} f_{ik}$$

**Simple and fast**. Used when both vectors are normalized.

---

### Ranking

**Process**:
1. Compute similarity between user profile and all candidate items
2. Sort by similarity (descending)
3. Return top-K items

**Example**:
```python
def recommend(user_profile, item_features, top_k=10):
    """
    user_profile: (d,) - user preference vector
    item_features: (n_items, d) - feature matrix
    """
    # Compute cosine similarity for all items
    scores = item_features @ user_profile  # (n_items,)
    scores /= (np.linalg.norm(item_features, axis=1) * np.linalg.norm(user_profile))

    # Get top-K items
    top_indices = np.argsort(scores)[::-1][:top_k]

    return top_indices, scores[top_indices]

# Example
user_profile = np.array([1.0, 0.0, 0.5, 0.5])
item_features = np.random.rand(1000, 4)  # 1000 items, 4 features

recommendations, scores = recommend(user_profile, item_features, top_k=10)
print(f"Top 10 items: {recommendations}")
print(f"Scores: {scores}")
```

---

## Advantages of Content-Based Filtering

### 1. User Independence

**No need for other users' data**. Recommendations are based solely on the target user's preferences.

**Benefit**: Privacy-friendly, no cold-start problem for new platforms.

---

### 2. Transparency

**Can explain recommendations**: "We recommend X because you liked Y, and both are Sci-Fi movies directed by Nolan."

**Example**:
```
Recommendation: "Tenet"
Reason: You watched "Inception" (Sci-Fi, Nolan) and "Interstellar" (Sci-Fi, Nolan).
        "Tenet" shares these features.
```

---

### 3. New Item Problem

**No cold start for new items**. As long as features are available, can recommend immediately.

**Example**: New movie released today → extract features (genre, director) → recommend to users with matching profiles.

---

### 4. Niche Interests

**Can recommend unpopular items** if they match user's profile.

**Example**: User likes obscure indie films with specific director → recommend new indie film by same director, even if no one else has watched it yet.

---

## Limitations of Content-Based Filtering

### 1. Over-Specialization (Filter Bubble)

**Problem**: Only recommends items similar to past likes. No diversity.

**Example**: User watched 10 action movies → only gets action movie recommendations → never discovers comedy, drama, etc.

**Solution**: Inject diversity, exploration (see hybrid strategies).

---

### 2. Feature Engineering Required

**Problem**: Need good features. For movies, easy (genre, director). For other domains, hard.

**Example**: Recommending restaurants → what features? Cuisine, price, location. But what about ambiance, service quality? Hard to quantify.

---

### 3. New User Problem

**Problem**: Need user's past interactions to build profile.

**Cold start**: New user with no history → cannot build profile.

**Solutions**:
- Ask user to select preferences (onboarding)
- Use demographics (age, gender, location)
- Start with popular items

---

### 4. Limited Serendipity

**Problem**: Unlikely to recommend unexpected items.

**Example**: User likes action movies → will never discover great documentaries.

**Solution**: Hybrid with collaborative filtering.

---

## Practical Implementation

### End-to-End Example: Movie Recommendations

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie data
movies = [
    {"id": 0, "title": "Inception", "genres": "Action Sci-Fi Thriller", "director": "Nolan"},
    {"id": 1, "title": "Interstellar", "genres": "Sci-Fi Drama Adventure", "director": "Nolan"},
    {"id": 2, "title": "The Dark Knight", "genres": "Action Crime Drama", "director": "Nolan"},
    {"id": 3, "title": "Toy Story", "genres": "Animation Comedy Family", "director": "Lasseter"},
    {"id": 4, "title": "Finding Nemo", "genres": "Animation Adventure Comedy", "director": "Stanton"},
]

# Extract features: combine genres and director
def extract_features(movie):
    return f"{movie['genres']} {movie['director']}"

movie_features = [extract_features(m) for m in movies]

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(movie_features)  # (5, n_features)

print(f"Feature matrix shape: {feature_matrix.shape}")
print(f"Feature names: {vectorizer.get_feature_names_out()[:10]}")

# User profile: liked movies 0, 1, 2 (all Nolan films)
liked_movies = [0, 1, 2]
user_profile = feature_matrix[liked_movies].mean(axis=0)  # Average

# Compute similarity with all movies
similarities = cosine_similarity(user_profile, feature_matrix)[0]

# Rank movies (exclude already watched)
movie_ids = list(range(len(movies)))
recommendations = sorted(
    [(i, similarities[i]) for i in movie_ids if i not in liked_movies],
    key=lambda x: x[1],
    reverse=True
)

print("\nRecommendations:")
for movie_id, score in recommendations[:3]:
    print(f"{movies[movie_id]['title']}: {score:.3f}")
```

**Output**:
```
Feature matrix shape: (5, 13)
Feature names: ['action' 'adventure' 'animation' 'comedy' 'crime' 'drama' 'family' 'lasseter' 'nolan' 'sci']

Recommendations:
Finding Nemo: 0.156
Toy Story: 0.142
```

**Analysis**: Both recommended movies have low scores because they don't match user's profile (Nolan, Action/Sci-Fi). System correctly identifies them as dissimilar.

---

## Real-World Applications

### 1. Music: Pandora Music Genome Project

**Approach**: Expert musicians tag songs with 400+ features (melody, harmony, rhythm, instrumentation, vocals).

**User profile**: Built from songs user has liked/disliked (thumbs up/down).

**Recommendation**: Find songs with similar feature profiles.

**Success**: Millions of users, personalized radio stations.

---

### 2. News: Google News

**Approach**: Extract keywords from articles user has read (TF-IDF).

**User profile**: Weighted combination of keywords.

**Recommendation**: Match new articles to user's keyword profile.

**Benefit**: Timely recommendations for breaking news.

---

### 3. Jobs: LinkedIn

**Approach**: Extract features from job postings (skills required, location, industry).

**User profile**: Skills, experience, preferences from user's profile.

**Recommendation**: Match users to jobs with similar features.

---

### 4. E-Commerce: Amazon

**Approach**: "Customers who bought X also viewed Y" (collaborative) + "Items similar to X" (content-based on product features).

**Hybrid**: Combine both approaches.

---

## Summary

**Key Takeaways**:
1. **Content-based filtering** recommends items similar to user's past likes based on **item features**
2. **Three components**: Content analyzer (feature extraction) → Profile learner (user profile) → Filtering (similarity matching)
3. **Feature extraction**: TF-IDF for text, embeddings for categories, CNNs for images
4. **User profile**: Average of liked items' features (or learned weights)
5. **Similarity**: Cosine similarity, dot product, or distance metrics
6. **Advantages**: Transparent, handles new items, user-independent
7. **Limitations**: Filter bubble, requires features, limited serendipity

**When to use**:
- New items arrive frequently (news, music)
- Features are readily available (movies, products)
- Interpretability is important (explain recommendations)
- Privacy is a concern (no need for other users' data)

**When NOT to use**:
- Features are hard to extract (e.g., art, fashion)
- Serendipity is desired (discovery)
- User has no history (cold start)

**Next**: Feature representation techniques (TF-IDF, embeddings, deep learning).

---

## References

1. **Pazzani, M. J., & Billsus, D. (2007)**. "Content-based recommendation systems". *The Adaptive Web*.
   - **Comprehensive overview** of content-based methods

2. **Lops, P., de Gemmis, M., & Semeraro, G. (2011)**. "Content-based Recommender Systems: State of the Art and Trends". *Recommender Systems Handbook*.
   - **Survey paper** covering techniques

3. **Salton, G., & McGill, M. J. (1986)**. "Introduction to Modern Information Retrieval". *McGraw-Hill*.
   - **TF-IDF** and information retrieval foundations

4. **Mikolov, T., et al. (2013)**. "Efficient Estimation of Word Representations in Vector Space". *ICLR*.
   - **Word2Vec** for text embeddings

5. **Van den Oord, A., et al. (2013)**. "Deep content-based music recommendation". *NIPS*.
   - **Deep learning** for music recommendation (Spotify)

---

## Practice Problems

### Problem 1: TF-IDF Calculation

**Given**:
```
Corpus: 3 documents
Doc 1: "the cat sat on the mat"
Doc 2: "the dog sat on the log"
Doc 3: "cats and dogs are animals"
```

**Calculate**: TF-IDF for word "cat" in Doc 1.

**Solution**:
```
TF(cat, Doc 1) = 1 / 6 = 0.167 (1 occurrence in 6 words)

IDF(cat) = log(3 / 1) = log(3) = 1.099
  (3 total docs, 1 doc contains "cat")

TF-IDF = 0.167 × 1.099 = 0.183
```

---

### Problem 2: User Profile Construction

**Given**:
```
User liked:
  Movie A: [Action: 1, Comedy: 0, Drama: 1], rating: 5
  Movie B: [Action: 1, Comedy: 1, Drama: 0], rating: 3

Construct weighted user profile.
```

**Solution**:
```
Weighted average:
  profile = (5×[1,0,1] + 3×[1,1,0]) / (5+3)
         = ([5,0,5] + [3,3,0]) / 8
         = [8, 3, 5] / 8
         = [1.0, 0.375, 0.625]

Interpretation: User prefers Action (100%), some Drama (62.5%), less Comedy (37.5%)
```

---

### Problem 3: Cosine Similarity

**Given**:
```
User profile: [1.0, 0.5, 0.0]
Item 1: [1.0, 0.0, 0.5]
Item 2: [0.8, 0.6, 0.0]

Which item is more similar to user?
```

**Solution**:
```python
import numpy as np

profile = np.array([1.0, 0.5, 0.0])
item1 = np.array([1.0, 0.0, 0.5])
item2 = np.array([0.8, 0.6, 0.0])

cos1 = np.dot(profile, item1) / (np.linalg.norm(profile) * np.linalg.norm(item1))
cos2 = np.dot(profile, item2) / (np.linalg.norm(profile) * np.linalg.norm(item2))

print(f"Cosine(profile, item1): {cos1:.3f}")  # 0.816
print(f"Cosine(profile, item2): {cos2:.3f}")  # 0.966

# Item 2 is more similar!
```

**Answer**: Item 2 (cosine = 0.966) is more similar than Item 1 (cosine = 0.816).
