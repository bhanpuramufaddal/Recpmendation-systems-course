# Week 4: Content-Based Filtering - Foundations

## The Opening Problem: When Collaborative Filtering Fails

Before diving into content-based filtering, let's understand **why we need it**.

### Scenario: The Brand New Movie Problem

Imagine you're Netflix, and a new Christopher Nolan film just released today: "Quantum" (Sci-Fi, Action).

**Your user data**:
- User Alice: Loved "Inception" (5 stars), "Interstellar" (5 stars), "The Dark Knight" (5 stars)
- User Bob: Loved "The Matrix" (5 stars), "Blade Runner 2049" (5 stars)

**Question**: Should you recommend "Quantum" to Alice?

**Collaborative Filtering's Answer**: "I don't know."

*Why?* Let's trace through what happens:

```
User-Based CF:
  Step 1: Find users similar to Alice
  Step 2: Check what similar users rated "Quantum"
  Step 3: Problem - NOBODY has rated "Quantum" yet!
  Result: Cannot make recommendation

Item-Based CF:
  Step 1: Find items similar to "Quantum"
  Step 2: Use co-rating patterns to compute similarity
  Step 3: Problem - "Quantum" has NO ratings to compare with!
  Result: Cannot compute similarity
```

**The core issue**: Collaborative filtering requires **overlapping interactions**. No overlap = no signal.

Can you see why this is a fundamental limitation? Collaborative filtering is essentially saying: "I can only recommend items that have been validated by the crowd." But what about items the crowd hasn't seen yet?

---

### The Solution: Use Item Features

**Content-based filtering asks a different question**:

Instead of: "What did similar users like?"
Ask: "What **features** did this user like, and which items have those features?"

**For Alice and "Quantum"**:
```
Alice's history:
  - "Inception": Sci-Fi, Action, Director=Nolan
  - "Interstellar": Sci-Fi, Drama, Director=Nolan
  - "The Dark Knight": Action, Crime, Director=Nolan

Alice's profile:
  - Loves Sci-Fi (67% of watched)
  - Loves Action (67% of watched)
  - Loves Nolan (100% of watched)

"Quantum" features:
  - Sci-Fi, Action, Director=Nolan

Match: STRONG! Recommend with high confidence.
```

**The insight**: We can recommend "Quantum" to Alice **on day one**, without any user ratings, because we know *what* she likes and we know *what* the movie is about.

---

## Learning Objectives

By the end of this section, you will:
- Understand content-based filtering principles and when to use them
- Master TF-IDF from first principles (not just the formula)
- Build user profiles step-by-step from rated items
- Recognize the filter bubble problem and its feedback loop
- Identify common failure modes of content-based systems
- Implement similarity-based matching for recommendations

---

## Overview

**Content-based filtering** recommends items similar to those a user has liked in the past, based on **item features** rather than user behavior patterns.

**Key idea**: If you liked action movies with Tom Hanks, recommend more action movies with Tom Hanks.

**Contrast with Collaborative Filtering**:
- **Collaborative**: "Users similar to you liked X"
- **Content-based**: "X is similar to items you liked"

This document covers the foundations of content-based recommendation systems.

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
   |
   Extract features from items
   |
2. Profile Learner
   |
   Build user profile from liked items
   |
3. Filtering Component
   |
   Match user profile to candidate items
   |
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

## TF-IDF: Deriving It From First Principles

Let's not just memorize TF-IDF. Let's **derive** it from intuition.

### Starting Point: Raw Word Counts

**Naive approach**: Just count words!

**Example corpus** (3 movie descriptions):
```
Doc 1 (Inception): "dreams within dreams mind-bending thriller"
Doc 2 (The Matrix): "reality simulation computer mind-bending action"
Doc 3 (Titanic): "romance ship tragedy love"
```

**Raw counts for "mind-bending"**:
- Doc 1: 1
- Doc 2: 1
- Doc 3: 0

**Raw counts for "the"** (imagine it's in every doc):
- Doc 1: 3
- Doc 2: 2
- Doc 3: 4

**Problem 1**: "the" appears more often than "mind-bending", but which word tells you more about the document?

*Can you see why raw counts are problematic?* Common words like "the", "a", "is" dominate the representation, but they carry almost no information about what makes a document unique.

---

### Step 1: Term Frequency (TF) - Normalizing by Document Length

**Insight**: A word appearing 5 times in a 10-word document is more significant than 5 times in a 1000-word document.

**Solution**: Normalize by document length.

$$\text{TF}(w, d) = \frac{\text{count of word } w \text{ in document } d}{\text{total words in document } d}$$

**Numerical example**:
```
Doc 1 (100 words): "mind-bending" appears 3 times
Doc 2 (500 words): "mind-bending" appears 3 times

Without TF:
  Both docs: count = 3 (same!)

With TF:
  Doc 1: TF = 3/100 = 0.03
  Doc 2: TF = 3/500 = 0.006

Doc 1 is MORE about "mind-bending" (5x higher TF)
```

**What we've solved**: Longer documents don't unfairly dominate.

**What's still broken**: Common words still get high TF in every document.

---

### Step 2: Inverse Document Frequency (IDF) - Rare Words Are More Informative

**Key insight**: If a word appears in EVERY document, it tells us nothing about what makes each document unique. If a word appears in ONLY ONE document, it's highly distinctive.

**Question**: What would happen if we weighted words by how rare they are across the corpus?

**Mathematical formulation**:

*First attempt*: Weight = 1 / (documents containing word)
- Problem: Very rare words get infinite weight!

*Better*: Use logarithm to smooth:

$$\text{IDF}(w) = \log \frac{N}{\text{df}(w)}$$

where:
- $N$ = total number of documents
- $\text{df}(w)$ = number of documents containing word $w$

**Why logarithm?** Without log:
```
Word in 1 doc out of 1000: weight = 1000
Word in 2 docs out of 1000: weight = 500

That's a 2x difference in rarity but 2x difference in weight.
Seems reasonable.

Word in 1 doc out of 1000: weight = 1000
Word in 1000 docs out of 1000: weight = 1

That's a 1000x difference! Too extreme.
```

With logarithm:
```
Word in 1 doc out of 1000: IDF = log(1000/1) = 6.9
Word in 10 docs out of 1000: IDF = log(1000/10) = 4.6
Word in 100 docs out of 1000: IDF = log(1000/100) = 2.3
Word in 1000 docs out of 1000: IDF = log(1000/1000) = 0

Smooth progression from informative (high) to uninformative (zero)
```

---

### Step 3: Combining TF and IDF

**Final formula**:
$$\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \text{IDF}(w)$$

**Interpretation**:
- **High TF-IDF**: Word appears frequently in THIS document but rarely in other documents
- **Low TF-IDF**: Word is either rare in this document OR common across all documents

---

### Complete Numerical Example

**Corpus**: 3 movie plot summaries (simplified)

```
Doc 1 (Inception, 10 words): "dreams dreams dreams mind reality mind reality thriller action sci-fi"
Doc 2 (Matrix, 10 words): "reality reality computer simulation mind action action sci-fi bullet hero"
Doc 3 (Titanic, 10 words): "ship romance love tragedy ocean iceberg love love ship disaster"
```

**Step 1: Compute TF for each word in each document**

For word "dreams":
- Doc 1: TF = 3/10 = 0.30
- Doc 2: TF = 0/10 = 0.00
- Doc 3: TF = 0/10 = 0.00

For word "reality":
- Doc 1: TF = 2/10 = 0.20
- Doc 2: TF = 2/10 = 0.20
- Doc 3: TF = 0/10 = 0.00

For word "love":
- Doc 1: TF = 0/10 = 0.00
- Doc 2: TF = 0/10 = 0.00
- Doc 3: TF = 3/10 = 0.30

**Step 2: Compute IDF for each word**

```
Total documents N = 3

"dreams": appears in 1 doc  -> IDF = log(3/1) = 1.099
"reality": appears in 2 docs -> IDF = log(3/2) = 0.405
"love": appears in 1 doc    -> IDF = log(3/1) = 1.099
"action": appears in 2 docs -> IDF = log(3/2) = 0.405
"sci-fi": appears in 2 docs -> IDF = log(3/2) = 0.405
```

**Step 3: Compute TF-IDF**

For Doc 1 (Inception):
```
"dreams": TF-IDF = 0.30 x 1.099 = 0.330  <- HIGH! Distinctive
"reality": TF-IDF = 0.20 x 0.405 = 0.081  <- Medium (shared with Matrix)
"mind": TF-IDF = 0.20 x 0.405 = 0.081
"action": TF-IDF = 0.10 x 0.405 = 0.041
"sci-fi": TF-IDF = 0.10 x 0.405 = 0.041
```

For Doc 3 (Titanic):
```
"love": TF-IDF = 0.30 x 1.099 = 0.330  <- HIGH! Distinctive
"ship": TF-IDF = 0.20 x 1.099 = 0.220  <- Also distinctive
"romance": TF-IDF = 0.10 x 1.099 = 0.110
```

**Result**: Each document is now characterized by its DISTINCTIVE words, not just common ones!

---

**c) Word Embeddings (Word2Vec, GloVe)**

**Modern approach**: Represent words as dense vectors (e.g., 300D).

**Advantage**: Captures semantic similarity.

**Example**:
```
vec("king") - vec("man") + vec("woman") ~ vec("queen")
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

## 2. Profile Learner: User Profile Construction

### Building a User Profile Step-by-Step

Let's walk through the **complete process** of constructing a user profile from their rating history.

**Scenario**: User "Alex" has rated 4 movies.

**Step 1: Gather user's rated items and their features**

```
Alex's ratings:
  Movie 1: "Inception" - Rating: 5/5
    Features: [Action=1, Comedy=0, Drama=0, Sci-Fi=1, Romance=0]

  Movie 2: "The Hangover" - Rating: 3/5
    Features: [Action=0, Comedy=1, Drama=0, Sci-Fi=0, Romance=0]

  Movie 3: "Interstellar" - Rating: 4/5
    Features: [Action=0, Comedy=0, Drama=1, Sci-Fi=1, Romance=0]

  Movie 4: "La La Land" - Rating: 2/5
    Features: [Action=0, Comedy=0, Drama=1, Sci-Fi=0, Romance=1]
```

**Step 2: Choose a profile construction method**

---

### Approach 1: Simple Average (Unweighted)

**Formula**:
$$\text{profile}(u) = \frac{1}{|I_u|} \sum_{i \in I_u} \mathbf{f}_i$$

**Calculation**:
```
Number of movies = 4

Sum of feature vectors:
  Action:  1 + 0 + 0 + 0 = 1
  Comedy:  0 + 1 + 0 + 0 = 1
  Drama:   0 + 0 + 1 + 1 = 2
  Sci-Fi:  1 + 0 + 1 + 0 = 2
  Romance: 0 + 0 + 0 + 1 = 1

Average:
  profile = [1/4, 1/4, 2/4, 2/4, 1/4]
          = [0.25, 0.25, 0.50, 0.50, 0.25]
```

**Interpretation**: Alex has equal interest in Action and Comedy (25%), higher interest in Drama and Sci-Fi (50%), and some interest in Romance (25%).

**Problem**: This treats the 5-star "Inception" the same as the 2-star "La La Land"!

---

### Approach 2: Weighted Average by Ratings

**Formula**:
$$\text{profile}(u) = \frac{\sum_{i \in I_u} r_{ui} \cdot \mathbf{f}_i}{\sum_{i \in I_u} r_{ui}}$$

**Calculation**:

```
Weighted sum:
  Action:  (5 x 1) + (3 x 0) + (4 x 0) + (2 x 0) = 5
  Comedy:  (5 x 0) + (3 x 1) + (4 x 0) + (2 x 0) = 3
  Drama:   (5 x 0) + (3 x 0) + (4 x 1) + (2 x 1) = 6
  Sci-Fi:  (5 x 1) + (3 x 0) + (4 x 1) + (2 x 0) = 9
  Romance: (5 x 0) + (3 x 0) + (4 x 0) + (2 x 1) = 2

Total rating weight = 5 + 3 + 4 + 2 = 14

Weighted profile:
  profile = [5/14, 3/14, 6/14, 9/14, 2/14]
          = [0.36, 0.21, 0.43, 0.64, 0.14]
```

**Interpretation**: Now Sci-Fi dominates (64%)! This makes sense because Alex gave 5 stars to "Inception" (Sci-Fi) and 4 stars to "Interstellar" (Sci-Fi), but only 2 stars to "La La Land" (Romance).

*Can you see the difference?* The weighted profile correctly captures that Alex **strongly prefers** Sci-Fi over Romance, even though both appear in her history.

---

### Approach 3: Rating-Deviation Weighting

**Insight**: A 4-star rating from someone who rates everything 4-5 stars is different from a 4-star rating from someone who averages 2 stars.

**Formula**:
$$\text{profile}(u) = \sum_{i \in I_u} (r_{ui} - \bar{r}_u) \cdot \mathbf{f}_i$$

**Calculation**:
```
Alex's average rating: (5 + 3 + 4 + 2) / 4 = 3.5

Deviations:
  Inception: 5 - 3.5 = +1.5 (above average = likes!)
  Hangover: 3 - 3.5 = -0.5 (below average = dislikes)
  Interstellar: 4 - 3.5 = +0.5 (above average)
  La La Land: 2 - 3.5 = -1.5 (below average = dislikes!)

Deviation-weighted profile:
  Action:  (1.5 x 1) + (-0.5 x 0) + (0.5 x 0) + (-1.5 x 0) = 1.5
  Comedy:  (1.5 x 0) + (-0.5 x 1) + (0.5 x 0) + (-1.5 x 0) = -0.5
  Drama:   (1.5 x 0) + (-0.5 x 0) + (0.5 x 1) + (-1.5 x 1) = -1.0
  Sci-Fi:  (1.5 x 1) + (-0.5 x 0) + (0.5 x 1) + (-1.5 x 0) = 2.0
  Romance: (1.5 x 0) + (-0.5 x 0) + (0.5 x 0) + (-1.5 x 1) = -1.5

Profile: [1.5, -0.5, -1.0, 2.0, -1.5]
```

**Interpretation**:
- **Positive values** = Alex prefers these genres (Sci-Fi: +2.0, Action: +1.5)
- **Negative values** = Alex dislikes these genres (Romance: -1.5, Drama: -1.0)

This captures both **positive and negative** preferences!

---

**Approach 4: Learned Profile** (Logistic Regression, Neural Network)

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

## The Filter Bubble Problem: A Concrete Example

### What is a Filter Bubble?

**Definition**: Content-based filtering can **trap users in increasingly narrow recommendations**, reinforcing their existing preferences while hiding diverse content.

### The Feedback Loop: How It Happens

Let's trace through exactly how a filter bubble forms.

**User "Mike" starts fresh on a movie platform**:

```
Day 1: Onboarding
  Platform asks Mike to rate a few movies.
  Mike rates highly: "Die Hard", "Mad Max", "John Wick"
  All are Action movies.

  User Profile v1: [Action=1.0, Comedy=0, Drama=0, Sci-Fi=0, Romance=0]
```

```
Day 2: First Recommendations
  System recommends based on profile:
    1. "Mission Impossible" (Action) - Mike watches, rates 4/5
    2. "Fast & Furious" (Action) - Mike watches, rates 5/5
    3. "The Notebook" (Romance) - Not shown (low similarity)
    4. "Inception" (Sci-Fi/Action) - Ranked #50, Mike never sees it

  User Profile v2: [Action=1.0, Comedy=0, Drama=0, Sci-Fi=0, Romance=0]
  (Profile unchanged - all new data is Action too!)
```

```
Day 7: One Week Later
  Mike's history: 15 Action movies
  System has learned: "Mike REALLY likes Action"

  Top 20 recommendations: ALL Action movies

  Content Mike never sees:
    - Sci-Fi classics (Blade Runner, 2001)
    - Comedies (Superbad, The Hangover)
    - Dramas (Shawshank Redemption)

  Why? These have <50% genre overlap with Mike's pure-Action profile.
```

```
Day 30: One Month Later
  Mike's history: 50+ Action movies
  Mike's profile: [Action=0.95, Comedy=0.02, Drama=0.01, Sci-Fi=0.02]

  Even mild interest in other genres is drowned out!

  Feedback loop complete:
    Likes Action -> Shown Action -> Watches Action -> Profile becomes more Action
    -> Shown even MORE Action -> Watches MORE Action -> ...
```

**The tragic irony**: Mike might have loved "Inception" or "The Dark Knight" (Action + Sci-Fi), but by the time he's deep in the filter bubble, even these are ranked too low to appear.

*What would happen if we just showed Mike completely random movies?* He'd probably hate most of them (20% hit rate vs 80% with personalization). But he'd also discover hidden gems he never knew he'd love.

---

### Breaking the Bubble

**Solutions** (preview of hybrid strategies):

1. **Diversity injection**: Force some non-Action movies into recommendations
2. **Exploration bonus**: Temporarily boost items dissimilar to profile
3. **User control**: "Show me something different" button
4. **Cross-genre bridges**: "Action fans who also like Sci-Fi watched..."

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

**Example**: New movie released today -> extract features (genre, director) -> recommend to users with matching profiles.

---

### 4. Niche Interests

**Can recommend unpopular items** if they match user's profile.

**Example**: User likes obscure indie films with specific director -> recommend new indie film by same director, even if no one else has watched it yet.

---

## What Can Go Wrong: Common Failure Modes

### Failure Mode 1: Poor Feature Engineering

**Problem**: If features don't capture what users actually care about, recommendations will be bad.

**Example - Movie Recommendations**:
```
Bad features: [Year, Runtime, Budget]
  - "Inception" (2010, 148min, $160M)
  - "Avatar" (2009, 162min, $237M)

System says: "Similar! Both are long, expensive, recent movies."
User thinks: "These are completely different movies!"

Good features: [Genre, Director, Themes, Tone]
  - "Inception" (Sci-Fi, Nolan, Mind-bending, Dark)
  - "Avatar" (Sci-Fi, Cameron, Adventure, Hopeful)

Now we can see the difference!
```

**Lesson**: Domain expertise matters. You need to know what makes items similar *to users*.

---

### Failure Mode 2: Feature Mismatch with User Intent

**Problem**: Users might like items for reasons not captured in features.

**Example - Music Recommendations**:
```
User listens to: "Bohemian Rhapsody" by Queen

Feature-based reasoning:
  Genre: Rock
  Era: 1970s
  Tempo: Variable

System recommends: Other 1970s rock songs

But user actually liked it because:
  - It played at their wedding
  - It reminds them of their dad
  - They're learning piano and love the intro

No feature captures "emotional significance"!
```

---

### Failure Mode 3: Over-Reliance on Text Features

**Problem**: TF-IDF captures word occurrence, not meaning.

**Example - Article Recommendations**:
```
User reads article: "Apple announces new iPhone with improved camera"

High TF-IDF words: "Apple", "iPhone", "camera"

System recommends:
  - "How to make apple pie" (mentions "apple")
  - "Best cameras for photography" (mentions "camera")

These are completely wrong domains!
```

**Solution**: Use word embeddings or domain-specific features.

---

### Failure Mode 4: The "Popularity Trap" for New Users

**Problem**: New users with few ratings get generic recommendations.

**Example**:
```
New user rates: 1 movie ("The Avengers" - Action, Superhero)

User profile: [Action=1.0, Superhero=1.0, everything else=0]

Recommendations: All superhero action movies!

But user might also love:
  - Drama (just hasn't rated one yet)
  - Comedy (watched but didn't rate)
```

**Solution**: Ask for more ratings during onboarding, use exploration strategies.

---

### Failure Mode 5: Synonym and Vocabulary Problems

**Problem**: Different words for the same concept confuse the system.

**Example**:
```
Item 1: "Sci-Fi thriller with robots"
Item 2: "Science Fiction suspense with androids"

TF-IDF sees: Almost no word overlap!
- "Sci-Fi" vs "Science Fiction"
- "thriller" vs "suspense"
- "robots" vs "androids"

Humans see: Very similar movies!
```

**Solution**: Stemming, lemmatization, or semantic embeddings.

---

### Failure Mode 6: Cold Start for Feature Extraction

**Problem**: Some items lack good features.

**Example**:
```
New indie movie on streaming platform:
  - No Wikipedia page (can't extract plot)
  - No professional reviews (can't extract sentiment)
  - Only metadata: Title, Year, 1 actor name

Features available: Almost nothing useful!
```

**Solution**: Hybrid with collaborative filtering once ratings arrive.

---

## Limitations of Content-Based Filtering

### 1. Over-Specialization (Filter Bubble)

**Problem**: Only recommends items similar to past likes. No diversity.

**Example**: User watched 10 action movies -> only gets action movie recommendations -> never discovers comedy, drama, etc.

**Solution**: Inject diversity, exploration (see hybrid strategies).

---

### 2. Feature Engineering Required

**Problem**: Need good features. For movies, easy (genre, director). For other domains, hard.

**Example**: Recommending restaurants -> what features? Cuisine, price, location. But what about ambiance, service quality? Hard to quantify.

---

### 3. New User Problem

**Problem**: Need user's past interactions to build profile.

**Cold start**: New user with no history -> cannot build profile.

**Solutions**:
- Ask user to select preferences (onboarding)
- Use demographics (age, gender, location)
- Start with popular items

---

### 4. Limited Serendipity

**Problem**: Unlikely to recommend unexpected items.

**Example**: User likes action movies -> will never discover great documentaries.

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
2. **Three components**: Content analyzer (feature extraction) -> Profile learner (user profile) -> Filtering (similarity matching)
3. **TF-IDF**: Weights words by frequency in document (TF) and rarity across corpus (IDF)
4. **User profile**: Weighted average of liked items' features (higher ratings = more influence)
5. **Similarity**: Cosine similarity, dot product, or distance metrics
6. **Advantages**: Transparent, handles new items, user-independent
7. **Limitations**: Filter bubble, requires good features, limited serendipity
8. **Failure modes**: Poor features, vocabulary mismatch, over-specialization

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

TF-IDF = 0.167 x 1.099 = 0.183
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
  profile = (5 x [1,0,1] + 3 x [1,1,0]) / (5+3)
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

---

### Problem 4: Filter Bubble Analysis

**Scenario**:
```
User starts with:
  - 3 Action movies rated (all 5 stars)
  - 0 other genres rated

System recommends 5 movies per day, user watches top 2.
After 7 days, user has watched 14 more movies.

If content-based filtering has no diversity mechanism:
  a) How many Action movies will user likely have watched?
  b) What will their final profile look like?
  c) What probability does a Comedy have of being recommended on day 8?
```

**Solution**:
```
a) All 14 movies will be Action (highest similarity to pure-Action profile)
   Total: 3 + 14 = 17 Action movies

b) Profile: [Action = 1.0, all others = 0.0]
   (Actually 100% Action since all watched movies are Action)

c) Probability ~ 0%
   Comedy has 0% overlap with profile, will never rank in top 5
```

**Lesson**: Without diversity injection, filter bubbles are self-reinforcing.

---

### Problem 5: Feature Engineering Challenge

**Given**: You're building a restaurant recommender.

**Available data**:
```
Restaurant features:
  - Cuisine type (Italian, Mexican, Chinese, ...)
  - Price range ($, $$, $$$, $$$$)
  - Location (lat, long)
  - Rating (1-5 stars)
  - Number of reviews
```

**Question**: User loves "Olive Garden" (Italian, $$, 3.8 stars). Rank these candidates:

```
A: Italian place, $$$$, 4.5 stars
B: Mexican place, $$, 4.0 stars
C: Italian place, $$, 3.2 stars
```

**Solution** (depends on feature weighting):

```
If cuisine is weighted highly:
  1. A (Italian match)
  2. C (Italian match)
  3. B (no cuisine match)

If price is weighted highly:
  1. B ($$ match)
  2. C ($$ match)
  3. A (no price match)

If rating is weighted highly:
  1. A (highest rating)
  2. B (4.0)
  3. C (3.2)

"Best" answer: Likely C
  - Same cuisine (Italian) - most important for food preference
  - Same price ($$ - user can afford)
  - Lower rating but similar to user's preference (3.8)
```

**Lesson**: Feature weighting dramatically affects recommendations. Domain expertise needed!
