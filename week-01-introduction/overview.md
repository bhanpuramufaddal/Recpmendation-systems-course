# Week 1: The Recommendation Problem - Overview

## Learning Objectives

By the end of this section, you will:
- Understand what recommendation systems are and their core formulations
- Distinguish between different types of feedback and recommendation tasks
- Recognize the evolution and impact of recommendation systems
- Identify key application domains

---

## 1.1 What Are Recommendation Systems?

### Definition

**Recommendation systems** are software tools and techniques that provide suggestions for items that are most likely to be of interest to a particular user. They help users discover items they might not find on their own from a vast collection.

### Core Formulations

#### Prediction vs. Ranking

**Prediction Formulation**:
- **Goal**: Predict the rating/score a user would give to an item
- **Output**: Numerical prediction (e.g., "You'd rate this movie 4.2 stars")
- **Evaluation**: Root Mean Square Error (RMSE), Mean Absolute Error (MAE)

$$\text{RMSE} = \sqrt{\frac{1}{|T|} \sum_{(u,i) \in T} (r_{ui} - \hat{r}_{ui})^2}$$

where:
- $r_{ui}$ = actual rating by user $u$ for item $i$
- $\hat{r}_{ui}$ = predicted rating
- $T$ = test set

**Ranking Formulation**:
- **Goal**: Rank items in order of relevance/preference
- **Output**: Ordered list of items (e.g., "Top 10 movies for you")
- **Evaluation**: Precision@K, NDCG, MAP

$$\text{Precision@K} = \frac{\text{number of relevant items in top K}}{K}$$

**Key Insight**: Modern systems focus more on ranking (what to show) rather than prediction (exact ratings).

---

### Explicit vs. Implicit Feedback

#### Explicit Feedback

**Definition**: Direct user ratings or preferences

**Examples**:
- Star ratings (1-5 stars on Netflix, Amazon)
- Thumbs up/down (YouTube, Spotify)
- Like/dislike buttons (Facebook, Reddit)
- Written reviews with scores

**Advantages**:
- Clear user intent
- Quantifiable preferences
- Less ambiguity

**Disadvantages**:
- **Sparse**: Most users don't rate most items
- **Biased**: Users tend to rate extreme experiences
- **Costly**: Requires explicit user action

#### Implicit Feedback

**Definition**: Inferred preferences from user behavior

**Examples**:
- Clicks, views, plays
- Purchase history
- Time spent on item
- Browse patterns
- Search queries
- Scrolling behavior

**Advantages**:
- **Abundant**: Naturally generated during usage
- **No user effort**: Passive collection
- **Scale**: Billions of signals

**Disadvantages**:
- **Ambiguous**: Did they like it or just accidentally click?
- **Noisy**: May not reflect true preferences
- **Negative feedback unclear**: Absence doesn't mean dislike

**Mathematical Representation**:

Explicit: $R \in \mathbb{R}^{|U| \times |I|}$ where $r_{ui}$ is the rating

Implicit: $Y \in \{0,1\}^{|U| \times |I|}$ where $y_{ui} = 1$ if interaction exists

---

### Top-N Recommendation vs. Rating Prediction

#### Top-N Recommendation

**Task**: Given user $u$, recommend $N$ items most likely to interest them

**Process**:
1. Score all unseen items for user
2. Rank by score
3. Return top $N$ items

**Example**: "Here are 10 movies we think you'll love"

**Metrics**: Precision@N, Recall@N, NDCG@N

#### Rating Prediction

**Task**: Given user $u$ and item $i$, predict $\hat{r}_{ui}$

**Process**:
1. Learn model from historical ratings
2. Predict score for (user, item) pair

**Example**: "You'll rate this movie 4.2 out of 5 stars"

**Metrics**: RMSE, MAE

**Industry Trend**:
- Netflix Prize focused on rating prediction (RMSE)
- Modern systems focus on Top-N (engagement, clicks, watch time)

**Why the shift?**
- Users care about discovering great content, not predicted ratings
- Top-N directly optimizes for business metrics
- Implicit feedback dominates (more data, better predictions)

---

## Key Terminology

| Term | Definition | Example |
|------|------------|---------|
| **User** | Entity receiving recommendations | Customer, viewer, listener |
| **Item** | Entity being recommended | Product, movie, song, article |
| **Interaction** | User-item engagement | Rating, click, purchase, view |
| **User-Item Matrix** | Matrix $R$ where $r_{ui}$ is interaction | Ratings matrix, purchase matrix |
| **Sparsity** | Ratio of unknown to total interactions | 99.5% of Netflix matrix is empty |
| **Cold Start** | New user/item with no interaction history | New movie, new user account |

---

## The User-Item Interaction Matrix

```
           Item1  Item2  Item3  Item4  Item5
User1        5      ?      3      ?      ?
User2        ?      4      ?      5      ?
User3        4      ?      ?      ?      2
User4        ?      5      4      ?      ?
User5        ?      ?      ?      3      4
```

**Properties**:
- **Sparse**: Most entries are unknown (?)
- **High-dimensional**: Millions of users × millions of items
- **Dynamic**: Continuously evolving

**Goal of RecSys**: Fill in the ? marks (or rank items without ratings)

---

## Mathematical Formulation

### Prediction Problem

Given:
- $U$ = set of users
- $I$ = set of items
- $R \in \mathbb{R}^{|U| \times |I|}$ = partially observed rating matrix

Find:
- Function $f: U \times I \rightarrow \mathbb{R}$ that predicts $\hat{r}_{ui}$ for unobserved $(u, i)$ pairs

### Ranking Problem

Given:
- User $u$
- Set of candidate items $C \subset I$

Find:
- Scoring function $s: U \times I \rightarrow \mathbb{R}$
- Return top-$N$ items: $\text{TopN}(u) = \arg\max_{i \in C, |L|=N} s(u, i)$

---

## Recommendation System Pipeline Preview

```
User Interaction → Data Collection → Candidate Generation → Ranking → Re-ranking → Display
     (clicks)          (logging)       (100s of items)    (scored)   (diversified)  (UI)
```

This will be covered in detail in the **pipeline.md** section.

---

## Real-World Scale

### Netflix (Pre-2024)
- 200M+ subscribers
- 15K+ titles
- 99.5% matrix sparsity
- Recommendations drive 80%+ of viewing

### Amazon
- 400M+ products
- 300M+ customers
- Billions of interactions daily
- 35% revenue from recommendations

### YouTube
- 2B+ users
- 800M+ videos
- 1B+ hours watched daily
- Homepage is 100% recommended

**Challenge**: How do you recommend from millions of items to billions of users in milliseconds?

---

## Summary

Recommendation systems are:
- **Prediction or ranking** tasks helping users discover items
- Based on **explicit ratings** (sparse, clear) or **implicit behavior** (abundant, noisy)
- Solving **Top-N recommendation** (industry focus) or **rating prediction** (academic roots)
- Operating at **massive scale** with **extreme sparsity**

**Next Steps**:
- **historical-context.md**: Evolution from GroupLens to modern deep learning systems
- **applications.md**: Domain-specific applications and use cases
- **pipeline.md**: End-to-end recommendation pipeline
- **challenges.md**: Cold start, sparsity, scalability, filter bubbles

---

## Further Reading

- Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Chapter 1.
- Aggarwal, C.C. (2016). *Recommender Systems: The Textbook*. Chapter 1.
- Koren, Y., & Bell, R. (2015). "Advances in Collaborative Filtering". *Recommender Systems Handbook*.
