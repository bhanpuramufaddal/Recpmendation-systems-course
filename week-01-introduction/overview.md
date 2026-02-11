# Week 1: The Recommendation Problem - Overview

## Opening Hook

> **"Netflix reports that 80% of watch time comes from recommendations. Not search. Not browsing. Recommendations."**

Let that sink in. Four out of every five hours you spend watching Netflix? You didn't find that content - an algorithm found it for you.

And it's not just Netflix. Amazon attributes 35% of their revenue to recommendations. YouTube's homepage? 100% recommended content. Spotify's Discover Weekly? 40 million users every week, letting an algorithm choose their music.

**Here's the question that should keep you up at night**: How do these systems know what you want before you do?

That's what this course is about. Welcome to Recommendation Systems.

---

## Learning Objectives

By the end of this section, you will:
- Understand what recommendation systems are and their core formulations
- Distinguish between different types of feedback and recommendation tasks
- Recognize the evolution and impact of recommendation systems
- Identify key application domains
- **Derive** why ranking matters more than prediction
- **Calculate** the signal strength tradeoff between explicit and implicit feedback

---

## 1.1 The Information Overload Problem

### Why Recommendations Exist

Let me give you some numbers that will make your head spin:

| Platform | Catalog Size | The Problem |
|----------|--------------|-------------|
| Amazon | **12 million** products | You can browse 100 items/day = 328 years to see everything |
| YouTube | **800 million** videos | At 10 minutes each = 15,220 years of content |
| Spotify | **100 million** tracks | Listening 24/7 = 1,141 years of music |
| Netflix | **15,000** titles | More manageable, but still 41 years at 1 movie/day |

**The fundamental problem**: Human attention is finite. Catalog size is exploding.

> **Socratic Question**: "If you had unlimited time, would you need recommendations?"
>
> Think about it. The scarcity isn't content - it's your attention. Recommendations are really attention allocation systems.

### Definition

**Recommendation systems** are software tools and techniques that provide suggestions for items that are most likely to be of interest to a particular user. They help users discover items they might not find on their own from a vast collection.

But here's a better definition for engineers:

> **Recommendation systems solve the information overload problem by predicting which items from a massive catalog will maximize user satisfaction, given limited user attention.**

---

## 1.2 Core Formulations: Prediction vs. Ranking

### The Naive Approach: Predict Everything

**Prediction Formulation**:
- **Goal**: Predict the rating/score a user would give to an item
- **Output**: Numerical prediction (e.g., "You'd rate this movie 4.2 stars")
- **Evaluation**: Root Mean Square Error (RMSE), Mean Absolute Error (MAE)

$$\text{RMSE} = \sqrt{\frac{1}{|T|} \sum_{(u,i) \in T} (r_{ui} - \hat{r}_{ui})^2}$$

where:
- $r_{ui}$ = actual rating by user $u$ for item $i$
- $\hat{r}_{ui}$ = predicted rating
- $T$ = test set of (user, item) pairs with known ratings

### Why Prediction Isn't Enough: A Derivation

**Let me show you why the industry moved away from pure prediction.**

Imagine you have a prediction model. For user Alice, it predicts:

| Movie | Predicted Rating | True Rating |
|-------|------------------|-------------|
| Movie A | 4.2 | 4.5 |
| Movie B | 4.1 | 4.3 |
| Movie C | 3.9 | 3.8 |
| Movie D | 3.8 | 4.0 |

Your RMSE looks great! Average error is only 0.2 stars.

But here's the problem: **You can only show the user K items** (say, K=2).

With prediction, you'd show: Movie A (4.2) and Movie B (4.1)
The user would actually prefer: Movie A (4.5) and Movie B (4.3)

**In this case, prediction worked!** But what if:

| Movie | Predicted Rating | True Rating |
|-------|------------------|-------------|
| Movie A | 4.2 | 3.0 |
| Movie B | 4.1 | 3.2 |
| Movie C | 3.9 | 4.8 |
| Movie D | 3.8 | 4.5 |

Same RMSE (average error ~0.8), but now you're showing the **wrong movies entirely**.

**The key insight**:

$$\text{User Satisfaction} \propto \text{Quality of Top-K items shown}$$

NOT:

$$\text{User Satisfaction} \propto \text{Average prediction accuracy across all items}$$

### Ranking Formulation

**Ranking Formulation**:
- **Goal**: Rank items in order of relevance/preference
- **Output**: Ordered list of items (e.g., "Top 10 movies for you")
- **Evaluation**: Precision@K, NDCG, MAP

$$\text{Precision@K} = \frac{\text{number of relevant items in top K}}{K}$$

> **Key Insight**: Modern systems focus on ranking (what to show) rather than prediction (exact ratings). Users never see items ranked #1001 through #1,000,000 - so why optimize prediction accuracy there?

---

## 1.3 Explicit vs. Implicit Feedback: The Great Tradeoff

### Explicit Feedback

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

### Implicit Feedback

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

### The 1% vs 100% Problem: A Numerical Derivation

Let me show you the real tradeoff with actual numbers.

**Scenario**: 1 million users, 100,000 items

**Explicit Feedback Reality**:
- Average user rates: **50 items** (that's generous!)
- Total explicit signals: 1M users x 50 ratings = **50 million ratings**
- Matrix density: 50M / (1M x 100K) = **0.05%**
- Only **1 in 2,000** cells filled

**Implicit Feedback Reality**:
- Average user interacts with: **5,000 items** (clicks, views, scrolls)
- Total implicit signals: 1M users x 5,000 = **5 billion signals**
- That's **100x more data**

But wait - which is more valuable?

### Signal Strength Analysis

**Explicit rating of 5 stars**:
- Confidence: ~95% the user likes it
- Information: High quality but rare

**Implicit signal (user watched 90% of movie)**:
- Confidence: ~70% the user likes it
- Information: Lower quality but abundant

**The math**:

Let $C_e$ = confidence of explicit signal = 0.95
Let $C_i$ = confidence of implicit signal = 0.70

Let $N_e$ = number of explicit signals = 50
Let $N_i$ = number of implicit signals = 5,000

**Total information from explicit**: $I_e = C_e \times N_e = 0.95 \times 50 = 47.5$ "confidence-weighted signals"

**Total information from implicit**: $I_i = C_i \times N_i = 0.70 \times 5,000 = 3,500$ "confidence-weighted signals"

$$\frac{I_i}{I_e} = \frac{3,500}{47.5} \approx 74$$

> **Implicit feedback gives us ~74x more total information, even accounting for lower per-signal confidence!**

This is why Netflix moved from star ratings to thumbs up/down to purely implicit (what you watch, when you stop, what you rewatch).

### Socratic Interlude

> **"If a user watched a movie but didn't rate it, did they like it?"**
>
> This is the fundamental ambiguity of implicit feedback. They might have:
> - Loved it (but too lazy to rate)
> - Thought it was okay (not worth rating)
> - Hated it (left it playing while doing something else)
> - Fell asleep (oops)
>
> **How do you distinguish these?** Hint: time spent, completion rate, rewatches, similar content engagement afterward...

---

## 1.4 Understanding a Single User: A Worked Example

Let's follow one user through both feedback types.

### User: Alice

**Alice's Explicit Feedback** (5 ratings total):
| Movie | Rating | What We Learn |
|-------|--------|---------------|
| The Matrix | 5 stars | Loves sci-fi action |
| Inception | 5 stars | Loves mind-bending plots |
| Titanic | 2 stars | Dislikes romance |
| The Notebook | 1 star | Really dislikes romance |
| Interstellar | 4 stars | Likes sci-fi, maybe not as much |

**What can we infer?**
- Likes: Sci-fi, action, complex plots
- Dislikes: Romance
- Limited data: Only 5 points to work with!

**Alice's Implicit Feedback** (1,000 signals):
| Behavior | Count | What We Learn |
|----------|-------|---------------|
| Sci-fi movies watched to completion | 47 | Strong preference signal |
| Sci-fi movies started but abandoned <10min | 12 | Not all sci-fi! |
| Romance movies scrolled past | 203 | Consistent avoidance |
| Action movie trailers watched | 89 | Active interest |
| Christopher Nolan director page visited | 5 | Director preference |
| "Movies like Inception" searched | 3 | Direct intent |
| Friday night viewing sessions | 48 | Time pattern |
| Weekend binge sessions (3+ movies) | 12 | Engagement pattern |
| Rewatched The Matrix | 2 | Strong positive signal |
| Added to watchlist but never watched | 34 | Aspirational vs actual |

**With 1,000 implicit signals we learn**:
- Not just "likes sci-fi" but **which sci-fi** (completed 47, abandoned 12)
- Negative preferences without explicit thumbs-down (203 romance scrolls)
- **Behavioral context** (Friday nights, weekend binges)
- Intent signals (searches, watchlist)
- Strength of preference (rewatches)

> **The explicit feedback told us WHAT Alice likes. The implicit feedback tells us HOW MUCH, WHEN, and in WHAT CONTEXT.**

---

## 1.5 The User-Item Matrix: Visualizing the Problem

### The Matrix View

Here's how we represent the recommendation problem mathematically:

```
                   Items (I)
              i1    i2    i3    i4    i5    i6    i7    i8    ...  i100000
         +----------------------------------------------------------
    u1   |   5     ?     ?     3     ?     ?     ?     ?    ...    ?
    u2   |   ?     4     ?     ?     5     ?     ?     ?    ...    ?
U   u3   |   4     ?     ?     ?     ?     ?     2     ?    ...    ?
s   u4   |   ?     5     4     ?     ?     ?     ?     ?    ...    ?
e   u5   |   ?     ?     ?     ?     ?     3     4     ?    ...    ?
r   .    |   .     .     .     .     .     .     .     .    ...    .
s   .    |   .     .     .     .     .     .     .     .    ...    .
    u1M  |   ?     ?     ?     ?     ?     ?     ?     ?    ...    3
         +----------------------------------------------------------
```

**Properties**:
- **Extremely Sparse**: 99.5%+ entries are unknown (?)
- **High-dimensional**: Millions of users x Millions of items
- **Dynamic**: Continuously evolving as users interact
- Each **row** = one user's preferences across all items
- Each **column** = one item's ratings from all users

### Visualizing Sparsity

Let's make sparsity concrete. For Netflix:
- 200 million users
- 15,000 titles
- Total possible ratings: 200M x 15K = **3 trillion cells**
- Average user has rated: ~200 movies
- Total actual ratings: ~200M x 200 = **40 billion ratings**

**Sparsity**: (3T - 40B) / 3T = **98.7% empty**

If we visualized this matrix where filled = black pixel, empty = white pixel:

```
Real Netflix Matrix (if you could see it):

[                                                              ]
[                                                              ]
[                                                              ]
[                          Mostly white                        ]
[                         (98.7% empty)                        ]
[                                                              ]
[                     Occasional black dots                    ]
[                    (the 1.3% we know)                        ]
[                                                              ]
[                                                              ]
```

**Goal of RecSys**: Fill in those empty cells (or at least rank which empty cells should become filled next!)

### Mathematical Representation

**Explicit feedback matrix**:

$$R \in \mathbb{R}^{|U| \times |I|}$$

where $r_{ui}$ is the rating user $u$ gave item $i$ (if observed)

**Implicit feedback matrix**:

$$Y \in \{0,1\}^{|U| \times |I|}$$

where $y_{ui} = 1$ if user $u$ interacted with item $i$

**The key difference**: $R$ has meaningful missing values (we don't know). $Y$ treats missing as 0 (no interaction observed).

---

## 1.6 Top-N Recommendation vs. Rating Prediction

### Top-N Recommendation

**Task**: Given user $u$, recommend $N$ items most likely to interest them

**Process**:
1. Score all unseen items for user
2. Rank by score
3. Return top $N$ items

**Example**: "Here are 10 movies we think you'll love"

**Metrics**: Precision@N, Recall@N, NDCG@N

### Rating Prediction

**Task**: Given user $u$ and item $i$, predict $\hat{r}_{ui}$

**Process**:
1. Learn model from historical ratings
2. Predict score for (user, item) pair

**Example**: "You'll rate this movie 4.2 out of 5 stars"

**Metrics**: RMSE, MAE

### The Industry Shift

**Historical (Netflix Prize era, 2006-2009)**:
- Focused on rating prediction (minimize RMSE)
- Academic challenge: predict exact ratings
- Netflix prize: improve RMSE by 10% = $1 million

**Modern (2015-present)**:
- Focus on Top-N recommendation
- Optimize engagement: clicks, watch time, conversions
- Implicit feedback dominates

**Why the shift?**
1. Users care about **discovery**, not predicted ratings
2. Top-N directly optimizes for **business metrics**
3. Implicit feedback provides **100x more data**
4. You can be wrong about the exact rating but still **recommend the right item**

---

## 1.7 The Mathematical Formulation

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
- Return top-$N$ items: $\text{TopN}(u) = \underset{S \subset C, |S|=N}{\arg\max} \sum_{i \in S} s(u, i)$

> **The beauty of this formulation**: We don't need to know exact preferences - we just need to get the **relative ordering** right for the top-K items the user will actually see.

---

## 1.8 Key Terminology

| Term | Definition | Example |
|------|------------|---------|
| **User** | Entity receiving recommendations | Customer, viewer, listener |
| **Item** | Entity being recommended | Product, movie, song, article |
| **Interaction** | User-item engagement | Rating, click, purchase, view |
| **User-Item Matrix** | Matrix $R$ where $r_{ui}$ is interaction | Ratings matrix, purchase matrix |
| **Sparsity** | Ratio of unknown to total interactions | 99.5% of Netflix matrix is empty |
| **Cold Start** | New user/item with no interaction history | New movie, new user account |
| **Feedback** | Signal about user preference | Explicit (rating) or implicit (click) |

---

## 1.9 Recommendation System Pipeline Preview

```
User Interaction --> Data Collection --> Candidate Generation --> Ranking --> Re-ranking --> Display
     (clicks)          (logging)       (1000s of items)       (scored)   (diversified)    (UI)
```

Each stage solves a different problem:
- **Candidate Generation**: From millions to thousands (fast, approximate)
- **Ranking**: From thousands to hundreds (accurate, slower)
- **Re-ranking**: From hundreds to tens (business rules, diversity)

This will be covered in detail in the **pipeline.md** section.

---

## 1.10 Real-World Scale

### Netflix
- 200M+ subscribers across 190+ countries
- 15K+ titles
- 99.5% matrix sparsity
- **Recommendations drive 80%+ of viewing**

### Amazon
- 12M+ products (marketplace expands to 400M+)
- 300M+ customers
- Billions of interactions daily
- **35% revenue from recommendations**

### YouTube
- 2B+ logged-in users monthly
- 800M+ videos
- 1B+ hours watched daily
- **Homepage is 100% recommended**

**The engineering challenge**: How do you recommend from millions of items to billions of users in **<100 milliseconds**?

---

## Summary

Recommendation systems are:
- Solutions to the **information overload problem** (12M products, 800M videos)
- **Ranking tasks** more than prediction tasks (users only see Top-K)
- Based on **implicit behavior** (100x more data) more than explicit ratings
- Operating at **massive scale** with **extreme sparsity** (99%+ unknown)

### What We Derived Today:
1. **Why ranking > prediction**: User satisfaction depends on Top-K quality, not average prediction accuracy
2. **The 74x implicit advantage**: Even with lower per-signal confidence, implicit feedback provides 74x more total information
3. **Sparsity quantified**: 98.7% of the Netflix matrix is empty

### Questions to Ponder:
- If implicit feedback is so powerful, why do platforms still ask for ratings?
- How do you handle the cold start problem when a new user has zero signals?
- Is there an ethical dimension to systems that know what you want before you do?

---

## Next Steps

- **historical-context.md**: Evolution from GroupLens to modern deep learning systems
- **applications.md**: Domain-specific applications and use cases
- **pipeline.md**: End-to-end recommendation pipeline
- **challenges.md**: Cold start, sparsity, scalability, filter bubbles

---

## Further Reading

- Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Chapter 1.
- Aggarwal, C.C. (2016). *Recommender Systems: The Textbook*. Chapter 1.
- Koren, Y., & Bell, R. (2015). "Advances in Collaborative Filtering". *Recommender Systems Handbook*.
- Covington, P., Adams, J., & Sargin, E. (2016). "Deep Neural Networks for YouTube Recommendations". *RecSys*.
