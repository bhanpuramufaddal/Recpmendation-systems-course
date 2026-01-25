# Week 2: Memory-Based Collaborative Filtering

## Overview

Memory-based collaborative filtering represents the earliest approaches to recommendation systems. These methods directly use the user-item interaction matrix without learning a model, hence "memory-based."

## Learning Objectives

By the end of this week, you will:
- Understand user-based and item-based collaborative filtering
- Master similarity measures (Pearson, cosine, Jaccard)
- Implement kNN-based recommendation algorithms
- Recognize scalability limitations and when to use memory-based methods

---

## Topics Covered

### [1. User-Based Collaborative Filtering](user-based-cf.md)
**Intuition**: "Users similar to you liked..."

**Key Concepts**:
- Pearson correlation, cosine similarity
- k-nearest neighbors
- Prediction formula
- Complexity: O(|U|² × |I|)

**Prerequisites**: Linear algebra basics, statistics

---

### [2. Item-Based Collaborative Filtering](item-based-cf.md)
**Intuition**: "You liked X, so you might like Y..."

**Key Concepts**:
- Item-item similarity
- Precomputation strategies
- Amazon's approach
- Better scalability than user-based

**Prerequisites**: User-based CF

---

### [3. Similarity Measures Deep Dive](similarity-measures.md)
**Comprehensive coverage** of similarity metrics.

**Topics**:
- Pearson correlation coefficient
- Cosine similarity
- Adjusted cosine similarity
- Jaccard similarity
- Significance weighting
- Variance weighting

**Prerequisites**: Statistics, linear algebra

---

### [4. Code Examples](code-examples.md)
**Hands-on implementations** in Python.

**Implementations**:
- User-based CF from scratch
- Item-based CF from scratch
- Using Surprise library
- Performance comparisons

**Prerequisites**: Python, NumPy, pandas

---

### [5. Practice Problems](practice-problems.md)
**Exercises** to reinforce understanding.

**Problem Types**:
- Similarity calculations
- Prediction computations
- Scalability analysis
- Implementation exercises

---

## Week Structure

**Day 1-2**: User-Based CF
- Read: user-based-cf.md
- Code: Implement user-user similarity

**Day 3-4**: Item-Based CF
- Read: item-based-cf.md
- Code: Implement item-item CF

**Day 5**: Similarity Measures
- Read: similarity-measures.md
- Practice: Compute various similarities

**Day 6-7**: Implementation & Practice
- Code: code-examples.md
- Solve: practice-problems.md

---

## Key Formulas

### User-Based Prediction

$$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} \text{sim}(u,v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N_k(u)} |\text{sim}(u,v)|}$$

### Item-Based Prediction

$$\hat{r}_{ui} = \frac{\sum_{j \in N_k(i)} \text{sim}(i,j) \cdot r_{uj}}{\sum_{j \in N_k(i)} |\text{sim}(i,j)|}$$

### Pearson Correlation

$$\text{Pearson}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}$$

---

## Datasets for Practice

1. **MovieLens 100K**: 100K ratings, 943 users, 1,682 movies
2. **MovieLens 1M**: 1M ratings, 6,040 users, 3,706 movies
3. **Book-Crossing**: 278K users, 271K books
4. **Jester**: Jokes dataset, 4.1M ratings

**Access**: https://grouplens.org/datasets/

---

## Required Reading

### Papers

1. **Resnick, P., et al. (1994)**. "GroupLens: An open architecture for collaborative filtering of netnews". *CSCW*.
   - First user-based CF system

2. **Sarwar, B., et al. (2001)**. "Item-based collaborative filtering recommendation algorithms". *WWW*.
   - Amazon's item-to-item approach
   - Comparison of item-based vs. user-based

3. **Herlocker, J. L., et al. (1999)**. "An algorithmic framework for performing collaborative filtering". *SIGIR*.
   - Comprehensive evaluation of CF techniques

### Textbook Chapters

- Aggarwal, C.C. (2016). *Recommender Systems*. Chapter 2: Neighborhood-Based Collaborative Filtering.
- Ricci et al. (2015). *Recommender Systems Handbook*. Chapter 5: Collaborative Filtering.

---

## Programming Libraries

### Surprise (recommended for beginners)
```python
from surprise import KNNBasic, Dataset
from surprise.model_selection import cross_validate

data = Dataset.load_builtin('ml-100k')
algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': True})
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

### Implicit (for large-scale)
```python
from implicit.nearest_neighbours import CosineRecommender
import scipy.sparse as sp

# User-item matrix (sparse)
user_item_data = sp.csr_matrix(...)

model = CosineRecommender(K=20)
model.fit(user_item_data)
recommendations = model.recommend(user_id, user_item_data[user_id])
```

---

## Common Pitfalls

1. **Ignoring sparsity**: Similarity based on 1-2 co-ratings is meaningless
   - **Solution**: Minimum overlap threshold (e.g., ≥10 co-ratings)

2. **Not mean-centering**: Users have different rating scales
   - **Solution**: Use Pearson (inherently mean-centered) or manually center

3. **Treating missing as zero**: Unknown ≠ dislike
   - **Solution**: Only compute over known ratings

4. **Scalability assumptions**: Works on MovieLens, fails on Netflix scale
   - **Solution**: Use item-based or move to model-based approaches

---

## Assessment

**Quiz**: End of week (20 questions)
**Coding Assignment**: Implement user-based and item-based CF on MovieLens
**Deliverable**: Jupyter notebook with RMSE comparison

---

## Next Week Preview

**Week 3**: Matrix Factorization
- Move from memory-based to model-based CF
- Learn latent factor models
- Understand SVD, ALS, SGD optimization

---

## Additional Resources

**Tutorials**:
- Building a Recommendation Engine with Surprise: https://surpriselib.com/
- Collaborative Filtering Tutorial (Kaggle): https://www.kaggle.com/code/gspmoreira/recommender-systems-in-python-101

**Videos**:
- Andrew Ng's "Collaborative Filtering" (Coursera Machine Learning)
- RecSys Summer School lectures on CF

**Blog Posts**:
- "The BellKor solution to the Netflix Prize" by Yehuda Koren
- "How Netflix's Recommendations System Works" by Netflix Tech Blog

---

*Return to [Main Course Page](../README.md)*
