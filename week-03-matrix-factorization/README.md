# Week 3: Matrix Factorization

## Overview

Matrix Factorization (MF) revolutionized recommendation systems by introducing **latent factor models**. Instead of comparing users/items directly, MF learns hidden features that explain rating patterns. This approach dominated the Netflix Prize and remains foundational today.

## Learning Objectives

By the end of this week, you will:
- Understand the matrix factorization framework and latent factors
- Master optimization techniques (SGD, ALS)
- Implement SVD++, TimeSVD++, and other advanced variants
- Handle implicit feedback with BPR and WRMF

---

## Topics Covered

### [1. The Matrix Factorization Framework](framework.md)
**Core Idea**: $R \approx U^T V$

**Key Concepts**:
- User and item latent factors
- Low-rank approximation
- Connection to SVD and PCA
- Dimensionality reduction

**Prerequisites**: Linear algebra (matrix multiplication, rank, eigenvalues)

---

### [2. Optimization Formulation](optimization.md)
**Objective**: Minimize reconstruction error + regularization

**Topics**:
- Squared error loss: $\min ||R - U^T V||^2_F$
- Observed entries only (not full matrix)
- L2 regularization: $\lambda(||U||^2 + ||V||^2)$
- Bias terms: $r_{ui} = \mu + b_u + b_i + u_i^T v_u$

**Prerequisites**: Calculus (gradients), optimization basics

---

### [3. Optimization Algorithms](algorithms.md)
**How to solve** the MF optimization problem.

**Algorithms**:
1. **Stochastic Gradient Descent (SGD)**: Most common
2. **Alternating Least Squares (ALS)**: Parallelizable
3. **Coordinate Descent**: Element-wise updates

**Topics**:
- Learning rate scheduling
- Convergence properties
- Trade-offs: Speed vs. accuracy

**Prerequisites**: Optimization, numerical methods

---

### [4. Advanced Variants](advanced-variants.md)
**Extensions** of basic MF.

**Models**:
- **SVD++**: Incorporating implicit feedback
- **TimeSVD++**: Temporal dynamics
- **Factorization Machines**: Feature interactions
- **Probabilistic MF (PMF)**: Bayesian approach

**Prerequisites**: Basic MF, probability theory

---

### [5. Handling Implicit Feedback](implicit-feedback.md)
**Adapting MF** for clicks, views, purchases (no ratings).

**Methods**:
- **WRMF**: Weighted Regularized MF
- **BPR**: Bayesian Personalized Ranking
- **Learning from positive-only feedback**

**Prerequisites**: Basic MF

---

### [6. Code Examples](code-examples.md)
**Implementations** in Python.

**Included**:
- Basic MF with SGD (from scratch)
- ALS implementation
- SVD++ implementation
- Using Surprise and Implicit libraries

**Prerequisites**: Python, NumPy

---

### [7. Practice Problems](practice-problems.md)
**Exercises** covering theory and implementation.

**Topics**:
- Gradient derivation
- Complexity analysis
- Implementation challenges
- Hyperparameter tuning

---

## Week Structure

**Day 1-2**: Framework & Formulation
- Read: framework.md, optimization.md
- Derive: Gradients for SGD

**Day 3-4**: Optimization Algorithms
- Read: algorithms.md
- Implement: SGD from scratch

**Day 5**: Advanced Variants
- Read: advanced-variants.md, implicit-feedback.md
- Study: SVD++, BPR

**Day 6-7**: Implementation & Practice
- Code: code-examples.md
- Solve: practice-problems.md

---

## Key Formulas

### Basic MF

**Model**:
$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i$$

**Objective**:
$$\min_{p,q,b} \sum_{(u,i) \in K} (r_{ui} - \hat{r}_{ui})^2 + \lambda(||p_u||^2 + ||q_i||^2 + b_u^2 + b_i^2)$$

### SGD Update

$$\mathbf{p}_u \leftarrow \mathbf{p}_u + \alpha \cdot (e_{ui} \cdot \mathbf{q}_i - \lambda \cdot \mathbf{p}_u)$$

$$\mathbf{q}_i \leftarrow \mathbf{q}_i + \alpha \cdot (e_{ui} \cdot \mathbf{p}_u - \lambda \cdot \mathbf{q}_i)$$

where $e_{ui} = r_{ui} - \hat{r}_{ui}$ is the prediction error.

### SVD++

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{q}_i^T \left( \mathbf{p}_u + |N(u)|^{-1/2} \sum_{j \in N(u)} \mathbf{y}_j \right)$$

where $N(u)$ = items rated by user $u$, $\mathbf{y}_j$ = implicit item factors.

---

## Netflix Prize Context

**Challenge** (2006-2009):
- Improve Cinematch by 10% RMSE
- Dataset: 100M ratings, 480K users, 17K movies

**Winning Solution** (BellKor's Pragmatic Chaos):
- Ensemble of 107 algorithms
- Heavy use of matrix factorization variants
- **Key innovations**: SVD++, TimeSVD++, regularization

**Impact**:
- MF became dominant paradigm in RecSys
- Proved value of latent factor models

**Irony**:
- Netflix never deployed the winning solution
- 10% RMSE improvement didn't justify engineering cost
- Lesson: Offline metrics ≠ online business value

---

## Datasets for Practice

1. **MovieLens 1M**: 1M ratings (good for MF practice)
2. **Netflix Prize Data**: 100M ratings (if you can access it)
3. **Last.fm**: Music listening data (implicit feedback)
4. **Goodreads**: Book ratings

---

## Required Reading

### Papers

1. **Koren, Y., Bell, R., & Volinsky, C. (2009)**. "Matrix factorization techniques for recommender systems". *Computer, IEEE*.
   - **Essential**: Comprehensive overview of MF for RecSys
   - Covers SVD++, temporal dynamics, bias terms

2. **Koren, Y. (2008)**. "Factorization meets the neighborhood: A multifaceted collaborative filtering model". *KDD*.
   - SVD++ original paper
   - Combining MF with neighborhood methods

3. **Rendle, S., et al. (2009)**. "BPR: Bayesian personalized ranking from implicit feedback". *UAI*.
   - **Important**: Handling implicit feedback
   - Pairwise ranking approach

4. **Hu, Y., Koren, Y., & Volinsky, C. (2008)**. "Collaborative filtering for implicit feedback datasets". *ICDM*.
   - WRMF (Weighted Regularized MF)
   - Confidence-weighted approach

### Textbook Chapters

- Aggarwal (2016). *Recommender Systems*. Chapter 3: Model-Based Collaborative Filtering.
- Ricci et al. (2015). *Recommender Systems Handbook*. Chapter 3: Matrix Factorization Techniques.

---

## Programming Libraries

### Surprise (User-Friendly)

```python
from surprise import SVD, Dataset
from surprise.model_selection import cross_validate

data = Dataset.load_builtin('ml-1m')
algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5)
```

### Implicit (Scalable, for Implicit Feedback)

```python
from implicit.als import AlternatingLeastSquares

model = AlternatingLeastSquares(factors=64, regularization=0.01, iterations=15)
model.fit(user_item_data)  # Sparse matrix
recommendations = model.recommend(user_id, user_item_data[user_id], N=10)
```

### LightFM (Hybrid MF)

```python
from lightfm import LightFM

model = LightFM(loss='warp', no_components=30)
model.fit(interactions, epochs=20)
```

---

## Common Pitfalls

1. **Overfitting**: Too many factors, too little regularization
   - **Solution**: Cross-validation for $k$ and $\lambda$

2. **Underfitting**: Too few factors, can't capture patterns
   - **Solution**: Start with $k = 20-100$, increase if needed

3. **Learning rate issues**: Divergence or slow convergence
   - **Solution**: Start with $\alpha = 0.01$, use schedule (decay over time)

4. **Cold start ignored**: MF can't handle new users/items
   - **Solution**: Hybrid with content-based, use features (Factorization Machines)

5. **Implicit feedback as explicit**: Treating views as ratings
   - **Solution**: Use BPR or WRMF specifically designed for implicit data

---

## Hyperparameter Tuning Guide

| Parameter | Typical Range | How to Tune |
|-----------|---------------|-------------|
| **Factors** $k$ | 20-200 | Cross-validation, more for complex data |
| **Regularization** $\lambda$ | 0.001-0.1 | Grid search, higher for overfitting |
| **Learning rate** $\alpha$ | 0.001-0.01 | Start high, decay over epochs |
| **Epochs** | 10-100 | Monitor validation RMSE, stop when plateaus |

---

## Assessment

**Quiz**: Matrix factorization theory (15 questions)
**Coding Assignment**: Implement SGD-based MF from scratch
**Project**: Compare SVD, SVD++, ALS on MovieLens
**Deliverable**: Report with RMSE, MAE, training time comparisons

---

## Next Week Preview

**Week 4**: Content-Based Recommendation
- Move beyond CF to content features
- TF-IDF, embeddings, metadata
- Hybrid models combining CF and content

---

## Additional Resources

**Tutorials**:
- "Matrix Factorization for Collaborative Filtering" (Google Developers)
- Surprise Library Documentation: https://surprise.readthedocs.io/

**Videos**:
- Coursera: "Low Rank Matrix Factorization" (Andrew Ng)
- RecSys Summer School: "Matrix Factorization and Beyond"

**Blog Posts**:
- "The BellKor Solution to the Netflix Prize" (Yehuda Koren)
- "Matrix Factorization Explained" (Ethan Rosenthal's blog)

**Code Repositories**:
- Netflix Prize solutions: https://github.com/search?q=netflix+prize
- Surprise examples: https://github.com/NicolasHug/Surprise

---

*Return to [Main Course Page](../README.md)*
