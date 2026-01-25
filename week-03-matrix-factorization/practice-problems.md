# Week 3: Matrix Factorization - Practice Problems

## Overview
These problems test your understanding of matrix factorization fundamentals, optimization algorithms (SGD, ALS), advanced variants (SVD++, BPR), and practical implementation challenges. Focus on gradient derivations, regularization, and handling implicit feedback.

---

## Problem 1: Gradient Derivation for SGD
**Difficulty:** Medium
**Topics:** Stochastic gradient descent, gradients, calculus

Given the basic matrix factorization objective:
$$L = \sum_{(u,i) \in K} (r_{ui} - \mathbf{p}_u^T \mathbf{q}_i)^2 + \lambda(||\mathbf{p}_u||^2 + ||\mathbf{q}_i||^2)$$

where:
- $\mathbf{p}_u$ = user latent factor vector (k dimensions)
- $\mathbf{q}_i$ = item latent factor vector (k dimensions)
- $\lambda$ = regularization parameter

**Derive:**
1. $\frac{\partial L}{\partial \mathbf{p}_u}$
2. $\frac{\partial L}{\partial \mathbf{q}_i}$
3. Write the SGD update rules

**Hints:**
- Let $e_{ui} = r_{ui} - \mathbf{p}_u^T \mathbf{q}_i$ (prediction error)
- Chain rule: $\frac{\partial}{\partial \mathbf{p}_u}(a - \mathbf{p}_u^T \mathbf{q}_i)^2 = -2(a - \mathbf{p}_u^T \mathbf{q}_i)\mathbf{q}_i$
- Don't forget the regularization term

**Learning Outcomes:**
- Master gradient computation for MF
- Understand SGD update mechanics
- Connect math to implementation

---

## Problem 2: Bias Terms in Matrix Factorization
**Difficulty:** Medium
**Topics:** Bias modeling, prediction formulas

The enhanced MF model with biases:
$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i$$

where:
- $\mu$ = global mean rating
- $b_u$ = user bias
- $b_i$ = item bias

**Given MovieLens data:**
- Global mean $\mu = 3.5$
- User Alice: $b_{\text{Alice}} = +0.5$ (rates higher than average)
- Movie "The Matrix": $b_{\text{Matrix}} = +0.8$ (rated higher than average)
- Latent factor interaction: $\mathbf{p}_{\text{Alice}}^T \mathbf{q}_{\text{Matrix}} = 0.3$

1. Predict Alice's rating for "The Matrix"
2. If the actual rating is 5.0, compute the prediction error
3. Explain what each term captures conceptually
4. Derive the gradient updates for $b_u$ and $b_i$

**Learning Outcomes:**
- Understand the role of biases
- Decompose predictions into interpretable components
- Work with the full MF model

---

## Problem 3: Regularization Impact
**Difficulty:** Easy
**Topics:** Regularization, overfitting, hyperparameter tuning

You train MF models with different $\lambda$ values:

| $\lambda$ | Training RMSE | Test RMSE | Latent Factor Norms |
|-----------|---------------|-----------|---------------------|
| 0.0       | 0.72          | 1.15      | 8.3                 |
| 0.01      | 0.78          | 0.93      | 2.1                 |
| 0.1       | 0.85          | 0.91      | 0.8                 |
| 1.0       | 0.95          | 0.96      | 0.3                 |

1. Which model is overfitting?
2. Which $\lambda$ would you choose for production?
3. What happens to latent factors as $\lambda$ increases?
4. Why doesn't $\lambda = 1.0$ perform best on test set?

**Learning Outcomes:**
- Recognize overfitting vs. underfitting
- Choose appropriate regularization
- Understand regularization effects

---

## Problem 4: ALS Derivation
**Difficulty:** Hard
**Topics:** Alternating least squares, optimization, linear algebra

In ALS, we alternate between fixing $\mathbf{Q}$ (item factors) and solving for $\mathbf{P}$ (user factors), then vice versa.

**Objective when $\mathbf{Q}$ is fixed:**
$$\min_{\mathbf{p}_u} \sum_{i \in I_u} (r_{ui} - \mathbf{p}_u^T \mathbf{q}_i)^2 + \lambda ||\mathbf{p}_u||^2$$

1. Write this in matrix form: $\min_{\mathbf{p}_u} ||\mathbf{r}_u - \mathbf{Q}_u^T \mathbf{p}_u||^2 + \lambda ||\mathbf{p}_u||^2$
2. Take the gradient and set to zero
3. Derive the closed-form solution: $\mathbf{p}_u = (\mathbf{Q}_u \mathbf{Q}_u^T + \lambda \mathbf{I})^{-1} \mathbf{Q}_u \mathbf{r}_u$
4. Explain why ALS is parallelizable but SGD is not

**Hints:**
- $\mathbf{Q}_u$ contains only items rated by user $u$
- Use matrix calculus: $\nabla_{\mathbf{x}} ||\mathbf{A}\mathbf{x} - \mathbf{b}||^2 = 2\mathbf{A}^T(\mathbf{A}\mathbf{x} - \mathbf{b})$
- Each user's update is independent in ALS

**Learning Outcomes:**
- Derive the ALS algorithm
- Understand closed-form solutions
- Recognize parallelization opportunities

---

## Problem 5: Number of Factors Selection
**Difficulty:** Medium
**Topics:** Model capacity, dimensionality, cross-validation

You're deciding how many latent factors $k$ to use. Dataset: MovieLens-1M

| k  | Train RMSE | Test RMSE | Training Time | Model Size |
|----|------------|-----------|---------------|------------|
| 10 | 0.92       | 0.95      | 30s           | 100 KB     |
| 50 | 0.84       | 0.89      | 120s          | 500 KB     |
| 100| 0.78       | 0.87      | 240s          | 1 MB       |
| 200| 0.72       | 0.88      | 480s          | 2 MB       |
| 500| 0.65       | 0.93      | 1200s         | 5 MB       |

1. Which $k$ would you choose? Why?
2. What is happening with $k=500$?
3. How would you systematically choose $k$?
4. In production with strict latency requirements, how might this affect your choice?

**Learning Outcomes:**
- Balance model capacity and generalization
- Recognize overfitting from too many factors
- Make practical engineering trade-offs

---

## Problem 6: SVD++ Understanding
**Difficulty:** Hard
**Topics:** SVD++, implicit feedback, advanced MF

Standard MF:
$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i$$

SVD++ incorporates implicit feedback:
$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{q}_i^T \left( \mathbf{p}_u + |N(u)|^{-1/2} \sum_{j \in N(u)} \mathbf{y}_j \right)$$

where $N(u)$ = items rated by user $u$, $\mathbf{y}_j$ = implicit item factors

1. What does the $\sum_{j \in N(u)} \mathbf{y}_j$ term capture?
2. Why divide by $|N(u)|^{-1/2}$ instead of $|N(u)|$?
3. How many additional parameters does SVD++ add?
4. Why did SVD++ win the Netflix Prize?

**Hints:**
- Implicit feedback: the fact that a user rated an item (regardless of rating value)
- Normalization prevents users with many ratings from dominating
- Additional parameters: one $\mathbf{y}_j$ vector per item

**Learning Outcomes:**
- Understand implicit feedback integration
- Recognize why SVD++ outperforms basic MF
- Think about parameter scaling

---

## Problem 7: BPR (Bayesian Personalized Ranking)
**Difficulty:** Hard
**Topics:** Implicit feedback, pairwise ranking, BPR

BPR optimizes for ranking rather than rating prediction. It uses pairwise comparisons:

**Objective:**
$$\max \sum_{(u,i,j) \in D_S} \ln \sigma(\hat{r}_{ui} - \hat{r}_{uj})$$

where:
- $(u,i,j)$: user $u$ interacted with $i$ but not $j$
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ (sigmoid)

1. What assumption does BPR make about user preferences?
2. Why is this better than treating missing interactions as negative?
3. Derive the gradient with respect to $\mathbf{p}_u$
4. How would you sample negative items $j$ efficiently?

**Hints:**
- BPR assumes: user prefers observed items over unobserved
- Missing ≠ negative (user might just not know about item)
- Gradient of $\ln \sigma(x) = \sigma(-x)$
- Uniform sampling or popularity-based sampling

**Learning Outcomes:**
- Understand pairwise ranking approaches
- Work with implicit feedback
- Implement BPR loss

---

## Problem 8: Computational Complexity Analysis
**Difficulty:** Medium
**Topics:** Complexity analysis, scalability

Compare the computational complexity of one epoch of training:

**SGD-based MF:**
- For each rating $(u,i)$ in dataset
- Update $\mathbf{p}_u$ and $\mathbf{q}_i$
- Cost per update: $O(k)$ (k = number of factors)

**ALS:**
- For each user: solve $(\mathbf{Q}_u \mathbf{Q}_u^T + \lambda \mathbf{I})^{-1} \mathbf{Q}_u \mathbf{r}_u$
- Matrix inversion: $O(k^3)$

**Given:**
- $|U|$ = 10,000 users
- $|I|$ = 5,000 items
- $|R|$ = 1,000,000 ratings
- $k$ = 100 factors
- Average ratings per user = 100

1. Calculate total FLOPs for one epoch of SGD
2. Calculate total FLOPs for one epoch of ALS
3. Which is faster for this dataset?
4. At what ratio of users to ratings does ALS become faster?

**Learning Outcomes:**
- Analyze algorithmic complexity
- Compare SGD vs. ALS practically
- Make informed algorithm choices

---

## Programming Exercises

### Exercise 1: Implement SGD-Based MF from Scratch
**Dataset:** MovieLens 100K
**Task:** Build basic MF with stochastic gradient descent

**Requirements:**
```python
class MatrixFactorization:
    def __init__(self, n_users, n_items, n_factors=20, learning_rate=0.01, reg=0.02):
        # Initialize user and item factors randomly
        self.P = np.random.normal(0, 0.1, (n_users, n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, n_factors))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.mu = 0  # Global mean

    def predict(self, user, item):
        # Implement prediction with biases
        pass

    def train_epoch(self, ratings):
        # One epoch of SGD
        # Shuffle ratings
        # For each rating: compute error, update factors
        pass

    def fit(self, train_ratings, valid_ratings, n_epochs=20):
        # Train for n_epochs
        # Track train and validation RMSE
        # Early stopping if validation RMSE increases
        pass
```

**Evaluation:**
1. Train for 20 epochs
2. Plot training and validation RMSE curves
3. Report final test RMSE (target: < 0.93)
4. Visualize learned latent factors (PCA to 2D)

---

### Exercise 2: Compare SGD vs. ALS
**Dataset:** MovieLens 1M
**Task:** Implement both optimizers and compare

**Implementation:**
- SGD: From Exercise 1
- ALS: Use closed-form solution per user/item

**Comparison Metrics:**
1. Training time per epoch
2. Convergence speed (epochs to reach RMSE < 0.90)
3. Final test RMSE
4. Memory usage

**Expected Results:**
```
| Algorithm | Time/Epoch | Epochs to Converge | Final RMSE | Memory |
|-----------|------------|-------------------|------------|--------|
| SGD       | 5s         | 15                | 0.88       | 50 MB  |
| ALS       | 12s        | 8                 | 0.87       | 80 MB  |
```

---

### Exercise 3: Implement SVD++
**Dataset:** MovieLens 100K
**Task:** Extend basic MF to include implicit feedback

**Key Changes:**
```python
class SVDPlusPlus:
    def __init__(self, n_users, n_items, n_factors=20):
        self.P = np.random.normal(0, 0.1, (n_users, n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, n_factors))
        self.Y = np.random.normal(0, 0.1, (n_items, n_factors))  # Implicit factors
        # ... biases ...

    def predict(self, user, item, user_items):
        # user_items: items rated by user (for implicit signal)
        # Implement SVD++ prediction formula
        pass
```

**Evaluation:**
1. Compare basic MF vs. SVD++
2. Measure RMSE improvement (expect ~2-3%)
3. Analyze when implicit feedback helps most

---

### Exercise 4: Hyperparameter Tuning with Grid Search
**Dataset:** MovieLens 100K
**Task:** Find optimal hyperparameters

**Parameters to Tune:**
- Number of factors: [10, 20, 50, 100]
- Learning rate: [0.001, 0.005, 0.01, 0.05]
- Regularization: [0.001, 0.01, 0.1]

**Implementation:**
```python
from itertools import product
from sklearn.model_selection import train_test_split

best_rmse = float('inf')
best_params = None

for k, lr, reg in product(factors, learning_rates, regularizations):
    model = MatrixFactorization(n_factors=k, learning_rate=lr, reg=reg)
    model.fit(train, valid)
    rmse = evaluate(model, valid)

    if rmse < best_rmse:
        best_rmse = rmse
        best_params = (k, lr, reg)
```

**Report:**
- Heatmaps of RMSE vs. each parameter pair
- Best parameter combination
- Validation curve (performance vs. training time)

---

### Exercise 5: Implement BPR for Implicit Feedback
**Dataset:** MovieLens 100K (binarized: rating ≥4 = positive)
**Task:** Implement Bayesian Personalized Ranking

**Requirements:**
```python
class BPR:
    def train_step(self, user, pos_item, neg_item):
        # Compute scores
        x_ui = self.predict(user, pos_item)
        x_uj = self.predict(user, neg_item)
        x_uij = x_ui - x_uj

        # Compute sigmoid
        sigmoid = 1 / (1 + np.exp(-x_uij))

        # Update factors
        # ... implement gradient updates ...

    def sample_negative(self, user, pos_item):
        # Sample item not interacted with by user
        pass
```

**Evaluation Metrics (not RMSE!):**
- AUC (Area Under ROC Curve)
- Precision@10
- Recall@10
- NDCG@10

**Expected:** AUC > 0.85, Precision@10 > 0.30

---

### Exercise 6: Using Surprise Library
**Dataset:** MovieLens 1M
**Task:** Leverage Surprise for advanced MF variants

**Algorithms to Compare:**
```python
from surprise import SVD, SVDpp, NMF
from surprise.model_selection import cross_validate, GridSearchCV

# Basic SVD
algo_svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)

# SVD++
algo_svdpp = SVDpp(n_factors=20, n_epochs=20)

# Non-Negative Matrix Factorization
algo_nmf = NMF(n_factors=15, n_epochs=50)

# Grid search for SVD
param_grid = {
    'n_factors': [50, 100, 150],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.02, 0.05, 0.1]
}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
gs.fit(data)
```

**Analysis:**
1. Compare RMSE: SVD vs. SVD++ vs. NMF
2. Training time comparison
3. Optimal hyperparameters from grid search

---

### Exercise 7: Temporal Dynamics (TimeSVD++)
**Dataset:** MovieLens with timestamps
**Task:** Incorporate temporal dynamics into MF

**Key Idea:** User preferences and item perceptions change over time

**Model:**
$$\hat{r}_{ui}(t) = \mu + b_u(t) + b_i(t) + \mathbf{p}_u(t)^T \mathbf{q}_i(t)$$

**Simplification:** Bin time into periods (e.g., months)

**Implementation:**
- Learn separate biases per time bin
- Compare temporal MF vs. static MF
- Analyze how biases change over time (plot trend)

---

## Discussion Questions

1. **Cold Start**: How does MF handle new users? New items? Compare to memory-based CF.

2. **Interpretability**: Can you interpret the k latent factors? What might they represent for a movie dataset?

3. **Explicit vs. Implicit**: Why is RMSE inappropriate for implicit feedback? What metrics should you use?

4. **Netflix Prize**: Why did Netflix never deploy the winning solution? What does this teach about research vs. production?

5. **Negative Sampling**: In BPR, how should you sample negative items? Uniform? Popularity-based? What are the trade-offs?

6. **Parallelization**: SGD is inherently sequential. How do systems like Hogwild! parallelize it anyway?

7. **Online Learning**: Can you update MF incrementally as new ratings arrive? Or must you retrain from scratch?

8. **Feature Engineering**: How would you incorporate side information (user age, movie genre) into MF? (Hint: Factorization Machines)

---

## Challenge Problem: Matrix Completion Guarantees

**Difficulty:** Very Hard
**Topics:** Theoretical guarantees, low-rank recovery

**Background:** Under certain conditions, matrix factorization can exactly recover a low-rank matrix from partial observations.

**Setup:**
- True rating matrix $R$ has rank $r$
- We observe $m$ random entries
- Goal: Recover $R$ exactly via MF

**Questions:**
1. Research the conditions under which exact recovery is possible (look up "matrix completion theory")
2. What is the minimum number of observations needed? (It's $O(nr \log n)$ for an $n \times n$ matrix)
3. Why do practical recommendation systems rarely satisfy these theoretical assumptions?
4. Design an experiment: Generate a synthetic low-rank matrix, observe random entries, run MF, measure recovery error

**Learning Outcomes:**
- Understand theoretical foundations of MF
- Connect theory to practice
- Recognize when theory doesn't apply

---

## References

### Papers
1. Koren, Y., Bell, R., & Volinsky, C. (2009). "Matrix factorization techniques for recommender systems". Computer, IEEE.
2. Koren, Y. (2008). "Factorization meets the neighborhood: A multifaceted collaborative filtering model". KDD.
3. Rendle, S., et al. (2009). "BPR: Bayesian personalized ranking from implicit feedback". UAI.
4. Hu, Y., Koren, Y., & Volinsky, C. (2008). "Collaborative filtering for implicit feedback datasets". ICDM.

### Textbooks
- Aggarwal (2016). Recommender Systems. Chapter 3: Model-Based Collaborative Filtering.
- Ricci et al. (2015). Recommender Systems Handbook. Chapter 3: Matrix Factorization Techniques.

### Datasets
- MovieLens: https://grouplens.org/datasets/movielens/
- Netflix Prize (if accessible): https://www.kaggle.com/netflix-inc/netflix-prize-data
- Last.fm: http://millionsongdataset.com/lastfm/
- Goodreads: https://mengtingwan.github.io/data/goodreads.html

---

*Return to [Week 3 Main Page](README.md)*
