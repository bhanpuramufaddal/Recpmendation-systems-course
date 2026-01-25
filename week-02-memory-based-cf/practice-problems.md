# Week 2: Memory-Based Collaborative Filtering - Practice Problems

## Overview
These problems test your understanding of user-based and item-based collaborative filtering, similarity measures, and the practical challenges of memory-based recommendation systems. Focus on computational complexity, similarity calculations, and implementation strategies.

---

## Problem 1: Pearson Correlation by Hand
**Difficulty:** Easy
**Topics:** Pearson correlation, similarity measures

Given the following user-item rating matrix:

| User | Movie A | Movie B | Movie C | Movie D |
|------|---------|---------|---------|---------|
| Alice| 5       | 3       | -       | 1       |
| Bob  | 4       | -       | -       | 1       |
| Carol| 1       | 1       | -       | 5       |
| Dave | 1       | -       | -       | 4       |

Calculate the Pearson correlation between Alice and Carol using only their co-rated items.

**Hints:**
- Only use items rated by both users (Movie A, Movie B, Movie D)
- Remember to mean-center the ratings for each user
- Pearson formula: $\text{Pearson}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}$

**Learning Outcomes:**
- Master manual similarity calculations
- Understand the importance of co-rated items
- Recognize how mean-centering affects similarity

---

## Problem 2: User-Based vs. Item-Based Trade-offs
**Difficulty:** Medium
**Topics:** User-based CF, item-based CF, scalability

You're building a recommendation system for:
- **Scenario A**: Netflix (500M users, 10K movies)
- **Scenario B**: A small book club (100 users, 50K books)

For each scenario:
1. Would you choose user-based or item-based CF?
2. Justify your choice based on computational complexity
3. What precomputation strategies would you use?

**Hints:**
- Consider the user-to-item ratio
- User-based: O(|U|² × |I|) for similarity computation
- Item-based: O(|I|² × |U|) for similarity computation
- Think about which similarities change more frequently

**Learning Outcomes:**
- Understand computational trade-offs
- Make architecture decisions based on scale
- Recognize when to use precomputation

---

## Problem 3: Cold Start Problem
**Difficulty:** Medium
**Topics:** Cold start, limitations of memory-based CF

A new movie "Inception 2" is added to your recommendation system. No users have rated it yet.

1. Can user-based CF recommend this movie? Why or why not?
2. Can item-based CF recommend this movie? Why or why not?
3. Propose three strategies to handle this cold start problem

**Hints:**
- Think about what user-based and item-based CF require
- Consider hybrid approaches
- Content-based features might help

**Learning Outcomes:**
- Recognize fundamental limitations of CF
- Understand the cold start problem
- Think about hybrid solutions

---

## Problem 4: Adjusted Cosine Similarity Derivation
**Difficulty:** Hard
**Topics:** Cosine similarity, adjusted cosine similarity, mathematical derivation

Standard cosine similarity for items i and j is:
$$\text{sim}(i,j) = \frac{\sum_{u \in U_{ij}} r_{ui} \cdot r_{uj}}{\sqrt{\sum_{u \in U_{ij}} r_{ui}^2} \sqrt{\sum_{u \in U_{ij}} r_{uj}^2}}$$

**Adjusted cosine similarity** subtracts user mean ratings:
$$\text{sim}_{adj}(i,j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u) \cdot (r_{uj} - \bar{r}_u)}{\sqrt{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{u \in U_{ij}} (r_{uj} - \bar{r}_u)^2}}$$

1. Why is adjusted cosine similarity preferred for item-based CF?
2. Give a concrete example where standard cosine fails but adjusted cosine succeeds
3. What is the relationship between adjusted cosine similarity and Pearson correlation?

**Hints:**
- Consider users with different rating scales (one rates 1-2, another rates 4-5)
- Think about what happens without mean-centering
- Compare the formulas carefully

**Learning Outcomes:**
- Understand why mean-centering matters
- Recognize bias in rating scales
- Connect different similarity measures mathematically

---

## Problem 5: Prediction Formula Understanding
**Difficulty:** Medium
**Topics:** Prediction formulas, k-nearest neighbors

Given the user-based prediction formula:
$$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} \text{sim}(u,v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N_k(u)} |\text{sim}(u,v)|}$$

1. Why do we use $|sim(u,v)|$ (absolute value) in the denominator?
2. What would happen if we removed the mean-centering terms ($\bar{r}_u$ and $\bar{r}_v$)?
3. How does k (number of neighbors) affect prediction quality and computation time?

**Hints:**
- Consider negative similarities
- Think about users with different rating scales
- Larger k = more computation but potentially more robust

**Learning Outcomes:**
- Understand components of prediction formulas
- Recognize the role of normalization
- Make informed choices about k

---

## Problem 6: Jaccard Similarity for Implicit Feedback
**Difficulty:** Easy
**Topics:** Jaccard similarity, implicit feedback

You have binary implicit feedback (watched/not watched) for 5 users and 6 movies:

| User | M1 | M2 | M3 | M4 | M5 | M6 |
|------|----|----|----|----|----|----|
| U1   | 1  | 1  | 0  | 1  | 0  | 0  |
| U2   | 1  | 1  | 1  | 0  | 0  | 1  |
| U3   | 0  | 1  | 1  | 1  | 0  | 0  |

Calculate the Jaccard similarity between U1 and U2.

**Formula:**
$$\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

**Learning Outcomes:**
- Work with implicit feedback
- Understand Jaccard similarity
- Recognize when to use different similarity measures

---

## Problem 7: Sparsity Analysis
**Difficulty:** Medium
**Topics:** Sparsity, data quality, practical considerations

MovieLens-1M dataset has:
- 6,040 users
- 3,706 movies
- 1,000,209 ratings

1. Calculate the sparsity of this matrix (% of empty cells)
2. If you require minimum 10 co-ratings for reliable similarity, estimate how many user pairs will have sufficient overlap
3. Propose strategies to handle extreme sparsity

**Hints:**
- Sparsity = 1 - (ratings / (users × items))
- Average ratings per user ≈ 1,000,209 / 6,040 ≈ 166
- Consider dimensionality reduction techniques

**Learning Outcomes:**
- Quantify data sparsity
- Understand its impact on similarity quality
- Think about practical solutions

---

## Problem 8: Significance Weighting
**Difficulty:** Hard
**Topics:** Significance weighting, similarity refinement

Standard Pearson correlation can be misleading with few co-ratings. **Significance weighting** adjusts similarity:

$$\text{sim}_{weighted}(u,v) = \text{sim}(u,v) \times \min\left(1, \frac{|I_{uv}|}{threshold}\right)$$

where $|I_{uv}|$ is the number of co-rated items.

1. If threshold = 50, calculate the weighted similarity for:
   - Users with Pearson = 0.9 and 10 co-ratings
   - Users with Pearson = 0.7 and 60 co-ratings
2. Which pair would you trust more? Why?
3. How would you choose an appropriate threshold?

**Hints:**
- Small overlap → unreliable correlation
- Weighting down-weights unreliable similarities
- Consider cross-validation for threshold selection

**Learning Outcomes:**
- Recognize unreliable similarities
- Apply significance weighting
- Understand statistical reliability

---

## Programming Exercises

### Exercise 1: Implement User-Based CF from Scratch
**Dataset:** MovieLens 100K
**Task:** Implement user-based collaborative filtering without using libraries

**Requirements:**
1. Compute user-user Pearson correlations
2. For each user, find top-k similar users (k=20)
3. Generate top-N recommendations (N=10)
4. Evaluate with RMSE on a test set

**Expected Output:**
- User similarity matrix (sparse)
- Predictions for test ratings
- RMSE score (target: < 1.0)

**Starter Code Structure:**
```python
def compute_user_similarities(ratings_matrix):
    # Compute Pearson correlation between all user pairs
    pass

def predict_rating(user_id, item_id, similarities, ratings, k=20):
    # Use k-nearest neighbors to predict rating
    pass

def evaluate_rmse(predictions, ground_truth):
    # Compute RMSE
    pass
```

---

### Exercise 2: Item-Based CF with Precomputation
**Dataset:** MovieLens 100K
**Task:** Implement item-based CF with offline precomputation

**Requirements:**
1. Precompute item-item similarities (adjusted cosine)
2. Save similarity matrix to disk
3. At runtime, load similarities and generate recommendations
4. Compare speed with user-based CF

**Expected Output:**
- Item similarity matrix
- Recommendation generation time (< 10ms per user)
- Comparison table: user-based vs. item-based

---

### Exercise 3: Similarity Measure Comparison
**Dataset:** MovieLens 100K
**Task:** Compare different similarity measures

**Implement:**
1. Pearson correlation
2. Cosine similarity
3. Adjusted cosine similarity
4. Jaccard similarity (binarize ratings: ≥4 = liked)

**Evaluation:**
- RMSE for each similarity measure
- Precision@10 for top-N recommendations
- Analyze which works best and why

**Expected Output:**
```
| Similarity Measure    | RMSE  | Precision@10 |
|-----------------------|-------|--------------|
| Pearson              | 0.95  | 0.32         |
| Cosine               | 0.98  | 0.30         |
| Adjusted Cosine      | 0.93  | 0.34         |
| Jaccard (binary)     | N/A   | 0.28         |
```

---

### Exercise 4: Handling Cold Start
**Dataset:** MovieLens 100K
**Task:** Implement strategies for new users

**Scenario:** A new user rates 5 movies. Generate recommendations.

**Strategies to implement:**
1. **Popularity baseline**: Recommend most popular items
2. **Few-shot user-based**: Use the 5 ratings to find similar users
3. **Item-based**: Find items similar to the 5 rated items

**Compare:**
- Quality of recommendations (precision/recall)
- Coverage (% of catalog recommended)
- Diversity (genre distribution)

---

### Exercise 5: Scalability Analysis
**Dataset:** MovieLens 100K and 1M
**Task:** Analyze computational bottlenecks

**Measure:**
1. Time to compute all user-user similarities
2. Time to compute all item-item similarities
3. Memory usage for similarity matrices
4. Prediction time (per user, per item)

**Questions:**
1. At what scale does user-based become infeasible?
2. What percentage of similarities can you prune (below threshold) to save memory?
3. How much faster is item-based with precomputation?

**Expected Output:**
```
| Dataset | Users | Items | User-User Time | Item-Item Time | Memory (GB) |
|---------|-------|-------|----------------|----------------|-------------|
| 100K    | 943   | 1,682 | 12s            | 5s             | 0.01        |
| 1M      | 6,040 | 3,706 | 180s           | 25s            | 0.15        |
```

---

### Exercise 6: Using the Surprise Library
**Dataset:** MovieLens 100K
**Task:** Implement and tune memory-based CF using Surprise

**Implement:**
```python
from surprise import KNNBasic, KNNWithMeans, Dataset
from surprise.model_selection import cross_validate, GridSearchCV

# User-based with Pearson
algo_user = KNNBasic(sim_options={'name': 'pearson', 'user_based': True})

# Item-based with adjusted cosine
algo_item = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})

# Grid search for k
param_grid = {'k': [10, 20, 30, 40, 50]}
gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse'], cv=5)
```

**Tasks:**
1. Compare user-based vs. item-based
2. Find optimal k through cross-validation
3. Analyze impact of mean-centering (KNNBasic vs. KNNWithMeans)

---

## Discussion Questions

1. **Theoretical Limits**: What is the fundamental assumption of collaborative filtering? In what scenarios would this assumption break down completely?

2. **Privacy**: Memory-based CF stores all user ratings. What privacy concerns arise? How could you implement privacy-preserving CF?

3. **Attack Resistance**: How could malicious users manipulate memory-based CF (e.g., shilling attacks)? What defenses exist?

4. **Beyond Ratings**: How would you adapt user-based CF for implicit feedback (clicks, views, purchases)? What changes to similarity measures are needed?

5. **Temporal Dynamics**: User preferences change over time. How could you incorporate recency into memory-based CF? Should recent ratings be weighted more?

6. **Diversity vs. Accuracy**: Memory-based CF tends to recommend popular items similar to what users already like. How could you increase diversity while maintaining relevance?

7. **Explainability**: How would you explain a recommendation from user-based CF to an end user? What information would you surface?

8. **Hybrid Approaches**: When would you combine memory-based CF with content-based filtering? Design a hybrid system architecture.

---

## Challenge Problem: Neighborhood Refinement

**Difficulty:** Hard
**Topics:** Advanced similarity, graph-based methods

Standard k-NN uses the k most similar users/items. But what if those neighbors are themselves similar to each other (redundant)?

**Task:**
1. Implement a **diversified neighborhood** selection algorithm that:
   - Finds highly similar neighbors
   - Ensures neighbors are dissimilar to each other
   - Balances similarity and diversity

2. Formulate this as an optimization problem:
   - Maximize: average similarity to target user
   - Minimize: average similarity among neighbors
   - Subject to: |neighborhood| = k

3. Compare standard k-NN vs. diversified k-NN on MovieLens

**Hints:**
- This is related to Maximum Marginal Relevance (MMR)
- Consider greedy algorithms
- Balance parameter λ controls similarity vs. diversity trade-off

**Expected Outcome:**
- Diversified neighbors may reduce RMSE slightly
- But increase diversity and coverage significantly

---

## References

### Papers
1. Resnick, P., et al. (1994). "GroupLens: An open architecture for collaborative filtering of netnews". CSCW.
2. Sarwar, B., et al. (2001). "Item-based collaborative filtering recommendation algorithms". WWW.
3. Herlocker, J. L., et al. (1999). "An algorithmic framework for performing collaborative filtering". SIGIR.

### Textbooks
- Aggarwal, C.C. (2016). Recommender Systems. Chapter 2: Neighborhood-Based Collaborative Filtering.
- Ricci et al. (2015). Recommender Systems Handbook. Chapter 5: Collaborative Filtering.

### Datasets
- MovieLens 100K: https://grouplens.org/datasets/movielens/100k/
- MovieLens 1M: https://grouplens.org/datasets/movielens/1m/
- Book-Crossing: http://www2.informatik.uni-freiburg.de/~cziegler/BX/
- Jester: http://eigentaste.berkeley.edu/dataset/

---

*Return to [Week 2 Main Page](README.md)*
