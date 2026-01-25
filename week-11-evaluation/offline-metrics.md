# Week 11: Offline Evaluation Metrics

## Overview

**Offline evaluation** assesses recommendation quality using **historical data** before deploying to users.

**Advantages**:
- **Safe**: No risk to user experience
- **Fast**: Test many models quickly
- **Reproducible**: Same dataset → consistent results

**Limitation**: Offline ≠ online performance (distribution shift, feedback loops).

This document covers offline metrics for recommendation systems.

---

## Accuracy Metrics

### Rating Prediction

**RMSE** (Root Mean Squared Error):
$$\text{RMSE} = \sqrt{\frac{1}{|T|} \sum_{(u,i) \in T} (r_{ui} - \hat{r}_{ui})^2}$$

**MAE** (Mean Absolute Error):
$$\text{MAE} = \frac{1}{|T|} \sum_{(u,i) \in T} |r_{ui} - \hat{r}_{ui}|$$

**When to use**: Rating prediction tasks (Netflix Prize).

```python
import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
```

---

## Ranking Metrics

### Precision@K

**Definition**: Fraction of top-K recommendations that are relevant.

$$\text{Precision@K} = \frac{|\text{relevant} \cap \text{top-K}|}{K}$$

**Example**:
```
Recommended: [A, B, C, D, E]  (K=5)
Relevant:    [A, C, F, G]

Precision@5 = 2/5 = 0.4
```

```python
def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k
```

---

### Recall@K

**Definition**: Fraction of relevant items in top-K.

$$\text{Recall@K} = \frac{|\text{relevant} \cap \text{top-K}|}{|\text{relevant}|}$$

```python
def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant) if len(relevant) > 0 else 0
```

---

### F1@K

**Harmonic mean** of precision and recall:

$$\text{F1@K} = 2 \cdot \frac{\text{Precision@K} \cdot \text{Recall@K}}{\text{Precision@K} + \text{Recall@K}}$$

---

### MAP (Mean Average Precision)

**Average Precision** for single user:
$$\text{AP} = \frac{1}{|\text{relevant}|} \sum_{k=1}^K \text{Precision@k} \cdot \text{rel}(k)$$

where $\text{rel}(k) = 1$ if item at position $k$ is relevant, else 0.

**MAP**: Average AP across all users.

**Example**:
```
Recommended: [A, B, C, D, E]
Relevant:    [A, C, E]

Precision@1 = 1/1 = 1.0  (A relevant)
Precision@2 = 1/2 = 0.5  (B not relevant)
Precision@3 = 2/3 = 0.67 (C relevant)
Precision@5 = 3/5 = 0.6  (E relevant)

AP = (1.0 + 0.67 + 0.6) / 3 = 0.76
```

```python
def average_precision(recommended, relevant):
    score = 0.0
    num_hits = 0.0

    for i, item in enumerate(recommended):
        if item in relevant:
            num_hits += 1
            score += num_hits / (i + 1)

    return score / len(relevant) if len(relevant) > 0 else 0

def mean_average_precision(recommendations, relevances):
    return np.mean([average_precision(rec, rel)
                    for rec, rel in zip(recommendations, relevances)])
```

---

### NDCG (Normalized Discounted Cumulative Gain)

**DCG@K**: Discounted cumulative gain (values position).

$$\text{DCG@K} = \sum_{i=1}^K \frac{rel_i}{\log_2(i+1)}$$

**IDCG**: Ideal DCG (perfect ranking).

**NDCG@K**:
$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

**Benefits**: Accounts for relevance grading (not just binary) and position.

```python
def dcg_at_k(relevances, k):
    relevances = np.asarray(relevances)[:k]
    if relevances.size:
        return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
    return 0.0

def ndcg_at_k(predicted_relevances, true_relevances, k):
    dcg = dcg_at_k(predicted_relevances, k)
    idcg = dcg_at_k(sorted(true_relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0
```

---

### MRR (Mean Reciprocal Rank)

**Reciprocal Rank**: 1 / (position of first relevant item).

$$\text{MRR} = \frac{1}{|U|} \sum_{u=1}^{|U|} \frac{1}{\text{rank}_u}$$

**Use case**: When only 1 relevant item per query (search).

```python
def reciprocal_rank(recommended, relevant):
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0

def mean_reciprocal_rank(recommendations, relevances):
    return np.mean([reciprocal_rank(rec, rel)
                    for rec, rel in zip(recommendations, relevances)])
```

---

## Beyond-Accuracy Metrics

### Coverage

**Catalog Coverage**: % of items recommended at least once.

$$\text{Coverage} = \frac{|\bigcup_{u \in U} R_u|}{|I|}$$

where $R_u$ = items recommended to user $u$.

**Goal**: High coverage → diverse recommendations (not just popular items).

```python
def catalog_coverage(all_recommendations, n_items):
    unique_items = set()
    for recs in all_recommendations:
        unique_items.update(recs)
    return len(unique_items) / n_items
```

---

### Diversity

**Intra-list diversity**: How different are items within a single recommendation list?

$$\text{Diversity} = \frac{1}{K(K-1)} \sum_{i \neq j} d(i, j)$$

where $d(i, j)$ = dissimilarity (e.g., 1 - cosine similarity).

```python
from sklearn.metrics.pairwise import cosine_similarity

def intra_list_diversity(recommended_items, item_features):
    """
    item_features: (n_items, n_features) embeddings
    """
    rec_features = item_features[recommended_items]
    similarity_matrix = cosine_similarity(rec_features)

    # Average pairwise dissimilarity
    n = len(recommended_items)
    diversity = 0
    for i in range(n):
        for j in range(i+1, n):
            diversity += (1 - similarity_matrix[i, j])

    return diversity / (n * (n-1) / 2) if n > 1 else 0
```

---

### Novelty

**Novelty**: How surprising are recommendations?

$$\text{Novelty} = -\frac{1}{K} \sum_{i \in R} \log_2 p(i)$$

where $p(i)$ = popularity (% users who interacted with item $i$).

**High novelty**: Recommending niche items (low $p(i)$).

```python
def novelty(recommended_items, item_popularity):
    """
    item_popularity: dict {item_id: popularity score (0-1)}
    """
    scores = [-np.log2(item_popularity[i] + 1e-10)
              for i in recommended_items]
    return np.mean(scores)
```

---

### Serendipity

**Serendipity**: Relevant + unexpected.

$$\text{Serendipity} = \text{relevant} \cap \text{unexpected}$$

**Unexpectedness**: Not predictable by simple baseline (e.g., not popular).

**Challenging to measure**: Requires defining "expected" recommendations.

---

## Combining Metrics

### Multi-Objective Evaluation

**Reality**: Trade-offs between metrics.

**Example**:
- High precision → recommend only popular items
- High diversity → recommend unpopular items

**Solution**: Combine metrics:

$$\text{Score} = \alpha \cdot \text{NDCG} + \beta \cdot \text{Diversity} + \gamma \cdot \text{Coverage}$$

**Hyperparameters** ($\alpha, \beta, \gamma$): Domain-specific.

---

## Implementation: Evaluation Suite

```python
class RecommenderEvaluator:
    def __init__(self, item_features=None):
        self.item_features = item_features

    def evaluate(self, recommendations, ground_truth, k=10):
        """
        recommendations: List[List[item_ids]] (one list per user)
        ground_truth: List[List[item_ids]] (relevant items per user)
        """
        metrics = {}

        # Ranking metrics
        precisions = [precision_at_k(rec, gt, k)
                      for rec, gt in zip(recommendations, ground_truth)]
        recalls = [recall_at_k(rec, gt, k)
                   for rec, gt in zip(recommendations, ground_truth)]

        metrics['precision@{}'.format(k)] = np.mean(precisions)
        metrics['recall@{}'.format(k)] = np.mean(recalls)

        # MAP
        metrics['MAP'] = mean_average_precision(recommendations, ground_truth)

        # NDCG (assuming binary relevance for simplicity)
        ndcgs = []
        for rec, gt in zip(recommendations, ground_truth):
            relevances = [1 if item in gt else 0 for item in rec[:k]]
            ndcgs.append(ndcg_at_k(relevances, [1]*len(gt), k))
        metrics['NDCG@{}'.format(k)] = np.mean(ndcgs)

        # Coverage
        all_items = max(max(rec) for rec in recommendations) + 1
        metrics['coverage'] = catalog_coverage(recommendations, all_items)

        # Diversity (if features available)
        if self.item_features is not None:
            diversities = [intra_list_diversity(rec[:k], self.item_features)
                           for rec in recommendations]
            metrics['diversity'] = np.mean(diversities)

        return metrics


# Example
recommendations = [
    [1, 3, 5, 7, 9],
    [2, 4, 6, 8, 10]
]
ground_truth = [
    [1, 5, 11],
    [2, 12, 13]
]

evaluator = RecommenderEvaluator()
results = evaluator.evaluate(recommendations, ground_truth, k=5)
print(results)
```

---

## Summary

**Key Takeaways**:
1. **Accuracy**: RMSE/MAE for ratings, Precision/Recall for rankings
2. **Ranking**: MAP, NDCG (position matters), MRR
3. **Beyond accuracy**: Coverage, diversity, novelty, serendipity
4. **Trade-offs**: Balance relevance, diversity, novelty

**Best Practices**:
- **NDCG** for graded relevance, **MAP** for binary
- **Always report** coverage and diversity
- **Multiple metrics**: No single metric captures everything
- **Stratify**: Report metrics for different user groups (new vs. active)

**Next**: Experimental design and A/B testing.

---

## References

1. **Herlocker, J. L., et al. (2004)**. "Evaluating Collaborative Filtering Recommender Systems". *ACM TOIS*.
2. **Shani, G., & Gunawardana, A. (2011)**. "Evaluating Recommendation Systems". *Recommender Systems Handbook*.
3. **Vargas, S., & Castells, P. (2011)**. "Rank and Relevance in Novelty and Diversity Metrics for Recommender Systems". *RecSys*.
