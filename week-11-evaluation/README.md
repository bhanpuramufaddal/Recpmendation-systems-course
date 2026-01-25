# Week 11: Evaluation Methodologies

## Overview

Evaluating recommendation systems requires both offline metrics (computed on historical data) and online metrics (from A/B tests). This week covers evaluation methodologies, metrics, and best practices.

## Topics

### [1. Offline Evaluation Metrics](offline-metrics.md)
**Rating Prediction**: RMSE, MAE
**Ranking**: Precision@K, Recall@K, MAP, NDCG
**Beyond-Accuracy**: Diversity, novelty, coverage, serendipity

### [2. Experimental Design](experimental-design.md)
- Train-test splitting
- Temporal splitting
- Leave-one-out
- Cross-validation
- Avoiding data leakage

### [3. Online Evaluation (A/B Testing)](ab-testing.md)
**Metrics**: CTR, conversion, watch time, retention
**Design**: Control vs. treatment, sample size, duration
**Statistical testing**: t-tests, bootstrap
**Guardrail metrics**

### [4. Challenges in Evaluation](eval-challenges.md)
- Popularity bias
- Position bias
- Selection bias
- Correlation vs. causation
- Offline-online metric gap

## Key Metrics

### NDCG (Normalized Discounted Cumulative Gain)

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

$$\text{DCG@K} = \sum_{i=1}^K \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

### MAP (Mean Average Precision)

$$\text{MAP} = \frac{1}{|U|} \sum_{u} \frac{1}{|I_u|} \sum_{k=1}^{|I_u|} P(k) \cdot rel(k)$$

### A/B Test Sample Size

$$n = \frac{(z_{\alpha/2} + z_{\beta})^2 \cdot 2\bar{p}(1-\bar{p})}{(\delta)^2}$$

where $\delta$ = minimum detectable effect

*Return to [Main Course Page](../README.md)*
