# Week 18: Cross-Domain and Transfer Learning

## Overview

Transfer learning enables recommendations across domains and for new users with limited data. Meta-learning allows models to quickly adapt to new scenarios.

## Topics

### [1. Cross-Domain Recommendation](cross-domain.md)
- Domain adaptation techniques
- Shared user representations
- Transfer from data-rich to data-poor domains
- Cold-start alleviation

**Example**: Transfer from movie ratings to book recommendations

### [2. Meta-Learning for Recommendations](meta-learning.md)
- **MAML**: Model-Agnostic Meta-Learning
- Learning to learn user preferences
- Few-shot recommendation
- Rapid adaptation to new users

**Applications**:
- New user cold-start
- Personalization with limited data

### [3. Federated Learning](federated-learning.md)
- Privacy-preserving collaborative filtering
- Federated averaging
- Communication efficiency
- Non-IID data handling

**Use Case**: Mobile keyboard recommendations without sharing user data

## MAML Algorithm

```
1. Sample batch of tasks (users)
2. For each task:
     a. Compute task-specific gradient
     b. Take gradient step
3. Meta-update: Aggregate task gradients
4. Repeat until convergence
```

**Result**: Model that quickly adapts to new users with few examples

## Federated Averaging

```
Server:
  Initialize global model w

For each round:
  1. Send w to subset of clients
  2. Clients train locally on their data
  3. Clients send updates Δw to server
  4. Server aggregates: w ← w + (1/n)Σ Δw
```

**Benefit**: Privacy-preserving (data never leaves device)

*Return to [Main Course Page](../README.md)*
