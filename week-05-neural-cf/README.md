# Week 5: Neural Collaborative Filtering

## Overview

Deep learning revolutionized recommendation systems by replacing linear models with neural networks that learn complex, non-linear user-item interactions. This week covers the transition from matrix factorization to neural collaborative filtering.

## Learning Objectives

- Understand how neural networks extend matrix factorization
- Implement Neural Collaborative Filtering (NCF)
- Master deep matrix factorization variants (AutoRec, VAE-CF)
- Learn training techniques for deep recommenders

## Topics Covered

### [1. From MF to Neural Networks](mf-to-neural.md)
- MF as a shallow neural network
- Non-linear feature interactions
- Expressiveness arguments

### [2. Neural Collaborative Filtering (NCF)](ncf.md)
- Generalized Matrix Factorization (GMF)
- Multi-Layer Perceptron (MLP) pathway
- NeuMF: Fusion of GMF and MLP
- Pre-training strategies

**Paper**: He et al. (2017). "Neural collaborative filtering". *WWW*.

### [3. Deep Matrix Factorization Variants](deep-variants.md)
- **AutoRec**: Autoencoder-based CF
- **CDAE**: Collaborative Denoising Autoencoders
- **VAE-CF**: Variational Autoencoders
- **Adversarial training**

### [4. Training Deep Recommenders](training.md)
- Negative sampling strategies
- Point-wise vs. pair-wise vs. list-wise loss
- Batch construction
- Hard negative mining
- Regularization (dropout, batch norm)

### [5. Code Examples](code-examples.md)
- NCF from scratch (PyTorch)
- AutoRec implementation
- Comparison with MF

### [6. Practice Problems](practice-problems.md)

## Key Architecture: NeuMF

```
User ID → Embedding (GMF) ─┐
Item ID → Embedding (GMF) ─┤→ Element-wise Product ─┐
                           │                        │
User ID → Embedding (MLP) ─┤                        │
Item ID → Embedding (MLP) ─┤→ Concat → MLP Layers ─┤
                                                    │
                                                    ├→ Concat → Output Layer → Prediction
```

## Performance

On MovieLens 1M:
- **MF**: HR@10 = 0.692, NDCG@10 = 0.425
- **NCF**: HR@10 = 0.726, NDCG@10 = 0.445
- **Improvement**: ~5% over MF

## Required Reading

1. **He, X., et al. (2017)**. "Neural collaborative filtering". *WWW*.
2. **Sedhain, S., et al. (2015)**. "AutoRec: Autoencoders meet collaborative filtering". *WWW*.
3. **Liang, D., et al. (2018)**. "Variational autoencoders for collaborative filtering". *WWW*.

*Return to [Main Course Page](../README.md)*
