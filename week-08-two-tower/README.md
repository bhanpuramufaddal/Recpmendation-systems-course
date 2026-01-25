# Week 8: Two-Tower Models and Large-Scale Retrieval

## Overview

Two-tower architectures enable efficient retrieval by encoding users and items separately, allowing pre-computation and fast approximate nearest neighbor (ANN) search. This is the foundation of modern industrial recommendation systems.

## Topics

### [1. Two-Tower Architecture](two-tower.md)
- Separate user and item encoders
- Dot-product similarity
- Training with in-batch negatives
- Asymmetric architectures

### [2. Retrieval at Scale](retrieval-scale.md)
- Approximate Nearest Neighbors (ANN)
- **FAISS** (Facebook)
- **ScaNN** (Google)
- **HNSW** graphs
- Product quantization

### [3. YouTube Recommendation System](youtube-system.md)
**Industry Case Study**

**Architecture**:
- Candidate generation: Deep network reducing millions → hundreds
- Ranking: Rich features for top candidates

**Paper**: Covington et al. (2016). "Deep neural networks for YouTube recommendations". *RecSys*.

### [4. Multi-Task Learning](multi-task.md)
- Predicting multiple objectives
- **MMOE**: Multi-gate Mixture of Experts
- Hard parameter sharing
- Balancing tasks

## ANN Library Comparison

| Library | Developer | Best For | Speed | Accuracy |
|---------|-----------|----------|-------|----------|
| FAISS | Facebook | GPU acceleration | Very Fast | High |
| ScaNN | Google | TensorFlow integration | Fast | Very High |
| HNSW | Academic | Low latency | Fast | Very High |
| Annoy | Spotify | Simplicity | Medium | Medium |

*Return to [Main Course Page](../README.md)*
