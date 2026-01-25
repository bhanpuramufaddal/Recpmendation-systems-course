# Week 13: Production Systems and MLOps

## Overview

Deploying recommendation systems in production requires careful system design, scalability planning, and operational excellence. This week covers production architectures, MLOps practices, and the cold start problem.

## Topics

### [1. System Architecture](system-architecture.md)
- Batch vs. streaming processing
- Lambda and Kappa architectures
- Feature stores
- Model serving infrastructure
- Microservices design

### [2. Scalability Challenges](scalability.md)
- Distributed training (data parallelism, model parallelism)
- Online learning and incremental updates
- Caching strategies
- Load balancing
- Latency optimization

### [3. The Cold Start Problem](cold-start.md)
**New Users**:
- Popularity-based recommendations
- Demographic matching
- Onboarding questionnaires
- Explore-exploit strategies

**New Items**:
- Content-based bootstrapping
- Feature-based models
- Exploration boost

**Meta-Learning**:
- MAML for few-shot adaptation
- Learning to learn

### [4. Model Management](model-management.md)
- A/B testing infrastructure
- Model versioning and rollback
- Monitoring and alerting
- Shadow mode deployment
- Continuous training pipelines

## Production Stack Example

```
┌─────────────────────────────────────┐
│   Data Collection (Kafka, Kinesis)  │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│   Feature Engineering (Spark, Flink)│
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│   Feature Store (Feast, Tecton)     │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│ Model Training (PyTorch, TensorFlow)│
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│Model Serving (TF Serving, TorchServe)│
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│   Caching Layer (Redis, Memcached)  │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│  Monitoring (Prometheus, Grafana)   │
└─────────────────────────────────────┘
```

## Latency Budget (Example: YouTube)

| Component | Latency | Cumulative |
|-----------|---------|------------|
| Candidate Generation | 5ms | 5ms |
| Ranking | 40ms | 45ms |
| Re-ranking | 30ms | 75ms |
| Serving Overhead | 25ms | 100ms |
| **Total** | | **100ms** |

## Required Reading

1. **Amatriain, X., & Basilico, J. (2015)**. "Recommender systems in industry: A Netflix case study". *Recommender Systems Handbook*.
2. **Sculley, D., et al. (2015)**. "Hidden technical debt in machine learning systems". *NIPS*.

*Return to [Main Course Page](../README.md)*
