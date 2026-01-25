# CS 329R: Recommendation Systems - From Foundations to Frontiers

## Course Overview

This comprehensive course provides deep coverage of recommendation systems, spanning classical collaborative filtering approaches to cutting-edge deep learning techniques and production systems. The course combines theoretical foundations with practical implementations, industry case studies, and research paper discussions.

**Prerequisites**: Linear algebra, probability theory, machine learning fundamentals (CS 229 or equivalent)

## Course Structure

The course is organized into 8 modules spanning 19 weeks, with each week covering specific topics in depth.

---

## Module 1: Introduction and Fundamentals

### [Week 1: The Recommendation Problem](week-01-introduction/)
Introduction to recommendation systems, historical context, applications across domains, the recommendation pipeline, and key challenges.

**Topics**:
- What Are Recommendation Systems?
- Historical Context and Evolution
- Applications Across Domains
- The Recommendation Pipeline
- Key Challenges (Cold Start, Sparsity, Scalability)

---

## Module 2: Collaborative Filtering Foundations

### [Week 2: Memory-Based Collaborative Filtering](week-02-memory-based-cf/)
User-based and item-based collaborative filtering, similarity measures, and limitations of memory-based methods.

**Topics**:
- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Similarity Measures Deep Dive
- Limitations of Memory-Based Methods

### [Week 3: Matrix Factorization](week-03-matrix-factorization/)
Model-based collaborative filtering using matrix factorization, optimization algorithms, advanced variants, and handling implicit feedback.

**Topics**:
- Matrix Factorization Framework
- Optimization Formulation
- Optimization Algorithms (SGD, ALS)
- Advanced Variants (SVD++, TimeSVD++, Factorization Machines)
- Handling Implicit Feedback (BPR, WRMF)

---

## Module 3: Content-Based and Hybrid Approaches

### [Week 4: Content-Based Recommendation](week-04-content-based/)
Feature representation, profile learning methods, and hybrid recommendation strategies.

**Topics**:
- Foundations of Content-Based Filtering
- Feature Representation (TF-IDF, CNN embeddings, Audio features)
- Profile Learning Methods
- Advantages and Limitations
- Hybrid Recommendation Strategies

---

## Module 4: Deep Learning for Recommendations

### [Week 5: Neural Collaborative Filtering](week-05-neural-cf/)
Deep learning approaches to collaborative filtering, including neural matrix factorization and training techniques.

**Topics**:
- From Matrix Factorization to Neural Networks
- Neural Collaborative Filtering (NCF)
- Deep Matrix Factorization Variants (AutoRec, VAE-CF)
- Training Deep Recommenders

### [Week 6: Sequential and Session-Based Recommendations](week-06-sequential/)
Modeling user sequences with RNNs, transformers, and reinforcement learning.

**Topics**:
- Modeling User Sequences
- Recurrent Neural Networks (GRU4Rec, LSTM)
- Transformer-Based Architectures (BERT4Rec, SASRec)
- Conversational and Interactive Recommendations

### [Week 7: Graph Neural Networks](week-07-gnn/)
Graph-based recommendations and knowledge-enhanced methods.

**Topics**:
- Graph-Based Recommendation
- Graph Neural Networks (GCN, GraphSAGE, PinSage)
- LightGCN and Simplified GNNs
- Knowledge Graph Integration (KGAT)

---

## Module 5: Modern Architectures and Industry Systems

### [Week 8: Two-Tower Models and Large-Scale Retrieval](week-08-two-tower/)
Scalable retrieval architectures and multi-task learning.

**Topics**:
- Two-Tower Architecture
- Retrieval at Scale (ANN, FAISS, ScaNN, HNSW)
- YouTube Recommendation System
- Multi-Task Learning

### [Week 9: Embeddings and Pre-training](week-09-embeddings/)
Learning embeddings, pre-training strategies, and multi-modal recommendations.

**Topics**:
- Learning User and Item Embeddings (Item2Vec)
- Pre-training Strategies
- Large Language Models for Recommendations
- Multi-Modal Recommendations (CLIP)

### [Week 10: Context-Aware and Bandit-Based Recommendations](week-10-context-bandits/)
Context-aware methods and exploration-exploitation with bandits and reinforcement learning.

**Topics**:
- Context-Aware Recommendation
- Multi-Armed Bandits
- Contextual Bandits
- Reinforcement Learning for Long-Term Optimization

---

## Module 6: Evaluation, Fairness, and Production Systems

### [Week 11: Evaluation Methodologies](week-11-evaluation/)
Offline and online evaluation, metrics, and challenges.

**Topics**:
- Offline Evaluation Metrics
- Experimental Design
- Online Evaluation (A/B Testing)
- Challenges in Evaluation

### [Week 12: Bias, Fairness, and Ethics](week-12-bias-fairness/)
Understanding and mitigating bias, fairness considerations, and ethical implications.

**Topics**:
- Types of Bias in Recommendations
- Debiasing Techniques
- Fairness in Recommendations
- Ethical Considerations

### [Week 13: Production Systems and MLOps](week-13-production/)
Building and deploying production recommendation systems.

**Topics**:
- System Architecture
- Scalability Challenges
- The Cold Start Problem
- Model Management

---

## Module 7: Industry Case Studies

### [Week 14: Social Media Platforms](week-14-social-media/)
In-depth case studies of social media recommendation systems.

**Topics**:
- Facebook News Feed Ranking
- Instagram Explore and Reels
- TikTok's For You Page
- LinkedIn Feed and Job Recommendations

### [Week 15: Streaming and E-Commerce Platforms](week-15-streaming-ecommerce/)
Deep dives into streaming and e-commerce recommendation systems.

**Topics**:
- Netflix Recommendation System
- Spotify Music Recommendations
- YouTube Recommendations
- Amazon Product Recommendations

### [Week 16: Emerging Applications](week-16-emerging-apps/)
Specialized domains and novel applications.

**Topics**:
- Airbnb Search and Recommendations
- Uber Eats and Food Delivery
- Dating Apps (Tinder, Bumble, Hinge)
- News Aggregation
- GitHub and Developer Tools
- Cursor AI: Code Completion with Online RL
- Education Platforms

---

## Module 8: Advanced Topics and Research Frontiers

### [Week 17: Explainability and Interpretability](week-17-explainability/)
Making recommendations interpretable and explainable.

**Topics**:
- Why Explainability Matters
- Explainability Techniques
- Inherently Interpretable Models

### [Week 18: Cross-Domain and Transfer Learning](week-18-transfer-learning/)
Transfer learning and meta-learning for recommendations.

**Topics**:
- Cross-Domain Recommendation
- Meta-Learning for Recommendations
- Federated Learning

### [Week 19: Emerging Paradigms](week-19-emerging-paradigms/)
Future directions and open research questions.

**Topics**:
- Generative Recommendations
- Foundation Models for Recommendations
- Interactive and Conversational Systems
- Emerging Challenges
- Open Research Questions

---

## Learning Resources

### Textbooks
1. Aggarwal, C.C. (2016). *Recommender Systems: The Textbook*. Springer.
2. Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook* (2nd ed.). Springer.
3. Jannach, D., & Zanker, M. (2024). *Recommender Systems: An Introduction*. Cambridge University Press.

### Key Research Papers
- **Classical**: Sarwar et al. (2001), Koren et al. (2009), Rendle (2010)
- **Deep Learning**: He et al. (2017), Covington et al. (2016), Hidasi et al. (2016)
- **Modern**: He et al. (2020), Sun et al. (2019), Yi et al. (2019)

### Code Repositories
- Official implementations linked in each week's code examples
- Open-source libraries: Surprise, LightFM, Implicit, RecBole, Microsoft Recommenders

---

## How to Use These Notes

Each week's folder contains:
- **Topic-specific markdown files** with detailed explanations, mathematical formulas, and algorithms
- **code-examples.md** with Python implementations (both from-scratch and using libraries)
- **paper-summaries.md** (when applicable) with research paper analyses
- **practice-problems.md** with exercises and solutions
- **industry-case.md** (for weeks 14-16) with production system deep-dives

**Recommended Approach**:
1. Read topic files sequentially within each week
2. Work through code examples hands-on
3. Read related research papers
4. Attempt practice problems
5. Explore industry case studies for real-world context

---

## Course Navigation

**Quick Links by Module**:
- [Module 1: Introduction](#module-1-introduction-and-fundamentals)
- [Module 2: Collaborative Filtering](#module-2-collaborative-filtering-foundations)
- [Module 3: Content-Based & Hybrid](#module-3-content-based-and-hybrid-approaches)
- [Module 4: Deep Learning](#module-4-deep-learning-for-recommendations)
- [Module 5: Modern Architectures](#module-5-modern-architectures-and-industry-systems)
- [Module 6: Evaluation & Production](#module-6-evaluation-fairness-and-production-systems)
- [Module 7: Industry Cases](#module-7-industry-case-studies)
- [Module 8: Advanced Topics](#module-8-advanced-topics-and-research-frontiers)

---

*Last updated: January 2026*
