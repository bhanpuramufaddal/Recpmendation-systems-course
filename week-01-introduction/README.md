# Week 1: The Recommendation Problem

## Overview

Welcome to Week 1 of CS 329R: Recommendation Systems! This week introduces the fundamental concepts, challenges, and landscape of recommendation systems.

## Learning Objectives

By the end of this week, you will:
- Understand what recommendation systems are and why they matter
- Recognize different types of feedback and formulations (prediction vs. ranking)
- Trace the historical evolution from information retrieval to modern deep learning approaches
- Identify key application domains and real-world impact
- Understand the end-to-end recommendation pipeline
- Recognize major challenges (cold start, sparsity, scalability, etc.)

---

## Week Structure

### 1.1 [Overview](overview.md)
**Core Concepts**
- What are recommendation systems?
- Prediction vs. ranking formulations
- Explicit vs. implicit feedback
- Content-based vs. collaborative filtering

**Key Takeaways:**
- Modern systems focus on ranking over prediction
- Implicit feedback is abundant but noisy
- Collaborative filtering leverages the "wisdom of the crowd"

---

### 1.2 [Historical Context](historical-context.md)
**Evolution of RecSys**
- Early information retrieval systems (1960s-1990s)
- Birth of collaborative filtering: GroupLens, MovieLens (1990s)
- Matrix factorization era: Netflix Prize (2006-2009)
- Deep learning revolution (2016-present)
- Modern trends: transformers, LLMs, multi-modal systems

**Key Milestones:**
- 1992: Tapestry system (first collaborative filtering)
- 1994: GroupLens (user-based CF)
- 2006: Netflix Prize launches ($1M grand prize)
- 2009: BellKor's Pragmatic Chaos wins with matrix factorization ensemble
- 2016: YouTube DNN, Wide & Deep (Google)
- 2020s: Foundation models, LLMs for recommendations

---

### 1.3 [Applications](applications.md)
**Real-World Domains**
- E-commerce: Amazon, Alibaba
- Streaming media: Netflix, Spotify, YouTube
- Social media: Facebook, Instagram, TikTok, LinkedIn
- News and content: Google News, Reddit
- Dating: Tinder, Bumble, Hinge
- Travel: Airbnb, Booking.com
- Food delivery: Uber Eats, DoorDash
- Developer tools: GitHub, Cursor AI

**Impact Metrics:**
- 35% of Amazon's revenue from recommendations
- 80% of Netflix watch time from recommendations
- 70% of YouTube watch time from recommendations

---

### 1.4 [Recommendation Pipeline](pipeline.md)
**End-to-End System Architecture**

```
User/Context → Candidate Generation → Ranking → Re-ranking → Serving → User Interaction → Logging
                     ↑                                                            ↓
                     └─────────────────────── Feedback Loop ─────────────────────┘
```

**Pipeline Stages:**
1. **Data Collection**: User interactions, item metadata, context
2. **Candidate Generation**: Retrieve hundreds/thousands from millions (retrieval phase)
3. **Ranking**: Score and rank candidates by relevance (scoring phase)
4. **Re-ranking**: Apply business rules, diversity, fairness constraints
5. **Serving**: Return top-K items to user with low latency (<100ms)
6. **Feedback Loop**: Log interactions, update models

**Key Concepts:**
- Multi-stage pipeline (funnel approach)
- Trade-off: recall (retrieval) vs. precision (ranking)
- Latency constraints drive architecture decisions

---

### 1.5 [Key Challenges](challenges.md)
**Major Problems in RecSys**

1. **Cold Start Problem**
   - New users: No interaction history
   - New items: No ratings/engagement
   - Solutions: Content-based, meta-learning, exploration

2. **Data Sparsity**
   - Most users rate/interact with <1% of items
   - Sparse user-item matrix
   - Solutions: Matrix factorization, embeddings, transfer learning

3. **Scalability**
   - Millions of users × millions of items
   - Real-time recommendations (<100ms)
   - Solutions: ANN search (FAISS, ScaNN), distributed systems

4. **Diversity vs. Accuracy**
   - Accurate ≠ useful (filter bubble problem)
   - Need variety, serendipity, exploration
   - Solutions: MMR, DPP, multi-objective optimization

5. **Bias and Fairness**
   - Popularity bias (rich get richer)
   - Demographic bias, filter bubbles
   - Solutions: Debiasing, fairness constraints, causal inference

6. **Evaluation Challenges**
   - Offline metrics don't correlate with online success
   - Position bias, selection bias in logged data
   - Solutions: A/B testing, counterfactual evaluation

7. **Temporal Dynamics**
   - User preferences drift over time
   - Item relevance decays (news, fashion)
   - Solutions: Recurrent models, time-aware features, online learning

8. **Privacy and Ethics**
   - User data collection and tracking
   - Manipulation and addiction concerns
   - Solutions: Federated learning, differential privacy, transparency

---

### 1.6 [Practice Problems](practice-problems.md)
**Hands-On Exercises**

Work through practical problems covering:
- Dataset exploration (MovieLens, Amazon reviews)
- Basic similarity computations
- Evaluation metric calculations
- Cold start scenario analysis
- System design questions

---

## Weekly Schedule

**Suggested Timeline:**
- **Day 1**: Overview + Historical Context (understand the landscape)
- **Day 2**: Applications (see real-world impact)
- **Day 3**: Pipeline (understand system architecture)
- **Day 4**: Challenges (recognize key problems)
- **Day 5**: Practice Problems (apply concepts)

---

## Key Concepts to Master

### Formulations
- [ ] Prediction vs. ranking
- [ ] Explicit vs. implicit feedback
- [ ] Content-based vs. collaborative filtering
- [ ] Rating prediction vs. top-K recommendation

### Metrics
- [ ] RMSE, MAE (prediction)
- [ ] Precision@K, Recall@K (ranking)
- [ ] NDCG, MAP (ranking with relevance grades)

### Challenges
- [ ] Cold start (user, item, system)
- [ ] Data sparsity
- [ ] Scalability bottlenecks
- [ ] Filter bubbles and diversity

### System Design
- [ ] Multi-stage pipeline (retrieval → ranking → re-ranking)
- [ ] Latency vs. quality trade-offs
- [ ] Feedback loops and model updates

---

## Prerequisites

### Required Background
- **Python Programming**: Intermediate level
- **Linear Algebra**: Vectors, matrices, matrix multiplication
- **Probability**: Basic concepts (expected value, distributions)
- **Machine Learning**: Supervised learning, train/test splits, evaluation

### Optional (Helpful)
- Deep learning basics (neural networks, backpropagation)
- Information retrieval concepts
- Distributed systems fundamentals

---

## Datasets for Practice

### Recommended Datasets
1. **MovieLens** (100K, 1M, 20M versions)
   - Movie ratings from users
   - Explicit feedback (1-5 stars)
   - Good for beginners

2. **Amazon Product Reviews**
   - Product ratings and reviews
   - Multi-domain (books, electronics, etc.)
   - Large scale

3. **Last.fm** (Music)
   - User listening history
   - Implicit feedback (play counts)
   - Temporal data

4. **Yelp** (Restaurants)
   - Business ratings and reviews
   - Location-aware
   - Rich metadata

---

## Additional Resources

### Classic Papers
1. **Goldberg et al. (1992)**: "Using Collaborative Filtering to Weave an Information Tapestry" - First CF system
2. **Resnick et al. (1994)**: "GroupLens: An Open Architecture for Collaborative Filtering of Netnews"
3. **Linden et al. (2003)**: "Amazon.com Recommendations: Item-to-Item Collaborative Filtering"
4. **Koren et al. (2009)**: "Matrix Factorization Techniques for Recommender Systems"

### Surveys
1. **Jannach et al. (2022)**: "Recommender Systems—Beyond Matrix Completion" (CACM)
2. **Zhang et al. (2019)**: "Deep Learning Based Recommender System: A Survey and New Perspectives"

### Books
1. **Aggarwal (2016)**: *Recommender Systems: The Textbook*
2. **Ricci et al. (2015)**: *Recommender Systems Handbook* (2nd ed.)

### Industry Blogs
1. Netflix Tech Blog: netflixtechblog.com
2. Spotify Engineering: engineering.atspotify.com
3. Meta AI Research: ai.facebook.com

---

## Next Week Preview

**Week 2: Memory-Based Collaborative Filtering**
- User-based CF: Find similar users, recommend what they liked
- Item-based CF: Find similar items, recommend similar items
- Similarity measures: Cosine, Pearson correlation, Jaccard
- Neighborhood selection and aggregation
- Implementation from scratch

Get ready to implement your first recommendation algorithm!

---

## Questions to Ponder

1. **Why do most modern systems use ranking instead of rating prediction?**
   - Hint: What do users actually care about?

2. **Is implicit feedback always inferior to explicit feedback?**
   - Hint: Consider quantity vs. quality trade-offs

3. **Why is the cold start problem harder for collaborative filtering than content-based filtering?**
   - Hint: What data does each approach require?

4. **Can you have a perfectly accurate recommendation system that still fails?**
   - Hint: Think about diversity, serendipity, filter bubbles

5. **How would you recommend items in a brand new system with zero users and zero items?**
   - Hint: This is the system cold start problem

---

## Getting Help

- **Office Hours**: Check course website for schedule
- **Discussion Forum**: Post questions and help peers
- **Study Groups**: Form groups of 3-4 students
- **Code Repository**: Reference implementations available

---

**Happy Learning!** 🎓

*This week lays the foundation for everything to come. Take time to understand these core concepts—they'll be referenced throughout the entire course.*
