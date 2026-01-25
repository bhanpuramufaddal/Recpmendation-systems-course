# Week 1: Historical Context and Evolution

## Learning Objectives

- Understand the evolution of recommendation systems from early research to modern industry
- Recognize the impact of the Netflix Prize on the field
- Identify key milestones and breakthroughs

---

## Timeline of Recommendation Systems

```
1990s          2000s              2010s              2020s
  |              |                  |                  |
Tapestry    Netflix Prize     Deep Learning      LLMs + RL
GroupLens    Amazon CF         YouTube DNN        BERT4Rec
            MovieLens         GRU4Rec            LightGCN
                              NCF                Foundation Models
```

---

## Early Collaborative Filtering (1990s)

### Tapestry (1992)

**First Collaborative Filtering System**
- Developed at Xerox PARC
- **Domain**: Email and document filtering
- **Approach**: Users manually annotated documents; system filtered based on annotations from similar users
- **Coined the term**: "Collaborative Filtering"

**Key Innovation**: Leveraging the "wisdom of the crowd"

**Reference**: Goldberg, D., et al. (1992). "Using collaborative filtering to weave an information tapestry". *Communications of the ACM*.

---

### GroupLens (1994)

**Automated Collaborative Filtering**
- University of Minnesota research project
- **Domain**: Usenet news articles
- **Approach**: Predicted article ratings based on ratings from similar users

**Technical Details**:
- **Algorithm**: Pearson correlation for user similarity
- **Prediction**: Weighted average of neighbors' ratings
- **Scale**: Thousands of users, tens of thousands of articles

**Formula** (User-based CF):

$$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N(u)} \text{sim}(u,v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N(u)} |\text{sim}(u,v)|}$$

where:
- $N(u)$ = k-nearest neighbors of user $u$
- $\text{sim}(u,v)$ = Pearson correlation between users $u$ and $v$
- $\bar{r}_u$ = average rating by user $u$

**Impact**: Established the foundation for memory-based collaborative filtering

**Reference**: Resnick, P., et al. (1994). "GroupLens: An open architecture for collaborative filtering of netnews". *CSCW*.

---

### MovieLens (1997-Present)

**Long-Running Research Platform**
- Developed by GroupLens Research Lab
- **Purpose**: Movie recommendation and dataset collection
- **Datasets**: Publicly available rating datasets

**Dataset Evolution**:
| Dataset | Year | Users | Movies | Ratings |
|---------|------|-------|--------|---------|
| ML-100K | 1998 | 943 | 1,682 | 100,000 |
| ML-1M | 2003 | 6,040 | 3,706 | 1M |
| ML-10M | 2009 | 71,567 | 10,681 | 10M |
| ML-20M | 2015 | 138,493 | 27,278 | 20M |
| ML-25M | 2019 | 162,541 | 62,423 | 25M |

**Impact**:
- Standard benchmark for recommendation research
- Enabled reproducible research
- Thousands of research papers use MovieLens

**Access**: https://grouplens.org/datasets/movielens/

---

## The Netflix Prize (2006-2009)

### The Challenge

**Announced**: October 2, 2006

**Goal**: Improve Netflix's recommendation algorithm (Cinematch) by 10% RMSE

**Prize**: \$1 million USD

**Dataset**:
- 100M ratings
- 480,189 users
- 17,770 movies
- Ratings from 1 to 5 stars
- Time period: 1999-2005

**Baseline (Cinematch)**: RMSE = 0.9514

**Target**: RMSE ≤ 0.8563 (10% improvement)

---

### The Competition

**Participants**:
- 40,000+ teams from 186 countries
- Academic researchers, industry practitioners, hobbyists
- Open leaderboard drove innovation

**Key Innovations**:

#### 1. **Ensemble Methods**
- Combining multiple algorithms outperformed any single approach
- Weighted blending of 100+ models

#### 2. **Matrix Factorization**
- SVD++ (Koren, 2008): Incorporated implicit feedback
- TimeSVD++: Temporal dynamics

$$\hat{r}_{ui}(t) = \mu + b_u(t) + b_i(t) + q_i^T \left[ p_u(t) + |I_u|^{-0.5} \sum_{j \in I_u} y_j \right]$$

#### 3. **Regularization Techniques**
- Prevented overfitting on sparse data
- L2 regularization became standard

#### 4. **Temporal Dynamics**
- User preferences drift over time
- Item popularity evolves
- Time-aware modeling improved accuracy

---

### The Winners

#### **BellKor's Pragmatic Chaos (2009)**

**Final RMSE**: 0.8567 (10.06% improvement)

**Team Members**:
- AT&T Research (BellKor)
- Commendo Research
- Pragmatic Theory

**Approach**:
- Ensemble of 107 different algorithms
- Matrix factorization (SVD++)
- Restricted Boltzmann Machines (RBMs)
- Neighborhood methods
- Temporal dynamics

**Close Competition**:
- "The Ensemble" team: RMSE = 0.8567 (tied!)
- BellKor won by submitting 20 minutes earlier

---

### Impact on the Field

**Algorithmic Advances**:
1. **Matrix Factorization** became the dominant approach
2. **Ensemble methods** proved essential for state-of-the-art performance
3. **Temporal modeling** incorporated time as a first-class citizen
4. **Regularization** techniques advanced

**Industry Impact**:
1. **Validated the importance** of recommendations for business
2. **Crowdsourcing innovation** through open competitions
3. **Public datasets** accelerated research
4. **Academic-industry collaboration** increased

**Ironic Outcome**:
- Netflix never deployed the winning solution in production
- **Reason**: 10% accuracy improvement didn't justify engineering cost
- **Lesson**: Research metrics (RMSE) ≠ business metrics (engagement, retention)

**Quote** (Netflix Blog, 2012):
> "We evaluated some of the new methods offline but the additional accuracy gains did not seem to justify the engineering effort needed to bring them into a production environment."

---

## Evolution to Industry Deployment (2010s)

### Amazon: Item-to-Item Collaborative Filtering (2003)

**Paper**: Linden, G., Smith, B., & York, J. (2003). "Amazon.com recommendations: Item-to-item collaborative filtering". *IEEE Internet Computing*.

**Innovation**:
- **Scalability**: Precompute item-item similarities
- **O(1) lookup** at serving time
- Works with billions of items

**Algorithm**:
```
For each item i:
  1. Find similar items based on co-purchases
  2. Store top-N similar items

For user u viewing item i:
  Return similar items to i that u hasn't purchased
```

**Impact**: Powers "Customers who bought this also bought" (still used today)

---

### YouTube: Deep Neural Networks (2016)

**Paper**: Covington, P., et al. (2016). "Deep neural networks for YouTube recommendations". *RecSys*.

**Architecture**:
- **Stage 1**: Candidate generation (DNN with billions of parameters)
- **Stage 2**: Ranking (rich features, predict watch time)

**Scale**:
- Billions of users
- Millions of videos
- Hundreds of billions of training examples

**Innovation**:
- Two-stage architecture (now industry standard)
- Watch time as primary metric (not clicks)
- Deep learning at unprecedented scale

---

### Spotify: Discover Weekly (2015)

**Launched**: July 2015

**Approach**:
- **Collaborative filtering**: 2B user-generated playlists
- **NLP**: Analyze playlist names, artist metadata
- **Audio analysis**: CNN for raw audio features

**Impact**:
- 40M users (2016) → 100M+ users (2024)
- 30%+ of Spotify streams from recommendations
- Personalized for every single user

---

## Deep Learning Era (2015-Present)

### Key Papers

**2015**: **AutoRec** (Sedhain et al.)
- Autoencoder-based collaborative filtering
- Outperformed matrix factorization

**2016**: **GRU4Rec** (Hidasi et al.)
- First RNN-based session recommendation
- Sequential pattern modeling

**2017**: **Neural Collaborative Filtering** (He et al.)
- Replaced inner product with neural architecture
- Foundation for deep learning in RecSys

**2018**: **PinSage** (Ying et al.)
- Graph neural network at Pinterest scale
- Billions of nodes and edges

**2019**: **BERT4Rec** (Sun et al.)
- Bidirectional transformers for sequential recommendation
- Masked item prediction

**2020**: **LightGCN** (He et al.)
- Simplified GNN for collaborative filtering
- 16% improvement over complex GNNs

---

## Modern Trends (2020s)

### 1. **Large Language Models**
- GPT-4, Claude, Llama for zero-shot recommendations
- Natural language explanations
- Prompt engineering for personalization

### 2. **Multi-Modal Learning**
- CLIP for vision-language understanding
- Combining text, image, audio, video
- Cross-modal retrieval

### 3. **Reinforcement Learning**
- Long-term user satisfaction
- Exploration vs. exploitation
- Online learning from production data
- **Example**: Cursor AI's online RL for code completion (2024)

### 4. **Foundation Models**
- Pre-trained on massive interaction data
- Transfer learning across domains
- Unified architectures for multiple tasks

### 5. **Fairness and Ethics**
- Debiasing algorithms
- Provider fairness (creator exposure)
- Consumer fairness (demographic parity)
- Regulation (GDPR, transparency requirements)

---

## Paradigm Shifts

| Era | Dominant Approach | Key Metric | Scale |
|-----|-------------------|------------|-------|
| 1990s | Memory-based CF | Prediction accuracy | Thousands |
| 2000s | Matrix Factorization | RMSE | Millions |
| 2010s | Deep Learning | Engagement | Billions |
| 2020s | Foundation Models + RL | Long-term satisfaction | Billions |

---

## Lessons Learned

1. **Accuracy ≠ Business Value**: Netflix Prize winner never deployed
2. **Scale Matters**: Algorithms must work with billions of users/items
3. **Online Metrics > Offline Metrics**: A/B testing is the ground truth
4. **Engineering is Critical**: 1% accuracy gain means nothing if you can't deploy it
5. **Data Quality > Algorithm Sophistication**: Garbage in, garbage out
6. **User Experience**: Diversity, novelty, explainability matter as much as relevance

---

## Summary

- **1990s**: Pioneering research (Tapestry, GroupLens, MovieLens)
- **2000s**: Netflix Prize catalyzed matrix factorization and ensembles
- **2010s**: Deep learning revolution (YouTube, Spotify, RNNs, GNNs)
- **2020s**: Foundation models, RL, multi-modal, fairness focus

**Next**: See **applications.md** for domain-specific use cases

---

## References

1. Goldberg, D., et al. (1992). "Using collaborative filtering to weave an information tapestry". *CACM*.
2. Resnick, P., et al. (1994). "GroupLens: An open architecture for collaborative filtering". *CSCW*.
3. Koren, Y. (2009). "The BellKor solution to the Netflix Grand Prize". *Netflix Prize documentation*.
4. Bennett, J., & Lanning, S. (2007). "The Netflix Prize". *KDD Cup*.
5. Gomez-Uribe, C. A., & Hunt, N. (2016). "The Netflix recommender system". *ACM TIST*.
