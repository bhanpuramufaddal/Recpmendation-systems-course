# CS 329R: Recommendation Systems - From Foundations to Frontiers

## Course Overview

This course provides a comprehensive treatment of recommendation systems, spanning classical approaches to cutting-edge deep learning techniques. Students will understand the mathematical foundations, algorithmic principles, and practical considerations for building systems that predict user preferences and personalize content at scale.

**Prerequisites**: Linear algebra, probability theory, machine learning fundamentals (CS 229 or equivalent)

---

## Module 1: Introduction and Fundamentals

### Week 1: The Recommendation Problem

**1.1 What Are Recommendation Systems?**
- Definition and scope
- The prediction vs. ranking formulation
- Explicit vs. implicit feedback
- Top-N recommendation vs. rating prediction

**1.2 Historical Context**
- Early collaborative filtering (GroupLens, MovieLens)
- The Netflix Prize and its impact
- Evolution from academic research to industry deployment

**1.3 Applications Across Domains**
- E-commerce (Amazon, eBay)
- Streaming platforms (Netflix, Spotify, YouTube)
- Social networks (Facebook, Twitter, LinkedIn)
- News and content aggregation
- Dating apps and matchmaking

**1.4 The Recommendation Pipeline**
- Data collection and user interaction logging
- Candidate generation
- Ranking and scoring
- Re-ranking and diversity
- Evaluation and online testing

**1.5 Key Challenges**
- Cold start problem (new users, new items)
- Sparsity in user-item interactions
- Scalability to billions of users/items
- Exploration vs. exploitation trade-offs
- Filter bubbles and echo chambers

---

## Module 2: Collaborative Filtering Foundations

### Week 2: Memory-Based Collaborative Filtering

**2.1 User-Based Collaborative Filtering**
- Intuition: "Users similar to you liked..."
- Computing user-user similarity (Pearson correlation, cosine similarity)
- k-nearest neighbors approach
- Prediction formulas and weighted averaging
- Computational complexity: O(|U|²|I|)

**2.2 Item-Based Collaborative Filtering**
- Intuition: "You liked X, so you might like Y..."
- Computing item-item similarity
- Advantages over user-based CF
- Amazon's item-to-item collaborative filtering
- Pre-computation and storage strategies

**2.3 Similarity Measures Deep Dive**
- Cosine similarity
- Pearson correlation coefficient
- Jaccard similarity for binary data
- Adjusted cosine similarity
- Significance weighting and variance weighting

**2.4 Limitations of Memory-Based Methods**
- Scalability issues
- Sparsity sensitivity
- No feature learning
- Limited personalization depth

---

### Week 3: Model-Based Collaborative Filtering - Matrix Factorization

**3.1 The Matrix Factorization Framework**
- User-item rating matrix R ≈ U^T V
- Latent factor interpretation
- Dimensionality reduction perspective
- Connection to SVD and PCA

**3.2 Optimization Formulation**
- Squared error loss: min ||R - U^T V||²_F
- Observed entries only
- Regularization (L2): λ(||U||² + ||V||²)
- Bias terms: r_ui = μ + b_u + b_i + u_i^T v_u

**3.3 Optimization Algorithms**
- Stochastic gradient descent (SGD)
- Alternating least squares (ALS)
- Coordinate descent
- Convergence properties and learning rate scheduling

**3.4 Advanced Matrix Factorization Variants**
- SVD++ incorporating implicit feedback
- TimeSVD++ for temporal dynamics
- Factorization machines for feature interactions
- Probabilistic matrix factorization (PMF)

**3.5 Handling Implicit Feedback**
- Confidence-weighted matrix factorization
- Bayesian personalized ranking (BPR)
- WRMF (Weighted Regularized Matrix Factorization)
- Learning from positive-only feedback

---

## Module 3: Content-Based and Hybrid Approaches

### Week 4: Content-Based Recommendation

**4.1 Foundations of Content-Based Filtering**
- Item profile construction
- User profile learning
- Matching profiles for recommendation

**4.2 Feature Representation**
- TF-IDF for text content
- Bag-of-words and n-grams
- Metadata and structured attributes
- Image and video features (CNN embeddings)
- Audio features (spectrograms, MFCCs)

**4.3 Profile Learning Methods**
- Weighted averaging of item features
- Rocchio algorithm
- Naive Bayes classifiers
- Logistic regression for preference prediction
- Decision trees and ensemble methods

**4.4 Advantages and Limitations**
- Advantages: No cold-start for items, transparency, user independence
- Limitations: Limited serendipity, over-specialization, feature engineering burden
- The "overspecialization" problem

**4.5 Hybrid Recommendation Strategies**
- Weighted hybridization
- Switching hybrid
- Mixed recommendations
- Feature combination
- Cascade hybrid
- Feature augmentation
- Meta-level hybrid approaches

---

## Module 4: Deep Learning for Recommendations

### Week 5: Neural Collaborative Filtering

**5.1 From Matrix Factorization to Neural Networks**
- MF as a shallow neural network
- Non-linear feature interactions
- The expressiveness argument

**5.2 Neural Collaborative Filtering (NCF)**
- Generalized matrix factorization (GMF)
- Multi-layer perceptron (MLP) pathway
- NeuMF: Fusion of GMF and MLP
- Pre-training strategies

**5.3 Deep Matrix Factorization Variants**
- AutoRec: Autoencoder-based CF
- Collaborative denoising autoencoders (CDAE)
- Variational autoencoders for CF (VAE-CF)
- Adversarial training for recommendations

**5.4 Training Deep Recommenders**
- Negative sampling strategies
- Pair-wise vs. point-wise loss functions
- Batch construction and hard negative mining
- Regularization techniques (dropout, batch norm)

---

### Week 6: Sequential and Session-Based Recommendations

**6.1 Modeling User Sequences**
- Markov chains and first-order models
- Higher-order Markov models
- Session-based vs. long-term modeling
- Next-item prediction formulation

**6.2 Recurrent Neural Networks for Recommendations**
- GRU4Rec: GRU-based session modeling
- LSTM for user behavior modeling
- Attention mechanisms over sequences
- Handling variable-length sessions

**6.3 Transformer-Based Architectures**
- Self-attention for sequential recommendation
- BERT4Rec: Bidirectional modeling with masked item prediction
- SASRec: Self-attentive sequential recommendation
- Positional encodings and temporal information

**6.4 Conversational and Interactive Recommendations**
- Multi-turn interaction modeling
- Reinforcement learning formulation
- Exploration strategies (Thompson sampling, ε-greedy)
- Contextual bandits for recommendation

---

### Week 7: Graph Neural Networks and Knowledge-Enhanced Methods

**7.1 Graph-Based Recommendation**
- User-item bipartite graph representation
- Graph structure and collaborative signal
- Random walks and personalized PageRank
- Metapath-based approaches

**7.2 Graph Neural Networks (GNNs) for Recommendations**
- Message passing framework
- Graph convolutional networks (GCN)
- GraphSAGE for inductive learning
- PinSage: GNN at Pinterest scale

**7.3 LightGCN and Simplified GNNs**
- Removing feature transformation and activation
- Neighborhood aggregation only
- Layer combination for final embeddings
- Empirical superiority and theoretical insights

**7.4 Knowledge Graph Integration**
- Knowledge graph embeddings (TransE, DistMult)
- Propagating knowledge in KG-enhanced recommendations
- KGAT: Knowledge graph attention networks
- Multi-task learning with KG auxiliary tasks

---

## Module 5: Modern Architectures and Industry Systems

### Week 8: Two-Tower Models and Large-Scale Retrieval

**8.1 Two-Tower Architecture**
- Separate user and item encoders
- Dot-product similarity in embedding space
- Training with in-batch negatives
- Asymmetric architectures for user/item complexity

**8.2 Retrieval at Scale**
- Approximate nearest neighbor search (ANN)
- Locality-sensitive hashing (LSH)
- Product quantization
- FAISS and ScaNN libraries
- Hierarchical navigable small world (HNSW) graphs

**8.3 YouTube Recommendation System**
- Candidate generation network
- Ranking network architecture
- Addressing selection bias
- Serving millions of candidates in milliseconds

**8.4 Multi-Task Learning**
- Predicting clicks, watch time, and engagement
- Hard parameter sharing
- MMOE (Multi-gate Mixture of Experts)
- Balancing multiple objectives

---

### Week 9: Embeddings and Pre-training

**9.1 Learning User and Item Embeddings**
- Word2Vec analogy: Item2Vec
- Skip-gram and CBOW adaptations
- Negative sampling in recommendation context
- Embedding quality evaluation

**9.2 Pre-training Strategies**
- Self-supervised learning for recommendations
- Contrastive learning (SimCLR adaptation)
- Masked prediction tasks
- Cross-domain transfer learning

**9.3 Large Language Models for Recommendations**
- LLMs as feature extractors
- Prompt engineering for recommendation
- Zero-shot and few-shot recommendation
- Fine-tuning LLMs on recommendation data

**9.4 Multi-Modal Recommendations**
- Vision-language models for product recommendations
- CLIP embeddings for content understanding
- Fusing text, image, and interaction signals
- Cross-modal retrieval

---

### Week 10: Context-Aware and Bandit-Based Recommendations

**10.1 Context-Aware Recommendation**
- Contextual dimensions (time, location, device, social)
- Tensor factorization methods
- Context-aware matrix factorization
- Feature-based contextual models

**10.2 Multi-Armed Bandits**
- Exploration-exploitation dilemma
- ε-greedy, UCB, and Thompson sampling
- Regret bounds and theoretical guarantees
- Practical implementation considerations

**10.3 Contextual Bandits**
- LinUCB and contextual Thompson sampling
- Neural bandits with deep networks
- Policy learning and off-policy evaluation
- Counterfactual risk minimization

**10.4 Reinforcement Learning for Long-Term Optimization**
- Markov decision process formulation
- Reward shaping for user satisfaction
- Policy gradient methods (REINFORCE, A3C)
- Offline RL for recommendation (batch RL)

---

## Module 6: Evaluation, Fairness, and Production Systems

### Week 11: Evaluation Methodologies

**11.1 Offline Evaluation Metrics**
- Rating prediction: RMSE, MAE
- Ranking metrics: Precision@K, Recall@K, MAP, NDCG
- Beyond accuracy: diversity, novelty, coverage
- Catalog coverage and Gini coefficient

**11.2 Experimental Design**
- Train-test splitting strategies
- Temporal splitting for sequential data
- Leave-one-out evaluation
- Cross-validation in recommendation context

**11.3 Online Evaluation (A/B Testing)**
- Metrics: CTR, conversion rate, user engagement
- Statistical significance testing
- Multi-armed bandit tests
- Interleaving experiments
- Long-term vs. short-term metrics

**11.4 Challenges in Evaluation**
- Popularity bias in metrics
- Position bias in rankings
- Selection bias from logged data
- Correlation vs. causation

---

### Week 12: Bias, Fairness, and Ethics

**12.1 Types of Bias in Recommendations**
- Popularity bias and Matthew effect
- Selection bias and self-selection
- Position bias in click data
- Conformity bias in social recommendations

**12.2 Debiasing Techniques**
- Inverse propensity scoring (IPS)
- Causal inference for unbiased learning
- Doubly robust estimation
- Unbiased learning from biased feedback

**12.3 Fairness in Recommendations**
- Consumer fairness (demographic parity, equal opportunity)
- Provider fairness (exposure, calibration)
- Individual vs. group fairness
- Multi-stakeholder optimization

**12.4 Ethical Considerations**
- Filter bubbles and echo chambers
- Radicalization and harmful content amplification
- Privacy and data protection (GDPR implications)
- Transparency and explainability
- Addictive design patterns

---

### Week 13: Production Systems and MLOps

**13.1 System Architecture**
- Batch vs. streaming processing
- Lambda and Kappa architectures
- Feature stores and embedding databases
- Model serving infrastructure

**13.2 Scalability Challenges**
- Distributed training (data and model parallelism)
- Online learning and incremental updates
- Caching strategies
- Load balancing and latency optimization

**13.3 The Cold Start Problem**
- New user strategies (popularity, demographic matching)
- New item strategies (content-based bootstrapping)
- Explore-exploit for cold starts
- Meta-learning approaches

**13.4 Model Management**
- A/B testing infrastructure
- Model versioning and rollback
- Monitoring and alerting
- Shadow mode deployment
- Continuous training pipelines

---

## Module 7: Industry Case Studies

### Week 14: Social Media Platforms

**14.1 Facebook News Feed Ranking**
- Evolution from EdgeRank to deep learning models
- Multi-objective optimization (engagement, meaningful interactions, well-being)
- Inventory management and diversity constraints
- Addressing misinformation and polarization
- Real-time feature computation at scale
- Position bias correction in feed ranking
- Balancing short-term engagement vs. long-term satisfaction

**14.2 Instagram Explore and Reels**
- Two-stage ranking: candidate generation + heavy ranking
- Visual understanding with computer vision models
- Engagement prediction (likes, saves, shares, watch time)
- Creator-viewer matching problem
- Handling rapid content velocity
- Interest diversification in Explore
- Short-form video ranking for Reels (competing with TikTok)
- Account recommendations and suggested users

**14.3 TikTok's For You Page**
- Cold start through initial interest selection
- Collaborative filtering on viewing behavior
- Video understanding (objects, scenes, music, effects)
- Retention and completion rate optimization
- Diversification to prevent fatigue
- Viral content amplification mechanisms
- Creator-side optimization and fairness
- Real-time A/B testing infrastructure

**14.4 LinkedIn Feed and Job Recommendations**
- Professional network graph structure
- Skills-based matching for job recommendations
- Endorsement and connection signals
- Content relevance for professional audiences
- Sponsored content integration
- People You May Know (PYMK) algorithm
- Balancing content types (jobs, posts, ads, learning)

---

### Week 15: Streaming and E-Commerce Platforms

**15.1 Netflix Recommendation System**
- Row generation: personalized genre rows
- Artwork personalization (contextual bandits for thumbnails)
- Ranking within rows vs. ranking rows
- Session-based viewing patterns
- Handling explicit ratings and implicit signals (viewing time, replays)
- Cold start for new content launches
- Regional and cultural personalization
- Addressing popularity bias and promoting diversity
- Offline evaluation vs. online A/B testing lessons

**15.2 Spotify Music Recommendations**
- Discover Weekly: collaborative filtering + audio analysis
- Daily Mix: clustering users by taste profiles
- Release Radar: new music from followed artists
- Radio: session-based sequential recommendations
- Natural Language Processing on playlists and metadata
- Audio feature extraction (tempo, energy, danceability)
- Balancing familiarity and discovery
- Podcast recommendations as a different modality
- Artist-side fairness and exposure

**15.3 YouTube Recommendations**
- Two-tower model for candidate generation
- Ranking with deep neural networks
- Watch time as primary optimization metric
- Addressing clickbait and misleading thumbnails
- Reducing borderline content and misinformation
- Handling the cold start for new channels
- Shorts feed vs. long-form recommendations
- Subscription feed vs. recommendation feed balance
- Advertiser-friendly content considerations

**15.4 Amazon Product Recommendations**
- Item-to-item collaborative filtering origins
- "Customers who bought this also bought"
- Personalized homepage and browse page
- Sponsored products and organic recommendations separation
- Search ranking as implicit recommendation
- Cross-category recommendations
- Session-based recommendations in shopping cart
- Post-purchase recommendations and replenishment
- Handling seasonal and trending products
- A/B testing at massive scale (millions of experiments)

---

### Week 16: Emerging Applications and Specialized Domains

**16.1 Airbnb Search and Recommendations**
- Two-sided marketplace considerations
- Location-based personalization
- Listing quality scoring
- Guest-host compatibility matching
- Price sensitivity and budget constraints
- Availability and booking probability
- Similar listings recommendations
- Experiences and restaurant recommendations
- Handling supply-demand imbalance

**16.2 Uber Eats and Food Delivery**
- Restaurant ranking based on cuisine preferences
- Real-time delivery time estimation impact
- Re-ordering and personalized shortcuts
- New restaurant discovery vs. favorites
- Contextual factors (time of day, weather, location)
- Dietary restrictions and preferences
- Search vs. browse behavior differences
- Promotional content integration

**16.3 Dating Apps (Tinder, Bumble, Hinge)**
- Two-sided matching problem
- Collaborative filtering on swipe data
- Profile completeness and quality signals
- Geographic constraints and preferences
- Preventing over-exposure and match exhaustion
- Balancing attractiveness and compatibility
- Gender imbalance handling
- Conversation likelihood prediction
- Ethical considerations in algorithmic matching

**16.4 News Aggregation (Google News, Apple News)**
- Freshness vs. relevance trade-off
- Topic diversification to prevent filter bubbles
- Source credibility and quality signals
- Personalization vs. editorial curation balance
- Breaking news detection and promotion
- Misinformation and fact-checking integration
- Reading time prediction
- Cross-device behavior tracking

**16.5 GitHub and Developer Tools**
- Repository recommendations based on stars and contributions
- Code completion and Copilot suggestions (LLM-based)
- Trending repositories and topics
- Developer network effects
- Issue and pull request recommendations
- Learning path recommendations for new technologies
- Package and dependency suggestions

**16.6 Cursor AI: Code Completion with Online RL**
- Tab completion as a recommendation problem
- Handling 400M+ requests per day
- Online reinforcement learning from user acceptance behavior
- Policy gradient methods vs. filtering approaches
- Learning when NOT to suggest (reducing noise by 21%)
- Improving accept rates (28% increase) through RL
- On-policy training with rapid deployment cycles (1.5-2 hour loops)
- Reward shaping: balancing acceptance, suggestion size, and user flow
- Continuous model updates multiple times per day
- Comparison with GitHub Copilot's logistic regression filter
- Infrastructure for real-time learning from production data

**16.7 Education Platforms (Coursera, Khan Academy)**
- Learning path personalization
- Prerequisite knowledge modeling
- Difficulty adaptation based on performance
- Course recommendations based on career goals
- Time commitment and completion likelihood
- Peer learner similarity
- Instructor quality signals
- Certification value and ROI considerations

---

## Module 8: Advanced Topics and Research Frontiers

### Week 17: Explainability and Interpretability

**17.1 Why Explainability Matters**
- User trust and transparency
- Debugging model failures
- Regulatory compliance
- Actionable feedback for users

**17.2 Explainability Techniques**
- Attention visualization
- LIME and SHAP for recommendations
- Influence functions
- Counterfactual explanations
- Post-hoc rationalization

**17.3 Inherently Interpretable Models**
- Matrix factorization interpretability
- Sparse linear models
- Decision trees and rule-based systems
- Trade-offs: accuracy vs. interpretability

---

### Week 18: Cross-Domain and Transfer Learning

**18.1 Cross-Domain Recommendation**
- Domain adaptation techniques
- Shared user representations across domains
- Transfer learning from data-rich to data-poor domains
- Cold-start alleviation through transfer

**18.2 Meta-Learning for Recommendations**
- Learning to learn user preferences
- MAML adaptation for few-shot recommendation
- Rapid adaptation to new users
- Personalized initialization strategies

**18.3 Federated Learning**
- Privacy-preserving collaborative filtering
- Federated averaging for recommendation
- Communication efficiency
- Handling non-IID user data

---

### Week 19: Emerging Paradigms and Future Directions

**19.1 Generative Recommendations**
- Diffusion models for recommendations
- GAN-based data augmentation
- Generating synthetic user behaviors
- Counterfactual data generation

**19.2 Foundation Models for Recommendations**
- Unified models across recommendation tasks
- Pre-training on large-scale interaction data
- Prompt-based recommendation with LLMs
- Towards general-purpose recommenders

**19.3 Interactive and Conversational Systems**
- Dialogue-based preference elicitation
- Natural language interfaces for recommendations
- Critique-based refinement
- Voice-based recommendation assistants

**19.4 Emerging Challenges**
- Recommendations in the metaverse
- Ephemeral and short-form content (TikTok, Reels)
- Real-time personalization at edge devices
- Carbon footprint and sustainable ML

**19.5 Open Research Questions**
- Causality in recommendation systems
- Long-term user satisfaction optimization
- Serendipity and diversity without sacrificing relevance
- Bridging online and offline experiences

---

## Assessment and Course Policies

**Grading Breakdown:**
- Problem Sets (35%): 4 problem sets covering theory and algorithms
- Midterm Exam (20%): Covering Modules 1-4
- Case Study Analysis (10%): Deep dive into one industry system
- Final Project (30%): Research-oriented project with paper and presentation
- Participation (5%): In-class discussions and paper reviews

**Required Readings:**
- Research papers from RecSys, KDD, WWW, SIGIR conferences
- Selected chapters from textbooks (Aggarwal, Ricci et al.)
- Industry blog posts from Netflix, YouTube, Spotify

**Office Hours:**
- Instructor: Tuesdays 2-4 PM
- TAs: Wednesdays and Thursdays 4-6 PM

---

## Recommended Textbooks

1. Aggarwal, C.C. (2016). *Recommender Systems: The Textbook*. Springer.
2. Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook* (2nd ed.). Springer.
3. Jannach, D., & Zanker, M. (2024). *Recommender Systems: An Introduction*. Cambridge University Press.

---

## Key Paper Reading List (Selected)

**Classical Foundations:**
- Sarwar et al. (2001): "Item-based collaborative filtering recommendation algorithms"
- Koren et al. (2009): "Matrix factorization techniques for recommender systems"
- Rendle (2010): "Factorization machines"

**Deep Learning Era:**
- He et al. (2017): "Neural collaborative filtering"
- Covington et al. (2016): "Deep neural networks for YouTube recommendations"
- Hidasi et al. (2016): "Session-based recommendations with recurrent neural networks"

**Modern Advances:**
- He et al. (2020): "LightGCN: Simplifying and powering graph convolution network"
- Sun et al. (2019): "BERT4Rec: Sequential recommendation with bidirectional encoder"
- Yi et al. (2019): "Sampling-bias-corrected neural modeling for large corpus item recommendations"

**Industry Case Studies:**
- Covington et al. (2016): "Deep neural networks for YouTube recommendations"
- Gomez-Uribe & Hunt (2016): "The Netflix recommender system: Algorithms, business value, and innovation"
- Davidson et al. (2010): "The YouTube video recommendation system"
- Chen et al. (2019): "Top-K off-policy correction for a REINFORCE recommender system" (YouTube)
- Amatriain & Basilico (2015): "Recommender systems in industry: A Netflix case study"
- Beutel et al. (2018): "Latent cross: Making use of context in recurrent recommender systems" (Google)
- Bharadhwaj et al. (2019): "Meta-learning for user cold-start recommendation" (LinkedIn)
- Jannach et al. (2016): "Leveraging multi-behavioral data for recommendations" (Zalando)
- Ying et al. (2018): "Graph convolutional neural networks for web-scale recommender systems" (Pinterest)
- Cursor Team (2024): "Improving Cursor Tab with online RL" - Online reinforcement learning for code completion

---

*Last updated: January 2026*
