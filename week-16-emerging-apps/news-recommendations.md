# System Design: News Recommendation System

## Problem Statement & Requirements

### Interview Prompt

> "Design a personalized news recommendation system for a platform like Google News or Apple News that serves 100M+ daily active users."

### Functional Requirements

1. **Personalized news feed**: Show articles matching user interests
2. **Breaking news**: Surface urgent/trending stories to all users
3. **Topic-based browsing**: Allow users to explore by category (Politics, Tech, Sports)
4. **Source diversity**: Show articles from multiple publishers
5. **Read later/save**: Bookmark articles for future reading
6. **Notifications**: Push alerts for breaking news or personalized updates

### Non-Functional Requirements

1. **Latency**: Feed generation < 100ms
2. **Freshness**: New articles indexed within 5 minutes of publication
3. **Availability**: 99.9% uptime (news is time-critical)
4. **Scale**: Handle 100M DAU, 50K new articles/hour
5. **Personalization**: Adapt to user preferences within 10-20 interactions

### Scope

**In scope**:
- Article recommendation and ranking
- User interest modeling
- Freshness and breaking news handling
- Diversity and filter bubble prevention

**Out of scope**:
- Full-text search (separate system)
- Comment moderation
- Subscription/paywall integration

---

## Scale Estimation (Back-of-Envelope)

### Users & Traffic

```
Users:
- Monthly Active Users (MAU): 300M
- Daily Active Users (DAU): 100M
- DAU/MAU ratio: 33% (typical for news apps)

Traffic:
- Sessions per user per day: 3
- Articles viewed per session: 10
- Total article views per day: 100M × 3 × 10 = 3B views/day

QPS Calculation:
- Feed requests per day: 100M × 3 = 300M requests/day
- Average QPS: 300M / 86,400 = 3,500 QPS
- Peak QPS (10x average): 35,000 QPS
```

### Content Volume

```
Article Ingestion:
- New articles per hour: 50,000
- New articles per day: 1.2M
- Active articles (48-hour window): 2.4M
- Total indexed articles (30 days): 36M

Sources:
- News publishers: 10,000+
- Languages: 40+
- Countries: 100+
```

### Storage

```
Article Embeddings:
- Active articles: 2.4M
- Embedding dimension: 768 (BERT-based)
- Bytes per embedding: 768 × 4 = 3KB
- Total embedding storage: 2.4M × 3KB = 7.2GB

User Embeddings:
- Active users: 300M
- Embedding dimension: 256
- Bytes per embedding: 256 × 4 = 1KB
- Total: 300M × 1KB = 300GB

Feature Store:
- User features: 300M users × 500 bytes = 150GB
- Article features: 36M articles × 2KB = 72GB
```

### Latency Budget Breakdown

```
Total budget: 100ms

Component breakdown:
- User feature lookup: 5ms
- Candidate retrieval (ANN): 15ms
- Article feature lookup: 10ms
- Ranking model inference: 30ms
- Diversity re-ranking: 15ms
- Freshness boost: 5ms
- Network overhead: 20ms
```

---

## High-Level Architecture

```
                                    ┌─────────────────┐
                                    │  Article        │
                                    │  Ingestion      │
                                    │  Pipeline       │
                                    └────────┬────────┘
                                             │
                                             ▼
┌──────────┐    ┌──────────┐    ┌─────────────────────────┐
│  User    │───▶│  API     │───▶│    Candidate            │
│  Request │    │  Gateway │    │    Generation           │
└──────────┘    └──────────┘    │  ┌─────────────────┐    │
                                │  │ Interest-Based  │    │
                                │  │ (User Embedding)│    │
                                │  └─────────────────┘    │
                                │  ┌─────────────────┐    │
                                │  │ Trending/       │    │
                                │  │ Breaking News   │    │
                                │  └─────────────────┘    │
                                │  ┌─────────────────┐    │
                                │  │ Topic-Based     │    │
                                │  │ (Explicit)      │    │
                                │  └─────────────────┘    │
                                └────────────┬────────────┘
                                             │ ~1000 candidates
                                             ▼
                                ┌─────────────────────────┐
                                │      Ranking Model      │
                                │  (Click, Read, Share)   │
                                └────────────┬────────────┘
                                             │ ~100 ranked
                                             ▼
                                ┌─────────────────────────┐
                                │     Re-Ranking Layer    │
                                │  - Topic Diversity      │
                                │  - Source Diversity     │
                                │  - Freshness Boost      │
                                │  - Filter Bubble        │
                                └────────────┬────────────┘
                                             │ ~50 final
                                             ▼
                                ┌─────────────────────────┐
                                │       Response          │
                                └─────────────────────────┘
```

---

## Data Model

### User Features

```python
class UserProfile:
    user_id: str

    # Explicit preferences
    followed_topics: List[str]         # ["technology", "politics"]
    followed_sources: List[str]        # ["NYT", "WSJ"]
    blocked_sources: List[str]         # Sources user doesn't want
    language_preferences: List[str]    # ["en", "es"]

    # Learned preferences (implicit)
    topic_interests: Dict[str, float]  # {"tech": 0.8, "sports": 0.2}
    source_affinity: Dict[str, float]  # {"NYT": 0.9, "BBC": 0.7}
    reading_patterns: Dict[str, Any]   # Time of day, session length

    # Embeddings
    interest_embedding: List[float]    # 256-dim learned vector

    # Engagement history
    recent_reads: List[str]            # Last 100 article IDs
    recent_topics: List[str]           # Topics from last 24 hours
```

### Article Features

```python
class Article:
    article_id: str

    # Content
    title: str
    summary: str
    body: str
    url: str

    # Metadata
    source: str                        # Publisher name
    source_credibility: float          # 0-1 trust score
    author: str
    publish_time: datetime
    category: str                      # Primary category
    topics: List[str]                  # Extracted topics
    entities: List[str]                # Named entities (people, places)

    # ML-derived features
    content_embedding: List[float]     # 768-dim BERT embedding
    topic_distribution: Dict[str, float]  # Topic probabilities

    # Engagement (aggregated)
    total_views: int
    click_through_rate: float
    avg_read_time: float
    share_rate: float

    # Quality signals
    freshness_score: float             # Decays over time
    breaking_news: bool                # Is this breaking news?
```

### Interaction Data

```python
class Interaction:
    user_id: str
    article_id: str
    timestamp: datetime

    # Engagement signals
    clicked: bool
    read_time_seconds: int
    scroll_depth: float                # 0-1, how far user scrolled
    shared: bool
    saved: bool

    # Negative signals
    skipped: bool                      # Shown but not clicked
    hidden: bool                       # User clicked "not interested"
```

---

## Candidate Generation (Retrieval)

### Multi-Source Retrieval Strategy

News recommendations require multiple retrieval sources due to the unique challenges of freshness and cold start.

#### Source 1: Interest-Based Retrieval (Two-Tower Model)

**Course Connection**: Week 8 (Two-Tower Architecture), Week 9 (Embeddings)

```python
class NewsTwoTower(nn.Module):
    def __init__(self, hidden_dim=256, embedding_dim=128):
        super().__init__()

        # User tower
        self.user_tower = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

        # Article tower (content-based for cold start)
        self.article_tower = nn.Sequential(
            nn.Linear(768, 512),  # BERT embedding input
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def encode_user(self, user_features):
        return F.normalize(self.user_tower(user_features), dim=1)

    def encode_article(self, article_embedding):
        return F.normalize(self.article_tower(article_embedding), dim=1)
```

**Why content-based for articles**: Every article is new (cold start problem). We use BERT embeddings of article content rather than collaborative signals.

**ANN Index**: FAISS with HNSW index for sub-millisecond retrieval
- Index updated every 5 minutes with new articles
- ~500 candidates retrieved per user

#### Source 2: Trending/Breaking News

```python
def get_trending_articles(country: str, k: int = 100) -> List[Article]:
    """
    Retrieve trending articles regardless of user preferences.

    Trending score = velocity × engagement × freshness
    """
    trending_score = (
        article.view_velocity *      # Views per hour
        article.engagement_rate *    # CTR × read_time
        freshness_decay(article.publish_time)
    )

    return top_k_by_score(articles, country, k)


def get_breaking_news() -> List[Article]:
    """
    Breaking news bypasses personalization.

    Detection signals:
    - Multiple sources covering same event
    - High velocity (views/hour > 10x normal)
    - Flagged by editorial team
    """
    return articles.filter(breaking_news=True)
```

#### Source 3: Topic-Based (Explicit Preferences)

```python
def get_topic_articles(user: UserProfile, k: int = 200) -> List[Article]:
    """
    Retrieve articles from user's followed topics.
    """
    candidates = []

    for topic in user.followed_topics:
        topic_articles = article_index.query(
            topic=topic,
            freshness_hours=48,
            k=k // len(user.followed_topics)
        )
        candidates.extend(topic_articles)

    return candidates
```

#### Source 4: Collaborative Filtering (Session-Based)

**Course Connection**: Week 6 (Sequential Recommendations)

```python
def get_similar_to_recent(user: UserProfile, k: int = 200) -> List[Article]:
    """
    Find articles similar to what user recently read.

    Uses session-based recommendations for short-term interests.
    """
    recent_embeddings = [
        get_article_embedding(aid)
        for aid in user.recent_reads[:10]
    ]

    # Average recent embeddings
    query_embedding = np.mean(recent_embeddings, axis=0)

    # ANN search
    return ann_index.search(query_embedding, k)
```

### Candidate Merging

```python
def generate_candidates(user: UserProfile) -> List[Article]:
    """
    Merge candidates from all sources with deduplication.
    """
    candidates = set()

    # Interest-based (primary)
    candidates.update(get_interest_based(user, k=500))

    # Trending (ensure diversity)
    candidates.update(get_trending_articles(user.country, k=100))

    # Breaking news (always include)
    candidates.update(get_breaking_news())

    # Topic-based (explicit follows)
    candidates.update(get_topic_articles(user, k=200))

    # Similar to recent reads
    candidates.update(get_similar_to_recent(user, k=200))

    return list(candidates)  # ~1000 candidates after dedup
```

---

## Ranking Model

### Model Architecture

**Course Connection**: Week 5 (Neural CF), Week 8 (Multi-Task Learning)

```python
class NewsRankingModel(nn.Module):
    """
    Multi-task ranking model predicting multiple engagement signals.
    """
    def __init__(self, user_dim=256, article_dim=768, context_dim=32):
        super().__init__()

        input_dim = user_dim + article_dim + context_dim

        # Shared bottom layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )

        # Task-specific towers
        self.click_tower = nn.Linear(256, 1)      # P(click)
        self.read_tower = nn.Linear(256, 1)       # P(read > 30s)
        self.share_tower = nn.Linear(256, 1)      # P(share)
        self.hide_tower = nn.Linear(256, 1)       # P(hide)

    def forward(self, user_features, article_features, context_features):
        x = torch.cat([user_features, article_features, context_features], dim=1)
        shared_repr = self.shared(x)

        return {
            'click': torch.sigmoid(self.click_tower(shared_repr)),
            'read': torch.sigmoid(self.read_tower(shared_repr)),
            'share': torch.sigmoid(self.share_tower(shared_repr)),
            'hide': torch.sigmoid(self.hide_tower(shared_repr))
        }
```

### Ranking Score Computation

```python
def compute_ranking_score(predictions: Dict[str, float],
                          article: Article) -> float:
    """
    Combine multi-task predictions into single ranking score.

    Weights determined by A/B testing.
    """
    # Base engagement score
    engagement_score = (
        0.3 * predictions['click'] +
        0.4 * predictions['read'] +    # Prioritize reading over clicking
        0.2 * predictions['share'] -
        0.3 * predictions['hide']      # Penalize predicted negative actions
    )

    # Freshness boost (news-specific)
    hours_since_publish = (now() - article.publish_time).hours
    freshness_weight = math.exp(-0.05 * hours_since_publish)

    # Source credibility
    credibility_weight = 0.5 + 0.5 * article.source_credibility

    # Final score
    score = engagement_score * freshness_weight * credibility_weight

    # Breaking news override
    if article.breaking_news:
        score *= 2.0

    return score
```

### Features Used

| Feature Category | Features | Description |
|-----------------|----------|-------------|
| **User-Article** | Topic overlap, Source affinity, Entity match | How well article matches user interests |
| **User** | Topic interests, Reading patterns, Session history | User preference model |
| **Article** | Freshness, Source credibility, Engagement rate | Article quality signals |
| **Context** | Time of day, Device type, Session position | Contextual relevance |
| **Cross** | User-topic × Article-topic, Historical CTR for user-source pair | Interaction features |

---

## Training Pipeline

### Data Collection

**Course Connection**: Week 3 (Implicit Feedback)

```python
class TrainingDataGenerator:
    def generate_training_examples(self,
                                   impressions: List[Impression]) -> Dataset:
        """
        Generate training data from impression logs.

        Positive: User clicked AND read > 30 seconds
        Negative: User saw but didn't click (with position bias correction)
        """
        examples = []

        for impression in impressions:
            user_features = get_user_features(impression.user_id)
            article_features = get_article_features(impression.article_id)
            context = get_context_features(impression)

            # Multi-task labels
            labels = {
                'click': impression.clicked,
                'read': impression.read_time > 30,
                'share': impression.shared,
                'hide': impression.hidden
            }

            # Position bias weight (IPS)
            position_weight = 1.0 / position_ctr[impression.position]

            examples.append({
                'features': (user_features, article_features, context),
                'labels': labels,
                'weight': position_weight
            })

        return examples
```

### Label Definition

```
Positive engagement:
- Click + Read > 30 seconds (primary)
- Share (strongest positive signal, but rare)
- Save for later (moderate positive)

Negative engagement:
- Shown but not clicked (weak negative, position-biased)
- "Not interested" feedback (strong negative)
- Rapid back-click (< 5 seconds) after clicking

Freshness handling:
- Only use articles from last 48 hours for training
- Weight more recent impressions higher
```

### Training Frequency

```
Model retraining schedule:
- Full model retrain: Weekly (Sunday night)
- Incremental updates: Daily (fine-tuning on last 24h data)
- Embedding index update: Every 5 minutes (new articles)

Why frequent updates matter for news:
- User interests shift (following a developing story)
- Article relevance changes (breaking news becomes stale)
- New topics emerge (elections, events)
```

---

## Serving Infrastructure

### Real-Time vs Batch Components

**Course Connection**: Week 13 (Production Systems)

```
┌─────────────────────────────────────────────────────────────┐
│                    BATCH PIPELINE                            │
│  (Daily)                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ User        │  │ Model       │  │ Article             │  │
│  │ Embedding   │  │ Training    │  │ Embedding           │  │
│  │ Generation  │  │             │  │ Generation          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    REAL-TIME PIPELINE                        │
│  (Per request)                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Feature     │  │ ANN         │  │ Ranking             │  │
│  │ Store       │  │ Search      │  │ Model               │  │
│  │ Lookup      │  │ (FAISS)     │  │ Inference           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 NEAR-REAL-TIME PIPELINE                      │
│  (Every 5 minutes)                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Article     │  │ Trending    │  │ Breaking News       │  │
│  │ Ingestion   │  │ Detection   │  │ Detection           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Caching Strategy

```python
class CachingLayers:
    """
    Multi-level caching for news recommendations.
    """

    # L1: User feature cache (Redis)
    user_features_ttl = 300  # 5 minutes

    # L2: Article feature cache (Redis)
    article_features_ttl = 60  # 1 minute (freshness changes)

    # L3: Trending articles cache (Redis)
    trending_cache_ttl = 60  # 1 minute (updates frequently)

    # L4: Pre-computed recommendations (for returning users)
    precomputed_recs_ttl = 300  # 5 minutes

    # Cache invalidation triggers:
    # - New article published → invalidate trending cache
    # - User interaction → invalidate user feature cache
    # - Breaking news → invalidate all user precomputed recs
```

### Latency Breakdown

```
Request flow timing:
────────────────────────────────────────────────────────
│ Step                          │ P50   │ P99   │
├───────────────────────────────┼───────┼───────┤
│ API Gateway + Auth            │ 5ms   │ 15ms  │
│ User feature lookup (Redis)   │ 2ms   │ 10ms  │
│ Candidate retrieval (FAISS)   │ 10ms  │ 25ms  │
│ Article feature lookup        │ 5ms   │ 15ms  │
│ Ranking model inference       │ 20ms  │ 40ms  │
│ Diversity re-ranking          │ 10ms  │ 20ms  │
│ Response serialization        │ 3ms   │ 8ms   │
├───────────────────────────────┼───────┼───────┤
│ Total                         │ 55ms  │ 133ms │
────────────────────────────────────────────────────────
```

---

## Metrics & Evaluation

### Offline Metrics

**Course Connection**: Week 11 (Evaluation)

| Metric | Description | Target |
|--------|-------------|--------|
| **NDCG@10** | Ranking quality | > 0.65 |
| **Recall@100** | Candidate retrieval | > 0.80 |
| **AUC (Click)** | Click prediction accuracy | > 0.75 |
| **AUC (Read)** | Read prediction accuracy | > 0.70 |
| **Coverage** | % of articles shown to at least 1 user | > 60% |

### Online Metrics (A/B Testing)

| Metric | Description | Type |
|--------|-------------|------|
| **CTR** | Click-through rate | Engagement |
| **Read-through rate** | % who read > 30s after click | Quality |
| **Session duration** | Time spent in app | Engagement |
| **Articles per session** | Depth of engagement | Engagement |
| **7-day retention** | Users returning in 7 days | Business |
| **DAU** | Daily active users | Business |

### Guardrail Metrics

**Course Connection**: Week 12 (Bias/Fairness)

```python
class GuardrailMetrics:
    """
    Metrics that must not degrade in A/B tests.
    """

    # Diversity metrics
    def topic_diversity(recommendations: List[Article]) -> float:
        """Entropy of topic distribution in recommendations."""
        topics = [a.category for a in recommendations]
        return entropy(Counter(topics))

    def source_diversity(recommendations: List[Article]) -> float:
        """Number of unique sources in top 20."""
        return len(set(a.source for a in recommendations[:20]))

    # Filter bubble prevention
    def viewpoint_diversity(recommendations: List[Article]) -> float:
        """For political news: balance of left/right sources."""
        # Requires source political leaning labels
        pass

    # Fairness metrics
    def publisher_exposure_gini(impressions: List) -> float:
        """Gini coefficient of impressions across publishers."""
        # Lower is better (more equal distribution)
        pass
```

---

## Trade-offs & Deep Dives

### Key Design Decisions

#### 1. Content-Based vs Collaborative Filtering

**Decision**: Heavy content-based focus with light collaborative signals

**Reasoning**:
- Every article is new (100% cold start for items)
- Article lifespan is 4-48 hours (no time to collect collaborative data)
- User interests can be inferred from article content

**Trade-off**: Less serendipity, but solves cold start problem

#### 2. Freshness vs Relevance

**Decision**: Exponential decay with freshness_score = e^(-0.05 * hours)

**Reasoning**:
- News value decays rapidly (breaking news → old news)
- Must balance with user interest (relevant old article > irrelevant new article)
- A/B tested multiple decay rates

**Trade-off**: May miss highly relevant older articles

#### 3. Personalization vs Filter Bubbles

**Decision**: 70% personalized, 20% trending, 10% exploration

**Reasoning**:
- Users expect personalization
- But need exposure to important news outside their interests
- Exploration helps discover new topics

**Trade-off**: Some users may dislike "off-topic" content

### Common Interview Follow-ups

**Q: How do you handle breaking news?**
> Breaking news detection system monitors article velocity and source coverage. When detected, breaking news bypasses personalization and is shown to all users in relevant geography. Detection signals: >10 sources covering same event within 1 hour, view velocity 10x normal.

**Q: How do you prevent filter bubbles?**
> Three strategies: (1) Inject 10% exploration content from outside user's interests, (2) Use MMR for topic diversity, (3) Mandatory inclusion of "top stories" regardless of personalization. We track viewpoint diversity as a guardrail metric.

**Q: How do you handle misinformation?**
> Multi-layered approach: (1) Source credibility scores based on fact-checking history, (2) NLP classifiers for clickbait/sensationalism, (3) Velocity checks (viral + unknown source = flag for review), (4) Integration with third-party fact-checkers. Flagged articles are de-ranked, not shown in trending.

**Q: What if a user's interests change?**
> Use time-decayed user interest model. Recent interactions weighted 2-3x more than older ones. Session-based signals capture short-term interests (following a developing story). Full user embedding updated daily, session context updated per-request.

---

## Course Concepts Applied

| Concept | Week | Application in News Recommendations |
|---------|------|-------------------------------------|
| **Collaborative Filtering** | 2-3 | Light CF signals from "users who read X also read Y" |
| **Content-Based Filtering** | 4 | Primary approach - BERT embeddings of articles for cold start |
| **Neural CF** | 5 | Deep ranking model with multi-task learning |
| **Sequential Models** | 6 | Session-based recommendations for developing stories |
| **Two-Tower Architecture** | 8 | User tower (interests) + Article tower (content) |
| **Multi-Task Learning** | 8 | Joint prediction of click, read, share, hide |
| **Embeddings** | 9 | User interest vectors, article content embeddings |
| **Contextual Bandits** | 10 | Exploration/exploitation for topic discovery |
| **Evaluation** | 11 | CTR vs read-through rate, offline vs online metrics |
| **Bias/Fairness** | 12 | Filter bubble prevention, source diversity |
| **Production Systems** | 13 | Feature store, latency optimization, caching |

---

## Implementation Checklist

### MVP (Week 1-2)
- [ ] Article ingestion pipeline with BERT embeddings
- [ ] Simple topic-based retrieval
- [ ] Basic ranking by freshness × relevance
- [ ] Source diversity constraints

### V1 (Week 3-4)
- [ ] Two-tower user-article embeddings
- [ ] FAISS index for ANN search
- [ ] Multi-task ranking model
- [ ] Trending detection

### V2 (Month 2)
- [ ] Real-time personalization
- [ ] Breaking news detection
- [ ] A/B testing infrastructure
- [ ] Filter bubble metrics

### V3 (Month 3+)
- [ ] Multi-language support
- [ ] Push notifications
- [ ] Advanced exploration (bandits)
- [ ] Publisher fairness optimization

---

## Summary

**News recommendation** presents unique challenges:

1. **Extreme cold start**: Every article is new, no collaborative signals at first
2. **Rapid freshness decay**: News becomes stale in hours, not days
3. **Content understanding required**: Must use NLP (BERT) heavily
4. **Filter bubble risks**: Personalization can create echo chambers
5. **Source credibility**: Must factor in trustworthiness

**Key architectural decisions**:
- Content-based retrieval with BERT embeddings
- Multi-task ranking (click, read, share, hide)
- Explicit freshness decay factor
- Mandatory diversity constraints
- Breaking news override system

**Metrics that matter**:
- Read-through rate > CTR (quality over clicks)
- Topic diversity (prevent filter bubbles)
- Source diversity (multiple perspectives)
- 7-day retention (long-term health)

---

## References

1. **Das, A., et al. (2007)**. "Google News Personalization: Scalable Online Collaborative Filtering". *WWW*.
   - Early Google News approach

2. **Liu, J., et al. (2010)**. "Personalized News Recommendation Based on Click Behavior". *IUI*.
   - Click-based personalization

3. **Carbonell, J., & Goldstein, J. (1998)**. "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries". *SIGIR*.
   - MMR for diversity

4. **Devlin, J., et al. (2019)**. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". *NAACL*.
   - BERT embeddings for content understanding

5. **Apple News Engineering Blog**. "Personalization at Apple News" (2021).
   - Industry practices

6. **Google AI Blog**. "How Google News uses machine learning" (2023).
   - Modern Google News architecture
