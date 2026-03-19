# System Design: News Aggregation (Google News, Apple News)

## Problem Statement & Requirements

### Interview Prompt

> "Design a news aggregation and recommendation system like Google News or Apple News that personalizes news feeds for 100M+ users."

### Functional Requirements

1. **Personalized news feed**: Articles matching user interests
2. **Breaking news**: Surface urgent/trending stories
3. **Topic browsing**: Explore by category (Politics, Tech, Sports)
4. **Source diversity**: Show articles from multiple publishers
5. **Local news**: Geographically relevant stories
6. **Notifications**: Push alerts for breaking news

### Non-Functional Requirements

1. **Latency**: Feed generation < 100ms
2. **Freshness**: New articles indexed within 5 minutes
3. **Scale**: 100M DAU, 50K new articles/hour
4. **Availability**: 99.9% uptime

### Scope

**In scope**: Article recommendation, freshness handling, diversity
**Out of scope**: Full-text search, content hosting, subscriptions

---

## Scale Estimation (Back-of-Envelope)

### Users & Traffic

```
Users:
- Daily Active Users: 100M
- Sessions per user: 3
- Articles viewed per session: 10

Content:
- New articles per hour: 50,000
- New articles per day: 1.2M
- Active articles (48-hour window): 2.4M
- Total indexed (30 days): 36M
- Sources: 10,000+

Traffic:
- Feed requests per day: 100M × 3 = 300M
- Average QPS: 3,500
- Peak QPS: 35,000
```

### Storage

```
Article Embeddings:
- Active articles: 2.4M
- Embedding dimension: 768 (BERT)
- Storage: 2.4M × 768 × 4 = 7.2GB

User Embeddings:
- Users: 100M
- Embedding dimension: 256
- Storage: 100M × 256 × 4 = 100GB

Feature Store:
- User features: 100M × 500 bytes = 50GB
- Article features: 36M × 2KB = 72GB
```

### Latency Budget

```
Total budget: 100ms

User feature lookup: 5ms
Candidate retrieval (ANN): 15ms
Article feature lookup: 10ms
Ranking model: 30ms
Diversity re-ranking: 15ms
Freshness boost: 5ms
Network overhead: 20ms
```

---

## High-Level Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         Article Ingestion Pipeline       │
                    │  - RSS feeds, APIs, web scraping         │
                    │  - NLP processing (BERT embeddings)      │
                    │  - Entity extraction, categorization     │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │         Content Understanding            │
                    │  - BERT embeddings (768-dim)             │
                    │  - Topic classification                  │
                    │  - Source credibility scoring            │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────────┐
                    ▼                                          ▼
             ┌──────────────┐                          ┌──────────────┐
             │ User         │                          │ Trending     │
             │ Interest     │                          │ Detection    │
             │ Model        │                          │              │
             └──────┬───────┘                          └──────┬───────┘
                    │                                         │
                    └──────────────────┬──────────────────────┘
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │              Ranking Model               │
                    │  - Relevance × Freshness × Credibility  │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │           Diversity Layer                │
                    │  - Topic diversity (MMR)                 │
                    │  - Source diversity                      │
                    │  - Viewpoint balance                     │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │           Final Feed                     │
                    └─────────────────────────────────────────┘
```

---

## Unique Challenges

### 1. Freshness: News Expires in Hours

```python
def freshness_score(article):
    """
    News value decays exponentially with time.

    λ = 0.05 gives:
    - 1 hour old: 0.95
    - 12 hours old: 0.55
    - 24 hours old: 0.30
    - 48 hours old: 0.09
    """
    hours_since_publish = (now() - article.publish_time).total_seconds() / 3600
    return math.exp(-0.05 * hours_since_publish)
```

### 2. Cold Start: Every Article is New

**Problem**: No collaborative signals for new articles.

**Solution**: Content-based embeddings using BERT.

```python
def get_article_embedding(article):
    """
    BERT embedding from title + first 512 chars of body.
    Enables similarity matching without engagement history.
    """
    text = article.title + " " + article.body[:512]
    embedding = bert_model.encode(text)  # 768-dim
    return embedding
```

### 3. Diversity: Avoid Filter Bubbles

**Problem**: Pure personalization creates echo chambers.

**Solution**: MMR (Maximal Marginal Relevance).

---

## Freshness vs. Relevance Trade-off

### Recency Boost

**Problem**: Old articles less relevant.

**Solution**: Decay score by time.

$$\text{Score} = \text{Relevance} \times e^{-\lambda t}$$

where $t$ = hours since publication, $\lambda$ = decay rate (e.g., 0.05).

---

### Breaking News Detection

**Course Connection**: Week 10 (Bandits - exploration override)

**Signals**:
- Sudden spike in publications on topic
- Social media mentions increase
- Multiple reputable sources covering

**Action**: Override personalization, show to all users.

```python
def is_breaking_news(topic):
    """
    Detect breaking news events.
    """
    # Count articles in last hour
    recent_count = count_articles(topic, hours=1)
    baseline = get_baseline_rate(topic)

    # 5x surge = breaking news
    if recent_count > 5 * baseline:
        # Verify multiple sources
        sources = get_unique_sources(topic, hours=1)
        if len(sources) >= 3:
            return True

    return False
```

---

## Diversification

### Topic Diversification (MMR)

**Course Connection**: Week 12 (Bias/Fairness)

**Goal**: Show multiple topics, not just user's favorites.

**Method**: **MMR** (Maximal Marginal Relevance)

$$\text{MMR} = \arg\max_{d_i} \left[ \lambda \cdot \text{Sim}(d_i, q) - (1-\lambda) \cdot \max_{d_j \in S} \text{Sim}(d_i, d_j) \right]$$

where:
- $\text{Sim}(d_i, q)$ = relevance to user query/interests
- $\max_{d_j \in S}$ = similarity to already-selected articles
- $\lambda$ = diversity parameter (0.5-0.7)

**Effect**: Penalize articles similar to already-shown ones.

```python
def select_diverse_articles(candidates, user_embedding, k=20, lambda_param=0.6):
    """
    MMR selection for topic diversity.
    """
    selected = []
    remaining = candidates.copy()

    while len(selected) < k and remaining:
        scores = []

        for article in remaining:
            # Relevance to user
            relevance = cosine_similarity(article.embedding, user_embedding)

            # Max similarity to already selected
            if selected:
                max_sim = max(
                    cosine_similarity(article.embedding, s.embedding)
                    for s in selected
                )
            else:
                max_sim = 0

            # MMR score
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            scores.append((article, mmr))

        # Select highest MMR
        best = max(scores, key=lambda x: x[1])[0]
        selected.append(best)
        remaining.remove(best)

    return selected
```

---

### Source Diversity

**Goal**: Show articles from multiple sources.

**Constraint**: Max 2 articles from same source in top-10.

```python
def diversify_sources(articles, k=10):
    selected = []
    source_count = {}

    for article in articles:
        source = article['source']

        # Limit per source
        if source_count.get(source, 0) >= 2:
            continue

        selected.append(article)
        source_count[source] = source_count.get(source, 0) + 1

        if len(selected) >= k:
            break

    return selected
```

---

## Source Credibility

### Trust Signals

**Indicators**:
1. **Domain authority**: Established news organizations (NYT, BBC)
2. **Fact-checking**: No history of misinformation
3. **Journalistic standards**: Professional reporters, editors

**Scoring**:
```python
def credibility_score(source):
    score = 0

    # Established organization
    if source in TRUSTED_SOURCES:
        score += 0.5

    # Fact-checking partnerships
    if source['fact_checked']:
        score += 0.3

    # User reports (low spam/misinformation)
    if source['user_report_rate'] < 0.01:
        score += 0.2

    return min(score, 1.0)
```

---

## Misinformation Detection

### Fact-Checking Integration

**Partners**: Snopes, PolitiFact, FactCheck.org.

**Process**:
1. Detect viral claims
2. Send to fact-checkers
3. Label articles: "False", "Partly false", "Missing context"

**Ranking**: De-rank false/misleading articles.

---

### NLP-Based Detection

**Features**:
- Sensational language ("You won't believe...")
- Clickbait patterns
- Lack of sources/citations
- Suspicious domain (unknown sites)

**Model**: Classify as credible vs. misinformation.

---

## Personalization

### Interest Modeling

**Course Connection**: Week 9 (Embeddings)

**Approach**: Topic modeling (LDA, BERT) on articles user clicked.

**Topics**: Politics, Technology, Sports, Entertainment, etc.

**User vector**: Distribution over topics.

```python
user_interests = {
    'politics': 0.4,
    'technology': 0.3,
    'sports': 0.2,
    'entertainment': 0.1
}

# Score article by topic overlap
def score_article(article, user_interests):
    article_topics = article['topics']  # {'politics': 0.6, 'technology': 0.4}

    score = sum(user_interests.get(topic, 0) * weight
                for topic, weight in article_topics.items())

    return score
```

---

### Implicit Feedback

**Course Connection**: Week 3 (Implicit Feedback)

**Signals**:
- **Click**: Interested in topic
- **Read time**: >30s → strong interest
- **Share**: Very high interest
- **Skip**: Not interested

**Update interests** based on feedback.

---

## Reading Time Prediction

### Model

**Predict**: How long user will read article.

**Features**:
- Article length (word count)
- Topic relevance
- Source credibility
- User's historical read time

**Use case**: Prioritize articles user will fully read (not just click).

---

## Cross-Device Behavior

### Sync Across Devices

**User reads on**:
- Mobile (commute): Short articles (2-3 min reads)
- Desktop (work break): Medium articles (5-10 min)
- Tablet (evening): Long-form (15+ min)

**Personalization**: Recommend article length based on device and time.

---

## Course Concepts Applied

| Concept | Week | Application in News Aggregation |
|---------|------|--------------------------------|
| **Collaborative Filtering** | 2-3 | Users who read X also read Y |
| **Content-Based** | 4 | BERT embeddings for article cold start |
| **Neural CF** | 5 | Deep ranking for read-through prediction |
| **Sequential Models** | 6 | Session-based interest detection |
| **Two-Tower** | 8 | User interests + Article content |
| **Multi-Task Learning** | 8 | Click, read, share prediction |
| **Embeddings** | 9 | Article content embeddings |
| **Contextual Bandits** | 10 | Exploration for new topics |
| **Evaluation** | 11 | CTR vs read-through rate metrics |
| **Bias/Fairness** | 12 | Filter bubble prevention, source diversity |
| **Production Systems** | 13 | Near-real-time ingestion, freshness |

---

## Summary

**Key Takeaways**:
1. **Freshness**: Recency boost, breaking news detection
2. **Cold start**: Content-based BERT embeddings (no CF signals)
3. **Diversity**: Topic and source diversification (MMR)
4. **Credibility**: Trust signals, fact-checking integration
5. **Misinformation**: NLP detection, de-ranking
6. **Personalization**: Topic modeling, implicit feedback

**Ethical considerations**: Balance personalization with diverse viewpoints.

---

## References

1. **Das, A., et al. (2007)**. "Google News Personalization: Scalable Online Collaborative Filtering". *WWW*.
   - Early Google News approach

2. **Resnick, P., et al. (2013)**. "Bursting Your (Filter) Bubble". *CHI*.
   - Filter bubble research

3. **Carbonell, J., & Goldstein, J. (1998)**. "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries". *SIGIR*.
   - MMR algorithm

4. **Google News Blog**. "How Google News Works" (2021).
   - Official Google News description
