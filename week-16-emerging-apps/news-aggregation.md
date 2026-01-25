# Week 16: News Aggregation (Google News, Apple News)

## Overview

**News aggregation**: Personalized news feeds.

**Challenges**:
1. **Freshness**: News is time-sensitive (breaking news)
2. **Diversity**: Avoid filter bubbles, show multiple perspectives
3. **Source credibility**: Prioritize reputable sources
4. **Personalization vs. serendipity**: Balance interests with discovery

---

## Freshness vs. Relevance Trade-off

### Recency Boost

**Problem**: Old articles less relevant.

**Solution**: Decay score by time.

$$\text{Score} = \text{Relevance} \times e^{-\lambda t}$$

where $t$ = hours since publication, $\lambda$ = decay rate (e.g., 0.1).

---

### Breaking News Detection

**Signals**:
- Sudden spike in publications on topic
- Social media mentions increase
- Multiple reputable sources covering

**Action**: Override personalization, show to all users.

---

## Diversification

### Topic Diversification

**Goal**: Show multiple topics, not just user's favorites.

**Method**: **MMR** (Maximal Marginal Relevance)

$$\text{MMR} = \arg\max_{d_i} \left[ \lambda \cdot \text{Sim}(d_i, q) - (1-\lambda) \cdot \max_{d_j \in S} \text{Sim}(d_i, d_j) \right]$$

where:
- $\text{Sim}(d_i, q)$ = relevance to user query/interests
- $\max_{d_j \in S}$ = similarity to already-selected articles
- $\lambda$ = diversity parameter (0.5-0.7)

**Effect**: Penalize articles similar to already-shown ones.

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

## Summary

**Key Takeaways**:
1. **Freshness**: Recency boost, breaking news detection
2. **Diversity**: Topic and source diversification (MMR)
3. **Credibility**: Trust signals, fact-checking integration
4. **Misinformation**: NLP detection, de-ranking
5. **Personalization**: Topic modeling, implicit feedback

**Ethical considerations**: Balance personalization with diverse viewpoints.

---

## References

1. **Google News Blog**: "How Google News Works" (2021).
2. **Resnick, P., et al. (2013)**. "Bursting Your (Filter) Bubble". *CHI*.
