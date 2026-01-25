# Week 14: Facebook News Feed Ranking

## Overview

**Facebook News Feed**: Personalized content stream for 3B+ users.

**Challenge**: Rank thousands of posts (friends, pages, ads) → show top 50-100.

**Objectives**:
1. Maximize meaningful interactions (not just clicks)
2. Diverse content (not all from one source)
3. Balance organic vs. ads

---

## Evolution

### EdgeRank (2006-2013)

**Formula**:
$$\text{EdgeRank} = \sum_{\text{edges}} \text{Affinity} \times \text{Weight} \times \text{Decay}$$

- **Affinity**: User-creator relationship strength
- **Weight**: Content type (photo > link > status)
- **Decay**: Time since posting

**Limitation**: Simple linear model, no personalization.

---

### Machine Learning Era (2013+)

**Models**: Logistic regression → GBDTs → Deep neural networks.

**Features** (1000+):
- User: demographics, past interactions
- Post: content type, creator, engagement
- Context: time of day, device
- Cross-features: user-post interactions

---

## Multi-Objective Optimization

### Predicted Metrics

**Facebook predicts**:
1. **Click probability**: $P(\text{click})$
2. **Like probability**: $P(\text{like})$
3. **Comment probability**: $P(\text{comment})$
4. **Share probability**: $P(\text{share})$
5. **Hide probability**: $P(\text{hide})$ (negative signal)

**Combined score**:
$$\text{Score} = w_1 P(\text{click}) + w_2 P(\text{like}) + w_3 P(\text{comment}) + w_4 P(\text{share}) - w_5 P(\text{hide})$$

**Weights learned** via offline experiments + online A/B tests.

---

### Value Model

**Beyond engagement**: Optimize for "time well spent".

**Facebook's shift (2018)**:
- De-emphasize passive consumption (watching videos)
- Prioritize active engagement (comments, shares)
- Boost posts from close friends/family

**Metric**: **Meaningful Social Interactions (MSI)**

---

## Architecture

### Two-Stage Ranking

**Stage 1: Candidate Generation**
- Input: ~10,000 posts
- Output: ~500 candidates
- Method: Lightweight model (logistic regression)
- Latency: <50ms

**Stage 2: Heavy Ranking**
- Input: 500 candidates
- Output: Top 50-100 ranked
- Method: Deep neural network
- Latency: <200ms

---

### Feature Engineering

**User features**:
- Demographics (age, location)
- Historical engagement (avg. likes per day)
- Connection graph (friend count, groups)

**Post features**:
- Content type (video, photo, link, text)
- Creator (friend, page, ad)
- Virality (existing engagement)
- Freshness (time since posting)

**Cross features**:
- User affinity to creator
- User's past engagement with similar content

---

## Handling Inventory Constraints

### Problem

**Inventory**: Limited slots (~100 posts shown per session).

**Demand**: Thousands of posts compete.

**Constraint**: Each creator gets fair exposure.

---

### Solution: Calibration

**Ensure** % of feed from each source matches user's historical preferences.

**Example**:
- User historically engages 40% friends, 30% pages, 30% groups
- Feed should reflect this distribution

---

## Addressing Challenges

### Filter Bubbles

**Problem**: Users only see content aligning with beliefs.

**Mitigation**:
1. **Diverse viewpoints**: Inject 10% posts from outside bubble
2. **Context labels**: "Why am I seeing this?"
3. **User control**: "Show more/less of this"

---

### Misinformation

**Problem**: False news spreads faster than truth.

**Mitigation**:
1. **Fact-checking**: Partner with third-party fact-checkers
2. **De-ranking**: Reduce viral misinformation
3. **Context**: Show "Missing context" labels
4. **User reports**: "Report as false news"

---

### Clickbait

**Problem**: Misleading headlines get clicks.

**Detection**:
- NLP: Analyze headline vs. article content
- Engagement signals: High clicks, low time-on-site

**Penalty**: Reduce ranking of clickbait posts.

---

## A/B Testing Infrastructure

### Scale

**Facebook runs**: 10,000+ experiments concurrently.

**Users in experiments**: Millions per variant.

**Duration**: 1-2 weeks for statistical significance.

---

### Metrics

**Primary**: User engagement (likes, comments, shares).

**Secondary**:
- Session length
- Return rate (7-day, 30-day)
- User satisfaction surveys

**Long-term**: User retention, revenue.

---

## Summary

**Key Takeaways**:
1. **Multi-objective**: Balance engagement, MSI, revenue
2. **Two-stage**: Fast candidate generation + heavy ranking
3. **Inventory management**: Fair exposure across sources
4. **Challenges**: Filter bubbles, misinformation, clickbait
5. **Scale**: 10K+ concurrent A/B tests

**Impact**: News Feed drives 50%+ of Facebook engagement.

---

## References

1. **Facebook Engineering Blog**: "News Feed Ranking" (multiple posts 2013-2021)
2. **Zhao, X., et al. (2019)**. "Recommending What Video to Watch Next: A Multitask Ranking System". *RecSys* (similar architecture).
