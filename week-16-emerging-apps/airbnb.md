# System Design: Airbnb Recommendation System

## Problem Statement & Requirements

### Interview Prompt

> "Design Airbnb's search ranking and recommendation system for a two-sided marketplace with 4M+ listings and 150M+ annual bookings."

### Functional Requirements

1. **Search ranking**: Rank listings for location + date queries
2. **Similar listings**: "Homes similar to this" on listing pages
3. **Personalized recommendations**: Homepage recommendations
4. **Experiences recommendations**: Local activities
5. **Wishlist recommendations**: Based on saved listings

### Non-Functional Requirements

1. **Latency**: Search results < 200ms
2. **Scale**: 4M listings, 150M bookings/year
3. **Two-sided optimization**: Balance guest and host objectives
4. **Location-aware**: Respect geographic constraints

### Scope

**In scope**: Search ranking, listing embeddings, personalization
**Out of scope**: Pricing algorithms, payment processing, messaging

---

## Scale Estimation (Back-of-Envelope)

### Users & Traffic

```
Users:
- Active guests: 100M/year
- Active hosts: 4M
- DAU (searching): 10M

Listings:
- Active listings: 4M
- Listings per city (median): 500
- Listings per city (major cities): 50,000+

Traffic:
- Searches per day: 50M
- Listing page views per day: 200M
- Bookings per day: 400K
- Average QPS: 600
- Peak QPS: 5,000
```

### Storage

```
Listing Embeddings:
- Listings: 4M
- Embedding dimension: 128
- Storage: 4M × 128 × 4 bytes = 2GB

User Embeddings:
- Active users: 100M
- Embedding dimension: 128
- Storage: 100M × 128 × 4 bytes = 50GB

Feature Store:
- Listing features: 4M × 5KB = 20GB
- User features: 100M × 1KB = 100GB
```

### Latency Budget (Search)

```
Total budget: 200ms

Location/date filtering: 20ms
User feature lookup: 15ms
Candidate retrieval (ANN): 30ms
Listing feature hydration: 30ms
Ranking model: 50ms
Position debiasing: 20ms
Response assembly: 20ms
Network overhead: 15ms
```

---

## Overview

Airbnb's recommendation system faces **unique challenges** as a **two-sided marketplace** for travel accommodations. Unlike Netflix (one-sided) or Amazon (retailer-controlled), Airbnb must match **guests with hosts** while respecting **location constraints**, **availability**, and **pricing**.

This document covers Airbnb's search ranking and recommendation systems, based on published research and engineering blog posts.

**Business impact**: Recommendations drive **40%+ of bookings**, worth billions in gross booking value (GBV).

---

## Learning Objectives

By the end of this section, you will:
- Understand two-sided marketplace recommendation challenges
- Master Airbnb's ranking framework (search + recommendations)
- Learn embedding-based similarity for listings
- Recognize location-aware personalization
- Apply lessons to marketplace and travel systems

---

## Airbnb at Scale (2024)

### The Numbers

- **Guests**: 150M+ bookings per year
- **Hosts**: 4M+ active listings
- **Countries**: 220+
- **Cities**: 100,000+
- **Nights booked annually**: 400M+

### The Challenge

**Unlike Netflix/Spotify**:
- **Location matters**: Can't recommend listings in wrong city
- **Availability**: Listings may be booked for desired dates
- **Pricing**: Dynamic, varies by season, demand
- **Two-sided**: Must satisfy both guests AND hosts

---

## The Two-Sided Marketplace Problem

### Guest Goals

- Find listing matching preferences (price, amenities, style)
- Good location (proximity to attractions)
- Available for desired dates
- Reasonable price
- Trustworthy host (reviews, response rate)

### Host Goals

- Maximize occupancy (no empty nights)
- Maximize revenue (higher prices when possible)
- Attract quality guests (good reviews, no damages)

### Platform Goals

- Maximize bookings (gross booking value)
- Balance supply and demand
- Grow both sides of marketplace
- Ensure quality experiences (reviews, safety)

**Challenge**: Optimize for all three!

---

## Search Ranking

### The Core Problem

**User query**: "Paris, France, Dec 1-5, 2 guests"

**Candidates**: 50,000+ listings in Paris

**Goal**: Rank top 20 to show in search results.

---

### Ranking Signals

**Published by Airbnb** (Haldar et al., 2019):

**1. Location**:
- Distance to query location (city center, neighborhood)
- Walkability score
- Proximity to attractions

**2. Price**:
- Listing price vs. user's budget (inferred from past searches/bookings)
- Price competitiveness (compared to similar listings)

**3. Availability**:
- Available for requested dates? (Hard constraint)
- Calendar density (how often listing is booked)

**4. Quality**:
- Review score (average rating)
- Number of reviews
- Superhost status (badge for top hosts)
- Instant Book (can book without host approval)

**5. Guest-Listing Fit**:
- User preferences (learned from past behavior)
- Listing amenities (WiFi, kitchen, pool, etc.)
- Listing type (entire home, private room, shared room)

**6. Booking Probability**:
- Predicted P(book | view)
- Trained on historical data

---

### Learning to Rank

**Model**: Gradient Boosted Decision Trees (GBDT) or Deep Neural Network

**Features** (100+ total):
- **User features**: Past bookings, searches, clicks, demographics
- **Listing features**: Price, location, amenities, reviews, photos
- **Context features**: Search query, dates, number of guests, device

**Target**:
$$y = \begin{cases} 1 & \text{if user booked listing} \\ 0 & \text{otherwise} \end{cases}$$

**Loss**: Cross-entropy (classification)

$$\mathcal{L} = -\sum_{(u,l)} y_{ul} \log(\hat{y}_{ul}) + (1 - y_{ul}) \log(1 - \hat{y}_{ul})$$

---

### Position Bias

**Problem**: Listings at top get more clicks (regardless of quality).

**Data bias**: Top-ranked listings have more bookings in training data → model learns to rank them higher → reinforcement.

**Solution**: Inverse Propensity Scoring (IPS)

**Weight training examples**:
$$w_{\text{position}} = \frac{1}{P(\text{click | position})}$$

**Example**:
- Position 1: $P(\text{click}) = 0.5$ → weight = 2
- Position 10: $P(\text{click}) = 0.05$ → weight = 20

**Result**: Model learns true listing quality, not position bias.

---

## Listing Embeddings

### Learning Similarities

**Goal**: Represent listings as vectors such that similar listings are close.

**Approach**: Embedding from click sessions (similar to word2vec).

**Paper**: Grbovic & Cheng, "Real-time Personalization using Embeddings for Search Ranking at Airbnb" (KDD 2018)

---

### Session-Based Embeddings

**Idea**: Users who click listing A then listing B → A and B are similar.

**Data**: Click sessions
```
Session 1: [listing_123, listing_456, listing_789]
Session 2: [listing_456, listing_101, listing_202]
...
```

**Model**: Skip-gram (word2vec variant)

**Objective**: Predict context listings given target listing.

$$\max \sum_{\text{sessions}} \sum_{l \in \text{session}} \sum_{c \in \text{context}(l)} \log P(c | l)$$

where:
$$P(c | l) = \frac{\exp(\mathbf{v}_l^T \mathbf{v}_c)}{\sum_{l'} \exp(\mathbf{v}_l^T \mathbf{v}_{l'})}$$

**Negative sampling**: Approximate denominator with negative samples.

---

### Booked Listing as Global Context

**Innovation**: Booked listing is added to context of ALL listings in session.

**Intuition**: Listings user considered are similar to the one they booked.

**Example**:
```
Session: [listing_A, listing_B, listing_C] → booked listing_C

Standard skip-gram:
- listing_A context: [listing_B, listing_C]
- listing_B context: [listing_A, listing_C]
- listing_C context: [listing_A, listing_B]

Airbnb's approach:
- listing_A context: [listing_B, listing_C (neighbor), listing_C (booked)]
- listing_B context: [listing_A, listing_C (neighbor), listing_C (booked)]
- listing_C context: [listing_A, listing_B, listing_C (booked - self)]

listing_C appears twice (as neighbor and as booked) → stronger signal.
```

**Result**: Embeddings capture booking intent, not just browsing.

---

### Implementation

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

class ListingEmbedding(nn.Module):
    def __init__(self, n_listings, embedding_dim=128):
        super().__init__()
        # Listing embeddings
        self.listing_emb = nn.Embedding(n_listings, embedding_dim)
        # Initialize
        nn.init.uniform_(self.listing_emb.weight, -0.1, 0.1)

    def forward(self, center_listing, context_listings, negative_samples):
        """
        center_listing: (batch,) - target listing IDs
        context_listings: (batch, n_context) - context listing IDs
        negative_samples: (batch, n_negatives) - negative listing IDs
        """
        # Embeddings
        center_emb = self.listing_emb(center_listing)  # (batch, dim)
        context_emb = self.listing_emb(context_listings)  # (batch, n_context, dim)
        negative_emb = self.listing_emb(negative_samples)  # (batch, n_negatives, dim)

        # Positive scores
        pos_scores = torch.bmm(context_emb, center_emb.unsqueeze(2)).squeeze()  # (batch, n_context)
        pos_loss = -F.logsigmoid(pos_scores).sum(dim=1).mean()

        # Negative scores
        neg_scores = torch.bmm(negative_emb, center_emb.unsqueeze(2)).squeeze()  # (batch, n_negatives)
        neg_loss = -F.logsigmoid(-neg_scores).sum(dim=1).mean()

        return pos_loss + neg_loss

# Usage
model = ListingEmbedding(n_listings=100000, embedding_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
for session in click_sessions:
    center = session['listing']
    context = session['context_listings']
    booked = session['booked_listing']  # Add to context

    # Add booked listing to context
    context_with_booked = context + [booked]

    # Sample negatives
    negatives = sample_negatives(n_negatives=10)

    # Forward
    loss = model(center, context_with_booked, negatives)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After training: use embeddings for similarity
listing_A_emb = model.listing_emb(torch.tensor([listing_A_id]))
listing_B_emb = model.listing_emb(torch.tensor([listing_B_id]))
similarity = F.cosine_similarity(listing_A_emb, listing_B_emb)
```

---

### Use Cases

**1. Similar Listings**:
- Show "Homes similar to this" on listing page
- Compute cosine similarity between embeddings

**2. Personalized Ranking**:
- User's embedding = average of listings they've clicked/booked
- Rank new listings by similarity to user embedding

**3. Explore Feature**:
- "Listings you might like" based on embedding similarity

---

## Location-Aware Personalization

### The Location Challenge

**Problem**: Users search different locations for each trip.

**Example**:
- User's past bookings: Paris, London, Tokyo
- Current search: New York
- Can't just recommend "listings similar to past bookings" (wrong city!)

**Solution**: Separate location-agnostic preferences from location.

---

### Decomposition

**User preferences** = **Location-agnostic** + **Location-specific**

**Location-agnostic**:
- Preferred price range
- Amenities (WiFi, kitchen, AC)
- Property type (entire home vs. room)
- Host characteristics (Superhost, response time)

**Location-specific**:
- Neighborhood preference (trendy vs. quiet)
- Distance to city center

**Ranking**: Combine both with location features for current search.

---

## Pricing Signals

### Dynamic Pricing

**Challenge**: Listing prices vary by:
- Season (summer vs. winter)
- Day of week (weekend vs. weekday)
- Events (conferences, holidays)
- Last-minute availability

**Implication for ranking**: Can't just use static price, must consider price elasticity.

---

### Price Sensitivity

**Observation**: Users have different price sensitivities.

**Signals**:
- Past booking prices (high/low)
- Search filter prices
- Clicks on expensive vs. budget listings

**Model**: Predict user's price sensitivity
$$P(\text{book | price}) = f(\text{price}, \text{user sensitivity})$$

**Ranking**: Boost listings matching user's budget.

---

## Supply-Demand Balancing

### The Marketplace Challenge

**Goal**: Keep both sides happy.

**Metrics**:
- **Guest metrics**: Booking success rate, search-to-book conversion
- **Host metrics**: Occupancy rate, revenue

**Trade-off**:
- Recommend popular listings → high booking rate BUT low occupancy for less popular listings
- Recommend diverse listings → lower booking rate BUT better occupancy distribution

**Solution**: Multi-objective optimization.

---

### Host Diversity

**Ensure** less popular listings get exposure:

**1. Exploration**:
- Inject 10-20% of less-shown but quality listings
- Use multi-armed bandits (Thompson Sampling)

**2. Quality Score**:
- Rank not just by booking probability, but also listing quality (reviews, photos)

**3. New Listing Boost**:
- Give new listings (< 3 bookings) temporary ranking boost
- Helps hosts get initial bookings

---

## Experiences Recommendations

### Beyond Accommodations

**Airbnb Experiences** (2016): Local activities hosted by locals (cooking classes, tours, workshops).

**Recommendation challenge**:
- Cold start (new experiences daily)
- Location-dependent
- Time constraints (experiences have fixed schedules)

---

### Ranking Experiences

**Signals**:
1. **Location fit**: Experience in same city as accommodation booking
2. **Interest fit**: User's past experience bookings, search queries
3. **Quality**: Review scores, completion rate
4. **Availability**: Available during user's trip dates
5. **Price**: Matches user's budget

**Model**: Similar to listing ranking (GBDT or DNN)

**Placement**: Show experiences on search results page, after booking confirmation.

---

## Cold Start Strategies

### New Listings

**Challenge**: No booking history, no reviews.

**Solutions**:

**1. Content-based features**:
- Price, amenities, photos (quality score from CV model)
- Location (neighborhood quality)
- Host profile (previous listings, response rate)

**2. Bootstrapping**:
- Give temporary ranking boost (first 30 days)
- Encourage hosts to lower prices initially (build reviews)

**3. Similar Listings**:
- Find similar listings (via embeddings) with bookings
- Use their engagement data as proxy

---

### New Users

**Challenge**: No search/booking history.

**Solutions**:

**1. Onboarding**:
- Ask: "Where do you want to go?"
- Collect: Dates, number of guests, budget

**2. Demographic defaults**:
- Age, gender, location → predict preferences

**3. Trending**:
- Show popular destinations, trending listings

**4. Rapid personalization**:
- After first search, update profile
- After 2-3 searches, have enough data for personalization

---

## A/B Testing at Airbnb

### Experimentation Culture

**Scale**: 100s of experiments running concurrently.

**Metrics**:
- **Primary**: Booking rate, gross booking value (GBV)
- **Secondary**: Search-to-book conversion, engagement (clicks, saves)
- **Quality**: Review scores, host satisfaction

**Example**: Embedding-based similar listings

**Test**:
- **Control**: Rule-based similar listings (same price range, location)
- **Treatment**: Embedding-based similar listings

**Results**:
- **Booking rate**: +5% (statistically significant)
- **GBV**: +\$10M annually (projected)
- **Decision**: Ship to 100%

---

## Lessons from Airbnb

### 1. Marketplace Complexity

**Two-sided marketplaces** require balancing guest AND host goals.

**Takeaway**: Optimize for multiple stakeholders, not just end users.

---

### 2. Location is a Hard Constraint

**Can't recommend listings in wrong city** (unlike movies on Netflix).

**Takeaway**: Respect hard constraints, personalize within constraints.

---

### 3. Embeddings from Implicit Feedback

**Click sessions** → embeddings (without explicit ratings).

**Booked listing as global context** → better embeddings.

**Takeaway**: Leverage behavioral data creatively.

---

### 4. Position Bias Matters

**Top-ranked items get more clicks** → biased training data.

**Inverse Propensity Scoring** corrects bias.

**Takeaway**: Always consider position bias in ranking systems.

---

### 5. Cold Start is Critical

**New listings and users** are common in marketplaces.

**Content-based features + bootstrapping** help cold start.

**Takeaway**: Design for cold start from day one.

---

## Technical Architecture (Estimated)

**Data Pipeline**:
- **Kafka**: Real-time events (searches, clicks, bookings)
- **Spark**: Batch processing for embeddings, features
- **Airflow**: Orchestration (daily embedding training)

**Model Training**:
- **XGBoost/LightGBM**: Ranking models (GBDT)
- **PyTorch**: Embedding models (word2vec-style)
- **TensorFlow**: Deep neural networks (ranking)

**Model Serving**:
- **Search ranking**: Microservice (Java/Scala)
- **Embedding similarity**: FAISS for ANN search
- **Latency**: <200ms for search ranking

**Offline Batch**:
- **Embedding training**: Daily (Spark cluster)
- **Feature generation**: Hourly (listing availability, prices)

---

## Summary

**Key Takeaways**:
1. **Two-sided marketplace**: Must balance guests, hosts, and platform
2. **Location-aware**: Hard constraint + personalization within location
3. **Embeddings from sessions**: Click sessions → similar listings
4. **Booked listing as global context**: Innovation for booking intent
5. **Position bias correction**: Inverse propensity scoring
6. **Multi-objective**: Booking rate + quality + diversity

**Architecture**:
```
User Search Query
       ↓
Candidate Retrieval (filter by location, dates, price)
       ↓
Feature Engineering (user, listing, context)
       ↓
Ranking Model (GBDT/DNN)
       ↓
Position Debiasing (IPS)
       ↓
Top-20 Results
```

**Technologies**:
- Models: GBDT (XGBoost), Word2vec (listings), Deep Neural Networks
- Scale: 4M listings, 150M bookings/year
- Latency: <200ms for search ranking
- Embeddings: 128D, trained on click sessions

**For Builders**:
- Start with content-based (location, price, amenities)
- Add collaborative filtering (embeddings from sessions)
- Correct position bias (IPS)
- Multi-objective optimization (bookings + quality + diversity)
- A/B test everything

---

---

## Course Concepts Applied

| Concept | Week | Application in Airbnb |
|---------|------|----------------------|
| **Collaborative Filtering** | 2-3 | Users who booked X also booked Y |
| **Matrix Factorization** | 3 | Listing embeddings from booking co-occurrence |
| **Content-Based** | 4 | Listing amenities, photos, description |
| **Neural CF** | 5 | Deep ranking model for booking prediction |
| **Sequential Models** | 6 | Session-based click embeddings |
| **Graph-Based** | 7 | Listing similarity graphs by location |
| **Two-Tower** | 8 | User preferences + Listing features |
| **Multi-Task Learning** | 8 | Click, inquiry, booking joint prediction |
| **Embeddings** | 9 | Listing embeddings from click sessions (Word2Vec-style) |
| **Contextual Bandits** | 10 | Exploration for new listings |
| **Evaluation** | 11 | Booking rate, GBV metrics |
| **Bias/Fairness** | 12 | Position bias (IPS), host diversity |
| **Production Systems** | 13 | Real-time inventory, latency optimization |

---

## References

1. **Haldar, M., et al. (2019)**. "Applying Deep Learning to Airbnb Search". *KDD*.
   - **Search ranking** at Airbnb

2. **Grbovic, M., & Cheng, H. (2018)**. "Real-time Personalization using Embeddings for Search Ranking at Airbnb". *KDD*.
   - **Listing embeddings**, breakthrough work

3. **Abdool, S., et al. (2020)**. "Managing Diversity in Airbnb Search". *KDD*.
   - **Supply-demand balancing**, host diversity

4. **Chen, L., et al. (2019)**. "A Study on Deep Learning Approaches for Session-Based Recommendations in E-Commerce". *RecSys Workshop*.
   - Session-based techniques applicable to Airbnb

5. **Airbnb Engineering Blog**. Various posts on search ranking, embeddings, pricing.
   - **Technical deep dives** from Airbnb engineers
