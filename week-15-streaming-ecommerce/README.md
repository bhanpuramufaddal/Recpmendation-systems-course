# Week 15: Streaming and E-Commerce Platforms

## Overview

Streaming platforms (Netflix, Spotify, YouTube) and e-commerce (Amazon) have different objectives and constraints than social media. This week explores these industry-leading systems.

## Topics

### [1. Netflix Recommendation System](netflix.md)
**80%+ of viewing from recommendations**

**Key Features**:
- Row generation (personalized genres)
- Artwork personalization (contextual bandits)
- Ranking within rows vs. ranking rows
- Session-based patterns

**2024 Updates**:
- Hydra: Multi-task unified model
- Reinforcement learning integration
- Regional personalization

**Papers**:
- Gomez-Uribe & Hunt (2016). "The Netflix recommender system". *ACM TIST*.

### [2. Spotify Music Recommendations](spotify.md)
**30%+ of streams from recommendations**

**Flagship Features**:
- **Discover Weekly**: CF + audio + NLP
- **Daily Mix**: Taste clustering
- **Release Radar**: New music
- **Radio**: Sequential recommendations

**Technical Stack**:
- Collaborative filtering on 2B playlists
- Audio CNN (tempo, energy, danceability)
- NLP on playlist names

### [3. YouTube Recommendations](youtube-detailed.md)
**70%+ of watch time from recommendations**

**Architecture**:
- Two-tower candidate generation
- Deep ranking network
- Watch time optimization

**Challenges**:
- Clickbait and misleading content
- Borderline content reduction
- Shorts vs. long-form

**Paper**: Covington et al. (2016). "Deep neural networks for YouTube recommendations". *RecSys*.

### [4. Amazon Product Recommendations](amazon.md)
**35% of revenue from recommendations**

**Key Systems**:
- Item-to-item collaborative filtering
- "Customers who bought this also bought"
- Personalized homepage
- Search ranking

**2024 Generative AI**:
- Personalized product descriptions
- Contextual recommendations
- Activity-based personalization

**Scale**: Millions of A/B experiments

## Comparison

| Platform | Primary Metric | Content Type | Session Length | Key Innovation |
|----------|---------------|--------------|----------------|----------------|
| **Netflix** | Watch time | Long-form video | 1-3 hours | Artwork personalization |
| **Spotify** | Stream count | Audio | 30-60 min | Audio feature analysis |
| **YouTube** | Watch time | Short+long video | 20-40 min | Two-stage architecture |
| **Amazon** | Revenue | Products | 10-15 min | Item-to-item CF |

*Return to [Main Course Page](../README.md)*
