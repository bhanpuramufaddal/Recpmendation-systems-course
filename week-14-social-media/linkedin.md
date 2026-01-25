# Week 14: LinkedIn Feed and Job Recommendations

## Overview

**LinkedIn**: Professional network with 900M+ users.

**Recommendation types**:
1. **Feed**: Posts from connections, companies
2. **Jobs**: Personalized job listings
3. **People You May Know (PYMK)**: Connection suggestions
4. **Learning**: Course recommendations

**Unique challenge**: Professional context (not entertainment).

---

## Feed Ranking

### Objectives

**Goals**:
1. Professional value (not just engagement)
2. Diverse content types (posts, articles, videos, jobs)
3. Creator fairness (give voice to all members)

**Metrics**:
- Engagement (likes, comments, shares)
- Dwell time (time spent reading)
- Professional actions (job applications, connections)

---

### GLMix Model

**LinkedIn's approach**: Generalized Linear Mixed-effects model.

**Components**:
1. **Fixed effects**: Global patterns (content type, time of day)
2. **Random effects**: User-specific, item-specific biases

**Formula**:
$$\text{score} = \beta^T x + u_i + v_j$$

where:
- $\beta^T x$: Fixed effects (global features)
- $u_i$: User random effect
- $v_j$: Item random effect

**Benefits**: Handles sparsity, interpretable.

---

## Job Recommendations

### Two-Sided Matching

**Dual optimization**:
1. **Job seekers**: Find relevant jobs
2. **Recruiters**: Find qualified candidates

**Matching score**:
$$\text{Score} = P(\text{apply} | \text{user, job}) \times P(\text{hire} | \text{user, job})$$

---

### Features

**User (job seeker)**:
- Skills (from profile)
- Experience level
- Location preferences
- Job search history

**Job (posting)**:
- Required skills
- Seniority level
- Location
- Company

**Cross features**:
- Skill match score
- Location distance
- Salary compatibility

---

### Economic Graph

**LinkedIn's knowledge graph**:
- **Nodes**: Members, companies, jobs, skills, schools
- **Edges**: Connections, employment, education

**Use**: Infer missing skills, predict career transitions.

**Example**:
```
User worked at Company A
Company A employees have Skill X → Infer user has Skill X
```

---

## People You May Know (PYMK)

### Algorithm

**Goal**: Suggest relevant connections.

**Signals**:
1. **Mutual connections**: Friends of friends
2. **Same company/school**: Coworkers, alumni
3. **Similar skills**: Professional similarity
4. **Geographic proximity**: Same city

**Scoring**:
$$\text{PYMK score} = w_1 \cdot \text{mutual} + w_2 \cdot \text{company} + w_3 \cdot \text{skills} + w_4 \cdot \text{location}$$

---

### Graph-Based Approach

**Random walks** on LinkedIn graph.

**Personalized PageRank**:
1. Start at user node
2. Random walk with restart
3. Nodes visited frequently → good suggestions

---

## Learning Recommendations

### Course Suggestions

**Objectives**:
1. **Skill gaps**: Identify missing skills for career goals
2. **Trending skills**: Popular in user's industry
3. **Personalization**: Match learning style, time commitment

**Example**:
```
User: Software Engineer at Startup
Goal: Become Engineering Manager
Missing skills: Leadership, Project Management

Recommend: "Management 101", "Agile for Managers"
```

---

### Learning Paths

**Sequential recommendations**: Course A → Course B → Course C.

**Approach**: Model as sequence-to-sequence problem.

**Input**: Current skills
**Output**: Optimal learning sequence

---

## Challenges

### Cold Start

**New users**: No connections, no job history.

**Solution**:
1. **Onboarding**: Ask for current role, skills, interests
2. **Demographic matching**: Similar users by role/industry
3. **Popular content**: Trending posts, top jobs

---

### Spam and Low-Quality Content

**Problem**: Promotional posts, clickbait.

**Detection**:
- **Engagement signals**: High impressions, low engagement
- **Content analysis**: Spammy keywords, excessive links
- **User reports**: "Mark as spam"

**Action**: De-rank or remove.

---

## A/B Testing

### Metrics

**Engagement**: Likes, comments, shares per session.

**Professional value**:
- Job applications submitted
- Connections made
- Messages sent

**Long-term**:
- User retention (monthly active users)
- Premium conversions

---

## Summary

**Key Takeaways**:
1. **GLMix**: Mixed-effects model for feed ranking
2. **Job matching**: Two-sided optimization (seeker + recruiter)
3. **Economic Graph**: Knowledge graph for skill inference
4. **PYMK**: Graph-based connection suggestions
5. **Learning**: Skill gap analysis, learning paths

**Professional context**: Unlike Facebook/Instagram (entertainment), LinkedIn optimizes for professional value.

---

## References

1. **Zhang, Y., et al. (2019)**. "GLMix: Generalized Linear Mixed Models for Large-Scale Response Prediction". *KDD* (LinkedIn).
2. **Kenthapadi, K., et al. (2017)**. "Personalized Job Recommendation System at LinkedIn". *RecSys*.
3. **Gupta, P., et al. (2013)**. "PYMK: Friend Recommendation at LinkedIn". *SIGMOD*.
