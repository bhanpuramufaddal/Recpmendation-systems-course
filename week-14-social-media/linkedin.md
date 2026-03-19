# System Design: LinkedIn Feed and Job Recommendations

## Problem Statement & Requirements

### Interview Prompt

> "Design LinkedIn's recommendation system for feed ranking, job recommendations, and People You May Know (PYMK) with 900M+ professional users."

### Functional Requirements

1. **Feed ranking**: Posts from connections, companies, groups
2. **Job recommendations**: Personalized job listings
3. **People You May Know (PYMK)**: Connection suggestions
4. **Learning recommendations**: Course suggestions for skill gaps
5. **Company recommendations**: Companies to follow
6. **Content/Article recommendations**: Professional content discovery

### Non-Functional Requirements

1. **Latency**: Feed load < 200ms
2. **Scale**: 900M users, 50M companies, 40M job postings
3. **Professional context**: Optimize for career value, not entertainment
4. **Availability**: 99.9% uptime

### Scope

**In scope**: Feed ranking, job matching, connection suggestions
**Out of scope**: Messaging, LinkedIn Learning infrastructure, Premium features

---

## Scale Estimation (Back-of-Envelope)

### Users & Traffic

```
Users:
- Total members: 900M+
- Monthly Active Users: 300M
- Daily Active Users: 50M
- Premium subscribers: 50M

Content:
- Posts created per day: 10M
- Job postings active: 40M
- Companies: 50M+
- Skills in taxonomy: 50K+

Traffic:
- Feed loads per user per day: 5
- Job searches per day: 100M
- PYMK impressions per day: 500M
- Total recommendation requests: 50M × 10 = 500M/day
- Average QPS: 6,000
- Peak QPS (business hours): 20,000
```

### Storage

```
User Embeddings:
- Users: 900M
- Embedding dimension: 256
- Storage: 900M × 256 × 4 bytes = 900GB

Job Embeddings:
- Active jobs: 40M
- Embedding dimension: 256
- Storage: 40M × 256 × 4 bytes = 40GB

Economic Graph (Knowledge Graph):
- Nodes: 1B+ (members, companies, skills, schools, jobs)
- Edges: 10B+
- Storage: Distributed graph DB (several TB)

Feature Store:
- User features: 900M × 2KB = 1.8TB
- Job features: 40M × 1KB = 40GB
```

### Latency Budget (Feed)

```
Total budget: 200ms

User feature lookup: 20ms
Social graph query: 30ms
Candidate retrieval: 40ms
Feature hydration: 30ms
Ranking model (GLMix): 40ms
Diversity re-ranking: 20ms
Network overhead: 20ms
```

---

## High-Level Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           Recommendation Surfaces        │
                    │  ┌──────┐ ┌──────┐ ┌──────┐ ┌────────┐  │
                    │  │ Feed │ │ Jobs │ │ PYMK │ │Learning│  │
                    │  └──────┘ └──────┘ └──────┘ └────────┘  │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │           Economic Graph                 │
                    │  (Members, Companies, Skills, Schools)   │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────────┐
                    ▼                  ▼                      ▼
             ┌──────────┐       ┌──────────┐           ┌──────────┐
             │ Feed     │       │ Job      │           │ PYMK     │
             │ Ranking  │       │ Matching │           │ Graph    │
             │ (GLMix)  │       │ (Two-Way)│           │ Search   │
             └──────────┘       └──────────┘           └──────────┘
```

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

**Course Connection**: Week 5 (Neural CF), Week 3 (Matrix Factorization)

**LinkedIn's approach**: Generalized Linear Mixed-effects model.

```python
class GLMixModel:
    """
    Generalized Linear Mixed-effects Model for feed ranking.

    Combines:
    - Fixed effects: Global patterns (content type, time of day)
    - Random effects: User-specific and item-specific biases

    Score = β^T x + u_i + v_j
    where:
    - β^T x: Fixed effects (global features)
    - u_i: User random effect (personalization)
    - v_j: Item random effect (item quality)
    """
    def __init__(self, n_users, n_items, n_features):
        self.beta = np.zeros(n_features)     # Fixed effects
        self.user_effects = np.zeros(n_users)  # Random user effects
        self.item_effects = np.zeros(n_items)  # Random item effects

    def score(self, user_id, item_id, features):
        fixed = np.dot(self.beta, features)
        user_effect = self.user_effects[user_id]
        item_effect = self.item_effects[item_id]
        return fixed + user_effect + item_effect

    def train(self, interactions):
        """
        Train using coordinate descent:
        1. Fix random effects, optimize fixed effects
        2. Fix fixed effects, optimize random effects
        """
        pass
```

**Benefits**:
- Handles sparsity well
- Interpretable coefficients
- Personalization without deep learning complexity

---

### Feature Engineering

| Category | Features | Description |
|----------|----------|-------------|
| **User** | Industry, seniority, company size, skills | Professional context |
| **Post** | Content type, author seniority, engagement rate | Post quality |
| **Author** | Connection degree, mutual connections, company | Author relevance |
| **Context** | Time of day, device, session depth | Temporal context |
| **Cross** | User-author affinity, industry match | Personalization |

---

## Job Recommendations

### Two-Sided Matching

**Course Connection**: Week 12 (Bias/Fairness - marketplace balance)

**Dual optimization**:
1. **Job seekers**: Find relevant jobs
2. **Recruiters**: Find qualified candidates

```python
class JobMatchingModel:
    """
    Two-sided marketplace matching.

    Optimize for both sides:
    - P(apply | user, job): Will user apply?
    - P(hire | user, job): Would recruiter hire?

    Final score balances both.
    """
    def __init__(self):
        self.apply_model = ApplyPredictionModel()
        self.hire_model = HirePredictionModel()

    def score(self, user: UserProfile, job: JobPosting) -> float:
        p_apply = self.apply_model.predict(user, job)
        p_hire = self.hire_model.predict(user, job)

        # Balance seeker and recruiter objectives
        # Higher weight on apply (seeker-side) for better UX
        score = 0.6 * p_apply + 0.4 * p_hire

        return score
```

### Job-User Features

**User (job seeker)**:
- Skills (from profile + inferred)
- Experience level (years, titles)
- Location preferences
- Salary expectations
- Job search activity

**Job (posting)**:
- Required skills
- Seniority level
- Location (remote, hybrid, onsite)
- Company
- Salary range

**Cross features**:
- Skill match score
- Location distance
- Salary compatibility
- Company affinity (alumni, followers)

---

### Economic Graph for Skill Inference

**Course Connection**: Week 7 (Graph-Based Methods)

**LinkedIn's knowledge graph**:
- **Nodes**: Members, companies, jobs, skills, schools
- **Edges**: Employment, education, has_skill, requires_skill

```python
class EconomicGraph:
    """
    Graph-based skill inference.

    Infer missing skills from:
    - Company employees (people at Company X have Skill Y)
    - Title patterns (Software Engineers have Python)
    - Education (Stanford CS grads have ML)
    """
    def infer_skills(self, user: UserProfile) -> List[str]:
        explicit_skills = user.listed_skills
        inferred_skills = []

        # From current/past companies
        for company in user.companies:
            common_skills = self.graph.query(
                """
                MATCH (c:Company {id: $company_id})<-[:WORKS_AT]-(m:Member)-[:HAS_SKILL]->(s:Skill)
                WITH s, count(m) as member_count
                WHERE member_count > 100
                RETURN s.name
                """,
                company_id=company.id
            )
            inferred_skills.extend(common_skills)

        # From title patterns
        for title in user.titles:
            title_skills = self.title_skill_mapping.get(title.normalized, [])
            inferred_skills.extend(title_skills)

        return list(set(inferred_skills) - set(explicit_skills))
```

---

## People You May Know (PYMK)

### Graph-Based Approach

**Course Connection**: Week 7 (Graph-Based Methods)

```python
class PYMKRecommender:
    """
    Connection recommendations using graph algorithms.
    """
    def __init__(self, graph):
        self.graph = graph

    def get_candidates(self, user_id: str, k: int = 100) -> List[str]:
        """
        Multi-signal candidate generation.
        """
        candidates = defaultdict(float)

        # 1. Mutual connections (strongest signal)
        mutual_friends = self.get_friends_of_friends(user_id)
        for candidate, mutual_count in mutual_friends:
            candidates[candidate] += 0.4 * min(mutual_count / 10, 1.0)

        # 2. Same company (current or past)
        coworkers = self.get_coworkers(user_id)
        for candidate in coworkers:
            candidates[candidate] += 0.3

        # 3. Same school (alumni)
        alumni = self.get_alumni(user_id)
        for candidate in alumni:
            candidates[candidate] += 0.15

        # 4. Similar skills
        skill_similar = self.get_skill_similar(user_id)
        for candidate, similarity in skill_similar:
            candidates[candidate] += 0.1 * similarity

        # 5. Geographic proximity
        nearby = self.get_nearby_members(user_id)
        for candidate in nearby:
            candidates[candidate] += 0.05

        # Sort and return
        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
        return [c for c, _ in sorted_candidates[:k]]

    def personalized_pagerank(self, user_id: str, k: int = 100) -> List[str]:
        """
        Random walk with restart from user node.
        """
        # Start at user node
        # Random walk with 15% restart probability
        # Nodes visited frequently → good suggestions
        return self.graph.personalized_pagerank(
            start_node=user_id,
            restart_prob=0.15,
            max_iterations=100
        )[:k]
```

### PYMK Scoring

```python
def pymk_score(user: UserProfile, candidate: UserProfile) -> float:
    """
    Score potential connections.
    """
    score = 0

    # Mutual connections (most important)
    mutual = count_mutual_connections(user, candidate)
    score += 0.4 * min(mutual / 10, 1.0)

    # Same company
    if any(c in candidate.companies for c in user.companies):
        score += 0.25

    # Same school
    if any(s in candidate.schools for s in user.schools):
        score += 0.15

    # Skill overlap
    skill_overlap = jaccard_similarity(user.skills, candidate.skills)
    score += 0.1 * skill_overlap

    # Geographic proximity
    if same_city(user, candidate):
        score += 0.05

    # Industry match
    if user.industry == candidate.industry:
        score += 0.05

    return score
```

---

## Learning Recommendations

### Skill Gap Analysis

```python
class LearningRecommender:
    """
    Recommend courses based on skill gaps.
    """
    def get_recommendations(self, user: UserProfile,
                           target_role: str = None) -> List[Course]:
        # Identify target skills
        if target_role:
            target_skills = self.get_role_skills(target_role)
        else:
            target_skills = self.get_trending_skills(user.industry)

        # Find skill gaps
        current_skills = set(user.skills)
        missing_skills = target_skills - current_skills

        # Recommend courses for missing skills
        courses = []
        for skill in missing_skills:
            skill_courses = self.get_courses_for_skill(skill)
            courses.extend(skill_courses)

        # Rank by relevance and quality
        ranked = self.rank_courses(courses, user)

        return ranked[:10]

    def get_learning_path(self, user: UserProfile,
                          goal: str) -> List[Course]:
        """
        Sequential course recommendations.

        Example:
        Current: Software Engineer
        Goal: Engineering Manager

        Path: Leadership 101 → Agile Management → People Management
        """
        # Model as sequence-to-sequence
        current_state = self.encode_skills(user.skills)
        target_state = self.encode_role(goal)

        # Find optimal path
        path = self.path_finder.find_path(current_state, target_state)

        return path
```

---

## Challenges

### Cold Start

**New users**: No connections, no job history.

**Solutions**:
1. **Onboarding**: Ask for current role, skills, interests
2. **Demographic matching**: Similar users by role/industry
3. **Popular content**: Trending posts, top jobs

### Spam and Low-Quality Content

**Problem**: Promotional posts, engagement bait.

**Detection**:
- Engagement signals: High impressions, low engagement
- Content analysis: Spammy keywords, excessive links
- User reports: "Mark as spam"

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

## Serving Infrastructure

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    REAL-TIME LAYER                           │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Feed         │  │ Job          │  │ PYMK             │   │
│  │ Service      │  │ Matching     │  │ Service          │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Feature      │  │ Economic     │  │ GLMix            │   │
│  │ Store        │  │ Graph        │  │ Model            │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    BATCH LAYER                               │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Embedding    │  │ Graph        │  │ Model            │   │
│  │ Generation   │  │ Analysis     │  │ Training         │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Course Concepts Applied

| Concept | Week | Application in LinkedIn |
|---------|------|-------------------------|
| **Collaborative Filtering** | 2-3 | Users with similar engagement patterns |
| **Matrix Factorization** | 3 | User and item effects in GLMix |
| **Content-Based** | 4 | Post content, job description matching |
| **Neural CF** | 5 | Deep job matching models |
| **Sequential Models** | 6 | Learning path recommendations |
| **Graph-Based** | 7 | Economic Graph, PYMK via PageRank |
| **Two-Tower** | 8 | Job-seeker matching |
| **Multi-Task Learning** | 8 | Engagement + professional value |
| **Embeddings** | 9 | Skill embeddings, company embeddings |
| **Contextual Bandits** | 10 | Exploration in job recommendations |
| **Evaluation** | 11 | Professional value metrics |
| **Bias/Fairness** | 12 | Two-sided marketplace balance |
| **Production Systems** | 13 | GLMix serving, graph queries |

---

## Summary

**Key Takeaways**:
1. **GLMix**: Mixed-effects model for feed ranking (interpretable, handles sparsity)
2. **Two-sided job matching**: Balance seeker and recruiter objectives
3. **Economic Graph**: Knowledge graph for skill inference and PYMK
4. **Graph-based PYMK**: Personalized PageRank, mutual connections
5. **Learning paths**: Sequential recommendations for career growth

**Professional context**: Unlike Facebook/Instagram (entertainment), LinkedIn optimizes for professional value.

---

## References

1. **Zhang, Y., et al. (2019)**. "GLMix: Generalized Linear Mixed Models for Large-Scale Response Prediction". *KDD* (LinkedIn).
2. **Kenthapadi, K., et al. (2017)**. "Personalized Job Recommendation System at LinkedIn". *RecSys*.
3. **Gupta, P., et al. (2013)**. "PYMK: Friend Recommendation at LinkedIn". *SIGMOD*.
4. **Xu, L., et al. (2020)**. "Understanding Searches Better: Query Understanding at LinkedIn". *SIGIR*.
