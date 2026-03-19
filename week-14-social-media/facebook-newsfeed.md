# System Design: Facebook News Feed Ranking

## Problem Statement & Requirements

### Interview Prompt

> "Design a personalized news feed for Facebook with 3 billion users, showing posts from friends, pages, and groups ranked by predicted engagement."

### Functional Requirements

1. **Personalized feed**: Show relevant posts from friends, pages, and groups
2. **Real-time updates**: New posts appear without refresh
3. **Multi-format content**: Text, photos, videos, links, events
4. **Social interactions**: Like, comment, share, save
5. **Advertising integration**: Blend sponsored content naturally
6. **Integrity**: Filter misinformation, spam, and harmful content

### Non-Functional Requirements

1. **Latency**: Feed generation < 200ms
2. **Freshness**: New posts visible within seconds
3. **Availability**: 99.99% uptime
4. **Scale**: 3B users, 2B DAU, 100M+ posts created per hour
5. **Personalization**: Adapt in real-time to user behavior

### Scope

**In scope**:
- Feed ranking algorithm
- Multi-objective optimization
- Integrity signal integration
- Real-time personalization

**Out of scope**:
- Ad auction system (separate system)
- Content creation/upload
- Notification system

---

## Scale Estimation (Back-of-Envelope)

### Users & Traffic

```
Users:
- Monthly Active Users (MAU): 3B
- Daily Active Users (DAU): 2B
- DAU/MAU ratio: 67% (highly engaged platform)

Content:
- Posts created per day: 2B+
- Active posts (eligible for feed): ~100M at any time
- Average connections per user: 300 friends + 100 pages/groups

Traffic:
- Feed loads per user per day: 10
- Total feed requests per day: 2B × 10 = 20B requests/day
- Average QPS: 20B / 86,400 = 230,000 QPS
- Peak QPS (3x): 700,000 QPS

Post inventory per user:
- Eligible posts per user: 300 friends × 2 posts + 100 pages × 5 posts = 1,100 posts
- After activity filtering: ~500 posts to consider
```

### Storage

```
User Embeddings:
- Users: 3B
- Embedding dimension: 256
- Storage: 3B × 256 × 4 bytes = 3TB

Post Embeddings:
- Active posts: 100M
- Embedding dimension: 256
- Storage: 100M × 256 × 4 bytes = 100GB

Feature Store:
- User features: 3B × 2KB = 6TB
- Post features: 100M × 1KB = 100GB
- Social graph: 3B users × 400 edges × 8 bytes = 10TB
```

### Latency Budget Breakdown

```
Total budget: 200ms

Component breakdown:
- Social graph query (inventory): 20ms
- User feature lookup: 10ms
- First-pass candidate scoring: 30ms
- Post feature lookup: 15ms
- Heavy ranking model: 50ms
- Integrity classifiers: 25ms
- Diversity re-ranking: 20ms
- Ad insertion: 15ms
- Network overhead: 15ms
```

---

## High-Level Architecture

```
                        ┌─────────────────────────────┐
                        │     Content Pool            │
                        │  (Friends, Pages, Groups)   │
                        │        ~10K posts           │
                        └────────────┬────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────┐
                        │    Inventory Selection      │
                        │  (Filter by recency, type)  │
                        │        ~2K posts            │
                        └────────────┬────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────┐
                        │    First-Pass Ranking       │
                        │  (Lightweight ML model)     │
                        │        ~500 posts           │
                        └────────────┬────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────┐
                        │    Integrity Filtering      │
                        │  (Misinformation, Spam)     │
                        │        ~450 posts           │
                        └────────────┬────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────┐
                        │    Heavy Ranking            │
                        │  (Deep Neural Network)      │
                        │  Multi-objective scoring    │
                        │        ~100 posts           │
                        └────────────┬────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────┐
                        │    Diversity & Blending     │
                        │  - Source diversity         │
                        │  - Content type mix         │
                        │  - Ad insertion slots       │
                        │        ~50 posts            │
                        └────────────┬────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────┐
                        │        Final Feed           │
                        └─────────────────────────────┘
```

---

## Data Model

### User Features

```python
class UserProfile:
    user_id: str

    # Demographics
    age: int
    gender: str
    location: str
    language: str

    # Social graph
    friend_ids: List[str]
    followed_pages: List[str]
    joined_groups: List[str]

    # Engagement patterns
    avg_session_duration: float
    posts_per_day_engagement: int
    preferred_content_types: Dict[str, float]  # {"video": 0.6, "photo": 0.3}

    # Historical engagement
    recent_likes: List[str]         # Last 100 post IDs liked
    recent_comments: List[str]      # Last 50 posts commented
    recent_shares: List[str]        # Last 20 posts shared
    recent_hides: List[str]         # Posts hidden

    # Embeddings
    interest_embedding: List[float]  # 256-dim
    social_embedding: List[float]    # 256-dim (from graph)
```

### Post Features

```python
class Post:
    post_id: str
    author_id: str

    # Content
    content_type: str              # "text", "photo", "video", "link", "event"
    text: str
    media_urls: List[str]
    link_url: Optional[str]

    # Metadata
    created_time: datetime
    updated_time: datetime
    privacy: str                   # "public", "friends", "custom"

    # Engagement (real-time)
    like_count: int
    comment_count: int
    share_count: int
    view_count: int

    # ML-derived
    content_embedding: List[float]  # 256-dim
    topic_labels: List[str]
    sentiment: float               # -1 to 1

    # Integrity signals
    spam_score: float
    misinformation_score: float
    clickbait_score: float
    violence_score: float
```

### Social Context Features

```python
class SocialContext:
    """
    Features capturing relationship between user and post author.
    """
    user_id: str
    author_id: str

    # Relationship
    relationship_type: str         # "friend", "page", "group"
    connection_strength: float     # 0-1, based on interactions

    # Interaction history
    total_interactions: int        # All-time likes/comments with author
    recent_interactions: int       # Last 30 days
    messages_exchanged: int

    # Social proof
    mutual_friends_engaged: int    # Friends who liked this post
    mutual_friends_count: int      # Total mutual friends with author

    # Affinity score
    affinity_score: float          # Pre-computed user-author affinity
```

---

## Candidate Generation (Inventory Selection)

### Multi-Source Inventory

**Course Connection**: Week 7 (Graph-Based Methods)

```python
def get_candidate_posts(user: UserProfile, k: int = 2000) -> List[Post]:
    """
    Retrieve candidate posts from all eligible sources.
    """
    candidates = []

    # Source 1: Friends' posts (social graph traversal)
    friend_posts = graph_db.query(
        """
        MATCH (u:User {id: $user_id})-[:FRIENDS]->(friend:User)-[:POSTED]->(p:Post)
        WHERE p.created_time > datetime() - duration('P7D')
        RETURN p
        ORDER BY p.created_time DESC
        LIMIT 1000
        """,
        user_id=user.user_id
    )
    candidates.extend(friend_posts)

    # Source 2: Followed pages
    page_posts = graph_db.query(
        """
        MATCH (u:User {id: $user_id})-[:FOLLOWS]->(page:Page)-[:POSTED]->(p:Post)
        WHERE p.created_time > datetime() - duration('P3D')
        RETURN p
        ORDER BY p.engagement_score DESC
        LIMIT 500
        """,
        user_id=user.user_id
    )
    candidates.extend(page_posts)

    # Source 3: Group posts
    group_posts = get_group_posts(user.joined_groups, limit=300)
    candidates.extend(group_posts)

    # Source 4: Suggested content (weak ties, viral content)
    suggested = get_suggested_content(user, limit=200)
    candidates.extend(suggested)

    return deduplicate(candidates)[:k]
```

### First-Pass Ranking (Lightweight Model)

```python
class FirstPassRanker:
    """
    Lightweight model for initial filtering.
    Uses simple logistic regression or shallow network.
    """
    def __init__(self):
        self.model = LogisticRegression()

    def score(self, user: UserProfile, posts: List[Post]) -> List[float]:
        features = []
        for post in posts:
            f = [
                user_author_affinity(user, post.author_id),
                post.like_count / (post.view_count + 1),
                hours_since_created(post),
                content_type_preference(user, post.content_type),
                mutual_friends_engaged(user, post),
            ]
            features.append(f)

        return self.model.predict_proba(features)[:, 1]

    def filter_top_k(self, user: UserProfile, posts: List[Post], k: int = 500):
        scores = self.score(user, posts)
        top_indices = np.argsort(scores)[-k:][::-1]
        return [posts[i] for i in top_indices]
```

---

## Multi-Objective Ranking Model

### Predicted Metrics

**Course Connection**: Week 8 (Multi-Task Learning)

Facebook predicts multiple engagement signals:

```python
class FeedRankingModel(nn.Module):
    """
    Multi-task deep learning model predicting multiple engagement types.
    """
    def __init__(self, user_dim=256, post_dim=256, context_dim=64):
        super().__init__()

        input_dim = user_dim + post_dim + context_dim

        # Shared bottom network
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Task-specific towers
        self.click_head = nn.Linear(256, 1)      # P(click)
        self.like_head = nn.Linear(256, 1)       # P(like)
        self.comment_head = nn.Linear(256, 1)    # P(comment)
        self.share_head = nn.Linear(256, 1)      # P(share)
        self.hide_head = nn.Linear(256, 1)       # P(hide) - negative
        self.watch_time_head = nn.Linear(256, 1) # E[watch_time] for videos

    def forward(self, user_features, post_features, context_features):
        x = torch.cat([user_features, post_features, context_features], dim=1)
        shared_repr = self.shared(x)

        return {
            'click': torch.sigmoid(self.click_head(shared_repr)),
            'like': torch.sigmoid(self.like_head(shared_repr)),
            'comment': torch.sigmoid(self.comment_head(shared_repr)),
            'share': torch.sigmoid(self.share_head(shared_repr)),
            'hide': torch.sigmoid(self.hide_head(shared_repr)),
            'watch_time': F.relu(self.watch_time_head(shared_repr))
        }
```

### Value Model (Combined Score)

```python
def compute_feed_score(predictions: Dict[str, float],
                       post: Post,
                       integrity_scores: Dict[str, float]) -> float:
    """
    Combine predictions into single ranking score.

    The weights reflect Facebook's 2018 shift to "Meaningful Social Interactions" (MSI):
    - Comments and shares weighted higher than likes
    - Active engagement > passive consumption
    """
    # Engagement score (MSI-weighted)
    engagement_score = (
        0.05 * predictions['click'] +      # Low weight (passive)
        0.15 * predictions['like'] +       # Moderate weight
        0.35 * predictions['comment'] +    # High weight (active)
        0.35 * predictions['share'] +      # High weight (active)
        0.10 * predictions['watch_time'] - # Video engagement
        0.50 * predictions['hide']         # Strong negative signal
    )

    # Integrity penalty
    integrity_penalty = (
        0.3 * integrity_scores['spam'] +
        0.4 * integrity_scores['misinformation'] +
        0.2 * integrity_scores['clickbait'] +
        0.1 * integrity_scores['violence']
    )

    # Freshness factor
    hours_old = hours_since_created(post)
    freshness = 1.0 / (1.0 + 0.1 * hours_old)

    # Final score
    score = engagement_score * (1 - integrity_penalty) * freshness

    return score
```

### Feature Engineering (1000+ Features)

| Category | Example Features | Count |
|----------|------------------|-------|
| **User** | Age, gender, language, session count, avg engagement | ~50 |
| **Post** | Content type, text length, media count, hashtags | ~100 |
| **Author** | Friend/page/group, post frequency, avg engagement | ~50 |
| **Social Context** | Mutual friends engaged, affinity score, interaction history | ~100 |
| **Engagement Signals** | Like count, comment count, share velocity | ~50 |
| **Cross Features** | User-author affinity × content type, topic match score | ~500 |
| **Embeddings** | User embedding, post embedding, dot product | ~150 |

---

## Meaningful Social Interactions (MSI)

### The 2018 Shift

**Before MSI (2013-2017)**:
- Optimized for time spent on platform
- Viral content and videos dominated
- Passive consumption prioritized

**After MSI (2018+)**:
- Prioritize posts that generate conversations
- Comments weighted 5x more than likes
- Posts from close friends > pages
- Reduce viral but low-quality content

### Implementation

```python
class MSIScorer:
    """
    Score posts for Meaningful Social Interactions potential.
    """
    def compute_msi_score(self, post: Post, user: UserProfile) -> float:
        # Connection strength (close friends > acquaintances > pages)
        connection_score = self.get_connection_strength(user, post.author_id)

        # Conversation potential
        comment_likelihood = self.predict_comment(user, post)

        # Reply chain depth (back-and-forth conversations)
        if post.comment_count > 0:
            avg_reply_depth = self.get_avg_reply_depth(post)
            conversation_quality = min(avg_reply_depth / 3, 1.0)
        else:
            conversation_quality = 0

        # MSI score
        msi = (
            0.4 * connection_score +
            0.3 * comment_likelihood +
            0.3 * conversation_quality
        )

        return msi
```

---

## Integrity Integration

### Content Classifiers

**Course Connection**: Week 12 (Bias/Fairness)

```python
class IntegritySystem:
    """
    Multi-classifier system for content integrity.
    """
    def __init__(self):
        self.spam_classifier = load_model("spam_v3")
        self.misinfo_classifier = load_model("misinformation_v2")
        self.clickbait_classifier = load_model("clickbait_v1")
        self.violence_classifier = load_model("violence_v1")
        self.hate_speech_classifier = load_model("hate_speech_v2")

    def score_post(self, post: Post) -> Dict[str, float]:
        # Extract features
        text_embedding = self.text_encoder(post.text)
        image_embedding = self.image_encoder(post.media_urls) if post.media_urls else None

        # Run classifiers
        scores = {
            'spam': self.spam_classifier(text_embedding),
            'misinformation': self.misinfo_classifier(text_embedding),
            'clickbait': self.clickbait_classifier(post.text, post.link_url),
            'violence': self.violence_classifier(image_embedding) if image_embedding else 0,
            'hate_speech': self.hate_speech_classifier(text_embedding)
        }

        return scores

    def should_demote(self, scores: Dict[str, float]) -> bool:
        """
        Determine if post should be demoted (reduced distribution).
        """
        return (
            scores['spam'] > 0.8 or
            scores['misinformation'] > 0.7 or
            scores['clickbait'] > 0.9 or
            scores['violence'] > 0.5 or
            scores['hate_speech'] > 0.6
        )

    def should_remove(self, scores: Dict[str, float]) -> bool:
        """
        Determine if post should be removed entirely.
        """
        return (
            scores['spam'] > 0.95 or
            scores['violence'] > 0.9 or
            scores['hate_speech'] > 0.9
        )
```

### Misinformation Handling

```
Misinformation pipeline:
1. ML classifier flags potential misinformation (score > 0.5)
2. High-velocity posts sent to third-party fact-checkers
3. If rated false:
   - Add "Missing Context" or "False Information" label
   - Reduce distribution by 80%
   - Show warning interstitial before sharing
4. Repeat offenders:
   - Page/profile flagged
   - All future posts start with penalty
```

---

## Diversity & Re-Ranking

### Source Diversity

```python
def apply_diversity(ranked_posts: List[Post], k: int = 50) -> List[Post]:
    """
    Ensure diversity in final feed.
    """
    final_feed = []
    author_counts = defaultdict(int)
    content_type_counts = defaultdict(int)

    for post in ranked_posts:
        # Max 3 posts from same author
        if author_counts[post.author_id] >= 3:
            continue

        # Max 5 consecutive posts of same type
        if len(final_feed) >= 5:
            recent_types = [p.content_type for p in final_feed[-5:]]
            if all(t == post.content_type for t in recent_types):
                continue

        final_feed.append(post)
        author_counts[post.author_id] += 1
        content_type_counts[post.content_type] += 1

        if len(final_feed) >= k:
            break

    return final_feed
```

### Feed Calibration

```python
def calibrate_feed(feed: List[Post], user: UserProfile) -> List[Post]:
    """
    Ensure feed distribution matches user's historical preferences.

    If user historically engages:
    - 40% friends, 30% pages, 30% groups
    Feed should approximate this distribution.
    """
    target_distribution = user.preferred_content_types

    # Adjust ranking to match distribution
    adjusted_feed = []
    source_quota = {
        'friend': int(len(feed) * 0.4),
        'page': int(len(feed) * 0.3),
        'group': int(len(feed) * 0.3)
    }

    for source_type, quota in source_quota.items():
        source_posts = [p for p in feed if get_source_type(p) == source_type]
        adjusted_feed.extend(source_posts[:quota])

    # Interleave by original score
    return interleave_by_score(adjusted_feed)
```

---

## Training Pipeline

### Data Collection

**Course Connection**: Week 11 (Evaluation)

```python
class FeedTrainingData:
    def generate_training_data(self, impressions: List[Impression]) -> Dataset:
        """
        Generate training data from feed impressions.

        Label hierarchy:
        - Share > Comment > Like > Click > View
        """
        examples = []

        for impression in impressions:
            # Features
            user_features = get_user_features(impression.user_id)
            post_features = get_post_features(impression.post_id)
            context = get_context(impression)

            # Multi-task labels
            labels = {
                'click': impression.clicked,
                'like': impression.liked,
                'comment': impression.commented,
                'share': impression.shared,
                'hide': impression.hidden,
                'watch_time': impression.video_watch_time or 0
            }

            # Position bias correction
            position_weight = 1.0 / self.position_ctr[impression.position]

            examples.append({
                'features': (user_features, post_features, context),
                'labels': labels,
                'weight': position_weight
            })

        return examples
```

### Training Infrastructure

```
Training scale:
- Training data: 100B+ impressions/day
- Model size: 10B+ parameters (distributed)
- Training hardware: 1000+ GPUs
- Training time: Continuous (incremental updates)

Update frequency:
- Full model retrain: Weekly
- Incremental updates: Hourly
- Real-time features: Streaming (Flink)
```

---

## Serving Infrastructure

### Architecture

**Course Connection**: Week 13 (Production Systems)

```
┌─────────────────────────────────────────────────────────────┐
│                    REAL-TIME LAYER                           │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Feed Service │  │ Ranking      │  │ Feature          │   │
│  │ (Thrift)     │  │ Service      │  │ Store (TAO)      │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Integrity    │  │ Embedding    │  │ Social Graph     │   │
│  │ Service      │  │ Service      │  │ (TAO)            │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    BATCH LAYER                               │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Model        │  │ User         │  │ Post             │   │
│  │ Training     │  │ Embedding    │  │ Embedding        │   │
│  │ (PyTorch)    │  │ Generation   │  │ Generation       │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ Integrity    │  │ Feature      │                         │
│  │ Model Train  │  │ Engineering  │                         │
│  └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### Latency Optimization

```python
class FeedServiceOptimizations:
    """
    Key optimizations for sub-200ms latency.
    """

    # 1. Parallel retrieval
    async def get_candidates_parallel(self, user_id: str):
        """Fetch from all sources concurrently."""
        results = await asyncio.gather(
            self.get_friend_posts(user_id),
            self.get_page_posts(user_id),
            self.get_group_posts(user_id),
            self.get_suggested_posts(user_id)
        )
        return flatten(results)

    # 2. Tiered caching
    cache_layers = {
        'user_features': ('redis', 300),      # 5 min TTL
        'post_features': ('redis', 60),       # 1 min TTL
        'affinity_scores': ('memcached', 3600), # 1 hour TTL
        'embeddings': ('local', 600)          # 10 min local cache
    }

    # 3. Model quantization
    def quantize_model(self, model):
        """INT8 quantization for 2-4x speedup."""
        return torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

    # 4. Batch inference
    def batch_score(self, user: UserProfile, posts: List[Post], batch_size=64):
        """Score posts in batches for GPU efficiency."""
        scores = []
        for i in range(0, len(posts), batch_size):
            batch = posts[i:i+batch_size]
            batch_scores = self.model.forward_batch(user, batch)
            scores.extend(batch_scores)
        return scores
```

---

## Metrics & Evaluation

### Online Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Engagement Rate** | (Likes + Comments + Shares) / Impressions | > 5% |
| **MSI Rate** | Comments with replies / Total comments | > 20% |
| **Time Spent** | Average session duration | Maintain |
| **DAU** | Daily active users | Growth |
| **7-Day Retention** | Users returning in 7 days | > 70% |

### Guardrail Metrics

```python
class FeedGuardrails:
    """
    Metrics that must not degrade in A/B tests.
    """

    # Content quality
    def misinformation_rate(self, impressions) -> float:
        """% of impressions from flagged misinformation."""
        flagged = sum(1 for i in impressions if i.post.misinformation_score > 0.7)
        return flagged / len(impressions)

    # Diversity
    def author_concentration(self, feed) -> float:
        """Gini coefficient of author distribution."""
        author_counts = Counter(p.author_id for p in feed)
        return gini_coefficient(list(author_counts.values()))

    # User wellbeing
    def negative_action_rate(self, impressions) -> float:
        """Hide + Report rate."""
        negative = sum(1 for i in impressions if i.hidden or i.reported)
        return negative / len(impressions)
```

### A/B Testing Scale

```
Facebook A/B testing:
- Concurrent experiments: 10,000+
- Users per experiment: Millions
- Typical duration: 1-2 weeks
- Statistical significance: p < 0.01

Long-term holdout groups:
- 1% of users held out from all experiments
- Used to measure cumulative experiment effects
- Prevents metric drift from many small experiments
```

---

## Trade-offs & Deep Dives

### Key Design Decisions

#### 1. Multi-Stage Ranking

**Decision**: 4-stage pipeline (Inventory → First-Pass → Heavy Ranking → Diversity)

**Reasoning**:
- Can't run heavy model on 10K posts (latency)
- First-pass filters 80% with 10% of compute
- Heavy model only runs on top 500

**Trade-off**: May miss good posts filtered early

#### 2. MSI Optimization

**Decision**: Weight comments/shares 7x more than likes

**Reasoning**:
- Active engagement correlates with user satisfaction
- Passive consumption can be addictive but unsatisfying
- Reduces viral but low-quality content

**Trade-off**: Reduced time-on-site initially (recovered after users adapted)

#### 3. Integrity Integration in Ranking

**Decision**: Integrity scores multiply final ranking score (not separate filter)

**Reasoning**:
- Borderline content should be demoted, not removed
- Binary filtering misses nuance
- Allows graceful degradation

**Trade-off**: Some problematic content still shown (lower in feed)

### Common Interview Follow-ups

**Q: How do you handle new users (cold start)?**
> Initial feed based on demographics and popular content. First 5-10 interactions rapidly update user profile. We also use "exploration" slots (10% of feed) to learn preferences faster.

**Q: How do you balance friends vs pages?**
> MSI naturally boosts friends (higher comment likelihood). But we also have explicit calibration ensuring minimum 30% friends content. Page posts need higher engagement to rank equally.

**Q: How do you prevent echo chambers?**
> Three mechanisms: (1) Integrity classifiers reduce hyperpartisan content, (2) "Related Articles" shows alternative perspectives, (3) Exploration slots expose users to diverse viewpoints. We measure viewpoint diversity as a guardrail.

**Q: How do you handle real-time events (elections, disasters)?**
> Special "crisis response" mode: boosted authoritative sources, reduced viral unverified content, banner with official information. ML models retrained to recognize crisis-related content.

---

## Course Concepts Applied

| Concept | Week | Application in Facebook News Feed |
|---------|------|-----------------------------------|
| **Collaborative Filtering** | 2-3 | User-user affinity scoring, "users like you" signals |
| **Matrix Factorization** | 3 | User and post embeddings |
| **Content-Based** | 4 | Post content features, topic matching |
| **Neural CF** | 5 | Deep ranking model with embeddings |
| **Sequential Models** | 6 | Session-based recent engagement patterns |
| **Graph Neural Networks** | 7 | Social graph traversal, affinity propagation |
| **Two-Tower** | 8 | User tower + Post tower for candidate retrieval |
| **Multi-Task Learning** | 8 | Joint click/like/comment/share prediction |
| **Embeddings** | 9 | User interest embeddings, post content embeddings |
| **Contextual Bandits** | 10 | Exploration for new content types |
| **Evaluation** | 11 | A/B testing, guardrail metrics, long-term holdouts |
| **Bias/Fairness** | 12 | Integrity classifiers, echo chamber prevention |
| **Production Systems** | 13 | Multi-stage pipeline, caching, latency optimization |

---

## Summary

**Facebook News Feed** is one of the most sophisticated recommendation systems:

1. **Scale**: 2B DAU, 700K QPS peak, 200ms latency budget
2. **Multi-stage pipeline**: Inventory → First-Pass → Heavy Ranking → Diversity
3. **Multi-objective**: Click, like, comment, share, hide predictions
4. **MSI focus**: Comments/shares weighted 7x more than likes (2018 shift)
5. **Integrity integration**: ML classifiers for spam, misinformation, hate speech
6. **Diversity**: Source, content type, and author diversity constraints
7. **A/B testing**: 10,000+ concurrent experiments

**Architecture**:
```
10K posts → 2K (Inventory) → 500 (First-Pass) → 100 (Heavy Ranking) → 50 (Final Feed)
```

**Key innovations**:
- Meaningful Social Interactions (MSI) optimization
- Integrity-aware ranking (not separate filter)
- Feed calibration for user preferences
- Real-time feature streaming

---

## References

1. **Facebook Engineering Blog**. "How News Feed Works" (multiple posts, 2013-2023).
   - Official explanations of ranking factors

2. **Huang, J., et al. (2020)**. "Embedding-based Retrieval in Facebook Search". *KDD*.
   - Two-tower architecture at scale

3. **Zhao, X., et al. (2019)**. "Recommending What Video to Watch Next: A Multitask Ranking System". *RecSys*.
   - Multi-task learning approach (similar to Feed)

4. **Mosseri, A. (2018)**. "Bringing People Closer Together". *Facebook Newsroom*.
   - MSI announcement and rationale

5. **Silverman, C. (2016)**. "This Analysis Shows How Viral Fake Election News Stories Outperformed Real News On Facebook". *BuzzFeed News*.
   - Context for integrity integration

6. **Facebook Transparency Center**. "How Facebook's Feed Works".
   - Current ranking signals
