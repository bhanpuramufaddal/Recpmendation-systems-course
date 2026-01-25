# Week 1: Practice Problems

## Conceptual Questions

### Problem 1: Explicit vs. Implicit Feedback

**Question**:
A music streaming service collects the following data:
- User star ratings (1-5 stars) for songs
- Number of times each song was played
- Whether user skipped song within first 30 seconds
- Whether user added song to playlist

(a) Classify each signal as explicit or implicit feedback.
(b) What are the advantages and disadvantages of each signal?
(c) Which signals would you prioritize for building a recommendation system? Why?

<details>
<summary>Solution</summary>

**(a) Classification**:
- **Star ratings**: Explicit (direct user preference statement)
- **Play count**: Implicit (inferred engagement)
- **Skip behavior**: Implicit (negative signal)
- **Playlist additions**: Implicit (but strong positive signal)

**(b) Advantages and Disadvantages**:

| Signal | Advantages | Disadvantages |
|--------|-----------|---------------|
| Star ratings | Clear intent, unambiguous | Sparse (few users rate), biased (extreme ratings) |
| Play count | Abundant, reflects actual behavior | Noisy (accidental plays, background music) |
| Skip behavior | Clear negative signal | May skip great songs in wrong context |
| Playlist additions | Strong engagement indicator | Less frequent than plays |

**(c) Prioritization**:

**Recommended priority**:
1. **Playlist additions**: Strongest signal (deliberate action)
2. **Play count + skip behavior**: Combined for engagement score
3. **Star ratings**: Supplementary (sparse but valuable)

**Reasoning**:
- Implicit signals are abundant (more training data)
- Combination of multiple implicit signals reduces noise
- Star ratings valuable but too sparse to be primary signal
- Modern approach: Train on all signals with different weights

**Example weighting**:
```
Engagement_Score = 1.0 × playlist_add
                  + 0.5 × (plays / max_plays)
                  - 0.3 × skip_rate
                  + 0.2 × (star_rating / 5) if rated
```

</details>

---

### Problem 2: Prediction vs. Ranking

**Question**:
Explain why modern recommendation systems (Netflix, YouTube, Spotify) focus on **ranking** rather than **rating prediction**, even though early research (Netflix Prize) focused on predicting ratings.

<details>
<summary>Solution</summary>

**Key Reasons**:

**1. Business Goal Alignment**:
- **Rating prediction**: "You'd rate this 4.2 stars"
- **Ranking**: "Here are your top 10 movies"
- Users care about **discovering great content**, not predicted ratings

**2. Metric Mismatch**:
- Netflix Prize winner: 10% RMSE improvement
- Never deployed because it didn't improve user engagement
- **RMSE** optimizes for accurate rating prediction
- **Business** needs engagement, watch time, retention

**3. Implicit Feedback Dominance**:
- Modern systems use clicks, views, watch time (no ratings)
- Can't predict star ratings if users don't provide them
- Ranking works naturally with implicit feedback

**4. Evaluation**:
- **Rating prediction**: Measured by RMSE, MAE
- **Ranking**: Measured by Precision@K, NDCG, MAP
- Ranking metrics better correlate with user satisfaction

**5. Presentation**:
- UIs show ordered lists (top picks, trending)
- Actual rating numbers often hidden or de-emphasized
- Users interact with rankings, not ratings

**Example**:
- Netflix removed star ratings in 2017
- Switched to binary thumbs up/down
- UI shows percentage match (ranking score), not predicted rating
- Focus shifted to "Top Picks for You" (ranked list)

**Conclusion**:
Ranking directly optimizes what users see and interact with, while rating prediction is an intermediate step that may not correlate with actual user satisfaction.

</details>

---

### Problem 3: Sparsity Calculation

**Question**:
Amazon has approximately:
- 300 million active users
- 350 million products
- 2 billion purchase interactions

(a) Calculate the sparsity of the user-item purchase matrix.
(b) If you use only users who purchased at least 5 items and items purchased by at least 10 users, estimate the new sparsity.
(c) What are the implications of this sparsity for collaborative filtering algorithms?

<details>
<summary>Solution</summary>

**(a) Original Sparsity**:

$$\text{Total possible interactions} = 300M \times 350M = 105 \times 10^{15}$$

$$\text{Actual interactions} = 2 \times 10^9$$

$$\text{Density} = \frac{2 \times 10^9}{105 \times 10^{15}} = 1.9 \times 10^{-5}$$

(approximately 0.0000019%)

$$\text{Sparsity} = 99.9999981$$

(99.9999981%)

**Interpretation**: Nearly 100% of the matrix is empty.

**(b) Filtered Sparsity**:

**Assumptions** (for estimation):
- ~50% of users have <5 purchases (power law distribution)
- ~80% of products have <10 purchases (long tail)

**Filtered matrix**:
- Users: 150M (50% of 300M)
- Items: 70M (20% of 350M)
- Interactions: ~1.8B (assuming we lose some)

$$\text{Total possible} = 150M \times 70M = 10.5 \times 10^{15}$$

$$\text{Density} = \frac{1.8 \times 10^9}{10.5 \times 10^{15}} = 1.7 \times 10^{-4}$$

$$\text{Sparsity} \approx 99.99983$$

(approximately 99.99983%)

**Result**: Still extremely sparse (improved by ~10x density, but still >99.99%)

**(c) Implications for Collaborative Filtering**:

**User-Based CF**:
- **Problem**: Finding users with overlapping purchases is rare
- **Example**: Two users each bought 10 items out of 350M → probability of overlap is tiny
- **Solution**: Matrix factorization to learn latent patterns

**Item-Based CF**:
- **Problem**: Most item pairs never co-purchased
- **Better than user-based**: Items have more stable patterns
- **Solution**: Focus on popular items, precompute similarities

**General Implications**:
1. **Memory-based CF fails** at this scale
2. **Matrix factorization essential**: Learns patterns from sparse data
3. **Implicit feedback critical**: Augment purchases with views, clicks
4. **Content-based features help**: Metadata compensates for lack of interactions
5. **Cold start severe**: Most users/items have very few interactions

**Modern Approach**:
- Two-tower neural networks
- Learn embeddings from all available signals (clicks, views, searches, purchases)
- Hybrid models combining CF and content features

</details>

---

## Algorithmic Problems

### Problem 4: User-Based Collaborative Filtering

**Question**:
Given the following rating matrix (1-5 scale, ? = unknown):

```
        Movie1  Movie2  Movie3  Movie4
Alice     5       3       ?       1
Bob       4       ?       ?       2
Carol     1       2       5       ?
Dave      ?       3       4       4
```

(a) Compute the Pearson correlation between Alice and each other user.
(b) Using k=2 nearest neighbors, predict Alice's rating for Movie3.
(c) What are the limitations of this approach?

<details>
<summary>Solution</summary>

**(a) Pearson Correlation**:

$$\text{Pearson}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}$$

where $I_{uv}$ = items rated by both users.

**Alice vs. Bob**:
- Co-rated items: Movie1, Movie4
- Alice: [5, 1], mean = 3
- Bob: [4, 2], mean = 3
- Numerator: (5-3)(4-3) + (1-3)(2-3) = 1×1 + (-2)×(-1) = 1 + 2 = 3
- Denom: $\sqrt{(2^2 + (-2)^2)} \times \sqrt{(1^2 + (-1)^2)} = \sqrt{8} \times \sqrt{2} = 4$
- **Pearson(Alice, Bob) = 3/4 = 0.75**

**Alice vs. Carol**:
- Co-rated items: Movie1, Movie2, Movie4
- Alice: [5, 3, 1], mean = 3
- Carol: [1, 2, ?] - Carol hasn't rated Movie4
- Co-rated: Movie1, Movie2
- Alice: [5, 3], mean = 4
- Carol: [1, 2], mean = 1.5
- Numerator: (5-4)(1-1.5) + (3-4)(2-1.5) = 1×(-0.5) + (-1)×0.5 = -0.5 - 0.5 = -1
- Denom: $\sqrt{1^2 + (-1)^2} \times \sqrt{0.5^2 + 0.5^2} = \sqrt{2} \times \sqrt{0.5} = 1$
- **Pearson(Alice, Carol) = -1.0** (perfect negative correlation)

**Alice vs. Dave**:
- Co-rated items: Movie2, Movie4
- Alice: [3, 1], mean = 2
- Dave: [3, 4], mean = 3.5
- Numerator: (3-2)(3-3.5) + (1-2)(4-3.5) = 1×(-0.5) + (-1)×0.5 = -0.5 - 0.5 = -1
- Denom: $\sqrt{1^2 + (-1)^2} \times \sqrt{0.5^2 + 0.5^2} = \sqrt{2} \times \sqrt{0.5} = 1$
- **Pearson(Alice, Dave) = -1.0**

**(b) Prediction for Movie3**:

**k=2 nearest neighbors**: Bob (0.75) and either Carol or Dave (-1.0)
Let's use Bob and Carol.

**Formula**:
$$\hat{r}_{Alice,Movie3} = \bar{r}_{Alice} + \frac{\sum_{v \in N} \text{sim}(Alice,v) \times (r_{v,Movie3} - \bar{r}_v)}{\sum_{v \in N} |\text{sim}(Alice,v)|}$$

- $\bar{r}_{Alice} = (5 + 3 + 1) / 3 = 3$
- Bob hasn't rated Movie3 → skip
- Carol rated Movie3 = 5, $\bar{r}_{Carol} = (1 + 2 + 5) / 3 = 2.67$
- Dave rated Movie3 = 4, $\bar{r}_{Dave} = (3 + 4 + 4) / 3 = 3.67$

Using Carol and Dave (both rated Movie3):
$$\hat{r} = 3 + \frac{(-1) \times (5 - 2.67) + (-1) \times (4 - 3.67)}{|-1| + |-1|}$$
$$= 3 + \frac{(-1) \times 2.33 + (-1) \times 0.33}{2}$$
$$= 3 + \frac{-2.33 - 0.33}{2} = 3 + \frac{-2.66}{2} = 3 - 1.33 = 1.67$$

**Predicted rating: 1.67** (Alice would probably not like Movie3)

**(c) Limitations**:

1. **Sparsity**: Very few co-rated items → unreliable correlations
2. **Scalability**: $O(|U|^2 \times |I|)$ for all user pairs
3. **Cold start**: Can't recommend for new users
4. **No learning**: Similarity recomputed from scratch
5. **Popularity bias**: Popular items dominate overlaps

</details>

---

### Problem 5: Candidate Generation Pipeline

**Question**:
Design a candidate generation system for a video platform with:
- 2 billion users
- 800 million videos
- Target: Return top 20 recommendations in <100ms

Your design should include:
(a) Number of stages and their purpose
(b) How many candidates pass through each stage
(c) Techniques used at each stage
(d) Latency budget for each stage

<details>
<summary>Solution</summary>

**Proposed Architecture: Three-Stage Pipeline**

---

**Stage 1: Candidate Generation (Multiple Sources)**

**Goal**: Reduce 800M videos → ~500 candidates

**Latency Budget**: 10ms

**Approach**: Multiple retrieval sources in parallel

**Source 1: Collaborative Filtering (200 candidates)**
- User embedding: $\mathbf{u} \in \mathbb{R}^{128}$
- Video embeddings: $\mathbf{v}_i \in \mathbb{R}^{128}$ (precomputed)
- **ANN search** (FAISS): Top-200 by $\mathbf{u}^T \mathbf{v}_i$
- Latency: ~5ms

**Source 2: Content-Based (200 candidates)**
- Recent watch history: Last 10 videos
- For each video, retrieve 20 similar videos (item-item similarity, precomputed)
- Union: ~200 unique candidates
- Latency: ~2ms (simple lookup)

**Source 3: Trending/Popular (50 candidates)**
- Region-specific trending videos (last 24 hours)
- Precomputed daily
- Latency: <1ms (cache lookup)

**Source 4: Subscriptions (50 candidates)**
- New uploads from subscribed channels
- Precomputed per user
- Latency: <1ms

**Total**: ~500 candidates, 10ms latency

---

**Stage 2: Ranking**

**Goal**: Score 500 candidates → Top 100

**Latency Budget**: 40ms

**Model**: Deep Neural Network (DNN)

**Features** (per candidate):
- **User features** (50 dims): Demographics, watch history embeddings, engagement patterns
- **Video features** (100 dims): Title/description embeddings, category, upload date, popularity
- **Contextual features** (20 dims): Time of day, device, location, session length
- **Interaction features** (30 dims): User-video similarity, channel subscription status

**Architecture**:
```
Concat(user, video, context, interaction) → 200 dims
   ↓
Dense(512) + ReLU
   ↓
Dense(256) + ReLU
   ↓
Dense(128) + ReLU
   ↓
Multi-Task Outputs:
  ├─> P(click)
  ├─> E[watch_time]
  └─> P(like)
```

**Scoring**:
$$\text{Score} = 0.3 \times P(\text{click}) + 0.6 \times E[\text{watch\_time}] + 0.1 \times P(\text{like})$$

**Inference**:
- Batch size: 500
- GPU inference: ~40ms for 500 videos
- Return top 100

---

**Stage 3: Re-ranking**

**Goal**: Optimize final 100 → Display 20

**Latency Budget**: 30ms

**Objectives**:
1. **Diversity**: Mix topics, channels, lengths
2. **Freshness**: Boost recent uploads
3. **Exploration**: Include 2 random videos (10%)

**Algorithm**: Maximal Marginal Relevance (MMR)

```python
selected = []
candidates = top_100_from_ranking

for i in range(20):
    scores = {}
    for video in candidates:
        relevance = ranking_score[video]

        # Diversity penalty
        if selected:
            max_similarity = max(similarity(video, s) for s in selected)
        else:
            max_similarity = 0

        # MMR score
        scores[video] = 0.7 * relevance - 0.3 * max_similarity

    # Add best video
    best = argmax(scores)
    selected.append(best)
    candidates.remove(best)

# Replace 2 videos with random exploration
selected[-2:] = random.sample(all_candidates, 2)
```

**Latency**: ~30ms (similarity lookups precomputed)

---

**Stage 4: Display**

**Goal**: Present to user with personalized thumbnails

**Latency Budget**: 20ms

**Thumbnail Personalization**:
- Contextual bandit: Choose best thumbnail per user
- A/B test different thumbnails, learn CTR
- Example: Action fan sees explosion scene, romance fan sees couple

**Total Latency**: 10 + 40 + 30 + 20 = **100ms** ✓

---

**Summary Table**:

| Stage | Input | Output | Technique | Latency |
|-------|-------|--------|-----------|---------|
| Candidate Gen | 800M videos | 500 candidates | ANN, content-based, trending | 10ms |
| Ranking | 500 candidates | Top 100 | DNN with multi-task learning | 40ms |
| Re-ranking | 100 candidates | Top 20 | MMR (diversity) | 30ms |
| Display | 20 videos | UI | Thumbnail personalization | 20ms |
| **Total** | | | | **100ms** |

---

**Key Design Decisions**:

1. **Parallelization**: Multiple candidate sources run concurrently
2. **Precomputation**: Item embeddings, similarities, trending lists
3. **Caching**: User embeddings, subscription lists
4. **GPU**: Batch inference for ranking
5. **Trade-offs**:
   - Candidate generation: High recall, low precision (broad net)
   - Ranking: High precision (expensive model on fewer items)
   - Re-ranking: Diversity, business goals

</details>

---

## System Design Problems

### Problem 6: Cold Start Strategy

**Question**:
You're launching a new book recommendation app. On day 1:
- 1000 books in catalog
- 0 user interaction data
- Users can rate books 1-5 stars

Design a 4-week strategy to address the cold start problem, including:
(a) Week 1 approach (no data)
(b) Week 2-3 approach (some data)
(c) Week 4+ approach (sufficient data)
(d) Specific techniques and metrics for each phase

<details>
<summary>Solution</summary>

**Phase 1: Week 1 (Zero Data)**

**Approach**: Non-personalized + Rapid Data Collection

**Strategies**:

**1. Onboarding Flow**:
```
Step 1: Ask user to select 3 favorite genres
Step 2: Show 10 popular books per genre
Step 3: Ask user to rate 5 books (any rating)
Reward: Unlock personalized recommendations
```

**Benefit**: Collect ~5 ratings per user immediately

**2. Initial Recommendations**:
- **Popular books**: Bestsellers, high average ratings (from external sources like Goodreads)
- **Category-based**: Show popular books in selected genres
- **Editorial curation**: Staff picks, "Books everyone should read"

**3. Data Collection Incentives**:
- "Rate 10 books to unlock feature X"
- Gamification: Badges for ratings
- Social: "See what your friends are reading" (requires ratings)

**Metrics**:
- % users completing onboarding
- Average ratings per user
- User retention day 1 → day 7

---

**Phase 2: Weeks 2-3 (Initial Data)**

**Approach**: Hybrid (Content-Based + Simple CF)

**Strategies**:

**1. Content-Based Recommendations**:
- Extract book features: Genre, author, year, page count, keywords from description
- TF-IDF on book descriptions
- For user who rated book $i$ highly, recommend books with high cosine similarity

**Formula**:
$$\text{score}(i, j) = \cos(\mathbf{book}_i, \mathbf{book}_j) = \frac{\mathbf{book}_i \cdot \mathbf{book}_j}{||\mathbf{book}_i|| \cdot ||\mathbf{book}_j||}$$

**2. Simple Collaborative Filtering**:
- Item-based CF: "Users who liked X also liked Y"
- Only for books with ≥10 ratings
- Cosine similarity on rating vectors

**3. Hybrid Approach**:
$$\text{score}_{hybrid} = \alpha \cdot \text{score}_{CF} + (1 - \alpha) \cdot \text{score}_{content}$$

- Books with sufficient ratings: $\alpha = 0.7$
- Books with few ratings (cold items): $\alpha = 0.3$

**4. Exploration**:
- 20% of recommendations: Random popular books
- Learn user preferences beyond initial genres

**Metrics**:
- Coverage: % of books recommended at least once
- Rating collection rate
- User engagement (ratings, browsing time)

---

**Phase 3: Week 4+ (Sufficient Data)**

**Approach**: Full Collaborative Filtering + Personalization

**Strategies**:

**1. Matrix Factorization**:
- Train on all ratings collected
- SVD or ALS with $k = 50$ latent factors
- Regularization: $\lambda = 0.01$

**Model**:
$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i$$

**2. Neural Collaborative Filtering (if sufficient data)**:
- User embedding + Book embedding
- MLP for non-linear interactions
- Train on implicit feedback (views, clicks) + explicit (ratings)

**3. Personalized Ranking**:
- Move from rating prediction to ranking
- BPR (Bayesian Personalized Ranking) for implicit feedback
- Optimize for correct pairwise ordering

**4. Continuous Cold Start Handling**:
- **New users**: Onboarding + popular books + exploration
- **New books**: Content-based until sufficient ratings
- **Hybrid**: Always maintain content-based component

**5. Diversity and Exploration**:
- Not just similar books to past likes
- Introduce users to new genres (serendipity)
- Multi-armed bandit for exploration (ε-greedy or Thompson sampling)

**Metrics**:
- Engagement: Daily active users, session length
- Rating prediction accuracy: RMSE on held-out set
- Business: User retention, books read per user
- Diversity: Average genre diversity in recommendations

---

**Summary Table**:

| Phase | Data Availability | Primary Approach | Secondary Approach | Exploration |
|-------|-------------------|------------------|-------------------|-------------|
| Week 1 | None | Popular books, editorial | Genre-based | N/A |
| Weeks 2-3 | Sparse (~10-50 ratings per book) | Content-based | Item-based CF | 20% random |
| Week 4+ | Sufficient (100+ ratings per book) | Matrix factorization | Content-based (cold items) | 10% bandit |

---

**Long-Term Enhancements**:

**1. Author-Based**:
- "More by this author"
- Author similarity (writing style, genre)

**2. Social**:
- Friends' recommendations
- Reading clubs and shared lists

**3. Contextual**:
- Time-aware: Beach reads in summer, cozy mysteries in winter
- Mood-based: "Looking for something light?" vs. "Want a challenge?"

**4. Explanations**:
- "Because you loved Harry Potter"
- "Popular in Science Fiction"
- Build user trust in recommendations

</details>

---

## Reflection Questions

### Problem 7: Ethical Considerations

**Question**:
A social media platform uses a recommendation algorithm to maximize user engagement (time spent on platform). This leads to:
- Increased ad revenue
- High user addiction rates
- Amplification of divisive content (generates strong reactions)
- Mental health concerns among teenagers

(a) What are the ethical issues with optimizing purely for engagement?
(b) Propose alternative objective functions that balance business goals with user well-being.
(c) How would you measure the success of these alternative objectives?

<details>
<summary>Solution</summary>

**(a) Ethical Issues**:

**1. Exploitation of Psychological Vulnerabilities**:
- Engagement optimization exploits dopamine loops
- Infinite scroll and autoplay prevent natural stopping points
- Particularly harmful for adolescents with developing brains

**2. Amplification of Harmful Content**:
- Divisive, outrageous content generates more engagement (clicks, comments, shares)
- Algorithm learns to promote conflict and polarization
- Echo chambers and radicalization pipelines

**3. Misalignment of Incentives**:
- Platform profits from addiction
- User's short-term engagement ≠ long-term well-being
- "Time well spent" vs. "time spent"

**4. Societal Harm**:
- Misinformation spreads faster than truth (engagement-driven)
- Political polarization increased
- Mental health crisis (comparison, FOMO, cyberbullying)

**5. Lack of Transparency**:
- Users unaware they're being manipulated
- No control over recommendation algorithm
- Informed consent not possible

---

**(b) Alternative Objective Functions**:

**1. Time Well Spent**:
$$\text{Objective} = \alpha \cdot \text{Engagement} + \beta \cdot \text{Satisfaction} - \gamma \cdot \text{Regret}$$

- **Engagement**: Time on platform
- **Satisfaction**: User survey: "Was this time valuable?"
- **Regret**: User survey: "Do you regret spending time here?"

**2. Multi-Stakeholder Optimization**:
$$\text{Objective} = w_1 \cdot \text{Revenue} + w_2 \cdot \text{User\_Well-being} + w_3 \cdot \text{Societal\_Health}$$

**Metrics**:
- **Revenue**: Ad clicks, conversions
- **User Well-being**: Session satisfaction, mental health surveys, usage patterns
- **Societal Health**: Reduce misinformation spread, promote diverse viewpoints

**3. Long-Term Retention Over Short-Term Engagement**:
$$\text{Objective} = \sum_{t=1}^{T} \gamma^t \cdot \text{Engagement}_t$$

- Discount factor $\gamma < 1$ prioritizes sustainable engagement
- Prevent burnout and addiction

**4. Meaningful Interactions**:
$$\text{Objective} = w_{comment} \cdot \text{Comments} + w_{share} \cdot \text{Shares} - w_{passive} \cdot \text{Passive\_Scrolling}$$

- Encourage active participation over passive consumption
- Weight quality interactions (thoughtful comments) over quantity

**5. Content Diversity**:
$$\text{Objective} = \text{Engagement} - \lambda \cdot \text{Homogeneity}$$

- **Homogeneity**: Similarity of recommended content
- Force diversity to prevent filter bubbles

---

**(c) Measuring Success**:

**Short-Term Metrics (A/B Testing)**:

| Metric | Description | Target |
|--------|-------------|--------|
| Session Satisfaction | "How do you feel after this session?" (1-5) | >4.0 |
| Time Well Spent | "Was this time valuable?" (%) | >70% |
| Regret | "Do you regret time spent?" (%) | <10% |
| Diversity Score | Avg cosine distance between shown items | >0.6 |

**Medium-Term Metrics (Weeks)**:

| Metric | Description | Target |
|--------|-------------|--------|
| Retention Rate | % users active after 30 days | >60% |
| Healthy Usage Patterns | Sessions per day, avg session length | 3 sessions, 20min each |
| Content Diversity | Unique content types engaged with | >5 types/week |

**Long-Term Metrics (Months)**:

| Metric | Description | Target |
|--------|-------------|--------|
| User Well-Being Index | Composite: satisfaction, mental health, sleep | Positive trend |
| Misinformation Exposure | % of viewed content flagged as false | <1% |
| Societal Impact | Public surveys, academic research | Positive perception |

**Business Metrics (ensuring sustainability)**:

| Metric | Description | Acceptable Trade-off |
|--------|-------------|----------------------|
| Revenue | Ad revenue, subscriptions | -10% acceptable if retention improves |
| DAU (Daily Active Users) | Users per day | Stable or slight decrease OK |
| Lifetime Value (LTV) | Long-term user value | Should increase |

---

**Implementation Strategy**:

**1. Gradual Rollout**:
- A/B test new objectives on 5% of users
- Monitor both old and new metrics
- Expand if well-being improves without severe business impact

**2. User Control**:
- "Take a break" reminders after X minutes
- Daily time limits (optional)
- "Show me diverse content" toggle
- Transparency: "Why am I seeing this?"

**3. Content Policy**:
- Demote divisive/misinformation content (regardless of engagement)
- Promote authoritative sources for news
- Limit reach of borderline content

**4. Research Collaboration**:
- Partner with mental health researchers
- Publish transparency reports
- External audits of algorithm impact

---

**Conclusion**:

**Ethical recommendation systems** require:
1. **Multi-objective optimization**: Balance business, user, and societal goals
2. **Long-term thinking**: Retention and well-being over immediate engagement
3. **Transparency and control**: Users understand and can influence recommendations
4. **Accountability**: Measure and report societal impact
5. **Iterative improvement**: Continuously refine based on research and feedback

**Trade-off**: Short-term revenue decrease acceptable for long-term sustainability and social responsibility.

</details>

---

## Advanced Problem

### Problem 8: Exploration-Exploitation with Bandits

**Question**:
You're building a news recommendation system. Each article has an unknown click-through rate (CTR). You want to maximize total clicks over 1000 user visits.

Given 3 articles with true CTRs (unknown to algorithm):
- Article A: CTR = 0.1
- Article B: CTR = 0.15
- Article C: CTR = 0.05

(a) Simulate ε-greedy with ε=0.1 for 1000 rounds. Report total clicks and regret.
(b) Explain why pure exploitation (ε=0) would fail.
(c) Propose a better exploration strategy than fixed ε-greedy.

<details>
<summary>Solution</summary>

**(a) ε-Greedy Simulation**:

**Algorithm**:
```python
import numpy as np

np.random.seed(42)
true_ctrs = {'A': 0.1, 'B': 0.15, 'C': 0.05}
articles = ['A', 'B', 'C']

# Initialize
clicks = {a: 0 for a in articles}
impressions = {a: 0 for a in articles}
total_clicks = 0
epsilon = 0.1
rounds = 1000

for t in range(rounds):
    # ε-greedy policy
    if np.random.random() < epsilon or t < 3:  # Explore or initial phase
        article = np.random.choice(articles)
    else:  # Exploit
        ctrs = {a: clicks[a] / impressions[a] if impressions[a] > 0 else 0
                for a in articles}
        article = max(ctrs, key=ctrs.get)

    # Simulate user click
    clicked = np.random.random() < true_ctrs[article]

    # Update
    impressions[article] += 1
    if clicked:
        clicks[article] += 1
        total_clicks += 1

print(f"Total clicks: {total_clicks}")
print(f"Impressions per article: {impressions}")
print(f"Observed CTRs: {[clicks[a]/impressions[a] for a in articles]}")
```

**Expected Results** (will vary due to randomness):
- Total clicks: ~140-145
- Article B (best) will be shown ~850-900 times
- Regret ≈ 1000 × 0.15 - 140 = 150 - 140 = ~10 clicks

**Regret Calculation**:
$$\text{Regret} = T \cdot \mu^* - \sum_{t=1}^T r_t$$
where $\mu^* = 0.15$ (best article's CTR), $T = 1000$

Optimal: 1000 × 0.15 = 150 clicks
Achieved: ~140 clicks
**Regret**: ~10 clicks

**(b) Why Pure Exploitation (ε=0) Fails**:

**Scenario**:
1. Round 1: Random choice (say Article C, no click)
2. Round 2: Random choice (say Article A, click!)
3. Round 3: Exploit → Article A (best so far)
4. Rounds 4-1000: Always Article A

**Result**:
- Article A shown 998 times → ~100 clicks
- Article B shown 0 times → 0 clicks
- Article C shown 1-2 times → 0 clicks
- **Total**: ~100 clicks

**Optimal**: 1000 × 0.15 = 150 clicks with Article B
**Regret**: 150 - 100 = **50 clicks lost**

**Problem**: Algorithm never discovers Article B is better because it commits to A after early luck.

**(c) Better Exploration Strategies**:

**1. UCB (Upper Confidence Bound)**

**Idea**: Balance estimated reward with uncertainty

$$\text{UCB}(a) = \hat{\mu}_a + \sqrt{\frac{2 \ln t}{n_a}}$$

- $\hat{\mu}_a$ = estimated CTR of article $a$
- $n_a$ = times article $a$ shown
- $t$ = total rounds

**Algorithm**:
```python
for t in range(1, rounds + 1):
    if t <= len(articles):  # Initial exploration
        article = articles[t-1]
    else:
        ucb_scores = {}
        for a in articles:
            mean = clicks[a] / impressions[a]
            confidence = np.sqrt(2 * np.log(t) / impressions[a])
            ucb_scores[a] = mean + confidence
        article = max(ucb_scores, key=ucb_scores.get)

    # ... (same click simulation and update)
```

**Expected Performance**:
- Total clicks: ~145-148 (better than ε-greedy)
- Regret: ~5-7 clicks
- Automatically balances exploration and exploitation

---

**2. Thompson Sampling**

**Idea**: Bayesian approach, sample from posterior

**Model**: Beta distribution for click probability
- Prior: Beta(α=1, β=1) (uniform)
- Update: Success → α+1, Failure → β+1

**Algorithm**:
```python
alpha = {a: 1 for a in articles}
beta = {a: 1 for a in articles}

for t in range(rounds):
    # Sample CTR from posterior for each article
    sampled_ctrs = {a: np.random.beta(alpha[a], beta[a]) for a in articles}

    # Choose article with highest sample
    article = max(sampled_ctrs, key=sampled_ctrs.get)

    # Simulate click
    clicked = np.random.random() < true_ctrs[article]

    # Update posterior
    if clicked:
        alpha[article] += 1
        total_clicks += 1
    else:
        beta[article] += 1
```

**Expected Performance**:
- Total clicks: ~146-149 (best!)
- Regret: ~3-5 clicks
- Naturally explores less as confidence grows

---

**3. Decreasing ε-Greedy**

**Idea**: Explore more early, exploit more later

$$\epsilon(t) = \min\left(1, \frac{c}{t}\right)$$

where $c$ is a tunable constant (e.g., $c = 100$).

**Benefits**:
- Early: High exploration (ε≈1)
- Late: Low exploration (ε→0)
- Better than fixed ε

---

**Comparison Table**:

| Strategy | Expected Clicks | Regret | Pros | Cons |
|----------|----------------|--------|------|------|
| ε-greedy (ε=0.1) | ~142 | ~8 | Simple | Wastes exploration on bad arms |
| Pure Exploitation | ~100 | ~50 | Fast (no exploration) | Misses better options |
| UCB | ~147 | ~3-5 | Principled, confidence-based | More complex |
| Thompson Sampling | ~148 | ~2-4 | **Best empirical performance** | Requires Bayesian updates |
| Decreasing ε | ~145 | ~5-7 | Adaptive | Tuning required |

**Recommendation**: **Thompson Sampling** for best performance in practice.

---

**Real-World Considerations**:

1. **Contextual Features**: User demographics, time, location → Contextual bandits (LinUCB)
2. **Non-Stationary**: CTRs change over time → Use sliding window or discounted updates
3. **Delayed Feedback**: Clicks may arrive later → Batch updates
4. **Multiple Objectives**: CTR + engagement time + revenue → Multi-objective bandits

</details>

---

## Summary

This problem set covers:
- **Conceptual understanding**: Feedback types, prediction vs. ranking, sparsity
- **Algorithmic skills**: CF computations, pipeline design
- **System design**: Cold start strategies, exploration-exploitation
- **Ethical reasoning**: Well-being vs. engagement trade-offs

**Next**: Proceed to Week 2 for collaborative filtering algorithms in depth.
