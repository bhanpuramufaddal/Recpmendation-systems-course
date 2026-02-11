# Week 1: Key Challenges in Recommendation Systems

## Learning Objectives

- Understand the fundamental challenges in building recommendation systems
- Recognize trade-offs and their implications
- Learn strategies to address each challenge

---

## Opening: "These Are the Problems That Will Keep You Up at Night"

*[Professor walks to whiteboard, draws a circle]*

"Before we dive into algorithms and equations, I want you to understand something fundamental: **recommendation systems fail in spectacular ways**, and today we're going to explore why.

Let me paint a picture for you:

- **2016**: YouTube's recommendation algorithm leads users from mainstream political videos to extremist content in just 3-5 clicks. Congressional hearings follow.
- **2018**: Amazon's recruiting AI, trained on historical hiring data, learns to systematically downrank women applicants.
- **2019**: A user watches ONE baby video on YouTube. For the next 6 months, their homepage is flooded with parenting content.
- **2021**: Spotify's 'Discover Weekly' accidentally recommends the same 30 songs to millions of users due to a feedback loop bug.

*[Pause]*

These aren't edge cases. These are the natural consequences of the eight challenges we'll cover today. By the end of this lecture, you'll understand exactly why these failures happen, and more importantly, how to prevent them.

**Here's my promise**: After today, you'll never look at a recommendation the same way again. Every time Netflix suggests a show, you'll ask: 'How did it handle cold start? What about the filter bubble? What's the exploration rate?'

Let's begin with the problem that every new recommendation system faces on day one..."

---

## Challenge 1: The Cold Start Problem

### Definition
Inability to make accurate recommendations for new users or items with no interaction history.

### "How Many Interactions Before We Can Trust Our Predictions?"

*[Professor writes on board: n = ?]*

"Here's a question that should bother you: **How many interactions does a user need before our recommendations become reliable?**

Let me show you the math that keeps data scientists up at night..."

### Mathematical Formulation: The Reliability Threshold

**The Central Question**: Given a new user $u$, after how many interactions $n$ can we achieve prediction error $\epsilon$ with confidence $\delta$?

**Theorem (Informal)**: For a matrix factorization model with $k$ latent factors:

$$n \geq \frac{k \cdot \log(1/\delta)}{\epsilon^2}$$

**Concrete Example**:
- Latent factors $k = 50$ (typical for Netflix)
- Desired error $\epsilon = 0.5$ stars
- Confidence $\delta = 0.05$ (95% confidence)

$$n \geq \frac{50 \cdot \log(20)}{0.25} = \frac{50 \cdot 3}{0.25} = 600 \text{ interactions}$$

*[Professor circles the number]*

"**600 interactions!** That's 600 movies rated before Netflix can confidently predict your preferences. The average user rates maybe 20-50 items total. Do you see the problem now?"

**The Variance Problem**:

After $n$ interactions, the variance of our user embedding estimate is:

$$\text{Var}(\hat{p}_u) \approx \frac{\sigma^2}{n} \cdot I_k$$

For $n = 5$ (typical new user): Variance is 120x higher than for $n = 600$.

This means early recommendations are essentially **educated guesses with high uncertainty**.

### Types

#### 1. **User Cold Start**
**Problem**: New user with no historical data

**Example**: New Netflix account - what to recommend?

**Strategies**:

**A. Onboarding Questionnaire**
- Ask user to rate a few items
- Select favorite genres/categories
- Example: Netflix asks to rate 3 movies initially

**B. Demographic Information**
- Age, gender, location
- Recommend based on similar demographic groups
- Privacy concerns limit this approach

**C. Popularity-Based**
- Show trending/popular items
- Safe but not personalized
- Used as fallback

**D. Explore-Exploit**
- **$\epsilon$-greedy**: Show popular items + random exploration
- **Multi-armed bandits**: Balance learning user preferences with showing relevant content

**E. Cross-Domain Transfer**
- Leverage data from other platforms
- Example: Spotify using Facebook likes

---

#### 2. **Item Cold Start**
**Problem**: New item with no user interactions

**Example**: Newly released movie on Netflix

**Strategies**:

**A. Content-Based Features**
- Use item metadata (genre, director, cast)
- Match to users with similar content preferences
- No interactions needed

**B. Feature-Based Models**
- Factorization machines
- Neural networks with side information

**C. Exploration Boost**
- Temporarily boost new items in rankings
- Gather initial interactions quickly

**D. Expert Curation**
- Editorial picks for new content
- Human-curated lists (e.g., "New Releases")

---

#### 3. **System Cold Start**
**Problem**: Brand new platform with minimal data

**Strategies**:
- Start with non-personalized recommendations
- Heavy reliance on content-based methods
- Aggressive data collection (incentivize ratings/reviews)
- Seed with external data

---

### Hybrid Model Solution

**Problem**: Predict $\hat{r}_{ui}$ where user $u$ or item $i$ is new

**Traditional CF**: Fails because no similar users/items to compare

**Solution**: Hybrid model
$$\hat{r}_{ui} = \alpha \cdot \text{CF}(u,i) + (1-\alpha) \cdot \text{Content}(u,i)$$

For cold items: $\alpha \approx 0$ (rely on content)
For warm items: $\alpha \approx 1$ (rely on CF)

### What Can Go Wrong: Cold Start Disasters

*[Professor leans forward]*

"Let me tell you about some real cold start failures:

**The Quibi Catastrophe (2020)**:
- New streaming service, complete system cold start
- Solution: Show same recommendations to everyone
- Result: Users felt 'the app didn't understand them'
- Shutdown after 6 months, losing $1.75 billion

**Spotify's 'Release Radar' Bug (2019)**:
- New songs (item cold start) assigned to random user clusters
- Heavy metal fans received children's music
- 'Unwanted notifications' complaints spiked 340%

**The Amazon Baby Trap**:
- User buys ONE baby gift for a friend
- Cold start algorithm assumes: new parent!
- User receives 6 months of baby product recommendations
- No easy way to signal 'this was a gift'

The lesson? **Cold start isn't just about accuracy - it's about trust.** Get it wrong early, and users never come back."

---

## Challenge 2: Data Sparsity

### Definition
User-item interaction matrix is extremely sparse (99%+ missing values).

### A Numerical Example: Why 99.9% Missing Breaks Everything

*[Professor draws matrix on board]*

"Let me show you exactly why sparsity kills simple methods. Here's a small user-item matrix:

```
           Movie1  Movie2  Movie3  Movie4  Movie5  Movie6  Movie7  Movie8  Movie9  Movie10
Alice         5       ?       ?       4       ?       ?       ?       ?       ?       ?
Bob           ?       3       ?       ?       ?       ?       5       ?       ?       ?
Carol         ?       ?       ?       4       ?       ?       ?       ?       2       ?
Dave          5       ?       ?       ?       ?       ?       ?       ?       ?       ?
Eve           ?       ?       ?       ?       ?       3       ?       ?       ?       ?
```

**Sparsity**: 10 ratings out of 50 cells = **80% sparse**

(And this is generous - real systems are 99.9%+ sparse!)

Now, let's compute **User-Based Collaborative Filtering** for Alice:

**Step 1**: Find users similar to Alice

For similarity, we need **co-rated items** (items both users rated).

| User Pair | Co-rated Items | Overlap |
|-----------|----------------|---------|
| Alice-Bob | None | 0 |
| Alice-Carol | Movie4 only | 1 |
| Alice-Dave | Movie1 only | 1 |
| Alice-Eve | None | 0 |

**Problem**: Only 1 overlapping rating with Carol and Dave!

**Step 2**: Compute Pearson correlation with Carol

$$\rho_{Alice,Carol} = \frac{(4-4.5)(4-3)}{...} = \text{undefined with 1 point!}$$

With a single co-rated item, Pearson correlation is mathematically **undefined** (or arbitrarily +1 or -1).

**Step 3**: Try to predict Alice's rating for Movie3

We need similar users who rated Movie3.
- Bob: didn't rate Movie3
- Carol: didn't rate Movie3
- Dave: didn't rate Movie3
- Eve: didn't rate Movie3

**Result**: Cannot make ANY prediction!

*[Professor underlines this]*

**This is why simple collaborative filtering fails at scale.** With real sparsity (99.9%), most users share ZERO common ratings."

### Scale of the Problem

**Example: Netflix**
```
- 260M users
- 15K titles
- Potential interactions: 3.9 trillion
- Actual ratings: ~100M (historical)
- Sparsity: 99.997% empty
```

### Visualization
```
           Item1  Item2  Item3  ... Item10000
User1        5      ?      ?    ...    ?
User2        ?      ?      3    ...    ?
User3        ?      4      ?    ...    ?
...
User1M       ?      ?      ?    ...    2
```

**Result**: Most users have rated <10 items out of thousands.

### Consequences

**1. Insufficient Similar Users/Items**
- User-based CF: Few users with overlapping ratings
- Item-based CF: Few items co-rated by same users

**2. Unreliable Similarity Estimates**
- Pearson correlation based on 1-2 overlapping ratings is meaningless
- High variance in predictions

**3. Long-Tail Items**
- Popular items have many ratings
- Niche items have few/no ratings
- **80/20 rule**: 20% of items account for 80% of interactions

### Solutions

**A. Dimensionality Reduction**
- **Matrix Factorization**: Learn low-rank approximation
  - $R \approx U^T V$ where $U \in \mathbb{R}^{k \times |U|}, V \in \mathbb{R}^{k \times |I|}$
  - $k \ll \min(|U|, |I|)$
- Captures latent patterns despite sparsity

**B. Regularization**
- Prevents overfitting on sparse data
- $\min ||R - U^T V||^2 + \lambda(||U||^2 + ||V||^2)$

**C. Implicit Feedback**
- Use clicks, views, plays (abundant) instead of ratings (sparse)
- Trades sparsity for noise

**D. Transfer Learning**
- Leverage data from related domains
- Pre-trained embeddings from rich domains

**E. Data Augmentation**
- Generate synthetic interactions
- Carefully to avoid introducing bias

### What Can Go Wrong: Sparsity Disasters

"Here's what happens when you ignore sparsity:

**The 'Irrelevant Recommendations' Problem**:
- E-commerce site uses raw collaborative filtering
- User A and User B both bought batteries (common item)
- System assumes they're similar
- User A (photographer) gets recommended diapers (User B is a parent)
- Conversion rate: 0.001%

**The Long-Tail Death Spiral**:
- Niche items have few ratings
- Few ratings = low confidence = not recommended
- Not recommended = no new ratings
- No new ratings = item disappears from recommendations
- Result: 60% of catalog never recommended

**Netflix's 'Missing Middle' Problem (reported in research)**:
- Highly-rated niche films had too few ratings
- Algorithm couldn't find similar users who watched them
- These films were systematically under-recommended
- User satisfaction surveys showed demand that algorithms couldn't detect"

---

## Challenge 3: Scalability

### The Scale Problem

Modern platforms operate at:
- **Users**: Billions (YouTube, Facebook)
- **Items**: Millions to billions (Amazon, Google)
- **Interactions**: Trillions (daily)
- **Latency requirement**: <100ms

### Computational Complexity: The Mathematical Reality

*[Professor writes on board]*

"Let me derive why you CANNOT use brute-force methods at scale."

#### The O(n^2) to O(n log n) Derivation

**User-Based CF - Brute Force**:

To recommend for user $u$:
1. Compute similarity with ALL other users: $O(|U|)$ similarities
2. Each similarity requires comparing ratings: $O(|I|)$ operations

**Total**: $O(|U| \cdot |I|)$ per user, $O(|U|^2 \cdot |I|)$ for all users

**For YouTube** (2B users, 800M videos):
$$O(2 \times 10^9 \times 2 \times 10^9 \times 8 \times 10^8) = O(10^{27})$$

*[Professor circles this]*

"That's $10^{27}$ operations. If you had a computer doing $10^{15}$ operations per second, it would take **30 billion years**. The universe is only 14 billion years old!"

---

**Approximate Nearest Neighbors - The Solution**:

**Key Insight**: We don't need EXACT nearest neighbors. 95% accuracy is fine.

**Locality Sensitive Hashing (LSH)**:
- Hash similar items to same bucket with high probability
- Query time: $O(\log |I|)$ instead of $O(|I|)$

**Hierarchical Navigable Small Worlds (HNSW)**:
- Build navigable graph of embeddings
- Greedy search through graph layers

**Complexity Comparison**:

| Method | Preprocessing | Query Time | Accuracy |
|--------|--------------|------------|----------|
| Brute Force | $O(1)$ | $O(n)$ | 100% |
| KD-Tree | $O(n \log n)$ | $O(\log n)$ average | 100% |
| LSH | $O(n)$ | $O(1)$ expected | ~95% |
| HNSW | $O(n \log n)$ | $O(\log n)$ | ~99% |

**The Trade-off**:
$$\text{Speedup} = \frac{O(n)}{O(\log n)} = \frac{n}{\log n}$$

For $n = 10^9$: Speedup = $\frac{10^9}{30} \approx 33$ million times faster!

"This is why **approximate methods aren't optional - they're required**. Exact methods at scale are physically impossible."

### Solutions

**A. Two-Stage Architecture**
```
Stage 1: Candidate Generation (fast, recalls 100-1000 items)
Stage 2: Ranking (expensive, scores 100-1000 items)
```

**B. Approximate Nearest Neighbors (ANN)**
- **Exact kNN**: $O(|I|)$ per query
- **ANN (HNSW, FAISS)**: $O(\log|I|)$ with 95%+ recall

**C. Caching**
- Precompute recommendations for popular users
- Cache item embeddings
- Redis/Memcached for fast lookups

**D. Distributed Computing**
- **Training**: Spark, Parameter servers, GPUs
- **Serving**: Load balancers, sharding

**E. Model Compression**
- **Knowledge distillation**: Large model -> small model
- **Quantization**: 32-bit -> 8-bit weights
- **Pruning**: Remove less important connections

**F. Batch Processing**
- Update recommendations daily/hourly (not real-time)
- Acceptable for many use cases (e.g., email campaigns)

### What Can Go Wrong: Scalability Disasters

"Scalability failures are usually invisible until they're catastrophic:

**Twitter's 'Fail Whale' Era (2008-2013)**:
- Recommendation queries took 2+ seconds
- During high traffic: system timeout
- Users saw error page (the famous whale)
- Lost millions in engagement

**Pinterest's Black Friday Meltdown (2015)**:
- Recommendation system couldn't scale with traffic
- Fell back to showing SAME recommendations to everyone
- Personalization: 0% for 6 hours
- Estimated revenue loss: $10M+

**The Latency Tax**:
- Amazon found: 100ms latency = 1% revenue loss
- Google found: 500ms delay = 20% drop in searches
- Your perfect algorithm is worthless if it takes 2 seconds"

---

## Challenge 4: Exploration vs. Exploitation

### The Dilemma

**Exploitation**: Show items you know the user will like (maximize immediate reward)

**Exploration**: Show items to learn user preferences (maximize long-term reward)

**Trade-off**:
- Pure exploitation -> filter bubble, stale recommendations
- Pure exploration -> poor user experience, irrelevant items

### Multi-Armed Bandit Formulation

**Setting**: $K$ arms (items), unknown reward distributions

**Goal**: Maximize cumulative reward over $T$ rounds

**Regret**: Difference from optimal strategy

$$\text{Regret} = T \cdot \mu^* - \sum_{t=1}^T r_t$$

where $\mu^*$ = expected reward of best arm

### Algorithms

#### **$\epsilon$-Greedy**
```
With probability epsilon: Choose random arm (explore)
With probability 1-epsilon: Choose best arm so far (exploit)
```

**Pros**: Simple
**Cons**: Wastes exploration on obviously bad arms

#### **UCB (Upper Confidence Bound)**
$$\text{UCB}(i) = \hat{\mu}_i + \sqrt{\frac{2 \ln t}{n_i}}$$

- $\hat{\mu}_i$ = average reward of arm $i$
- $n_i$ = number of times arm $i$ pulled
- $t$ = total rounds

**Intuition**: Choose arm with highest potential (mean + uncertainty)

#### **Thompson Sampling**
- Bayesian approach
- Sample from posterior distribution of rewards
- Naturally balances exploration and exploitation

**Best in practice** (empirically outperforms UCB, $\epsilon$-greedy)

### Contextual Bandits

User and item features available -> better targeting

**LinUCB**: Linear model + UCB principle
**Neural Bandits**: Deep learning + Thompson sampling

### Production Example: Netflix Artwork

**Problem**: Which thumbnail to show for a movie?

**Approach**: Contextual bandits
- Context: User features, time, device
- Arms: Different thumbnail images
- Reward: Click-through rate

**Result**: 20-30% increase in engagement

---

## Challenge 5: Filter Bubbles and Echo Chambers

### Definition

**Filter Bubble**: Algorithmic personalization reinforces existing preferences, limiting exposure to diverse content.

**Echo Chamber**: Users surrounded by similar viewpoints, amplifying beliefs.

### A 5-Round Feedback Loop: Watching Popularity Bias Amplify

*[Professor draws diagram]*

"Let me trace exactly how a feedback loop creates a popularity death spiral:

**Setup**: Music streaming service with 1000 songs. Song A is slightly more popular initially.

---

**Round 1 - Initial State**:
- Song A: 1000 plays (popular)
- Song B: 800 plays (good but less popular)
- Algorithm weight: Popularity contributes 30% to ranking score

**Recommendation output**: Song A ranked #1, Song B ranked #5

---

**Round 2 - After 1 week**:
- Song A: 1500 plays (+50% because it was recommended more)
- Song B: 850 plays (+6% organic discovery only)
- Gap widens: A has 1.76x more plays than B

**Recommendation output**: Song A now appears on 'Top Hits' playlist

---

**Round 3 - After 2 weeks**:
- Song A: 3000 plays (doubling - 'Top Hits' effect)
- Song B: 900 plays (barely growing)
- Gap: A has 3.3x more plays than B

**New users see**: Song A in their first recommendations (cold start uses popularity)

---

**Round 4 - After 1 month**:
- Song A: 10,000 plays (viral momentum)
- Song B: 950 plays (stagnant)
- Gap: A has 10.5x more plays than B

**Algorithm behavior**: Song B's ranking score falls below recommendation threshold

---

**Round 5 - After 2 months**:
- Song A: 50,000 plays (cultural phenomenon)
- Song B: 980 plays (invisible to algorithm)
- Gap: A has 51x more plays than B

**Final state**: Song B will NEVER be recommended, regardless of quality.

---

*[Professor steps back]*

**The Math of Amplification**:

If clicks increase recommendations by factor $\alpha$, and recommendations increase clicks by factor $\beta$:

After $n$ rounds: $\text{Popularity ratio} = \left(\frac{p_A}{p_B}\right)^{(\alpha \beta)^n}$

For $\alpha = 1.2$, $\beta = 1.3$, initial ratio 1.25:
- Round 1: 1.25
- Round 2: 1.95
- Round 3: 4.8
- Round 4: 35.7
- Round 5: 1,987

**This is exponential amplification**. Small initial differences become insurmountable."

### Socratic Moment: Can You Escape?

*[Professor pauses, looks at class]*

"Here's a question I want you to really think about:

**Can you ever escape a filter bubble if you only see what the algorithm recommends?**

Think about it:
- The algorithm shows you content based on your history
- You can only interact with content you're shown
- Your interactions become your new history
- The algorithm uses this history to choose what to show next

*[Pause]*

It's a closed loop. The algorithm defines your reality, and your reality defines the algorithm.

**Some follow-up questions**:
- If you've never been shown jazz music, can you discover you love jazz?
- If political content is always left-leaning, do you know right-leaning arguments exist?
- If you're only shown beginner content, can you find advanced material?

**The uncomfortable answer**: Without explicit intervention (diversity injection, exploration, user control), **filter bubbles are mathematically inevitable**.

This is why diversity objectives aren't just nice-to-have - they're ethically necessary."

### How Recommendations Create Bubbles

1. **Optimize for engagement** -> show familiar content
2. **Feedback loop**: User clicks similar content -> model learns to show more similar content
3. **Limited exploration**: Exploitation-heavy strategies

### Consequences

**Individual Level**:
- Reduced serendipity and discovery
- Narrow worldview
- Confirmation bias reinforcement

**Societal Level**:
- Political polarization
- Misinformation spread
- Radicalization pipelines (e.g., YouTube conspiracy videos)

### Mitigation Strategies

**A. Diversity Objectives**
- Optimize for diversity alongside relevance
- **Maximal Marginal Relevance (MMR)**: Balance similarity and diversity

**B. Serendipity**
- Intentionally recommend surprising but relevant items
- "You might also like" with low similarity but high predicted rating

**C. Explanation and Control**
- Show users why items recommended
- Allow users to adjust recommendation weights
- Example: Instagram's "Your Algorithm" tool (2025)

**D. Limit Amplification**
- Cap percentage of similar content
- Break recommendation chains (e.g., stop suggesting conspiracy videos)

**E. Promote Authoritative Sources**
- For news, prioritize credible sources
- Fact-checking integration

**F. Transparency**
- Show users their recommendation profile
- Allow editing interests/preferences

### What Can Go Wrong: Filter Bubble Disasters

"The consequences of filter bubbles extend far beyond 'I keep seeing the same music':

**YouTube's Radicalization Pipeline (2016-2019)**:
- Researchers found: mainstream political videos -> extreme content in 5 clicks
- Algorithm optimized for watch time
- Extreme content had higher engagement
- Result: Algorithmic pathway to radicalization

**Facebook's Myanmar Crisis (2018)**:
- Recommendation algorithm amplified hate speech
- Users in closed information bubbles
- UN report: Facebook 'substantively contributed' to genocide
- Content moderation couldn't keep up with algorithmic amplification

**Spotify's 'Taste Freeze' Phenomenon**:
- Users complained their Discover Weekly stopped evolving
- Analysis: After 2 years, recommendations stabilized on narrow taste profile
- Algorithm confident it 'knew' the user
- Users felt 'trapped' in their past preferences"

---

## Challenge 6: Privacy vs. Personalization Tradeoff

### The Fundamental Tension

*[Professor draws scale on board]*

"Here's the uncomfortable truth that every recommendation system designer must face:

**Better data = Better recommendations = Less privacy**

Let me show you exactly what data enables what recommendations:

### The Data-Capability Matrix

| Data Type | Enables | Privacy Risk |
|-----------|---------|--------------|
| **Anonymous clicks** | Basic popularity, trending | Low |
| **Session history** | Short-term preferences, 'continue watching' | Medium |
| **Account history** | Long-term taste modeling, personalization | High |
| **Demographics** | Cold start, cohort recommendations | High |
| **Location** | Local recommendations, context | Very High |
| **Social graph** | 'Friends liked', viral prediction | Very High |
| **Cross-platform** | Complete user model, life events | Extreme |

### What Each Privacy Level Gets You

**Level 1: No Personal Data (Anonymous)**
- Recommendations: Global popularity only
- Quality: Same for everyone
- Example: 'Top 10 in your country'

**Level 2: Session Only (Ephemeral)**
- Recommendations: 'Because you just watched X'
- Quality: Good within session, resets daily
- Example: YouTube incognito mode

**Level 3: Account History (Standard)**
- Recommendations: Full personalization
- Quality: Improves over months/years
- Example: Netflix logged-in experience

**Level 4: Cross-Platform (Complete)**
- Recommendations: Predicts needs before you know them
- Quality: 'Eerily accurate'
- Example: Amazon knowing you need diapers before you announce pregnancy

*[Professor pauses]*

**The Question You Must Answer**:
How much privacy are users willing to trade for how much personalization improvement?

Research shows:
- 10% improvement in relevance requires 2x more data
- Going from 'good' to 'great' recommendations requires location, social, and behavioral data
- Most users say they value privacy but behave as if they don't

This is the **privacy paradox** - and your system design must navigate it."

### Regulatory Requirements

**User Data Collection**:
- Every click, view, purchase tracked
- Sensitive information (health, politics, location)

**Regulatory Requirements**:
- **GDPR** (Europe): Right to deletion, consent
- **CCPA** (California): Opt-out, transparency

### Technical Solutions

**A. Federated Learning**
- Train models on-device
- Only share model updates, not raw data
- Privacy-preserving aggregation

**B. Differential Privacy**
- Add noise to data/models
- Guarantee individual privacy
- Trade-off: Accuracy vs. privacy

$$\text{Privacy budget } \epsilon: \Pr[\text{Output} | \text{User in data}] \leq e^\epsilon \cdot \Pr[\text{Output} | \text{User not in data}]$$

**C. Anonymization**
- Hash user IDs
- Aggregate data (cohort-level)

**D. User Control**
- Opt-out options
- Data deletion on request
- Transparency in data usage

### What Can Go Wrong: Privacy Disasters

"Privacy failures in recommendation systems have ended companies:

**Target's Pregnancy Prediction (2012)**:
- Algorithm detected pregnancy from purchase patterns
- Sent baby coupons to teenager's home
- Father learned daughter was pregnant from Target
- Massive PR disaster, congressional attention

**Cambridge Analytica (2018)**:
- Used Facebook's social graph for political targeting
- 87 million users' data harvested without consent
- Recommendation-like targeting for political manipulation
- Result: $5B fine for Facebook, company dissolved

**Apple's Siri Privacy Revelations (2019)**:
- Contractors listened to Siri recordings for 'quality'
- Included private conversations, medical information
- Voice-based recommendations required human review
- User trust severely damaged"

---

## Challenge 7: Evaluation Metrics

### The Problem
Optimizing for the wrong metric can harm user experience.

### Offline vs. Online Metrics

**Offline (RMSE, Precision@K)**:
- Computed on historical data
- Fast, cheap, reproducible
- **Limitation**: Doesn't capture real user behavior

**Online (CTR, watch time, retention)**:
- Real user feedback via A/B tests
- Expensive, slow, noisy
- **Ground truth** for business value

### Discrepancy Example

**Netflix Prize**: 10% RMSE improvement didn't translate to better user experience.

**Why?**
- RMSE measures rating prediction accuracy
- Users care about finding great content (ranking, not ratings)
- Diversity, novelty matter but not in RMSE

### Beyond-Accuracy Metrics

**Diversity**: How varied are recommendations?
**Novelty**: How unexpected are items?
**Coverage**: % of catalog recommended
**Serendipity**: Surprising + relevant

**Challenge**: Hard to measure offline, trade-off with accuracy

### What Can Go Wrong: Metric Disasters

"Goodhart's Law: 'When a measure becomes a target, it ceases to be a good measure.'

**YouTube's Watch Time Optimization (2016)**:
- Optimized for watch time as primary metric
- Algorithm discovered: conspiracy videos maximize watch time
- Users fell down 'rabbit holes' of increasingly extreme content
- Watch time up, societal harm up

**Facebook's Engagement Metric (2018)**:
- Optimized for reactions, comments, shares
- Anger and outrage drove highest engagement
- Algorithm amplified divisive content
- Result: 'Angry react' became signal for 'important content'"

---

## Challenge 8: Concept Drift and Temporal Dynamics

### Definition
User preferences and item popularity change over time.

### Examples

**Seasonal Trends**:
- Halloween movies in October
- Tax software in April

**User Preferences Drift**:
- New parent -> baby product recommendations
- Student graduates -> professional content

**Item Popularity Decay**:
- Viral video -> trending for days -> forgotten

### Mathematical Model

**TimeSVD++**:
$$\hat{r}_{ui}(t) = \mu + b_u(t) + b_i(t) + q_i^T p_u(t)$$

- $b_u(t)$, $p_u(t)$: User bias and factors evolve with time
- $b_i(t)$: Item bias evolves with time

### Solutions

**A. Online Learning**
- Update model incrementally with new data
- Sliding window (last N days)

**B. Time-Aware Features**
- Include time as explicit feature
- Day of week, season, recency

**C. Short-Term vs. Long-Term**
- Long-term model: User's general preferences
- Short-term model: Current session/context
- **Combine**: $\text{Score} = \alpha \cdot \text{Long} + (1-\alpha) \cdot \text{Short}$

**D. Recency Weighting**
- Weight recent interactions more heavily
- Exponential decay: $w(t) = e^{-\lambda t}$

**E. Continuous Retraining**
- Retrain model daily/weekly
- Balance freshness with stability

### What Can Go Wrong: Temporal Disasters

"Static models in a dynamic world fail spectacularly:

**COVID-19 Recommendation Collapse (March 2020)**:
- All recommendation models trained on pre-pandemic data
- Suddenly: no one wants travel, everyone wants home office supplies
- Amazon reported 'weeks of irrelevant recommendations'
- Models took months to adapt

**The 'Divorced User' Problem**:
- User's spouse leaves the account
- Historical data: shared preferences
- Current reality: completely different person
- System keeps recommending ex-spouse's interests for months

**Seasonal Blindness**:
- User searches for 'Halloween costumes' in October
- Next October: System recommends... nothing
- Model trained on recency, forgot annual patterns
- User goes to competitor"

---

## Challenge Summary Table

| Challenge | Impact | Key Solutions |
|-----------|--------|---------------|
| **Cold Start** | Can't recommend for new users/items | Onboarding, content-based, exploration |
| **Sparsity** | Unreliable similarity estimates | Matrix factorization, implicit feedback |
| **Scalability** | Latency, cost | Two-stage architecture, ANN, caching |
| **Exploration-Exploitation** | Filter bubbles vs. learning | Bandits, Thompson sampling |
| **Filter Bubbles** | Limited diversity, polarization | Diversity objectives, serendipity |
| **Evaluation** | Wrong metric optimization | Online A/B testing, beyond-accuracy metrics |
| **Concept Drift** | Stale recommendations | Online learning, time-aware features |
| **Privacy** | User trust, legal compliance | Federated learning, differential privacy |

---

## Trade-Offs in RecSys

Recommendation system design involves constant trade-offs:

1. **Accuracy vs. Diversity**: Relevant but repetitive vs. varied but less relevant
2. **Personalization vs. Privacy**: Better recommendations vs. user data protection
3. **Exploration vs. Exploitation**: Learning vs. immediate reward
4. **Complexity vs. Latency**: Better models vs. faster serving
5. **Short-term vs. Long-term**: Engagement today vs. retention tomorrow

**No universal solution** - context and business goals determine priorities.

---

## Summary

*[Professor returns to the opening]*

"We started with the question: What will keep you up at night?

Now you know the eight challenges:

1. **Cold Start**: New users/items with no history - and you need 600+ interactions for reliable predictions
2. **Sparsity**: 99%+ of user-item matrix is empty - and simple methods mathematically cannot work
3. **Scalability**: Billions of users/items, <100ms latency - and exact methods are physically impossible
4. **Exploration-Exploitation**: Balance learning and performance - or trap users forever
5. **Filter Bubbles**: Avoid over-personalization and echo chambers - with exponential amplification working against you
6. **Evaluation**: Offline metrics != online business value - and wrong metrics cause real harm
7. **Concept Drift**: Preferences and trends change over time - and static models decay
8. **Privacy**: Data protection and regulatory compliance - while users want personalization

*[Professor writes on board]*

**The Meta-Challenge**: All eight challenges interact. Solving one often makes another worse.
- More data (helps sparsity) -> worse privacy
- More exploration -> slower to exploit good recommendations
- Faster serving (scalability) -> simpler models (accuracy loss)

**Your job as a recommendation system designer**: Navigate these trade-offs thoughtfully, understanding that **every design choice has consequences**."

---

## Looking Ahead

**Next Steps**:
- **practice-problems.md**: Exercises to test understanding
- **Week 2**: Dive into collaborative filtering algorithms

---

## References

1. Schein, A. I., et al. (2002). "Methods and metrics for cold-start recommendations". *SIGIR*.
2. Lam, X. N., et al. (2008). "Addressing cold-start problem in recommendation systems". *Ubicomp*.
3. Pariser, E. (2011). *The Filter Bubble*. Penguin Press.
4. Li, L., et al. (2010). "A contextual-bandit approach to personalized news article recommendation". *WWW*.
5. Dwork, C., & Roth, A. (2014). "The algorithmic foundations of differential privacy". *FNT in TCS*.
