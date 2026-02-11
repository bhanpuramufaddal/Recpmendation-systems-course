# Week 13: Cold Start Problem

## The Opening Problem

> **"Your best model gives random recommendations to 10% of users."**

Let me show you why this happens with a concrete user journey.

**Sarah joins Netflix on Monday evening:**

```
Day 1, 7:00 PM:
- Sarah creates account
- System has ZERO information about Sarah
- Our collaborative filtering model: "I need interaction history to work"
- Matrix factorization: "I need ratings to learn latent factors"
- Neural CF: "I need user embeddings, but Sarah has none"

What do we show Sarah?
```

**The system's dilemma:**

$$P(\text{Sarah likes item } i) = \text{???}$$

We have no $r_{Sarah,*}$ in our rating matrix. Sarah is a **zero vector** in our user-item space.

**What actually happens:**

```
Option A: Show popular items (The Office, Stranger Things)
         - Not personalized
         - Sarah might hate comedies
         - Wasted opportunity to learn

Option B: Show random items
         - Even worse engagement
         - Sarah might leave the platform

Option C: Show nothing
         - Not an option
```

**The business impact:**
- Netflix: New users who don't find something to watch within 90 seconds have **high churn**
- Spotify: Users who don't discover good music in their first session have **50% lower retention**
- Amazon: New users who see irrelevant products have **30% lower conversion**

**Today we solve this:**
1. How do we mathematically define "cold start"?
2. What's the minimum interactions needed before CF works?
3. How do we smoothly transition users from cold to warm?

---

## Overview

**Cold start**: New users/items with no interaction history.

**Types**:
1. **User cold start**: New user, no preferences known
2. **Item cold start**: New item, no interactions yet
3. **System cold start**: New platform, no data

---

## Mathematical Formulation of Cold Start

### Defining Cold Start Formally

**Definition**: A user $u$ is in **cold start** if their interaction count $n_u$ falls below a threshold $\tau$:

$$\text{ColdStart}(u) = \mathbb{1}[n_u < \tau]$$

**But what is the right threshold $\tau$?**

### Deriving the Minimum Interactions Bound

Consider a matrix factorization model where we learn user embedding $\mathbf{p}_u \in \mathbb{R}^k$ with $k$ latent factors.

**Intuition**: To reliably estimate $k$ parameters, we need at least $k$ observations.

**Formal bound**: For user embedding $\mathbf{p}_u$ with regularization $\lambda$, the estimation variance is:

$$\text{Var}(\hat{\mathbf{p}}_u) = \frac{\sigma^2}{n_u + \lambda}$$

where:
- $\sigma^2$ = noise variance in ratings
- $n_u$ = number of ratings by user $u$
- $\lambda$ = regularization strength

**For reliable estimation**, we need variance below threshold $\epsilon$:

$$\frac{\sigma^2}{n_u + \lambda} < \epsilon$$

Solving for $n_u$:

$$n_u > \frac{\sigma^2}{\epsilon} - \lambda$$

**Practical rule of thumb**:

$$\tau \approx 2k \text{ to } 5k$$

where $k$ is the number of latent factors.

**Example calculation**:

```
Latent factors: k = 50
Noise variance: sigma^2 = 1.0
Desired variance: epsilon = 0.1
Regularization: lambda = 0.01

Minimum interactions: n_u > 1.0/0.1 - 0.01 = 9.99

With safety margin: tau = 20-30 interactions
```

**Industry standards**:

| Platform | Latent Dims | Cold Start Threshold |
|----------|-------------|---------------------|
| Netflix | 100-200 | 20-50 ratings |
| Spotify | 50-100 | 10-30 songs played |
| Amazon | 100+ | 5-15 purchases |

---

## User Cold Start

### Problem

**New user**: No ratings, clicks, or purchases -> can't personalize.

**Naive solution**: Show popular items (not personalized).

---

### The Information Gap

**What we know about new user Sarah:**
- $n_{Sarah} = 0$ (no interactions)
- $\mathbf{p}_{Sarah} = ?$ (unknown embedding)

**What we need:**
- $P(\text{Sarah likes item } i)$ for all items $i$

**The information gap:**

$$\Delta I = H(\text{Sarah's preferences}) - I(\text{known signals})$$

where $H$ is entropy (uncertainty) and $I$ is mutual information.

**Goal of cold start solutions**: Minimize $\Delta I$ as quickly as possible.

---

### Solutions: Addressing the Information Gap

**Solution 1: Onboarding Survey**

**Information gain**: Direct preference elicitation reduces entropy immediately.

$$I(\text{onboarding}) = H(\text{preferences}) - H(\text{preferences} | \text{survey answers})$$

**Step-by-step derivation**:

```
Before onboarding:
- P(Sarah likes action) = 0.5 (uniform prior)
- P(Sarah likes romance) = 0.5
- Entropy: H = -2 * 0.5 * log(0.5) = 1.0 bit

After Sarah selects "action" as favorite:
- P(Sarah likes action) = 0.9 (posterior)
- P(Sarah likes romance) = 0.2
- Entropy drops significantly

Information gained: ~0.6 bits per genre selected
```

**Implementation**:

```python
def onboarding_recommendations(selected_genres, selected_items):
    """Recommend based on initial preferences"""
    candidates = []

    # Items from selected genres
    for genre in selected_genres:
        candidates.extend(get_items_by_genre(genre, k=20))

    # Similar to selected items
    for item in selected_items:
        similar = get_similar_items(item, k=10)
        candidates.extend(similar)

    return rank_by_popularity(candidates)[:50]
```

**Optimal number of onboarding questions**:

The information gain per question decreases (diminishing returns):

$$I(q_n) \approx I(q_1) \cdot \alpha^{n-1}$$

where $\alpha \approx 0.7$ (each question gives ~70% of previous question's value).

**Rule of thumb**: 3-5 questions optimal (more causes survey fatigue).

---

**Solution 2: Demographic Matching**

**Key insight**: Users with similar demographics often have similar preferences.

**Bayesian formulation**:

$$P(\text{Sarah likes } i) = \sum_{u \in \text{similar}} P(u | \text{demographics}) \cdot P(\text{user } u \text{ likes } i)$$

**Step-by-step**:

```
Sarah's demographics: Female, Age 28, San Francisco

Step 1: Find demographic cluster
- Find users with: Age in [25, 31], Female, Bay Area

Step 2: Compute cluster preferences
- Action movies: 35% popularity in cluster
- Rom-coms: 45% popularity in cluster
- Documentaries: 20% popularity in cluster

Step 3: Weight by cluster similarity
- Sarah's predicted preference = cluster average
```

**Implementation**:

```python
def demographic_cold_start(age, location, gender):
    """Find similar users by demographics"""
    similar_users = find_users(age_range=(age-5, age+5),
                                location=location,
                                gender=gender)

    # Aggregate their preferences
    popular_items = get_popular_items_for_users(similar_users)
    return popular_items
```

---

**Solution 3: Hybrid: Content + CF**

Start with content-based, transition to CF as data accumulates.

**The hybrid weight formula** (derived later in Hybrid Strategies section):

$$\hat{r}_{ui} = w(n_u) \cdot \hat{r}_{ui}^{CB} + (1 - w(n_u)) \cdot \hat{r}_{ui}^{CF}$$

where $w(n_u)$ decreases as $n_u$ increases.

---

**Solution 4: Active Learning**

Strategically ask for ratings on **diverse** items.

**Why diversity matters** (entropy-based derivation below).

---

## Item Cold Start

### Problem

**New item**: Just released, no clicks/ratings.

**Challenge**: Can't use CF (no user-item interactions).

---

### Numerical Example: New Movie Added

**Scenario**: "Cosmic Dawn" (new sci-fi film) is added to catalog.

**What we know about the movie:**

| Feature | Value | Feature Vector |
|---------|-------|---------------|
| Genre | Sci-Fi | [1, 0, 0, 0, 0] |
| Director | Denis Villeneuve | [0.8, 0.7, 0.9] (embedding) |
| Year | 2024 | [0.95] (normalized) |
| Runtime | 142 min | [0.71] (normalized) |

**Combined feature vector**:

$$\mathbf{f}_{\text{Cosmic Dawn}} = [1, 0, 0, 0, 0, 0.8, 0.7, 0.9, 0.95, 0.71]$$

**User Sarah's profile** (from her watch history):

| Movie | Rating | Feature Vector |
|-------|--------|---------------|
| Blade Runner 2049 | 5 | [1, 0, 0, 0, 0, 0.8, 0.7, 0.9, 0.7, 0.82] |
| Arrival | 4 | [1, 0, 0, 0, 0, 0.8, 0.7, 0.9, 0.6, 0.58] |
| The Martian | 5 | [1, 0, 0, 0, 0, 0.5, 0.6, 0.5, 0.65, 0.71] |

**Step 1: Compute user profile vector**

$$\mathbf{u}_{\text{Sarah}} = \frac{1}{|H|} \sum_{i \in H} w_i \cdot \mathbf{f}_i$$

where $w_i = \frac{r_i - \bar{r}}{\sigma}$ (rating-weighted).

```python
weights = [1.0, 0.5, 1.0]  # normalized ratings above mean
user_profile = np.average(feature_vectors, weights=weights, axis=0)
# u_Sarah = [1.0, 0, 0, 0, 0, 0.72, 0.68, 0.80, 0.66, 0.72]
```

**Step 2: Compute similarity to new movie**

$$\text{sim}(\text{Sarah}, \text{Cosmic Dawn}) = \frac{\mathbf{u}_{\text{Sarah}} \cdot \mathbf{f}_{\text{Cosmic Dawn}}}{||\mathbf{u}_{\text{Sarah}}|| \cdot ||\mathbf{f}_{\text{Cosmic Dawn}}||}$$

```python
similarity = cosine_similarity(u_sarah, f_cosmic_dawn)
# similarity = 0.94 (very high!)
```

**Step 3: Predict rating**

$$\hat{r}_{\text{Sarah}, \text{Cosmic Dawn}} = \bar{r}_{\text{Sarah}} + \text{sim} \cdot (\text{max\_rating} - \bar{r}_{\text{Sarah}})$$

```python
predicted_rating = 4.67 + 0.94 * (5 - 4.67) = 4.98
```

**Recommendation decision**: Show "Cosmic Dawn" to Sarah!

---

### Solutions

**1. Content-Based**

Use item features (title, category, description).

```python
def cold_item_recommendations(new_item, user_history):
    """Recommend new item based on content similarity"""
    new_item_features = extract_features(new_item)

    # Find similar items user liked
    scores = []
    for hist_item in user_history:
        hist_features = extract_features(hist_item)
        similarity = cosine_similarity(new_item_features, hist_features)
        scores.append(similarity)

    # If user liked similar items, recommend new item
    avg_similarity = np.mean(scores)
    return avg_similarity > 0.7
```

**2. Exploration**

Show new items to random sample of users (A/B test).

**3. Transfer Learning**

Use pre-trained embeddings (e.g., BERT for text, CLIP for images).

---

## Hybrid Strategies: Mathematical Weighting

### How to Weight CF vs Content-Based Based on Interaction Count

**Key insight**: As user accumulates more interactions, CF becomes more reliable.

**The confidence-based weighting function:**

$$w(n_u) = \frac{\tau}{\tau + n_u}$$

where:
- $n_u$ = number of interactions for user $u$
- $\tau$ = cold start threshold (e.g., 20)

**Derivation**:

We want a weight function that:
1. $w(0) = 1$ (pure content-based for new users)
2. $w(\tau) = 0.5$ (50-50 split at threshold)
3. $w(\infty) \rightarrow 0$ (pure CF for established users)

The function $w(n) = \frac{\tau}{\tau + n}$ satisfies all three:

```
n=0:   w = 20/(20+0) = 1.0    (100% content-based)
n=20:  w = 20/(20+20) = 0.5   (50% content, 50% CF)
n=100: w = 20/(20+100) = 0.17 (17% content, 83% CF)
n=500: w = 20/(20+500) = 0.04 (4% content, 96% CF)
```

**Alternative: Sigmoid weighting**

$$w(n_u) = \frac{1}{1 + e^{(n_u - \tau)/\sigma}}$$

This provides smoother transition around the threshold.

**Implementation**:

```python
class AdaptiveHybridRecommender:
    def __init__(self, tau=20, sigma=5):
        self.tau = tau
        self.sigma = sigma
        self.content_model = ContentBasedModel()
        self.cf_model = CollaborativeFilteringModel()

    def compute_weight(self, n_interactions, method='confidence'):
        """Compute content-based weight based on interaction count."""
        if method == 'confidence':
            return self.tau / (self.tau + n_interactions)
        elif method == 'sigmoid':
            return 1 / (1 + np.exp((n_interactions - self.tau) / self.sigma))

    def predict(self, user_id, item_id):
        n_u = self.get_interaction_count(user_id)
        w = self.compute_weight(n_u)

        cb_score = self.content_model.predict(user_id, item_id)
        cf_score = self.cf_model.predict(user_id, item_id)

        return w * cb_score + (1 - w) * cf_score
```

**Visual representation**:

```
Weight for Content-Based
  1.0 |******
      |      ****
  0.5 |          ****
      |              ****
  0.0 |                  ********
      +--------------------------> Interactions
        0    20    40    60    80
```

---

## Active Learning Derivation: Why Diverse Items are Optimal

### The Entropy-Based Selection Principle

**Goal**: Select items to ask about that maximize information gain.

**Setup**: User has unknown preference vector $\mathbf{p} \in \mathbb{R}^k$ (k latent factors).

**Key insight**: Each rating reveals information about one direction in preference space.

### Information-Theoretic Derivation

**Expected information gain** from asking about item $i$:

$$I(i) = H(\mathbf{p}) - H(\mathbf{p} | r_i)$$

where:
- $H(\mathbf{p})$ = entropy of preference distribution before
- $H(\mathbf{p} | r_i)$ = entropy after observing rating $r_i$

**For Gaussian prior** $\mathbf{p} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:

$$I(i) = \frac{1}{2} \log \left( 1 + \frac{\mathbf{q}_i^T \boldsymbol{\Sigma} \mathbf{q}_i}{\sigma^2} \right)$$

where $\mathbf{q}_i$ is item $i$'s feature vector.

**Critical observation**: Information is maximized when $\mathbf{q}_i$ is **orthogonal** to previously selected items!

### Why Diversity is Optimal

**Intuition**: Asking about similar items is redundant.

**Example with 2D preferences** (action vs. romance):

```
Scenario A: Ask about 3 action movies
- All items have feature vectors [1, 0]
- We learn a lot about action preference
- We learn NOTHING about romance preference
- Entropy reduction: ~1 dimension

Scenario B: Ask about 1 action, 1 romance, 1 documentary
- Items have vectors [1,0,0], [0,1,0], [0,0,1]
- We learn about ALL preferences
- Entropy reduction: ~3 dimensions
```

**Mathematical proof**:

For $n$ items, total information gain is:

$$I_{\text{total}} = \frac{1}{2} \log \det \left( \mathbf{I} + \frac{1}{\sigma^2} \mathbf{Q}^T \boldsymbol{\Sigma} \mathbf{Q} \right)$$

This is maximized when $\mathbf{Q}$ (matrix of selected item features) has orthogonal columns.

### Optimal Active Learning Algorithm

```python
class ActiveLearningSelector:
    def __init__(self, item_features, n_questions=5):
        """
        item_features: (n_items, k) matrix of item feature vectors
        n_questions: number of items to select
        """
        self.item_features = item_features
        self.n_questions = n_questions

    def select_diverse_items(self):
        """Select maximally diverse items using greedy entropy maximization."""
        n_items, k = self.item_features.shape
        selected = []
        remaining = set(range(n_items))

        # Initialize covariance (prior uncertainty)
        Sigma = np.eye(k)

        for _ in range(self.n_questions):
            best_item = None
            best_gain = -np.inf

            for i in remaining:
                q_i = self.item_features[i]
                # Information gain for item i
                gain = 0.5 * np.log(1 + q_i.T @ Sigma @ q_i)

                if gain > best_gain:
                    best_gain = gain
                    best_item = i

            selected.append(best_item)
            remaining.remove(best_item)

            # Update covariance (posterior after observing rating)
            q = self.item_features[best_item]
            Sigma = Sigma - np.outer(Sigma @ q, q @ Sigma) / (1 + q @ Sigma @ q)

        return selected

# Example usage
item_features = np.random.randn(1000, 50)  # 1000 items, 50 latent dims
selector = ActiveLearningSelector(item_features, n_questions=5)
questions = selector.select_diverse_items()
print(f"Ask user about items: {questions}")
```

**Result**: Selected items span different genres, directors, time periods, maximizing learning efficiency.

---

## Meta-Learning

### Concept

**Learn to learn**: Model adapts quickly to new users/items with few examples.

**MAML** (Model-Agnostic Meta-Learning):

**Idea**: Train model initialization that adapts fast to new tasks.

**Process**:
1. **Meta-train**: Learn good initialization across many users
2. **Adapt**: Fine-tune on new user's few interactions

---

### Implementation (Simplified)

```python
class MetaRecommender:
    def __init__(self, base_model):
        self.meta_model = base_model  # Meta-learned initialization

    def adapt_to_user(self, user_interactions, n_steps=5):
        """Quickly adapt to new user"""
        user_model = copy.deepcopy(self.meta_model)
        optimizer = torch.optim.SGD(user_model.parameters(), lr=0.01)

        for _ in range(n_steps):
            for item, rating in user_interactions:
                pred = user_model(item)
                loss = (pred - rating) ** 2
                loss.backward()
                optimizer.step()

        return user_model
```

---

## Bandit Approaches

### Exploration for Cold Start

**Multi-armed bandits**: Explore new items, exploit known good ones.

**Thompson Sampling**: Naturally explores new items (high uncertainty).

```python
# New item has uniform prior Beta(1, 1)
# Gets explored more than well-known items
```

---

## What Can Go Wrong

### 1. Survey Fatigue

**Problem**: Asking too many onboarding questions causes users to quit.

**The math**:

$$P(\text{user completes onboarding}) \approx e^{-\gamma \cdot n_{\text{questions}}}$$

where $\gamma \approx 0.15$ for typical users.

| Questions | Completion Rate |
|-----------|----------------|
| 3 | 64% |
| 5 | 47% |
| 10 | 22% |
| 20 | 5% |

**Solution**: Maximum 3-5 questions, use "skip" option, make it feel like a game.

---

### 2. Demographic Stereotyping

**Problem**: Over-relying on demographics leads to unfair stereotypes.

**Example**:
```
System: "User is 65-year-old male -> recommend golf and war documentaries"
Reality: User loves K-pop and anime

Impact: User feels misunderstood, reduced trust, potential legal issues
```

**Why it happens**:
- $P(\text{likes war docs} | \text{65M}) = 0.3$ in training data
- But $P(\text{65M} | \text{likes war docs})$ is different (Bayes' theorem!)
- System conflates correlation with causation

**Solution**:
- Use demographics as weak prior, not strong constraint
- Quickly adapt based on actual behavior
- Audit for demographic bias

---

### 3. Content Feature Quality

**Problem**: Content-based fallback only works if features are good.

**Example**:
```
Movie metadata says: Genre = "Drama"
Reality: It's a horror film with dramatic elements

Content-based system: Recommends to drama lovers
Result: Horror-hating user gets scared, churns
```

**Feature quality issues**:
- Missing metadata (new indie films)
- Incorrect labels (user-generated tags)
- Stale features (actors not updated)
- Ambiguous categories ("thriller" vs "action")

**Solution**:
- Multi-modal features (text + images + audio)
- Pre-trained embeddings (BERT, CLIP)
- Human curation for important items
- Feedback loops to correct errors

---

### 4. Cold-Start-Cold-Start: New User + New Item

**The nightmare scenario**:

```
New user Sarah (0 interactions) + New movie "Cosmic Dawn" (0 ratings)

Content-based: "Sarah's profile is empty, can't compute similarity"
Collaborative: "No interaction data for either"
Demographics: "Generic recommendations only"
```

**This is the hardest case** - quadratic uncertainty:

$$\text{Uncertainty} \propto \sigma_u^2 + \sigma_i^2 + \sigma_u^2 \sigma_i^2$$

**Solutions**:
1. **Never cold-start both simultaneously** - Show new items to established users first
2. **Transfer learning** - Use external data (IMDB ratings, social media buzz)
3. **Editorial curation** - Human-selected "featured" items for new users
4. **Contextual bandits** - Treat as exploration with massive uncertainty bonus

---

### 5. Exploration vs Exploitation Tradeoff

**Problem**: Too much exploration annoys users, too little prevents learning.

**The regret bound**:

$$\text{Regret}(T) \geq \Omega(\sqrt{T \cdot \log K})$$

where $T$ = time horizon, $K$ = number of arms (items).

**Translation**: You WILL make mistakes while learning. The question is how many.

**Solution**: UCB or Thompson Sampling with carefully tuned exploration parameter.

---

## Transition Timeline: From Cold to Warm

### Numerical Example: Sarah's First 50 Interactions

**Day 0: Pure Cold Start**

```
Interaction count: n = 0
Weight w(0) = 20/(20+0) = 1.0

Recommendation strategy: 100% content-based
- Uses demographics + onboarding survey
- Shows popular items in preferred genres
```

**Day 1: First interactions (n = 3)**

```
Sarah watched:
  - "Inception" (rated 5)
  - "The Matrix" (rated 4)
  - "Tenet" (rated 3)

Weight w(3) = 20/(20+3) = 0.87

Recommendation mix:
  - 87% content-based (similar to sci-fi/thriller)
  - 13% collaborative (limited signal)

CF starts learning:
  - User embedding: p_Sarah = [0.8, 0.3, 0.1, ...] (noisy)
  - Variance: Var(p_Sarah) = 0.33 (high uncertainty)
```

**Day 3: Building history (n = 10)**

```
Sarah has watched 10 movies
Genres: 7 sci-fi, 2 drama, 1 comedy

Weight w(10) = 20/(20+10) = 0.67

Recommendation mix:
  - 67% content-based
  - 33% collaborative

CF status:
  - User embedding more stable
  - Variance: 0.18 (moderate uncertainty)
  - Can start finding "users like Sarah"
```

**Week 1: Approaching threshold (n = 20)**

```
Sarah has 20 interactions
Clear pattern: Loves Nolan, Villeneuve; dislikes rom-coms

Weight w(20) = 20/(20+20) = 0.50

Recommendation mix:
  - 50% content-based (still important for new items)
  - 50% collaborative (now reliable)

CF status:
  - Found similar users: Bob, Alice, Charlie
  - Variance: 0.10 (acceptable)
  - Collaborative signals becoming dominant
```

**Week 2: Warm user (n = 35)**

```
Weight w(35) = 20/(20+35) = 0.36

Recommendation mix:
  - 36% content-based (for long-tail items)
  - 64% collaborative (main driver)

Sarah is no longer "cold":
  - CF works well
  - Can discover items unlike her history (serendipity)
  - Content-based still helps for brand new releases
```

**Month 1: Established user (n = 50)**

```
Weight w(50) = 20/(20+50) = 0.29

Recommendation mix:
  - 29% content-based
  - 71% collaborative

Sarah's position in system:
  - Robust user embedding
  - Part of user neighborhood
  - Influencing recommendations for new users like her
  - Variance: 0.04 (low uncertainty)
```

**Visual Timeline**:

```
Strategy Mix over Time

100% |CB****
     |    **
 75% |      **CB
     |        **
 50% |          **------ CF starts dominating
     |            **
 25% |        CF****
     |      **    **
  0% |****        ****
     +-------------------> Interactions
       0   10   20   50

Legend: CB = Content-Based, CF = Collaborative Filtering
```

**Transition Table**:

| Interactions | CB Weight | CF Weight | Strategy |
|-------------|-----------|-----------|----------|
| 0 | 100% | 0% | Demographics + Onboarding |
| 5 | 80% | 20% | Content-based dominant |
| 10 | 67% | 33% | Mixed, CB preferred |
| 20 | 50% | 50% | Equal weight (threshold) |
| 35 | 36% | 64% | CF dominant |
| 50 | 29% | 71% | Mostly CF |
| 100 | 17% | 83% | Established user |
| 500 | 4% | 96% | Power user |

---

## Summary

**Key Takeaways**:
1. **User cold start**: Onboarding, demographics, hybrid
2. **Item cold start**: Content-based, exploration, transfer learning
3. **Meta-learning**: MAML for fast adaptation
4. **Bandits**: Natural exploration of new items

**Mathematical Insights**:
1. **Cold start threshold**: $\tau \approx 2k$ to $5k$ where $k$ = latent dimensions
2. **Hybrid weighting**: $w(n) = \frac{\tau}{\tau + n}$ provides smooth transition
3. **Active learning**: Entropy maximization selects diverse items
4. **Information bound**: Each question provides ~70% of previous question's value

**What Can Go Wrong**:
1. Survey fatigue (too many questions)
2. Demographic stereotyping (over-reliance on demographics)
3. Content feature quality (garbage in, garbage out)
4. Cold-start-cold-start (new user + new item)
5. Exploration-exploitation tradeoff

**Best Strategy**: Combine multiple approaches
- **Day 1**: Demographics + onboarding
- **Week 1**: Hybrid (content + emerging CF signal)
- **Month 1+**: Full CF

**Next**: Model management and MLOps.

---

## References

1. **Vartak, M., et al. (2017)**. "Meta-Learning for User Cold-Start Recommendation". *IJCNN*.
2. **Finn, C., et al. (2017)**. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks". *ICML*.
3. **Schein, A. I., et al. (2002)**. "Methods and Metrics for Cold-Start Recommendations". *SIGIR*.
4. **Lika, B., et al. (2014)**. "Facing the cold start problem in recommender systems". *Expert Systems with Applications*.
5. **Rashid, A. M., et al. (2008)**. "Learning Preferences of New Users in Recommender Systems". *SIGIR*.
