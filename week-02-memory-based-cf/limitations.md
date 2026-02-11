# Week 2: Limitations of Memory-Based Methods

## Opening Question

> **"Memory-based collaborative filtering was a breakthrough that made personalized recommendations possible. So why did the entire field move on to something else?"**

Think about it. User-based CF powers the intuition "users like you enjoyed this." Item-based CF gave us Amazon's famous "customers who bought X also bought Y." These methods are interpretable, they work, and they were the foundation of recommendation systems for over a decade.

**Yet every major recommendation system today uses model-based approaches.**

Netflix? Matrix factorization and deep learning. YouTube? Two-tower neural networks. Spotify? Graph neural networks combined with matrix factorization. TikTok? Complex deep learning models.

**This lecture answers the question: What went wrong with memory-based CF?**

The answer lies in three fundamental barriers that no amount of clever engineering can overcome:
1. **Scalability** - The math doesn't work at internet scale
2. **Sparsity** - Most users have rated almost nothing
3. **Representation** - Raw ratings can't capture why users like things

Understanding these limitations isn't just academic. It's the motivation for everything we'll study in Weeks 3-19. By the end of this lecture, you'll understand exactly why matrix factorization was inevitable.

---

## Learning Objectives

By the end of this lecture, you will:
- Derive the exact computational complexity that makes memory-based CF infeasible at scale
- Calculate memory requirements for real-world systems (Netflix, Amazon)
- Understand why sparsity fundamentally breaks similarity computation
- Recognize the inability to learn latent representations as a core limitation
- See clearly why model-based methods were developed
- Be prepared to appreciate the elegance of matrix factorization (Week 3)

---

## 1. The Scalability Wall

### The Problem We're Solving

Let's start with the basic question: **How long does it take to make a recommendation?**

Memory-based methods need to compute similarity between users (or items) to find neighbors. Let's derive exactly how expensive this is.

### Computational Complexity Derivation

#### User-Based CF: Full Analysis

**Step 1: Computing similarity between two users**

For users $u$ and $v$, we need to compute (using Pearson correlation):

$$\text{sim}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}$$

This requires iterating over all items in $I_{uv}$ (items both users rated).

**Worst case**: Both users rated all items $\rightarrow O(n_i)$ operations per pair

**Step 2: Computing all pairwise similarities**

Number of user pairs: $\binom{n_u}{2} = \frac{n_u(n_u-1)}{2} \approx O(n_u^2)$

Each pair requires $O(n_i)$ operations.

**Total complexity for precomputing all similarities:**
$$\boxed{O(n_u^2 \cdot n_i)}$$

**Step 3: Finding k-nearest neighbors**

For each user, sort all other users by similarity: $O(n_u \log n_u)$

For all users: $O(n_u^2 \log n_u)$

**Total User-Based CF complexity:**
$$O(n_u^2 \cdot n_i + n_u^2 \log n_u) = O(n_u^2 \cdot n_i)$$

#### Item-Based CF: Full Analysis

By symmetry:

**Total Item-Based CF complexity:**
$$\boxed{O(n_i^2 \cdot n_u)}$$

### Real Numbers: Netflix Scale Analysis

Let's calculate what this means for Netflix (circa 2020):

**Netflix Numbers:**
- Users ($n_u$): 200 million = $2 \times 10^8$
- Items ($n_i$): 15,000 titles = $1.5 \times 10^4$

**User-Based CF:**
$$n_u^2 \cdot n_i = (2 \times 10^8)^2 \times (1.5 \times 10^4)$$
$$= 4 \times 10^{16} \times 1.5 \times 10^4$$
$$= 6 \times 10^{20} \text{ operations}$$

**How long would this take?**

Modern server: $\sim 10^{10}$ floating-point operations per second (10 GFLOPS realistic for this workload)

$$\text{Time} = \frac{6 \times 10^{20}}{10^{10}} = 6 \times 10^{10} \text{ seconds}$$

$$= \frac{6 \times 10^{10}}{3.15 \times 10^7} \approx 1,900 \text{ years}$$

> **Pause and absorb that: Computing user-based CF similarities for Netflix would take almost 2,000 years on a single server.**

Even with 1,000 servers running in parallel: **1.9 years** just to precompute similarities once.

**Item-Based CF:**
$$n_i^2 \cdot n_u = (1.5 \times 10^4)^2 \times (2 \times 10^8)$$
$$= 2.25 \times 10^8 \times 2 \times 10^8$$
$$= 4.5 \times 10^{16} \text{ operations}$$

$$\text{Time} = \frac{4.5 \times 10^{16}}{10^{10}} = 4.5 \times 10^{6} \text{ seconds} \approx 52 \text{ days}$$

**Item-based is better** (because $n_i \ll n_u$ for most systems), but still impractical for daily updates.

### Amazon Scale: Truly Intractable

**Amazon Numbers:**
- Users: 300 million = $3 \times 10^8$
- Products: 350 million = $3.5 \times 10^8$

**User-Based CF:**
$$n_u^2 \cdot n_i = (3 \times 10^8)^2 \times (3.5 \times 10^8) = 3.15 \times 10^{25} \text{ operations}$$

$$\text{Time} = \frac{3.15 \times 10^{25}}{10^{10}} = 3.15 \times 10^{15} \text{ seconds} \approx 100 \text{ million years}$$

**Item-Based CF:**
$$n_i^2 \cdot n_u = (3.5 \times 10^8)^2 \times (3 \times 10^8) = 3.67 \times 10^{25} \text{ operations}$$

Similarly intractable.

> **The fundamental issue**: Memory-based CF has polynomial complexity in the number of users AND items. Internet-scale means both numbers are huge.

### Socratic Moment: The Compression Insight

> **"What if, instead of storing the entire user-item matrix and computing similarity over raw ratings, we could compress each user into a small vector of, say, 50 numbers? And each item into another 50 numbers?"**

Then the computational complexity becomes:
- Storage: $O(k \cdot (n_u + n_i))$ instead of $O(n_u \cdot n_i)$
- Similarity: $O(k)$ per pair instead of $O(n_i)$

Where $k = 50 \ll n_i$.

**This is exactly what matrix factorization does.** But we're getting ahead of ourselves. First, let's understand more limitations.

---

## 2. Memory Requirements: The Storage Impossibility

### The Full Matrix Problem

Memory-based CF requires storing (or being able to access) the user-item interaction matrix.

**Storage for dense matrix:**
$$\text{Memory} = n_u \times n_i \times \text{bytes per entry}$$

Using 4-byte floats:

### Netflix Memory Requirements

$$\text{Memory} = 200M \times 15K \times 4 \text{ bytes}$$
$$= 2 \times 10^8 \times 1.5 \times 10^4 \times 4$$
$$= 1.2 \times 10^{13} \text{ bytes}$$
$$= 12 \text{ TB (terabytes)}$$

Just to store the ratings matrix in memory.

### Amazon Memory Requirements

$$\text{Memory} = 300M \times 350M \times 4 \text{ bytes}$$
$$= 3 \times 10^8 \times 3.5 \times 10^8 \times 4$$
$$= 4.2 \times 10^{17} \text{ bytes}$$
$$= 420 \text{ PB (petabytes)}$$

> **For context**: The entire data stored by Google in 2020 was estimated at 15 exabytes. Storing Amazon's full user-item matrix would require 42% of that.

### "But It's Sparse!"

True, we can use sparse matrix representations. Let's recalculate.

**Netflix sparsity**: ~98.8% (users rate ~1.2% of content)

**Actual ratings**: $\sim 100M$ ratings

**Sparse storage**: Each rating needs (user_id, item_id, rating) = ~12 bytes

$$\text{Sparse memory} = 100M \times 12 \text{ bytes} = 1.2 \text{ GB}$$

Much better. But here's the catch:

### The Similarity Matrix Problem

We also need to store precomputed similarities.

**Item-item similarity matrix** (for item-based CF):
$$\text{Memory} = n_i \times n_i \times 4 \text{ bytes}$$

For Netflix:
$$= 15K \times 15K \times 4 = 900 \text{ MB}$$

Manageable.

For Amazon:
$$= 350M \times 350M \times 4 = 4.9 \times 10^{17} \text{ bytes} = 490 \text{ PB}$$

Still impossible.

**User-user similarity matrix**:
$$= 200M \times 200M \times 4 = 1.6 \times 10^{17} \text{ bytes} = 160 \text{ PB}$$

Also impossible.

### The Latency Trap

Even if we could store everything, **real-time recommendation requires sub-100ms response times**.

**Option 1: Precompute and store all similarities**
- Space: Impossible at scale (as shown above)

**Option 2: Compute on-the-fly**
- Finding neighbors for one user: $O(n_u \cdot n_i)$ operations
- For Netflix: $200M \times 15K = 3 \times 10^{12}$ operations
- At 10 GFLOPS: 300 seconds = **5 minutes** per recommendation

Neither option works.

### Why Model-Based Wins on Memory

Matrix factorization stores:
- User embeddings: $n_u \times k$ floats
- Item embeddings: $n_i \times k$ floats

With $k = 100$ dimensions:

**Netflix:**
$$= (200M + 15K) \times 100 \times 4 \text{ bytes}$$
$$\approx 200M \times 100 \times 4 = 80 \text{ GB}$$

**Amazon:**
$$= (300M + 350M) \times 100 \times 4$$
$$= 650M \times 400 = 260 \text{ GB}$$

**260 GB vs 490 PB** - a factor of 1.8 million improvement.

---

## 3. The Sparsity Crisis

### Understanding Sparsity

**Definition**: Sparsity is the fraction of entries in the user-item matrix that are missing.

$$\text{Sparsity} = 1 - \frac{\text{number of ratings}}{n_u \times n_i}$$

**Typical Values:**
| System | Users | Items | Ratings | Sparsity |
|--------|-------|-------|---------|----------|
| MovieLens 100K | 943 | 1,682 | 100K | 93.7% |
| MovieLens 1M | 6,040 | 3,706 | 1M | 95.5% |
| Netflix Prize | 480K | 17K | 100M | 98.8% |
| Netflix (full) | 200M | 15K | ~3B | 99.9% |
| Amazon | 300M | 350M | ~10B | 99.9999% |

### The Overlap Problem: A Numerical Deep Dive

**The Setup:**

Consider two users, Alice and Bob, each with 10 ratings on a platform with 10,000 items.

**Question**: What's the probability they have rated at least one item in common?

**Calculation:**

Probability that Bob has NOT rated any of Alice's 10 items:
- For each of Alice's items, probability Bob didn't rate it: $\frac{9990}{10000}$
- For all 10 items: $\left(\frac{9990}{10000}\right)^{10} = (0.999)^{10} \approx 0.99$

Wait, that's not quite right. Let's be more precise.

Bob randomly rates 10 items from 10,000. Probability none overlap with Alice's 10:

$$P(\text{no overlap}) = \frac{\binom{9990}{10}}{\binom{10000}{10}}$$

Using the approximation for large numbers:

$$\approx \left(\frac{9990}{10000}\right)^{10} \times \left(\frac{9990}{10000}\right)^{10} / \text{correction}$$

More simply: Expected number of overlapping items:

$$E[\text{overlap}] = \frac{10 \times 10}{10000} = 0.01$$

**On average, two users who each rated 10 items from 10,000 will have 0.01 items in common.**

**Probability of at least one overlap** (using Poisson approximation):

$$P(\text{overlap} \geq 1) = 1 - e^{-0.01} \approx 0.01 = 1\%$$

> **Only 1% of user pairs have ANY overlap.** The other 99% have similarity = undefined.

### Scaling Up the Example

**Netflix-like scenario:**
- Users rate on average 200 items out of 15,000

$$E[\text{overlap}] = \frac{200 \times 200}{15000} = 2.67 \text{ items}$$

$$P(\text{overlap} \geq 1) \approx 1 - e^{-2.67} \approx 93\%$$

Better, but...

$$P(\text{overlap} \geq 10) \approx 1 - \sum_{k=0}^{9} \frac{2.67^k e^{-2.67}}{k!} \approx 3\%$$

> **Only 3% of user pairs have 10+ items in common** - the minimum for reliable similarity estimates.

### Why Small Overlap Destroys Similarity

**Statistical Problem:**

Correlation based on 2-3 data points has enormous variance.

**Example:**

Alice and Bob both rated only 2 movies in common:
- Inception: Alice 5, Bob 4
- Titanic: Alice 4, Bob 5

Pearson correlation:
$$\text{sim}(A, B) = \frac{(5-4.5)(4-4.5) + (4-4.5)(5-4.5)}{0.5 \times 0.5} = \frac{-0.25 - 0.25}{0.25} = -2$$

Wait, that's outside [-1, 1]. Let's recalculate properly:

$$\bar{r}_A = 4.5, \quad \bar{r}_B = 4.5$$
$$\text{sim} = \frac{(5-4.5)(4-4.5) + (4-4.5)(5-4.5)}{\sqrt{0.25 + 0.25}\sqrt{0.25 + 0.25}}$$
$$= \frac{-0.25 - 0.25}{0.5 \times 0.5} = \frac{-0.5}{0.707 \times 0.707} = \frac{-0.5}{0.5} = -1$$

**Perfect negative correlation!** Based on 2 data points where the difference is just 1 star.

Now imagine they had both rated the same:
- Inception: Alice 5, Bob 5
- Titanic: Alice 4, Bob 4

Pearson correlation: **undefined** (division by zero - no variance in the difference).

**The lesson**: Small overlap leads to either undefined, extremely noisy, or misleading similarity scores.

### Mathematical Formalization of the Sparsity Problem

Let $S$ be the set of co-rated items between users $u$ and $v$.

**Variance of estimated correlation** (assuming ratings are i.i.d.):

$$\text{Var}(\hat{\rho}) \approx \frac{(1 - \rho^2)^2}{|S|}$$

For $|S| = 2$ and true $\rho = 0.5$:

$$\text{Var}(\hat{\rho}) \approx \frac{(1 - 0.25)^2}{2} = \frac{0.5625}{2} = 0.28$$

Standard deviation: $\sqrt{0.28} \approx 0.53$

**95% confidence interval**: $0.5 \pm 1.96 \times 0.53 = [-0.54, 1.54]$

The interval spans from strong negative to impossibly positive. **The estimate is meaningless.**

For $|S| = 100$:
$$\text{Var}(\hat{\rho}) \approx \frac{0.5625}{100} = 0.0056$$

Standard deviation: $0.075$

**95% CI**: $[0.35, 0.65]$ - now useful.

> **Minimum overlap for reliable similarity**: ~50-100 co-rated items

At typical sparsity levels, this threshold is rarely met.

---

## 4. The Representation Problem: No Feature Learning

### What Memory-Based CF Actually Stores

Memory-based CF stores raw ratings:
```
User 123: {Inception: 5, Matrix: 4, Titanic: 3, ...}
User 456: {Avatar: 5, Interstellar: 4, Gravity: 5, ...}
```

**What can we compute?**
- Similarity based on overlapping ratings (if any exist)
- Weighted average of neighbor ratings

**What CAN'T we compute?**
- Why User 123 likes Inception (is it sci-fi? action? Nolan?)
- Whether User 123 would like a NEW sci-fi movie they haven't rated
- Hidden patterns connecting users and items

### The Latent Factor Insight

**Observation**: Users don't rate movies randomly. There are underlying factors:
- Genre preferences (action, comedy, drama)
- Style preferences (art-house vs. blockbuster)
- Actor/director preferences
- Mood-based preferences

**Memory-based CF cannot learn these factors.** It only sees:
- "User A rated Movie X highly"
- "User B also rated Movie X highly"
- Therefore: "A and B might be similar"

It cannot learn:
- "User A likes sci-fi action"
- "User B likes sci-fi action"
- Therefore: "A and B will both like this NEW sci-fi action movie"

### The Generalization Gap

**Example:**

**User profile (what memory-based CF sees):**
```
User liked: Inception (5), Matrix (5), Interstellar (5), Dark Knight (5)
User disliked: The Notebook (1), Titanic (2), Romeo & Juliet (1)
```

**Human interpretation:**
- Loves: Sci-fi, Christopher Nolan, action
- Hates: Romance

**Memory-based CF interpretation:**
- User is similar to others who rated these exact movies similarly

**New movie arrives: "Tenet" (Christopher Nolan, sci-fi action)**

No one has rated it yet.

- **Memory-based CF**: Cannot recommend (no ratings to compute similarity)
- **Model-based CF**: User's embedding is close to sci-fi/action/Nolan factors. Tenet's embedding (from content) is also close. Recommend with high confidence.

### The Mathematical View

**Memory-based representation:**
$$\mathbf{u} \in \mathbb{R}^{n_i} \quad \text{(one dimension per item, mostly zeros)}$$

**Model-based representation:**
$$\mathbf{u} \in \mathbb{R}^{k} \quad \text{(k latent factors, dense)}$$

Where $k \ll n_i$ (typically 20-200 vs. thousands to millions).

**Key insight**: The model-based representation **compresses** the information into a dense, meaningful vector that captures WHY users like things, not just WHAT they rated.

---

## 5. Latency Under the Microscope

### Real-Time Recommendation Requirements

Modern recommendation systems must return results in **<100ms** (ideally <50ms).

Let's break down what memory-based CF needs to do at request time:

### User-Based CF: Request-Time Computation

**Option A: Precomputed neighbors (but we showed this is impossible to store)**

**Option B: Compute on-the-fly**

For user $u$ requesting recommendations:

1. **Compute similarity with all other users**: $O(n_u \cdot n_i)$
2. **Sort to find top-k neighbors**: $O(n_u \log k)$
3. **Aggregate neighbor ratings**: $O(k \cdot n_i)$
4. **Sort items by predicted score**: $O(n_i \log n_i)$

**Dominant term**: $O(n_u \cdot n_i)$

**Netflix numbers:**
$$200M \times 15K = 3 \times 10^{12} \text{ operations}$$

At 10 GFLOPS: **300 seconds = 5 minutes**

This is for a SINGLE recommendation request.

### Item-Based CF: Better but Still Problematic

**Approach**: Precompute item-item similarities (only $n_i^2$)

For Netflix: $15K \times 15K = 225M$ entries = 900 MB (storable)

**At request time:**

1. Look up items user has rated: $O(|R_u|)$ where $|R_u|$ = items rated
2. For each rated item, look up similar items: $O(|R_u| \cdot k)$
3. Aggregate and deduplicate: $O(|R_u| \cdot k)$

With $|R_u| = 200$ rated items, $k = 50$ neighbors: $O(10,000)$ operations

**This is fast!** Item-based CF at inference time is $O(|R_u| \cdot k)$, which is constant w.r.t. system size.

**But**:
1. Precomputation is still $O(n_i^2 \cdot n_u)$ - days of computation
2. For Amazon with 350M items, storing item-item similarities is impossible
3. New items have no similarities computed

### Model-Based: The Latency Winner

**At request time:**
1. Look up user embedding: $O(1)$
2. Compute dot product with all item embeddings: $O(n_i \cdot k)$
3. Or use ANN (approximate nearest neighbor): $O(k \cdot \log n_i)$

**Netflix with ANN**: $100 \times \log(15K) \approx 100 \times 14 = 1,400$ operations

**Amazon with ANN**: $100 \times \log(350M) \approx 100 \times 28 = 2,800$ operations

**Latency**: <1ms with proper infrastructure

---

## 6. Additional Limitations

### 6.1 Cold Start Problem

**New User Cold Start**

**Problem**: New user has zero ratings.

**Memory-Based CF Response:**
- User-based: Cannot find similar users (no ratings to compute similarity)
- Item-based: Cannot recommend (no user preferences known)

**Fallback Strategies:**
- Show popular items (non-personalized)
- Ask user to rate a few items (onboarding)
- Use demographic information (not part of pure CF)

**Limitation**: Memory-based CF has **no principled way** to handle new users without ratings.

**New Item Cold Start**

**Problem**: New item has zero ratings.

**Memory-Based CF Response:**
- User-based: Can still recommend (based on user similarity), but won't recommend new item
- Item-based: Cannot recommend new item (no item similarity computed yet)

**Impact:**
- New movies, products, content get **zero exposure**
- Rich-get-richer effect (popular items stay popular)

### 6.2 No Incorporation of Side Information

**Limitation:**

Memory-based CF uses **only the user-item rating matrix**.

**What's Ignored:**
- **User features**: Age, gender, location, occupation
- **Item features**: Genre, director, actors, price, brand
- **Context**: Time, device, location, weather
- **Sequential patterns**: Order of interactions matters

**Example: Ignoring Context**

**Scenario**: User watches comedies on weekends, documentaries on weekdays.

**Memory-Based CF:**
- Treats all ratings equally
- **Cannot capture** temporal or contextual patterns
- Recommends comedies on Monday (suboptimal)

**Context-Aware Approaches (model-based):**
- Tensor factorization: User x Item x Context
- Contextual bandits
- Sequential models (RNNs, Transformers)

### 6.3 Inability to Handle Implicit Feedback

**The Challenge:**

Memory-based CF was designed for **explicit feedback** (ratings: 1-5 stars).

**Implicit Feedback** (clicks, views, purchases):
- No negative examples (only positive or missing)
- Missing does not equal negative (user may not know item exists)
- Varying confidence (10 views vs. 1 view)

**Why Memory-Based Struggles:**

```
User -> [item_1: click, item_2: click, ..., item_100000: no click]
```

**Problem:**
- How to compute similarity with binary data?
- How to interpret "no click"? (unknown vs. dislike)

**Cosine similarity with binary data:**
- Only considers overlap of positive items
- Ignores magnitude of preference
- Treats all clicks equally (1 view = 100 views)

**Better Approaches (model-based):**
- Weighted matrix factorization (Hu et al., 2008)
- Bayesian Personalized Ranking (BPR)
- Neural CF with implicit feedback

### 6.4 Popularity Bias

**The Problem:**

Memory-based CF suffers from **popularity bias**:
- Popular items get more ratings
- More ratings -> more similar items -> more recommendations
- Rich-get-richer feedback loop

**Example: MovieLens**

**Popular Movie (e.g., "The Shawshank Redemption"):**
- 50,000 ratings
- High similarity with many items
- Frequently recommended

**Niche Movie (e.g., indie film):**
- 100 ratings
- Low similarity with most items (insufficient overlap)
- Rarely recommended

**Consequences:**
- **Filter bubble**: Users only see popular items
- **Long-tail items** get no exposure
- **Diversity** decreases
- **Serendipity** (surprising discoveries) is lost

### 6.5 Shilling Attack Vulnerability

**Vulnerability:**

Memory-based CF is vulnerable to **shilling attacks** (fake profiles).

**Attack Scenario:**
- Competitor creates fake user accounts
- Fake users give low ratings to competitor products
- Fake users give high ratings to own products

**Impact on Item-Based CF:**
- Fake ratings inflate item similarity
- Malicious items get recommended more
- Legitimate items get downranked

**Why Memory-Based Is Vulnerable:**
- No model validation or anomaly detection
- Ratings are directly used (not learned/filtered)
- No outlier rejection

**Model-Based Defenses:**
- Regularization (penalizes extreme patterns)
- Outlier detection
- Robust matrix factorization

---

## 7. The Complete Comparison: Memory-Based vs. Model-Based

| **Aspect** | **Memory-Based CF** | **Model-Based CF (Preview)** |
|------------|---------------------|------------------------------|
| **Complexity** | $O(n_u^2 \cdot n_i)$ or $O(n_i^2 \cdot n_u)$ | $O(k \cdot (n_u + n_i))$ per epoch |
| **Training Time** | None (lazy learning) | Minutes to hours |
| **Prediction Time** | $O(n_u)$ or $O(n_i)$ | $O(k)$ or $O(\log n)$ with ANN |
| **Memory: Ratings** | $O(\|R\|)$ sparse | $O(\|R\|)$ sparse |
| **Memory: Model** | $O(n_u^2)$ or $O(n_i^2)$ for similarities | $O(k \cdot (n_u + n_i))$ |
| **Sparsity Handling** | Very sensitive | Robust (learns latent patterns) |
| **Feature Learning** | No | Yes (latent factors) |
| **Generalization** | Only to seen items/users | Can extrapolate patterns |
| **New Users/Items** | No solution | Partial (hybrid, features) |
| **Implicit Feedback** | Limited support | Designed for it (BPR, WRMF) |
| **Side Information** | Cannot use | Can incorporate |
| **Explainability** | High ("users who liked X") | Low (latent factors opaque) |
| **Attack Robustness** | Vulnerable | More robust (regularization) |

### Complexity Comparison with Real Numbers

| **System** | **Memory-Based (User)** | **Memory-Based (Item)** | **Model-Based (k=100)** |
|------------|------------------------|-------------------------|------------------------|
| **MovieLens 1M** | $10^{11}$ ops | $10^{10}$ ops | $10^{8}$ ops/epoch |
| **Netflix** | $10^{21}$ ops | $10^{17}$ ops | $10^{11}$ ops/epoch |
| **Amazon** | $10^{26}$ ops | $10^{26}$ ops | $10^{11}$ ops/epoch |

---

## 8. When Memory-Based CF Still Makes Sense

Despite limitations, memory-based CF is useful in certain scenarios:

### Good Use Cases

1. **Small datasets** (<10K users, <10K items)
   - Computational cost is manageable
   - Sparsity is often lower

2. **High-density matrices**
   - Users rate many items (e.g., internal employee tools)
   - Sparsity <80%

3. **Explainability is critical**
   - Medical, legal, financial domains
   - Users need to understand recommendations
   - "Users like you also chose this treatment"

4. **Rapidly changing catalogs**
   - News, events (items have short lifespan)
   - No time to retrain models
   - New item arrives -> immediately in similarity computation

5. **Quick prototyping**
   - Baseline for comparison
   - No infrastructure for model training
   - Prove concept before investing in ML infrastructure

---

## 9. Why This Matters: The Bridge to Week 3

### The Core Insight

Memory-based CF fails because it:
1. **Stores too much** (all ratings, all similarities)
2. **Computes too much** (pairwise comparisons)
3. **Learns nothing** (no latent structure)

### The Solution Preview

> **"What if we could represent each user as a small vector of 50-100 numbers that captures their preferences? And each item as another small vector that captures its characteristics?"**

**Then:**
- Storage: $O(k \cdot (n_u + n_i))$ instead of $O(n_u \cdot n_i)$
- Similarity: $O(k)$ per pair instead of $O(n_i)$
- Prediction: Simple dot product $\hat{r}_{ui} = \mathbf{p}_u^T \mathbf{q}_i$

**And crucially:**
- The vectors can capture WHY users like items
- They can generalize to new combinations
- They can be learned from sparse data

### Matrix Factorization in One Equation

$$\underbrace{\mathbf{R}}_{n_u \times n_i} \approx \underbrace{\mathbf{P}^T}_{n_u \times k} \times \underbrace{\mathbf{Q}}_{k \times n_i}$$

This is what we'll study in Week 3. Every limitation we discussed today is addressed:

| Limitation | How MF Addresses It |
|------------|---------------------|
| Scalability | $O(k)$ operations instead of $O(n)$ |
| Memory | Store $2k(n_u + n_i)$ instead of $n_u \cdot n_i$ |
| Sparsity | Learn patterns that generalize across gaps |
| Feature learning | Latent factors capture hidden structure |
| Cold start | Can incorporate features (Factorization Machines) |

### The Netflix Prize Validation

In 2006, Netflix offered $1 million to improve their recommendations by 10%.

**What won?** Matrix factorization and its variants (SVD++, TimeSVD++).

**Key insight from winners:**
> "The fundamental limitation of nearest-neighbor methods is their inability to learn latent factors. Matrix factorization discovers these hidden patterns automatically."

---

## Practice Problems

**Problem 1: Complexity Analysis**

Calculate the computational complexity of user-based CF for a system with 1 million users and 100K items. How long would it take to compute all user-user similarities on a machine that can do 1 billion operations/second?

**Problem 2: Memory Calculation**

For a system with 50 million users and 1 million items:
a) Calculate the memory needed to store the full user-item matrix (4 bytes per entry)
b) Calculate the memory needed for model-based approach with k=100 latent factors
c) What is the compression ratio?

**Problem 3: Sparsity and Overlap**

Given a user-item matrix with 99% sparsity where each user rates exactly 100 items:
a) Calculate the expected number of co-rated items between two random users
b) What fraction of user pairs have at least 5 co-rated items? (Use Poisson approximation)
c) Why does this matter for similarity computation?

**Problem 4: System Design**

A startup has 10,000 users and 50,000 products. Should they use memory-based or model-based CF? Justify with complexity analysis.

**Problem 5: Shilling Attack**

Describe a shilling attack scenario for an e-commerce site. How would you detect and prevent such attacks? Why are model-based methods more robust?

---

## Summary

Memory-based CF has fundamental limitations:

1. **Scalability**: $O(n^2)$ complexity is intractable at internet scale
2. **Memory**: Cannot store or precompute similarities for large systems
3. **Sparsity**: Most user pairs have insufficient overlap for reliable similarity
4. **No Learning**: Cannot discover latent factors or generalize beyond observed ratings
5. **Cold Start**: No principled handling of new users/items
6. **Latency**: Real-time computation is infeasible

These limitations motivated the development of **model-based collaborative filtering**, which we begin studying in Week 3 with matrix factorization.

---

## References

1. **Sarwar, B., et al. (2001)**. "Item-Based Collaborative Filtering Recommendation Algorithms". *WWW*.
   - Analysis of item-based CF scalability

2. **Herlocker, J. L., et al. (2004)**. "Evaluating Collaborative Filtering Recommender Systems". *ACM TOIS*.
   - Comprehensive coverage of CF limitations

3. **Koren, Y., et al. (2009)**. "Matrix Factorization Techniques for Recommender Systems". *IEEE Computer*.
   - Motivation for model-based approaches

4. **Su, X., & Khoshgoftaar, T. M. (2009)**. "A Survey of Collaborative Filtering Techniques". *Advances in Artificial Intelligence*.
   - Detailed comparison of memory-based vs. model-based

5. **Lam, S. K., & Riedl, J. (2004)**. "Shilling Recommender Systems for Fun and Profit". *WWW*.
   - Shilling attacks on memory-based CF

6. **Adomavicius, G., & Tuzhilin, A. (2005)**. "Toward the Next Generation of Recommender Systems". *IEEE TKDE*.
   - Context-aware and hybrid methods

---

**Next:** Week 3 introduces **Matrix Factorization**, which addresses all of these limitations by learning low-dimensional latent representations of users and items.

*The question that drives Week 3: How do we find those magical 50-100 numbers that capture a user's preferences?*
