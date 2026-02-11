# Week 2: Item-Based Collaborative Filtering

## Learning Objectives

- Understand item-based collaborative filtering intuition
- Compute item-item similarity matrices
- Recognize advantages over user-based CF
- Implement Amazon's item-to-item approach
- Master precomputation and storage strategies
- Identify failure modes specific to item-based approaches

---

## Opening: Why Might User-Based CF Fail for Large Catalogs?

*Before we learn item-based CF, let me show you exactly what breaks with user-based CF at scale.*

### The Amazon Challenge (Circa 2003): A Concrete Problem

**The numbers that broke user-based CF:**

```
Amazon's scale in 2003:
  - 20 million customers
  - 1 million products
  - Growing every day

User-Based CF would require:
  - User pairs: C(20,000,000, 2) = 199,999,990,000,000 pairs
                                 = 200 TRILLION pairs

  - If each similarity takes 1 microsecond:
    200 trillion * 1 microsecond = 200 million seconds = 6.3 YEARS
```

*Can you see the problem?* Even with aggressive optimization, you're looking at weeks of computation that needs to be repeated whenever user preferences change.

**But here's the deeper issue:** User preferences are UNSTABLE.

```
User behavior timeline:
  January: User buys programming books (learning to code)
  March: User gets married, buys kitchen appliances
  June: User has baby, buys baby products
  December: User's interests shift to parenting books

Result: User similarities computed in January are WRONG by March
        Need to recompute 200 trillion pairs every few weeks
```

**Amazon engineers asked**: *What if we flip the problem?*

---

### The Key Insight: Items Are Stable, Users Are Not

**User-Based**: "Find users like you"
- Users change constantly (marriage, kids, new job, new hobbies)
- Need to recompute frequently
- 20M users = 200T pairs

**Item-Based**: "Find items like what you bought"
- Items don't change (*The Matrix* is always similar to *Inception*)
- Compute once, use for weeks/months
- 1M items = 500B pairs (still big, but 400x smaller!)

**Let me show you this with numbers:**

| Approach | Entities | Pairs | Update Frequency | Feasibility |
|----------|----------|-------|-----------------|-------------|
| User-Based | 20M users | 200T | Weekly (users change) | Impossible |
| Item-Based | 1M items | 500B | Monthly (items stable) | Challenging but doable |

**The ratio**: $\frac{200T}{500B} = 400$

Item-based is **400x more tractable** for this catalog!

---

## Comparing User-Based vs Item-Based: Same Data, Different Approach

*Let me show you both approaches on the exact same dataset so you can see the difference.*

### Our Dataset

```
Rating Matrix (Users x Items):
         Movie1  Movie2  Movie3  Movie4  Movie5
Alice      5       3       4       ?       2
Bob        3       1       2       3       4
Carol      4       3       5       4       ?
Dave       3       3       3       4       3
Eve        1       ?       5       ?       5
```

**Task**: Predict Alice's rating for Movie4.

---

### User-Based Approach (What We Did Before)

**Step 1**: Find users similar to Alice (using their ratings on Movies 1,2,3,5)

```
Alice's ratings: [5, 3, 4, ?, 2]

Comparing to Bob on co-rated items (1,2,3,5):
  Alice: [5, 3, 4, 2], mean = 3.5
  Bob:   [3, 1, 2, 4], mean = 2.5

  Deviations: Alice [1.5, -0.5, 0.5, -1.5], Bob [0.5, -1.5, -0.5, 1.5]

  Pearson = (1.5*0.5 + -0.5*-1.5 + 0.5*-0.5 + -1.5*1.5) / (sqrt(5)*sqrt(5))
          = (0.75 + 0.75 - 0.25 - 2.25) / 5 = -1.0 / 5 = -0.20
```

**Similar calculations for Carol, Dave, Eve...**

**Step 2**: Use neighbors who rated Movie4 to predict Alice's rating.

**Problem**: We computed similarity for ALL user pairs, but most of that work is wasted when predicting for just one item!

---

### Item-Based Approach (The New Way)

**Step 1**: Find items similar to Movie4 (using how users rated them)

```
Movie4's ratings (column): [?, 3, 4, 4, ?]  (Alice and Eve didn't rate it)

Let's compare Movie4 to Movie1:
  Users who rated both: Bob, Carol, Dave

  Movie4 ratings: [3, 4, 4]
  Movie1 ratings: [3, 4, 3]

  User means (for adjusted cosine):
    Bob: (3+1+2+3+4)/5 = 2.6
    Carol: (4+3+5+4)/4 = 4.0
    Dave: (3+3+3+4+3)/5 = 3.2

  Adjusted Movie4: [3-2.6, 4-4.0, 4-3.2] = [0.4, 0.0, 0.8]
  Adjusted Movie1: [3-2.6, 4-4.0, 3-3.2] = [0.4, 0.0, -0.2]

  Adj. Cosine = (0.4*0.4 + 0*0 + 0.8*-0.2) / (sqrt(0.16+0+0.64) * sqrt(0.16+0+0.04))
              = (0.16 - 0.16) / (sqrt(0.8) * sqrt(0.2))
              = 0 / 0.4 = 0.0
```

**Let's try Movie4 vs Movie3:**

```
Users who rated both: Bob, Carol, Dave

Movie4 ratings: [3, 4, 4]
Movie3 ratings: [2, 5, 3]

Adjusted Movie4: [0.4, 0.0, 0.8]
Adjusted Movie3: [2-2.6, 5-4.0, 3-3.2] = [-0.6, 1.0, -0.2]

Adj. Cosine = (0.4*-0.6 + 0*1.0 + 0.8*-0.2) / (sqrt(0.8) * sqrt(0.36+1+0.04))
            = (-0.24 + 0 - 0.16) / (0.894 * 1.183)
            = -0.40 / 1.058 = -0.378
```

**Let's try Movie4 vs Movie5:**

```
Users who rated both: Bob, Dave (Carol and Eve didn't rate both)

Movie4 ratings: [3, 4]
Movie5 ratings: [4, 3]

Bob mean: 2.6, Dave mean: 3.2

Adjusted Movie4: [3-2.6, 4-3.2] = [0.4, 0.8]
Adjusted Movie5: [4-2.6, 3-3.2] = [1.4, -0.2]

Adj. Cosine = (0.4*1.4 + 0.8*-0.2) / (sqrt(0.16+0.64) * sqrt(1.96+0.04))
            = (0.56 - 0.16) / (0.894 * 1.414)
            = 0.40 / 1.264 = 0.316
```

**Item similarities to Movie4:**

| Item | Similarity to Movie4 | Alice's Rating |
|------|---------------------|----------------|
| Movie1 | 0.0 | 5 |
| Movie3 | -0.378 | 4 |
| Movie5 | 0.316 | 2 |

---

### Step 2: Predict Using Similar Items Alice Rated

**Using only positive similarities** (Movie5, sim = 0.316):

$$\hat{r}_{Alice, Movie4} = \frac{0.316 \times 2}{0.316} = 2.0$$

**Using all items** (with absolute similarities):

$$\hat{r}_{Alice, Movie4} = \frac{0.0 \times 5 + 0.378 \times 4 + 0.316 \times 2}{0.0 + 0.378 + 0.316} = \frac{0 + 1.512 + 0.632}{0.694} = \frac{2.144}{0.694} = 3.09$$

**Prediction**: Alice would rate Movie4 approximately **3.1 stars**.

---

### The Key Difference: Where Work Happens

| Aspect | User-Based | Item-Based |
|--------|-----------|------------|
| **What we compare** | Users (rows) | Items (columns) |
| **Similarity direction** | Horizontal slices | Vertical slices |
| **For prediction** | Find similar users, look at their ratings for target item | Find similar items, look at user's ratings for those items |
| **Precomputation** | Hard (users change) | Easy (items stable) |

*Notice*: Item-based lets us precompute ALL item-item similarities once and reuse them!

---

## The Algorithm: Complete Derivation

### Step 1: Compute Item-Item Similarity

**Intuition**: Two items are similar if users who rated one highly also rated the other highly.

**Formula** (Adjusted Cosine - preferred for item-based):

$$\text{sim}(i, j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)(r_{uj} - \bar{r}_u)}{\sqrt{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)^2} \cdot \sqrt{\sum_{u \in U_{ij}} (r_{uj} - \bar{r}_u)^2}}$$

**Why adjusted cosine instead of regular cosine?**

Regular cosine doesn't account for user rating scales:
- Alice always rates 4-5 (generous)
- Bob always rates 1-2 (harsh)
- Both might LOVE the same item, but their raw ratings look different

Adjusted cosine subtracts each user's mean, so we're comparing "how much did this user like this item relative to their baseline?"

### Step 2: Predict Rating

**Formula**:

$$\hat{r}_{ui} = \frac{\sum_{j \in N_k(i) \cap I_u} \text{sim}(i,j) \cdot r_{uj}}{\sum_{j \in N_k(i) \cap I_u} |\text{sim}(i,j)|}$$

**What each term means**:

| Term | Meaning |
|------|---------|
| $N_k(i)$ | The k items most similar to target item $i$ |
| $I_u$ | Items that user $u$ has rated |
| $N_k(i) \cap I_u$ | Similar items that user $u$ actually rated |
| $\text{sim}(i,j)$ | How similar item $j$ is to target item $i$ |
| $r_{uj}$ | User $u$'s rating for item $j$ |

**In plain English**: "Look at items similar to the target. For each one the user rated, weight that rating by similarity. Average the weighted ratings."

---

## Complete Numerical Walkthrough: Building the Item-Item Matrix

*Let me show you the full computation for a small system.*

### Setup

```
Rating Matrix:
         Item1  Item2  Item3  Item4
User1      5      3      ?      4
User2      4      ?      4      5
User3      ?      2      5      ?
User4      3      4      4      3

User means:
  User1: (5+3+4)/3 = 4.0
  User2: (4+4+5)/3 = 4.33
  User3: (2+5)/2 = 3.5
  User4: (3+4+4+3)/4 = 3.5
```

### Computing All Item-Item Similarities

**Item1 vs Item2:**

```
Co-raters: User1, User4

Raw ratings:
  Item1: [5, 3]
  Item2: [3, 4]

User means: [4.0, 3.5]

Adjusted:
  Item1: [5-4.0, 3-3.5] = [1.0, -0.5]
  Item2: [3-4.0, 4-3.5] = [-1.0, 0.5]

Numerator: 1.0*(-1.0) + (-0.5)*0.5 = -1.0 - 0.25 = -1.25
Denom: sqrt(1.0 + 0.25) * sqrt(1.0 + 0.25) = sqrt(1.25) * sqrt(1.25) = 1.25

sim(Item1, Item2) = -1.25 / 1.25 = -1.0
```

*Perfect negative correlation!* When users rate Item1 high, they rate Item2 low.

---

**Item1 vs Item3:**

```
Co-raters: User2, User4

Raw ratings:
  Item1: [4, 3]
  Item3: [4, 4]

User means: [4.33, 3.5]

Adjusted:
  Item1: [4-4.33, 3-3.5] = [-0.33, -0.5]
  Item3: [4-4.33, 4-3.5] = [-0.33, 0.5]

Numerator: (-0.33)*(-0.33) + (-0.5)*(0.5) = 0.109 - 0.25 = -0.141
Denom: sqrt(0.109 + 0.25) * sqrt(0.109 + 0.25) = 0.359

sim(Item1, Item3) = -0.141 / 0.359 = -0.393
```

---

**Item1 vs Item4:**

```
Co-raters: User1, User2, User4

Raw ratings:
  Item1: [5, 4, 3]
  Item4: [4, 5, 3]

User means: [4.0, 4.33, 3.5]

Adjusted:
  Item1: [5-4.0, 4-4.33, 3-3.5] = [1.0, -0.33, -0.5]
  Item4: [4-4.0, 5-4.33, 3-3.5] = [0.0, 0.67, -0.5]

Numerator: 1.0*0.0 + (-0.33)*0.67 + (-0.5)*(-0.5) = 0 - 0.221 + 0.25 = 0.029
Denom: sqrt(1.0 + 0.109 + 0.25) * sqrt(0 + 0.449 + 0.25)
     = sqrt(1.359) * sqrt(0.699) = 1.166 * 0.836 = 0.975

sim(Item1, Item4) = 0.029 / 0.975 = 0.030
```

---

**Item2 vs Item3:**

```
Co-raters: User4 only (User3 rated both but... wait, User3 rated Item2=2 and Item3=5)

Actually, co-raters: User3, User4

Raw ratings:
  Item2: [2, 4]
  Item3: [5, 4]

User means: [3.5, 3.5]

Adjusted:
  Item2: [2-3.5, 4-3.5] = [-1.5, 0.5]
  Item3: [5-3.5, 4-3.5] = [1.5, 0.5]

Numerator: (-1.5)*1.5 + 0.5*0.5 = -2.25 + 0.25 = -2.0
Denom: sqrt(2.25 + 0.25) * sqrt(2.25 + 0.25) = sqrt(2.5) * sqrt(2.5) = 2.5

sim(Item2, Item3) = -2.0 / 2.5 = -0.8
```

---

**Item2 vs Item4:**

```
Co-raters: User1, User4

Raw ratings:
  Item2: [3, 4]
  Item4: [4, 3]

User means: [4.0, 3.5]

Adjusted:
  Item2: [3-4.0, 4-3.5] = [-1.0, 0.5]
  Item4: [4-4.0, 3-3.5] = [0.0, -0.5]

Numerator: (-1.0)*0.0 + 0.5*(-0.5) = 0 - 0.25 = -0.25
Denom: sqrt(1.0 + 0.25) * sqrt(0 + 0.25) = sqrt(1.25) * sqrt(0.25) = 1.118 * 0.5 = 0.559

sim(Item2, Item4) = -0.25 / 0.559 = -0.447
```

---

**Item3 vs Item4:**

```
Co-raters: User2, User4

Raw ratings:
  Item3: [4, 4]
  Item4: [5, 3]

User means: [4.33, 3.5]

Adjusted:
  Item3: [4-4.33, 4-3.5] = [-0.33, 0.5]
  Item4: [5-4.33, 3-3.5] = [0.67, -0.5]

Numerator: (-0.33)*0.67 + 0.5*(-0.5) = -0.221 - 0.25 = -0.471
Denom: sqrt(0.109 + 0.25) * sqrt(0.449 + 0.25) = sqrt(0.359) * sqrt(0.699) = 0.599 * 0.836 = 0.501

sim(Item3, Item4) = -0.471 / 0.501 = -0.940
```

---

### The Complete Similarity Matrix

```
Item-Item Similarity Matrix:
         Item1   Item2   Item3   Item4
Item1     1.0    -1.0   -0.393   0.030
Item2    -1.0     1.0   -0.800  -0.447
Item3   -0.393  -0.800   1.0    -0.940
Item4    0.030  -0.447  -0.940   1.0
```

*Notice*: Most similarities are negative! This dataset has items that tend to be rated oppositely.

---

### Making a Prediction

**Task**: Predict User3's rating for Item1.

**User3's existing ratings**: Item2 = 2, Item3 = 5

**Similarities to Item1**:
- Item2: sim = -1.0, User3's rating = 2
- Item3: sim = -0.393, User3's rating = 5

**Using k=2 (both items):**

$$\hat{r}_{User3, Item1} = \frac{|-1.0| \times 2 + |-0.393| \times 5}{|-1.0| + |-0.393|}$$

$$= \frac{2.0 + 1.965}{1.393} = \frac{3.965}{1.393} = 2.85$$

**But wait!** The negative similarities contain information. If Item1 is negatively correlated with Item2, and User3 rated Item2 LOW (2), that suggests User3 might rate Item1 HIGH!

**Alternative formula accounting for negative correlations:**

For negative similarity, we "flip" the rating: $r'_{uj} = \text{max\_rating} - r_{uj} + 1$

```
Item2: sim = -1.0, raw rating = 2, flipped rating = 5 - 2 + 1 = 4
Item3: sim = -0.393, raw rating = 5, contribution weighted by |sim|
```

This is getting complex -- in practice, many systems just use positive similarities.

---

## Amazon's Item-to-Item Algorithm

### The 2003 Production System

**Offline Phase (Weekly)**:
1. Compute all item-item similarities
2. For each item, store only top-N (e.g., N=100) most similar items
3. Store in fast lookup table

**Online Phase (Real-time)**:
1. User views item X
2. Lookup: "What items are similar to X?" (O(1) lookup!)
3. Return top-k similar items
4. Display: "Customers who bought this also bought..."

**The magic**: Precomputation turns O(|I|^2) into O(1)!

### Storage Optimization

**Naive storage**: Store full |I| x |I| matrix

```
1 million items:
  1M x 1M = 1 trillion similarity values
  At 4 bytes each = 4 terabytes

  UNACCEPTABLE!
```

**Smart storage**: Store only top-N per item

```
1 million items, top-100 similar per item:
  1M x 100 = 100 million entries
  At 8 bytes each (item_id + similarity) = 800 megabytes

  MANAGEABLE!
```

*Can you see why* this is a 5000x improvement in storage?

---

## What Can Go Wrong: Failure Modes

### Failure Mode 1: Cold Start for New Items

**Symptom**: New item added to catalog, never gets recommended.

**Why it happens**:
- New item has no ratings
- Can't compute similarity to any existing item
- Never appears in "similar items" lists

**Concrete example**:
```
Day 1: "New Blockbuster Movie" added to catalog
       Ratings: []
       Similarity to all items: undefined

Day 7: Still no recommendations include it
       Few organic views -> few ratings -> can't bootstrap

Result: New items trapped in "cold start death spiral"
```

**Solutions**:
1. **Content-based bootstrapping**: Use movie genre/actors to estimate initial similarities
2. **Hybrid approach**: For new items, use content-based; for established items, use CF
3. **Editorial placement**: Manually feature new items until they get enough ratings
4. **Exploration bonus**: Occasionally recommend new items to collect data

**Detection**: Flag items with < 10 ratings and track their recommendation frequency.

---

### Failure Mode 2: Sparsity Kills Similarity Quality

**Symptom**: Similarity matrix is unreliable; recommendations seem random.

**Why it happens**: Most item pairs have very few co-raters.

**The math of sparsity**:
```
Catalog: 100,000 items
Average user rates: 50 items
Average item has: 500 ratings

Probability two random items share a rater:
  P(user rated item A) = 50/100,000 = 0.0005
  P(user rated both A and B) ~ 0.0005 * 0.0005 = 0.00000025

Expected co-raters for any pair: 0.00000025 * 1M users = 0.25 users

MOST ITEM PAIRS HAVE ZERO CO-RATERS!
```

**Consequences**:
- Can only compute similarity for ~1% of item pairs
- Remaining similarities based on 1-5 users (unreliable)
- Similarity matrix is sparse AND noisy

**Solutions**:
1. **Significance weighting**: Downweight similarities with few co-raters
2. **Minimum support threshold**: Only compute similarity if >= 10 co-raters
3. **Dimensionality reduction**: Use SVD to fill in missing similarities
4. **Implicit feedback**: Add view/click data to increase density

---

### Failure Mode 3: Assumes Item Similarity Is Stable

**Symptom**: Recommendations feel outdated; trending items missing.

**Why it happens**: Precomputed similarities don't capture temporal dynamics.

**Concrete examples**:

```
Example 1: Seasonal items
  - "Christmas Lights" is similar to "Halloween Decorations" (both rated by holiday shoppers)
  - But in December, you want to recommend OTHER Christmas items!

Example 2: Trending topics
  - News article about "Climate Change Documentary" goes viral
  - Suddenly everyone rating it has different profile than before
  - Old similarities are WRONG for the new user base

Example 3: Format changes
  - Book "1984" gets a new popular TV adaptation
  - Old similarity: similar to other dystopian books
  - New reality: similar to other shows people binge
  - Similarity needs updating!
```

**Solutions**:
1. **Time-decayed similarities**: Weight recent ratings higher
2. **Sliding window**: Only use ratings from last N months
3. **Trending detection**: Track velocity of ratings, recompute for "hot" items
4. **Hybrid**: Use content-based for temporal context

---

### Failure Mode 4: Popularity Bias Amplification

**Symptom**: Recommendations dominated by popular items; long-tail items never shown.

**Why it happens**: Popular items have more co-raters, so their similarities are more reliable, so they get recommended more, so they get MORE ratings...

**The feedback loop**:
```
Popular item has 10,000 ratings
  -> High-quality similarity estimates
  -> Gets recommended frequently
  -> Gets MORE ratings
  -> Even better similarities
  -> Recommended even MORE

Niche item has 50 ratings
  -> Noisy similarity estimates
  -> Rarely recommended
  -> No new ratings
  -> Similarities stay noisy
  -> NEVER recommended

Result: Rich get richer, poor get poorer
```

**Quantifying the bias**:
```
Top 1% of items: 50% of all recommendations
Top 10% of items: 90% of all recommendations
Bottom 50% of items: < 1% of recommendations
```

**Solutions**:
1. **Popularity normalization**: $\text{sim}_{norm}(i,j) = \frac{\text{sim}(i,j)}{\sqrt{|U_i| \cdot |U_j|}}$
2. **Diversity re-ranking**: After generating top-100, select diverse subset
3. **Explore-exploit**: Occasionally recommend low-confidence items
4. **Inverse propensity weighting**: Upweight rare items in training

---

## Computational Complexity

### Offline Phase

**Computing all item-item similarities:**

```
For each item pair (i, j):
  - Find co-raters: O(|U|)
  - Compute similarity: O(|co-raters|)

Total pairs: C(|I|, 2) ~ |I|^2 / 2
Total: O(|I|^2 * |U|)

With sparsity exploitation (only pairs with co-raters):
  - Much faster in practice
  - But still O(|I|^2) pairs to consider
```

**Storage:**
```
Full matrix: O(|I|^2) -- prohibitive
Top-N per item: O(|I| * N) -- manageable
```

### Online Phase

**Single item recommendation:**
```
1. Lookup similar items: O(1) (precomputed)
2. Return top-k: O(1)

Total: O(1) -- instant!
```

**User-personalized recommendations:**
```
1. Get user's rated items: O(|I_u|)
2. For each rated item, get similar items: O(|I_u| * N)
3. Aggregate and sort: O(|I_u| * N * log(|I_u| * N))

Total: O(|I_u| * N * log(|I_u| * N))

For typical user (50 items, N=100):
  O(5000 * log(5000)) ~ O(60,000) operations

At 1B ops/sec: 60 microseconds -- FAST!
```

---

## Complete Example: From Raw Data to Recommendation

### Step 1: Raw Data

```
         Movie1  Movie2  Movie3  Movie4  Movie5
User1      5      4       ?       5       3
User2      4      ?       4       ?       2
User3      ?      3       5       4       ?
User4      3      2       3       3       4
User5      5      5       4       5       5
```

### Step 2: Compute Similarities (showing Movie1 vs Movie3)

```
Co-raters: User4, User5

Movie1 ratings: [3, 5]
Movie3 ratings: [3, 4]

User means:
  User4: (3+2+3+3+4)/5 = 3.0
  User5: (5+5+4+5+5)/5 = 4.8

Adjusted Movie1: [3-3.0, 5-4.8] = [0, 0.2]
Adjusted Movie3: [3-3.0, 4-4.8] = [0, -0.8]

sim(Movie1, Movie3) = (0*0 + 0.2*-0.8) / (sqrt(0.04) * sqrt(0.64))
                    = -0.16 / (0.2 * 0.8)
                    = -0.16 / 0.16 = -1.0
```

### Step 3: Build Similarity Table (Top-2 per Movie)

```
Movie1: [(Movie4, 0.85), (Movie5, 0.72)]
Movie2: [(Movie4, 0.91), (Movie1, 0.68)]
Movie3: [(Movie5, 0.78), (Movie4, 0.65)]
Movie4: [(Movie2, 0.91), (Movie1, 0.85)]
Movie5: [(Movie3, 0.78), (Movie1, 0.72)]
```

### Step 4: Predict User2's Rating for Movie4

```
User2's ratings: Movie1=4, Movie3=4, Movie5=2

Items similar to Movie4 that User2 rated:
  Movie1: sim=0.85, User2's rating=4
  (Movie3 and Movie5 might also be similar, but let's use top-2)

Prediction = (0.85 * 4) / 0.85 = 4.0
```

**If we use more similar items:**

```
Assume sim(Movie4, Movie3)=0.65, sim(Movie4, Movie5)=0.30

Prediction = (0.85*4 + 0.65*4 + 0.30*2) / (0.85 + 0.65 + 0.30)
           = (3.4 + 2.6 + 0.6) / 1.8
           = 6.6 / 1.8 = 3.67
```

**User2 would probably rate Movie4 around 3.7 stars.**

---

## Summary

### Item-Based Collaborative Filtering

**Core Idea**: Items that are liked by the same users are similar.

**Algorithm**:
1. Compute item-item similarity using adjusted cosine
2. Store top-N similar items per item
3. Predict by weighted average of user's ratings for similar items

**Key Formula**:
$$\hat{r}_{ui} = \frac{\sum_{j \in N_k(i) \cap I_u} \text{sim}(i,j) \cdot r_{uj}}{\sum_{j \in N_k(i) \cap I_u} |\text{sim}(i,j)|}$$

### Advantages Over User-Based

| Aspect | User-Based | Item-Based |
|--------|-----------|------------|
| Complexity | O(|U|^2 * |I|) | O(|I|^2 * |U|) |
| Update frequency | Weekly (users change) | Monthly (items stable) |
| Online latency | O(|U|) | O(1) with precomputation |
| Cold start (new user) | Severe | Better (rate few items, get recs) |

### Failure Modes to Remember

1. **Cold start for new items** (no ratings = no similarity)
2. **Sparsity** (most item pairs have zero co-raters)
3. **Assumes stability** (misses trends and seasonal patterns)
4. **Popularity amplification** (rich get richer)

### When to Use Item-Based CF

- Large user base, moderate item catalog
- Items don't change frequently
- Need low-latency online serving
- Can tolerate weekly/monthly model updates

**Next**: See **similarity-measures.md** for detailed comparison of all similarity metrics.

---

## References

1. **Linden, G., Smith, B., & York, J. (2003)**. "Amazon.com recommendations: Item-to-item collaborative filtering". *IEEE Internet Computing*.
2. **Sarwar, B., et al. (2001)**. "Item-based collaborative filtering recommendation algorithms". *WWW*.
3. **Deshpande, M., & Karypis, G. (2004)**. "Item-based top-N recommendation algorithms". *ACM TOIS*.
