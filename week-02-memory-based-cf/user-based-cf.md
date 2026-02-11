# Week 2: User-Based Collaborative Filtering

## Learning Objectives

- Understand the intuition behind user-based collaborative filtering
- Compute user-user similarity using various metrics
- Implement k-nearest neighbors for recommendation
- Analyze computational complexity and limitations
- Recognize failure modes and when to avoid this approach

---

## Opening: Why Does Recommending Popular Items Fail?

*Before we dive into algorithms, let me show you exactly what breaks when we take the naive approach.*

### The Popularity Trap: A Concrete Failure

**Scenario**: You run a movie streaming service with 1 million users. Your boss says: "Just recommend the top 10 most-watched movies to everyone. Simple!"

**Let's see what happens with real numbers.**

**Top 10 Most-Watched Movies** (by total views):
1. Avengers: Endgame (850,000 views)
2. The Lion King (820,000 views)
3. Frozen (800,000 views)
4. Spider-Man (780,000 views)
5. Fast & Furious 9 (750,000 views)
... (you get the idea)

**Now let's look at three specific users:**

| User | Taste Profile | Movies They Love | Popular Recs Match? |
|------|--------------|------------------|-------------------|
| Alice | Indie films, documentaries | Moonlight, Free Solo, Lady Bird | 0/10 match |
| Bob | Classic noir, foreign cinema | Casablanca, Rashomon, The Third Man | 0/10 match |
| Charlie | Blockbuster superhero fan | All Marvel movies | 3/10 match |

**The math of failure:**

```
Alice's satisfaction: 0%
Bob's satisfaction: 0%
Charlie's satisfaction: 30%

Average satisfaction: 10%
```

**Can you see why** this is catastrophic? We're essentially telling 2 out of 3 users: "Your tastes don't matter. Watch what everyone else watches."

**The deeper problem**: Popular items are popular BECAUSE they appeal to the average person. Users with distinctive tastes are systematically ignored.

*What we need*: A way to find items that match EACH user's specific preferences.

---

### A Slightly Better Approach: Content-Based Filtering

**Idea**: If Alice liked Sci-Fi before, recommend more Sci-Fi.

**What goes wrong?**

1. **Feature engineering nightmare**: How do you describe a movie? Genre? Director? Actors? Cinematography style? Budget?

2. **The "Inception" problem**: This movie is... sci-fi? thriller? heist? romance? drama? All of the above?

3. **Discovery failure**: Alice has only watched sci-fi, but she might LOVE the documentary "Free Solo" -- content-based will never suggest it.

*What we need*: A way to discover preferences we don't explicitly know.

---

## The Collaborative Filtering Insight

**If users agreed in the past, they will likely agree in the future.**

*This is the key insight.* We don't need to understand WHY Alice likes certain movies (is it the cinematography? the acting? the themes?). We just need to find people who liked the SAME movies as Alice and see what ELSE they liked.

**Example**:
- Alice and Bob both loved *The Matrix*, *Inception*, and *Interstellar*
- Alice also loved *Blade Runner 2049*
- Bob hasn't seen *Blade Runner 2049*
- **Recommendation**: Suggest *Blade Runner 2049* to Bob

*Notice*: We never analyzed what makes these movies similar. We just observed that Alice and Bob have similar tastes.

### Why This Is Powerful

**No feature engineering needed**: The algorithm discovers hidden patterns.

**Serendipitous discovery**: Bob might discover he loves a movie he never would have found on his own.

**Scales with data**: More users = better recommendations (more "taste neighbors" to learn from).

---

## The Algorithm: Step-by-Step Derivation

### The Intuition: Building Up the Formula

*Let me derive the prediction formula from first principles, so you understand WHY each term is there.*

**Goal**: Predict how user $u$ would rate item $i$ they haven't seen.

**Naive Idea 1**: Just average what similar users rated it.

$$\hat{r}_{ui} = \frac{1}{k} \sum_{v \in N_k(u)} r_{vi}$$

**What's wrong?** Different users have different baselines!
- Alice rates everything 4-5 stars (generous)
- Bob rates everything 1-2 stars (harsh)
- If both give a movie their "top rating," Alice gives 5, Bob gives 2

**Naive Idea 2**: Use deviations from each user's average.

$$\hat{r}_{ui} = \bar{r}_u + \frac{1}{k} \sum_{v \in N_k(u)} (r_{vi} - \bar{r}_v)$$

**Better!** Now we're saying: "Start with Alice's baseline (4.2), then adjust based on whether similar users rated this item above or below THEIR baselines."

**What's still wrong?** Not all neighbors are equally similar!
- Carol is 95% similar to Alice
- Dave is 20% similar to Alice
- They shouldn't have equal influence

**Final Formula**: Weight by similarity.

$$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} \text{sim}(u,v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N_k(u)} |\text{sim}(u,v)|}$$

### Why Each Term?

Let me break down every piece:

| Term | Meaning | Why It's There |
|------|---------|----------------|
| $\bar{r}_u$ | User $u$'s average rating | Baseline -- Alice tends to rate ~4.2 |
| $r_{vi} - \bar{r}_v$ | How much neighbor $v$ rated above/below their average | Removes rating scale bias |
| $\text{sim}(u,v)$ | How similar neighbor $v$ is to user $u$ | More similar = more influence |
| $\sum \|\text{sim}(u,v)\|$ | Sum of all similarity weights | Normalization so result is on rating scale |

**Interpretation in plain English**:
> "Predict Alice's rating by starting with her average, then adjusting based on how her neighbors rated this item relative to THEIR averages, weighted by how similar each neighbor is to Alice."

---

## Complete Numerical Walkthrough

*Let's work through a complete example with actual numbers. I'll show every calculation.*

### The Setup: Our Rating Matrix

```
         Movie1  Movie2  Movie3  Movie4  Movie5
Alice      5       3       4       ?       ?      (target user)
Bob        3       1       2       2       4
Carol      4       3       5       3       ?
Dave       3       3       3       4       3
Eve        ?       ?       5       ?       3
```

**Task**: Predict Alice's rating for Movie4 using k=2 nearest neighbors with Pearson correlation.

---

### Step 1: Compute All Similarities to Alice

**Alice vs. Bob:**

Co-rated items: Movie1, Movie2, Movie3

```
Alice's ratings: [5, 3, 4]
Alice's mean (on these items): (5 + 3 + 4) / 3 = 4.0

Bob's ratings: [3, 1, 2]
Bob's mean (on these items): (3 + 1 + 2) / 3 = 2.0

Alice's deviations: [5-4, 3-4, 4-4] = [1, -1, 0]
Bob's deviations: [3-2, 1-2, 2-2] = [1, -1, 0]
```

**Pearson formula:**

$$\text{Pearson}(Alice, Bob) = \frac{\sum (a_i - \bar{a})(b_i - \bar{b})}{\sqrt{\sum(a_i - \bar{a})^2} \cdot \sqrt{\sum(b_i - \bar{b})^2}}$$

```
Numerator: (1)(1) + (-1)(-1) + (0)(0) = 1 + 1 + 0 = 2

Denominator: sqrt(1 + 1 + 0) * sqrt(1 + 1 + 0) = sqrt(2) * sqrt(2) = 2

Pearson(Alice, Bob) = 2 / 2 = 1.0
```

**Perfect correlation!** When Alice rates above her average, Bob also rates above his average. Their patterns match exactly.

---

**Alice vs. Carol:**

Co-rated items: Movie1, Movie2, Movie3

```
Alice's ratings: [5, 3, 4], mean = 4.0
Carol's ratings: [4, 3, 5], mean = 4.0

Alice's deviations: [1, -1, 0]
Carol's deviations: [0, -1, 1]
```

```
Numerator: (1)(0) + (-1)(-1) + (0)(1) = 0 + 1 + 0 = 1

Denominator: sqrt(1 + 1 + 0) * sqrt(0 + 1 + 1) = sqrt(2) * sqrt(2) = 2

Pearson(Alice, Carol) = 1 / 2 = 0.5
```

**Moderate correlation** -- they agree on some movies but not all.

---

**Alice vs. Dave:**

Co-rated items: Movie1, Movie2, Movie3

```
Alice's ratings: [5, 3, 4], mean = 4.0
Dave's ratings: [3, 3, 3], mean = 3.0

Alice's deviations: [1, -1, 0]
Dave's deviations: [0, 0, 0]  <- All zeros! Dave rates everything the same.
```

```
Numerator: (1)(0) + (-1)(0) + (0)(0) = 0

Denominator: sqrt(2) * sqrt(0) = 0  <- Division by zero!
```

**Pearson is undefined** when one user has zero variance. Dave rates everything 3, so we can't tell what he likes or dislikes.

*Notice*: This is a fundamental limitation -- users who rate everything the same are uninformative.

**We'll treat this as similarity = 0.**

---

**Alice vs. Eve:**

Co-rated items: Movie3 only

```
Only 1 co-rated item -- too few for reliable similarity!
```

**We skip Eve** (or set similarity = 0). With only one overlap, Pearson is undefined (can't compute correlation with a single point).

---

### Step 2: Select k=2 Nearest Neighbors

**Similarities (for users who rated Movie4):**

| User | Similarity | Rated Movie4? | Candidate? |
|------|------------|---------------|------------|
| Bob | 1.0 | Yes (rating = 2) | Yes |
| Carol | 0.5 | Yes (rating = 3) | Yes |
| Dave | 0.0 | Yes (rating = 4) | No (sim = 0) |
| Eve | N/A | No | No |

**Top-2 neighbors who rated Movie4**: Bob (sim = 1.0), Carol (sim = 0.5)

---

### Step 3: Predict the Rating

**Get each neighbor's full rating data for mean calculation:**

```
Bob's ALL ratings: [3, 1, 2, 2, 4]
Bob's mean: (3 + 1 + 2 + 2 + 4) / 5 = 2.4

Carol's ALL ratings: [4, 3, 5, 3] (no Movie5 rating)
Carol's mean: (4 + 3 + 5 + 3) / 4 = 3.75

Alice's mean (from her ratings): [5, 3, 4]
Alice's mean: (5 + 3 + 4) / 3 = 4.0
```

**Apply the formula:**

$$\hat{r}_{Alice, Movie4} = \bar{r}_{Alice} + \frac{\sum_{v} \text{sim}(Alice, v) \cdot (r_{v, Movie4} - \bar{r}_v)}{\sum_{v} |\text{sim}(Alice, v)|}$$

```
Bob's contribution:
  sim(Alice, Bob) = 1.0
  Bob's Movie4 rating = 2
  Bob's mean = 2.4
  Deviation = 2 - 2.4 = -0.4
  Weighted: 1.0 * (-0.4) = -0.4

Carol's contribution:
  sim(Alice, Carol) = 0.5
  Carol's Movie4 rating = 3
  Carol's mean = 3.75
  Deviation = 3 - 3.75 = -0.75
  Weighted: 0.5 * (-0.75) = -0.375

Numerator: -0.4 + (-0.375) = -0.775
Denominator: |1.0| + |0.5| = 1.5

Adjustment: -0.775 / 1.5 = -0.517

Prediction: 4.0 + (-0.517) = 3.48
```

**Final Prediction**: Alice would rate Movie4 approximately **3.5 stars**.

**Interpretation**: Both Bob and Carol rated Movie4 below their personal averages, suggesting it's not a standout film. Since Alice is similar to them, she'd probably also find it below her average (4.0), hence the prediction of 3.5.

---

## Choosing k: The Bias-Variance Tradeoff

### The Problem

**k too small (k = 1-3)**:
- Prediction depends on just 1-3 users
- High variance: one weird neighbor ruins everything
- Overfitting to noise

**k too large (k = 100+)**:
- Includes users who aren't really similar
- High bias: averaging over dissimilar users
- Prediction drifts toward population average

### Numerical Example: k's Effect

*Let's see how k changes our prediction for Alice's Movie4 rating.*

Assume we found more neighbors with these similarities and Movie4 ratings:

| User | Similarity | Movie4 Rating | Deviation from Mean |
|------|------------|---------------|-------------------|
| Bob | 1.0 | 2 | -0.4 |
| Carol | 0.5 | 3 | -0.75 |
| Frank | 0.3 | 4 | +0.2 |
| Grace | 0.1 | 5 | +1.5 |

**k = 1** (only Bob):
```
Prediction = 4.0 + (1.0 * -0.4) / 1.0 = 3.6
```

**k = 2** (Bob + Carol):
```
Prediction = 4.0 + (1.0 * -0.4 + 0.5 * -0.75) / 1.5 = 3.48
```

**k = 4** (all four):
```
Numerator = 1.0*(-0.4) + 0.5*(-0.75) + 0.3*(0.2) + 0.1*(1.5)
          = -0.4 - 0.375 + 0.06 + 0.15 = -0.565

Denominator = 1.0 + 0.5 + 0.3 + 0.1 = 1.9

Prediction = 4.0 + (-0.565/1.9) = 4.0 - 0.297 = 3.70
```

*Notice*: As k increases, less similar users (Grace with sim=0.1) pull the prediction. With enough dissimilar users, prediction approaches the global average.

### Practical Guidelines

| System Size | Recommended k | Rationale |
|-------------|---------------|-----------|
| < 1,000 users | 5-10 | Small pool, need most similar |
| 1,000 - 100,000 | 10-30 | Balanced |
| 100,000+ | 20-50 | Many neighbors available |

**Best Practice**: Use cross-validation to tune k for your specific dataset.

---

## Computational Complexity Analysis

### Breaking Down the Costs

**Step 1: Compute all pairwise user similarities**

For each pair of users, we iterate over co-rated items:
- Number of user pairs: $\binom{|U|}{2} \approx \frac{|U|^2}{2}$
- Cost per pair: $O(|I|)$ in worst case (all items co-rated)
- **Total: $O(|U|^2 \cdot |I|)$**

**Step 2: Find k nearest neighbors for each user**

- Sort similarities: $O(|U| \log |U|)$ per user
- For all users: $O(|U|^2 \log |U|)$

**Step 3: Predict ratings**

- For each (user, item) pair: $O(k)$
- Total pairs: $|U| \times |I|$
- **Total: $O(|U| \cdot |I| \cdot k)$**

**Dominant cost**: $O(|U|^2 \cdot |I|)$ -- quadratic in users!

### Real-World Scale Example: Netflix

Let's plug in Netflix's numbers (circa 2009):

```
|U| = 480,000 users (contest dataset)
|I| = 17,770 movies

Similarity computation:
  Pairs = 480,000^2 / 2 = 115 billion pairs
  Operations per pair ~ 100 (avg co-rated items)
  Total = 11.5 trillion operations

At 10 billion ops/second = 1,150 seconds = ~19 minutes
```

**For full Netflix (260M users today):**

```
Pairs = 260,000,000^2 / 2 = 33.8 quadrillion pairs
Even at 10 billion ops/second = 3.38 million seconds = 39 days
```

**Verdict**: Completely infeasible for large-scale systems!

---

## What Can Go Wrong: Failure Modes

### Failure Mode 1: The Cold Start Problem

**Symptom**: New user joins, gets terrible or no recommendations.

**Why it happens**:
- New user has 0 ratings
- Can't compute similarity with anyone
- Falls back to popularity (which we already know fails)

**Concrete example**:
```
New user Emma joins.
Emma's ratings: []

sim(Emma, anyone) = undefined (no co-rated items)

Fallback: Recommend top-10 popular movies
Emma actually loves indie documentaries
Result: 0% satisfaction
```

**Solutions**:
1. **Onboarding**: Ask new users to rate 10-20 items before showing recommendations
2. **Demographic bootstrapping**: Use age/location/gender to find initial neighbors
3. **Hybrid approach**: Use content-based for new users, CF for established users

**Rule of thumb**: Users need ~20 ratings before CF becomes reliable.

---

### Failure Mode 2: Popularity Bias in Similarity

**Symptom**: Popular items dominate similarity calculations; niche tastes ignored.

**Why it happens**: Popular items are rated by almost everyone, so they dominate co-rated sets.

**Concrete example**:
```
Alice and Bob both rated these movies:
  - Avengers (everyone rated it)
  - The Dark Knight (everyone rated it)
  - Spirited Away (niche anime)

Overlap = 3 items, but 2/3 are mainstream blockbusters

If Alice and Bob have identical ratings on blockbusters but
opposite tastes on anime... similarity still looks high!
```

**Numerical demonstration**:
```
             Avengers  DarkKnight  SpiritedAway
Alice           5          5            5
Bob             5          5            1

Pearson numerator: (5-5)(5-3.67) + (5-5)(5-3.67) + (5-5)(1-3.67)
                 = 0 + 0 + 0 = 0  [Alice has no variance on overlapping items!]

Actually, let's fix Alice's ratings:
Alice: [5, 4, 5], mean = 4.67
Bob: [5, 5, 1], mean = 3.67

Alice devs: [0.33, -0.67, 0.33]
Bob devs: [1.33, 1.33, -2.67]

Numerator: 0.33*1.33 + (-0.67)*1.33 + 0.33*(-2.67)
         = 0.44 - 0.89 - 0.88 = -1.33

Denominator: sqrt(0.11+0.45+0.11) * sqrt(1.77+1.77+7.13)
           = sqrt(0.67) * sqrt(10.67) = 0.82 * 3.27 = 2.68

Pearson = -1.33 / 2.68 = -0.50 (negative correlation!)
```

The anime preference (SpiritedAway) actually dominates here because it has high variance. But in practice, with 100 popular items and 3 niche items, popular items dominate.

**Solutions**:
1. **IDF weighting**: Downweight popular items (treat them like stopwords in text)
2. **Normalize by item popularity**: $\text{sim}_{adjusted} = \text{sim} / \sqrt{|U_i| \cdot |U_j|}$
3. **Focus on niche overlaps**: Only use non-popular items for similarity

---

### Failure Mode 3: Scalability Collapse (O(n^2))

**Symptom**: System works great with 10,000 users, crawls to a halt at 1 million.

**Why it happens**: Quadratic complexity means 100x users = 10,000x computation.

**Growth table**:

| Users | Pairs | Time (est.) |
|-------|-------|------------|
| 1,000 | 500K | 0.05 sec |
| 10,000 | 50M | 5 sec |
| 100,000 | 5B | 8 min |
| 1,000,000 | 500B | 14 hours |
| 10,000,000 | 50T | 58 days |

**Solutions**:
1. **Switch to item-based CF**: Usually $|I| \ll |U|$
2. **Locality-sensitive hashing (LSH)**: Approximate nearest neighbors in O(n)
3. **Matrix factorization**: Compress users to latent factors
4. **Cluster users**: Compute similarity only within clusters

---

### Failure Mode 4: Gray Sheep Users

**Symptom**: Certain users consistently get poor recommendations despite having many ratings.

**Why it happens**: Some users have genuinely unique tastes that don't match anyone else.

**Concrete example**:
```
Alice loves: Classical music + Death metal + Korean dramas + 1950s westerns

Most users who like classical music hate death metal.
Most users who like death metal hate classical music.
No one else combines all four interests.

Result: Alice has no good neighbors.
Best match might have sim = 0.1
Predictions are essentially random.
```

**Detection**: Flag users whose average neighbor similarity is below threshold (e.g., < 0.3).

**Solutions**:
1. **Content-based fallback**: For gray sheep, use content similarity instead
2. **Cluster-based**: Find micro-communities
3. **Diverse ensemble**: Combine multiple recommendation approaches
4. **Accept the limitation**: Some users are genuinely hard to model

---

### Failure Mode 5: Shilling Attacks

**Symptom**: Certain items suddenly appear in everyone's recommendations.

**Why it happens**: Malicious actors create fake profiles to manipulate similarity calculations.

**Attack types**:

**Push attack** (boost an item):
```
Create 100 fake users who all:
  - Rate popular items highly (to seem legitimate)
  - Rate target item 5 stars

Result: Target item appears similar to popular items
        Gets recommended to millions of users
```

**Nuke attack** (hurt a competitor):
```
Create fake users who:
  - Rate competitor's items 1 star
  - Rate random items highly

Result: Competitor items get negative similarity
        Never recommended
```

**Solutions**:
1. **Anomaly detection**: Flag users with suspicious patterns
2. **Trust modeling**: Weight ratings by user credibility
3. **Rate limiting**: Restrict how fast new users can rate
4. **Diversity requirements**: Don't let any single neighbor dominate

---

## Optimizations and Tricks

### 1. Sparsity Exploitation

**Observation**: Most users share very few rated items.

**Solution**: Only compute similarity for user pairs with minimum overlap (e.g., >= 5 co-rated items).

**Benefit**: Reduces computation and improves reliability.

### 2. Significance Weighting

**Problem**: Similarity based on 1-2 co-rated items is unreliable.

**Solution**: Weight similarity by number of co-rated items.

$$\text{sim}_{weighted}(u, v) = \text{sim}(u, v) \cdot \min\left(1, \frac{|I_{uv}|}{threshold}\right)$$

**Example**: threshold = 50
- If $|I_{uv}| = 10$, multiply similarity by 10/50 = 0.2 (downweight)
- If $|I_{uv}| \geq 50$, use full similarity

### 3. Variance Weighting

**Problem**: Users who rate everything the same (e.g., always 5 stars) are not informative.

**Solution**: Weight by user's rating variance.

$$\text{sim}_{weighted}(u, v) = \text{sim}(u, v) \cdot \text{var}(u) \cdot \text{var}(v)$$

**Example**:
- User A: ratings = [5, 5, 5, 5, 5], var = 0 --> similarity weighted to 0
- User B: ratings = [1, 3, 5, 2, 4], var = 2.5 --> keeps similarity

### 4. Negative Similarity Handling

**Issue**: Negative Pearson correlation

**Options**:
1. **Ignore**: Only use positive similarities
2. **Absolute value**: $|\text{sim}(u,v)|$
3. **Include**: Can be informative (users with opposite tastes)

**Recommendation**: Use only positive similarities (negative correlations often spurious with few co-ratings)

---

## Similarity Metrics Quick Reference

### 1. Pearson Correlation Coefficient

**Measures linear correlation** between two users' rating patterns.

$$\text{Pearson}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}$$

**Range**: [-1, +1]
- +1: Perfect positive correlation (users agree perfectly)
- 0: No correlation
- -1: Perfect negative correlation (users always disagree)

**Advantages**:
- Accounts for different rating scales
- Mean-centered: Focuses on deviations from average

**Disadvantages**:
- Sensitive to outliers
- Requires many co-rated items for reliability (>= 5 recommended, >= 20 ideal)
- Undefined if variance is zero

### 2. Cosine Similarity

**Measures angle** between rating vectors.

$$\text{cosine}(u, v) = \frac{r_u \cdot r_v}{||r_u|| \cdot ||r_v||}$$

**Range**: [0, +1] for ratings (always positive)

**Advantages**:
- Simple and fast to compute
- Works well when magnitude doesn't matter

**Disadvantages**:
- Doesn't account for different rating scales

### 3. Jaccard Similarity (for Binary Data)

**For implicit feedback** (watched/not watched, clicked/not clicked).

$$\text{Jaccard}(u, v) = \frac{|I_u \cap I_v|}{|I_u \cup I_v|}$$

**Range**: [0, 1]

**Advantages**:
- Simple for binary data
- No need for explicit ratings

**Disadvantages**:
- Ignores rating magnitudes
- Popularity bias

---

## Comparison: User-Based vs. Item-Based CF

| Aspect | User-Based CF | Item-Based CF |
|--------|---------------|---------------|
| **Similarity** | Between users | Between items |
| **Scalability** | Poor ($|U|$ often > $|I|$) | Better (fewer items) |
| **Stability** | Changes as user preferences drift | More stable (item features static) |
| **Serendipity** | Higher (diverse users) | Lower (similar items) |
| **Cold Start** | Severe for new users | Handles new users better |

**Modern practice**: Item-based CF preferred for large-scale systems (covered in next section).

---

## Summary

**User-Based Collaborative Filtering**:
- Find users with similar rating patterns
- Aggregate their ratings for prediction
- Uses Pearson correlation or cosine similarity
- Complexity: $O(|U|^2 \cdot |I|)$ --> doesn't scale

**Key Formula**:
$$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} \text{sim}(u,v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N_k(u)} |\text{sim}(u,v)|}$$

**Failure Modes to Remember**:
1. Cold start (new users have no neighbors)
2. Popularity bias (mainstream items dominate)
3. Scalability ($O(n^2)$ blows up)
4. Gray sheep (unique users have no matches)
5. Shilling attacks (fake profiles manipulate recs)

**Key Takeaway**: Intuitive but limited by scalability and sparsity. Modern systems use item-based CF or matrix factorization.

**Next**: See **item-based-cf.md** for a more scalable approach.

---

## References

1. Resnick, P., et al. (1994). "GroupLens: An open architecture for collaborative filtering". *CSCW*.
2. Breese, J. S., et al. (1998). "Empirical analysis of predictive algorithms for collaborative filtering". *UAI*.
3. Herlocker, J. L., et al. (1999). "An algorithmic framework for performing collaborative filtering". *SIGIR*.
