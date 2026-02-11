# Week 2: Similarity Measures Deep Dive

## Learning Objectives

- Master all major similarity metrics for collaborative filtering
- Understand when to use each metric
- Derive each measure from first principles
- Learn significance weighting and normalization techniques
- Recognize failure modes for each measure

---

## Opening: Why Does Raw Dot Product Fail?

*Before I show you the sophisticated similarity measures, let me demonstrate why the simplest approach fails catastrophically.*

### The Naive Approach: Just Multiply Ratings

**Intuition**: "If Alice and Bob both rated items highly, they must be similar. Let's just multiply their ratings and sum them up!"

$$\text{naive\_similarity}(A, B) = \sum_{i} r_{Ai} \cdot r_{Bi}$$

**Let's test this on real data.**

```
          Movie1  Movie2  Movie3  Movie4  Movie5  SUM
Alice        5       4       5       4       5     23 (ratings sum)
Bob          1       1       1       1       1      5 (ratings sum)
Carol        5       5       5       5       5     25 (ratings sum)
Dave         3       4       3       5       4     19 (ratings sum)
```

**Computing naive similarity:**

```
Alice vs Bob:
  5*1 + 4*1 + 5*1 + 4*1 + 5*1 = 5 + 4 + 5 + 4 + 5 = 23

Alice vs Carol:
  5*5 + 4*5 + 5*5 + 4*5 + 5*5 = 25 + 20 + 25 + 20 + 25 = 115

Alice vs Dave:
  5*3 + 4*4 + 5*3 + 4*5 + 5*4 = 15 + 16 + 15 + 20 + 20 = 86
```

**Results:**
- Alice-Carol: 115 (most similar?)
- Alice-Dave: 86
- Alice-Bob: 23 (least similar?)

**But wait! Let's look at what these users actually prefer:**

```
         Best Rated    Worst Rated    Pattern
Alice    M1,M3,M5 (5)  M2,M4 (4)     Prefers odd-numbered movies
Bob      All same (1)  All same (1)  Hates everything equally
Carol    All same (5)  All same (5)  Loves everything equally
Dave     M4 (5)        M1,M3 (3)     Prefers even-numbered movies
```

**The problem revealed:**
- Alice and Bob have OPPOSITE tastes (she loves, he hates)
- Alice and Carol have NO DISTINGUISHABLE pattern (Carol loves everything)
- Alice and Dave have DIFFERENT preferences (she likes odds, he likes evens)

**Yet our naive measure says Alice is most similar to Carol!**

*Can you see the two fundamental problems?*

1. **Magnitude dominates**: Carol's high ratings (all 5s) inflate similarity regardless of pattern
2. **No normalization**: Users who rate more items or rate higher always look "more similar"

*This is why we need proper similarity measures.*

---

## Deriving Similarity Measures from First Principles

*Let me walk you through the logical progression that leads to each similarity measure.*

### Problem 1: Activity Level Bias

**Observation**: Some users rate 1000 items, others rate 10.

**Failed attempt**: Sum of rating products (as shown above)

```
Heavy rater Alice: rated 1000 items, all 5s -> huge sum with everyone
Light rater Bob: rated 10 items, all 5s -> small sum with everyone

Even though both rate everything 5, Alice "looks more similar" to everyone!
```

**Solution**: Divide by magnitude. This gives us **Cosine Similarity**.

$$\text{cosine}(A, B) = \frac{\sum_i r_{Ai} \cdot r_{Bi}}{||\mathbf{r}_A|| \cdot ||\mathbf{r}_B||} = \frac{\mathbf{r}_A \cdot \mathbf{r}_B}{||\mathbf{r}_A|| \cdot ||\mathbf{r}_B||}$$

**Intuition**: Measure the *angle* between rating vectors, not their length. Two users pointing in the same direction are similar, regardless of how many items they rated.

---

### Problem 2: Rating Scale Bias

**Observation**: Some users are generous (rate 4-5), others are harsh (rate 1-2).

**Cosine fails here:**

```
          Movie1  Movie2  Movie3  Movie4
Alice        5       4       5       4     (generous, avg = 4.5)
Bob          2       1       2       1     (harsh, avg = 1.5)
Carol        5       5       5       5     (loves everything)

Alice's pattern: M1 > M2, M3 > M4
Bob's pattern:   M1 > M2, M3 > M4
Carol's pattern: M1 = M2 = M3 = M4

Alice and Bob have IDENTICAL preferences! They just use different scales.
```

**Computing cosine:**

```
Alice vs Bob:
  dot = 5*2 + 4*1 + 5*2 + 4*1 = 10 + 4 + 10 + 4 = 28
  ||Alice|| = sqrt(25+16+25+16) = sqrt(82) = 9.06
  ||Bob|| = sqrt(4+1+4+1) = sqrt(10) = 3.16
  cosine = 28 / (9.06 * 3.16) = 28 / 28.6 = 0.978

Alice vs Carol:
  dot = 5*5 + 4*5 + 5*5 + 4*5 = 25+20+25+20 = 90
  ||Carol|| = sqrt(100) = 10
  cosine = 90 / (9.06 * 10) = 90 / 90.6 = 0.993
```

**Result**: Alice-Carol (0.993) > Alice-Bob (0.978)

**But that's wrong!** Alice and Bob have the SAME preference pattern!

**Solution**: Subtract each user's mean before computing. This gives us **Pearson Correlation**.

$$\text{Pearson}(A, B) = \frac{\sum_i (r_{Ai} - \bar{r}_A)(r_{Bi} - \bar{r}_B)}{\sqrt{\sum_i (r_{Ai} - \bar{r}_A)^2} \cdot \sqrt{\sum_i (r_{Bi} - \bar{r}_B)^2}}$$

**After mean-centering:**

```
Alice (mean = 4.5): [5-4.5, 4-4.5, 5-4.5, 4-4.5] = [0.5, -0.5, 0.5, -0.5]
Bob (mean = 1.5):   [2-1.5, 1-1.5, 2-1.5, 1-1.5] = [0.5, -0.5, 0.5, -0.5]
Carol (mean = 5.0): [5-5, 5-5, 5-5, 5-5] = [0, 0, 0, 0]
```

Now Alice and Bob have **identical** centered vectors!

```
Pearson(Alice, Bob) = (0.5*0.5 + -0.5*-0.5 + 0.5*0.5 + -0.5*-0.5) / (1 * 1)
                    = (0.25 + 0.25 + 0.25 + 0.25) / 1 = 1.0

Pearson(Alice, Carol) = 0 / (1 * 0) = undefined (Carol has zero variance)
```

**Perfect!** Pearson correctly identifies Alice and Bob as identical, and can't be fooled by Carol's "loves everything" pattern.

---

### Problem 3: Binary Data Has No Magnitude

**Observation**: Sometimes we only know "interacted" vs "didn't interact" (clicked, purchased, watched).

**Pearson/Cosine fail here:**

```
User A: clicked items {1, 2, 3, 5, 8}
User B: clicked items {2, 3, 5, 7, 9}

There are no ratings to compute!
```

**Solution**: Count overlaps relative to total. This gives us **Jaccard Similarity**.

$$\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

```
A ∩ B = {2, 3, 5} -> 3 items
A ∪ B = {1, 2, 3, 5, 7, 8, 9} -> 7 items

Jaccard = 3/7 = 0.43
```

**Intuition**: Of all items either user interacted with, what fraction did BOTH interact with?

---

## The Complete Numerical Comparison

*Let me compute ALL similarity measures on the SAME user pair so you can see how they differ.*

### Our Test Data

```
          Movie1  Movie2  Movie3  Movie4  Movie5
Alice        5       3       4       5       3      (mean = 4.0, rates most items)
Bob          3       2       3       3       2      (mean = 2.6, harsh critic)
```

**What's the true relationship?** Let's look at patterns:
- Both rate Movie1 and Movie4 highest relative to their averages
- Both rate Movie2 and Movie5 lowest relative to their averages
- They have the SAME preference pattern, just different scales

---

### 1. Raw Dot Product (Naive)

$$\text{dot}(Alice, Bob) = 5 \times 3 + 3 \times 2 + 4 \times 3 + 5 \times 3 + 3 \times 2$$

$$= 15 + 6 + 12 + 15 + 6 = 54$$

**Problem**: This number means nothing on its own. Is 54 high or low?

---

### 2. Cosine Similarity

$$\text{cosine}(Alice, Bob) = \frac{\sum r_{Ai} \cdot r_{Bi}}{||\mathbf{r}_A|| \cdot ||\mathbf{r}_B||}$$

**Step 1**: Compute dot product = 54 (from above)

**Step 2**: Compute magnitudes

```
||Alice|| = sqrt(5² + 3² + 4² + 5² + 3²) = sqrt(25+9+16+25+9) = sqrt(84) = 9.17
||Bob|| = sqrt(3² + 2² + 3² + 3² + 2²) = sqrt(9+4+9+9+4) = sqrt(35) = 5.92
```

**Step 3**: Divide

$$\text{cosine} = \frac{54}{9.17 \times 5.92} = \frac{54}{54.3} = 0.994$$

**Interpretation**: 0.994 is very high, suggesting strong similarity. But wait -- this is partly because all ratings are positive! Cosine would be different if we had negative numbers.

---

### 3. Pearson Correlation

$$\text{Pearson}(Alice, Bob) = \frac{\sum (r_{Ai} - \bar{r}_A)(r_{Bi} - \bar{r}_B)}{\sigma_A \cdot \sigma_B}$$

**Step 1**: Mean-center the ratings

```
Alice mean = (5+3+4+5+3)/5 = 20/5 = 4.0
Bob mean = (3+2+3+3+2)/5 = 13/5 = 2.6

Alice centered: [5-4, 3-4, 4-4, 5-4, 3-4] = [1, -1, 0, 1, -1]
Bob centered: [3-2.6, 2-2.6, 3-2.6, 3-2.6, 2-2.6] = [0.4, -0.6, 0.4, 0.4, -0.6]
```

**Step 2**: Compute numerator (sum of products)

```
1*0.4 + (-1)*(-0.6) + 0*0.4 + 1*0.4 + (-1)*(-0.6)
= 0.4 + 0.6 + 0 + 0.4 + 0.6 = 2.0
```

**Step 3**: Compute denominator (product of standard deviations)

```
Alice variance = (1² + (-1)² + 0² + 1² + (-1)²) / 5 = 4/5 = 0.8
Alice std = sqrt(0.8) = 0.894
Actually, for Pearson we use sqrt(sum of squared deviations):
Alice: sqrt(1+1+0+1+1) = sqrt(4) = 2.0

Bob: sqrt(0.16+0.36+0.16+0.16+0.36) = sqrt(1.2) = 1.095
```

**Step 4**: Divide

$$\text{Pearson} = \frac{2.0}{2.0 \times 1.095} = \frac{2.0}{2.19} = 0.913$$

**Interpretation**: 0.913 correlation is very strong. Pearson correctly identifies that Alice and Bob have similar taste patterns despite different rating scales.

---

### 4. Jaccard Similarity

For Jaccard, we need to convert to binary (rated vs not rated, or above-average vs below-average).

**Using "rated this item" (binary):**

```
Alice rated: {M1, M2, M3, M4, M5} = all 5
Bob rated: {M1, M2, M3, M4, M5} = all 5

Intersection = 5 items
Union = 5 items

Jaccard = 5/5 = 1.0
```

**Problem**: This says they're identical, but ignores HOW they rated!

**Using "above average rating" (more informative):**

```
Alice's above-average movies (rating > 4.0): {M1, M4} (ratings 5 and 5)
Alice's below-or-equal: {M2, M3, M5} (ratings 3, 4, 3)

Bob's above-average movies (rating > 2.6): {M1, M3, M4} (ratings 3, 3, 3)
Bob's below-or-equal: {M2, M5} (ratings 2, 2)

Intersection of "liked" items = {M1, M4} ∩ {M1, M3, M4} = {M1, M4}
Union of "liked" items = {M1, M4} ∪ {M1, M3, M4} = {M1, M3, M4}

Jaccard (liked) = 2/3 = 0.67
```

**Interpretation**: 0.67 is moderate similarity, which might underestimate their true agreement.

---

### 5. Euclidean Distance (Inverse)

$$\text{distance}(Alice, Bob) = \sqrt{\sum_i (r_{Ai} - r_{Bi})^2}$$

```
distance = sqrt((5-3)² + (3-2)² + (4-3)² + (5-3)² + (3-2)²)
         = sqrt(4 + 1 + 1 + 4 + 1)
         = sqrt(11) = 3.32
```

$$\text{similarity} = \frac{1}{1 + \text{distance}} = \frac{1}{1 + 3.32} = 0.23$$

**Interpretation**: 0.23 is low! Euclidean sees them as very different because of the rating scale difference.

---

### Summary Comparison

| Measure | Alice-Bob Score | What It Captures |
|---------|----------------|------------------|
| Dot Product | 54 | Raw activity (meaningless alone) |
| Cosine | 0.994 | Direction similarity |
| Pearson | 0.913 | Pattern similarity (scale-invariant) |
| Jaccard (binary) | 1.0 | Overlap of items rated |
| Jaccard (liked) | 0.67 | Overlap of preferred items |
| Euclidean | 0.23 | Distance in rating space |

**Key insight**: Pearson is usually best for explicit ratings because it captures the pattern while ignoring scale differences.

---

## Detailed Formulas and Derivations

### 1. Cosine Similarity

**Formula:**

$$\text{cosine}(u, v) = \frac{\mathbf{r}_u \cdot \mathbf{r}_v}{||\mathbf{r}_u|| \cdot ||\mathbf{r}_v||} = \frac{\sum_i r_{ui} \cdot r_{vi}}{\sqrt{\sum_i r_{ui}^2} \cdot \sqrt{\sum_i r_{vi}^2}}$$

**Geometric intuition:**

```
Imagine users as arrows in |I|-dimensional space.

Alice: →→→→→ (direction based on ratings)
Bob:   →→→ (same direction, shorter arrow)
Carol: ↑↑↑ (different direction)

Cosine measures the ANGLE between arrows:
- Same direction (0°): cos(0°) = 1 (identical preferences)
- Perpendicular (90°): cos(90°) = 0 (no relationship)
- Opposite (180°): cos(180°) = -1 (opposite preferences)
```

**Range**: [-1, 1] in general, but [0, 1] when all ratings are positive.

**Properties:**
- Ignores magnitude (length of vector)
- Only considers direction
- Scale-independent for positive ratings

---

### 2. Pearson Correlation Coefficient

**Formula:**

$$\text{Pearson}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \cdot \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}$$

**Why mean-centering?**

```
Before centering:
  Alice: [5, 4, 5, 4] "I love things by rating 4-5"
  Bob:   [2, 1, 2, 1] "I love things by rating 1-2"

The numbers are completely different!

After centering (subtracting mean):
  Alice: [+1, 0, +1, 0] "I like movies 1 and 3 more than average"
  Bob:   [+0.5, -0.5, +0.5, -0.5] "I like movies 1 and 3 more than average"

Now the PATTERN is visible! Both prefer odd-numbered movies.
```

**Range**: [-1, +1]
- +1: Perfect positive correlation (identical patterns)
- 0: No correlation (random relationship)
- -1: Perfect negative correlation (opposite patterns)

**Properties:**
- Scale-invariant (handles generous/harsh raters)
- Mean-centered (removes user bias)
- Undefined if variance is zero (user rates everything the same)

---

### 3. Adjusted Cosine Similarity

**Formula:**

$$\text{adj\_cosine}(i, j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)(r_{uj} - \bar{r}_u)}{\sqrt{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)^2} \cdot \sqrt{\sum_{u \in U_{ij}} (r_{uj} - \bar{r}_u)^2}}$$

**Key difference from Pearson:**
- Pearson: Used for **user-user** similarity, subtracts **user** means
- Adjusted Cosine: Used for **item-item** similarity, subtracts **user** means

**Why subtract user means for items?**

```
Item1 and Item2 are both rated by Alice (generous) and Bob (harsh):

         Item1  Item2
Alice      5      5    (Alice's mean = 5)
Bob        2      2    (Bob's mean = 2)

Raw ratings suggest items have different "quality" because of user scales.

After adjusting:
Alice: [5-5, 5-5] = [0, 0]
Bob: [2-2, 2-2] = [0, 0]

Both users rated both items at exactly their average!
```

**When to use:**
- Item-based CF (computing item-item similarities)
- When different users have different rating scales

---

### 4. Jaccard Similarity

**Formula:**

$$\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

**Derivation from overlap intuition:**

```
User A interacted with items: {1, 2, 3, 5, 8}
User B interacted with items: {2, 3, 5, 7, 9}

Venn diagram:
     A only     Both     B only
    {1, 8}   {2, 3, 5}  {7, 9}
       2    +    3    +    2   = 7 total items

Jaccard = items in BOTH / items in EITHER = 3/7 = 0.43
```

**Range**: [0, 1]
- 0: No overlap at all
- 1: Perfect overlap (A = B)

**Properties:**
- Works with binary data (no ratings needed)
- Simple and intuitive
- Ignores how much users liked items

---

## When Each Measure Fails: Detailed Analysis

### Failure Mode 1: Pearson Fails with Few Overlapping Items

**Symptom**: Pearson gives extreme values (+1 or -1) that seem unreliable.

**Why it happens**: With only 2-3 co-rated items, any correlation looks perfect by chance.

**Concrete example:**

```
Alice and Bob only both rated 2 movies:

         Movie1  Movie2
Alice      5       3
Bob        4       2

Alice mean (on these 2): 4.0
Bob mean (on these 2): 3.0

Alice centered: [1, -1]
Bob centered: [1, -1]

Pearson = (1*1 + -1*-1) / (sqrt(2) * sqrt(2)) = 2/2 = 1.0

PERFECT CORRELATION! But it's based on only 2 data points!
```

**The statistical problem:**
- With 2 points, you can always fit a perfect line
- Need at least 5-10 co-rated items for reliability
- Even then, confidence intervals are wide

**Solutions:**
1. **Minimum overlap threshold**: Don't compute if < 5 co-rated items
2. **Significance weighting**: Multiply by $\min(1, |I_{uv}|/\tau)$ where $\tau \approx 50$
3. **Shrinkage toward zero**: $\text{sim}_{shrunk} = \text{sim} \times \frac{|I_{uv}|}{|I_{uv}| + \lambda}$

---

### Failure Mode 2: Cosine Ignores Rating Scales

**Symptom**: Users with different rating habits appear similar when they're not.

**We already saw this:**

```
Alice (generous): [5, 4, 5, 4]
Bob (harsh):      [2, 1, 2, 1]
Carol (uniform):  [5, 5, 5, 5]

Cosine(Alice, Carol) = 0.993
Cosine(Alice, Bob) = 0.978

But Alice and Bob have the SAME preferences!
```

**The deeper problem:**

```
What does a "4" mean?

For Alice: "It's okay, not my favorite" (below her average)
For Bob: "I absolutely loved it!" (way above his average)

Cosine sees both as "4" without context.
```

**When to use cosine anyway:**
- Binary data (all ratings are 0 or 1)
- Implicit feedback (all positive signals)
- When users are known to have similar scales

---

### Failure Mode 3: Jaccard Ignores Magnitude

**Symptom**: Users who "liked" items for different reasons appear similar.

**Concrete example:**

```
Both Alice and Bob watched (binary = 1) these movies:
{Inception, The Matrix, Interstellar, Arrival}

Jaccard might say they're similar (high overlap).

But what if we had ratings?

         Inception  Matrix  Interstellar  Arrival
Alice        5         5         5          5     (LOVED all of them)
Bob          2         2         2          2     (Watched but HATED all)

They "interacted" with the same items, but with opposite feelings!
```

**When Jaccard is still useful:**
- True binary data (click/no-click, purchase/no-purchase)
- When any interaction is a positive signal
- When you can't trust rating magnitudes

**Alternative: Weighted Jaccard**

$$\text{Weighted Jaccard}(A, B) = \frac{\sum_i \min(r_{Ai}, r_{Bi})}{\sum_i \max(r_{Ai}, r_{Bi})}$$

This gives credit for how much users agreed, not just whether they interacted.

---

### Failure Mode 4: Euclidean Is Dominated by Scale

**Symptom**: Similar users with different rating habits appear very dissimilar.

**Example:**

```
Alice: [5, 4, 5, 4, 5]  (generous, average = 4.6)
Bob:   [2, 1, 2, 1, 2]  (harsh, average = 1.6)

Euclidean distance:
sqrt((5-2)² + (4-1)² + (5-2)² + (4-1)² + (5-2)²)
= sqrt(9 + 9 + 9 + 9 + 9)
= sqrt(45) = 6.7

Similarity = 1/(1+6.7) = 0.13  VERY LOW!

But they have IDENTICAL preference patterns!
```

**When Euclidean might work:**
- Dense data where everyone rates on the same scale
- Normalized ratings (after z-score normalization)
- Embeddings that are already calibrated

---

### Failure Mode 5: Popularity Bias in Jaccard

**Symptom**: All users appear similar because they all interact with popular items.

**Concrete example:**

```
Popular items (everyone has them): {iPhone, Netflix, Amazon Prime}

User A: {iPhone, Netflix, Amazon Prime, Rare Jazz Album, Vintage Camera}
User B: {iPhone, Netflix, Amazon Prime, Gaming PC, Basketball}

Intersection: {iPhone, Netflix, Amazon Prime} = 3
Union: {iPhone, Netflix, Amazon Prime, Rare Jazz Album, Vintage Camera, Gaming PC, Basketball} = 7

Jaccard = 3/7 = 0.43

Not bad! But the similarity is entirely from POPULAR items.
User A likes jazz and photography; User B likes gaming and sports.
They have NOTHING in common in terms of actual taste!
```

**Solution: TF-IDF Weighting**

Downweight popular items:

$$\text{weight}(item) = \log\left(\frac{N}{n_{item}}\right)$$

where $N$ = total users, $n_{item}$ = users who interacted with item.

```
iPhone: log(1M / 800K) = log(1.25) = 0.1 (very popular, low weight)
Rare Jazz Album: log(1M / 500) = log(2000) = 7.6 (rare, high weight)
```

---

## Advanced Techniques

### Significance Weighting

**Problem**: Similarity based on few overlaps is unreliable.

**Solution**: Penalize similarities with low support.

$$\text{sim}_{weighted}(u, v) = \text{sim}(u, v) \cdot \min\left(1, \frac{|I_{uv}|}{\tau}\right)$$

**Example with** $\tau = 50$:

| Users | Raw Sim | Overlap | Weight | Weighted Sim |
|-------|---------|---------|--------|--------------|
| A-B | 0.95 | 5 | 5/50 = 0.1 | 0.095 |
| A-C | 0.60 | 100 | 1.0 | 0.60 |
| A-D | 0.80 | 30 | 30/50 = 0.6 | 0.48 |

*Notice*: A-B had the highest raw similarity but the lowest overlap, so it's heavily penalized!

---

### Variance Weighting

**Problem**: Users who rate everything the same provide no information.

**Solution**: Weight by user variance.

$$\text{sim}_{weighted}(u, v) = \text{sim}(u, v) \cdot \sigma_u \cdot \sigma_v$$

**Example:**

```
Alice: [5, 5, 5, 5, 5], variance = 0, std = 0
Bob: [1, 3, 5, 2, 4], variance = 2.5, std = 1.58
Carol: [2, 4, 3, 5, 1], variance = 2.5, std = 1.58

sim(Alice, Bob) = 0.8 * 0 * 1.58 = 0  (Alice uninformative)
sim(Bob, Carol) = 0.6 * 1.58 * 1.58 = 1.5 (both informative)
```

---

### Case Amplification

**Problem**: Similarities near 0.5 are ambiguous (weakly similar or weakly dissimilar?).

**Solution**: Amplify to sharpen distinctions.

$$\text{sim}_{amplified} = \text{sign}(\text{sim}) \cdot |\text{sim}|^\rho$$

where $\rho > 1$ (typically 2.5).

**Effect:**

| Original | Amplified (rho=2.5) |
|----------|---------------------|
| 0.9 | 0.90^2.5 = 0.87 |
| 0.7 | 0.70^2.5 = 0.48 |
| 0.5 | 0.50^2.5 = 0.18 |
| 0.3 | 0.30^2.5 = 0.05 |

*High similarities stay relatively high; low similarities get crushed.*

---

## Decision Guide: Which Metric Should I Use?

```
                         Start Here
                             |
              Is your data binary (yes/no)?
                    /            \
                  Yes             No
                   |               |
              Use Jaccard     Do users have different
                              rating scales?
                                /            \
                              Yes             No
                               |               |
                          Use Pearson     Use Cosine
                               |
                    Is this for items or users?
                         /            \
                      Items          Users
                        |              |
                   Adjusted         Pearson
                    Cosine

Additional considerations:
- Few co-rated items? -> Add significance weighting
- Many uninformative users? -> Add variance weighting
- Popularity bias? -> Use TF-IDF or normalized Jaccard
```

---

## Summary

### Quick Reference

| Metric | Formula | Range | Best For | Failure Mode |
|--------|---------|-------|----------|--------------|
| **Cosine** | $\frac{\mathbf{u} \cdot \mathbf{v}}{\\|\mathbf{u}\\| \\|\mathbf{v}\\|}$ | [0,1] | Binary, implicit | Ignores rating scales |
| **Pearson** | $\frac{\sum (r_i - \bar{r}_u)(r_i - \bar{r}_v)}{\sigma_u \sigma_v}$ | [-1,1] | Explicit ratings | Needs many overlaps |
| **Adjusted Cosine** | Cosine with user mean subtraction | [-1,1] | Item-based CF | Needs many co-raters |
| **Jaccard** | $\frac{\\|A \cap B\\|}{\\|A \cup B\\|}$ | [0,1] | Binary sets | Ignores magnitude |
| **Euclidean** | $\frac{1}{1 + \\|\mathbf{u} - \mathbf{v}\\|}$ | (0,1] | Normalized data | Dominated by scale |

### Key Takeaways

1. **No single measure is best** -- choose based on your data type
2. **Always add significance weighting** when overlaps are small
3. **Pearson is usually best for explicit ratings** (handles scale differences)
4. **Jaccard is best for pure binary data** (no magnitude to consider)
5. **Test multiple measures** on your specific dataset

**Next**: See **code-examples.md** for full implementations.

---

## References

1. **Herlocker, J. L., et al. (2002)**. "Evaluating collaborative filtering recommender systems". *ACM TOIS*.
2. **Sarwar, B., et al. (2001)**. "Item-based collaborative filtering recommendation algorithms". *WWW*.
3. **Breese, J. S., et al. (1998)**. "Empirical analysis of predictive algorithms for collaborative filtering". *UAI*.
