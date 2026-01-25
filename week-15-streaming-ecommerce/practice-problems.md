# Week 15: Streaming and E-Commerce Platforms - Practice Problems

## Overview
Study Netflix, Spotify, YouTube, and Amazon recommendation systems with different objectives (watch time, purchases, streams).

---

## Problem 1: Netflix Row Generation
**Difficulty:** Medium

Netflix homepage has personalized rows (genres):
- "Action Movies You'll Love"
- "Trending Now"
- "Because You Watched Inception"

**Design:**
1. How do you generate row titles?
2. How do you decide which rows to show?
3. How do you rank items within each row?
4. How do you rank rows themselves?

**Learning Outcomes:** Design multi-level recommendations, personalize categories, optimize layout

---

## Problem 2: Spotify Discover Weekly
**Difficulty:** Hard

**Inputs:**
- Collaborative filtering (2B playlists)
- Audio features (CNN on spectrograms)
- NLP on playlist names

**Fusion:**
1. How do you combine three signal types?
2. Weight each signal? (CF=0.5, Audio=0.3, NLP=0.2?)
3. Cold start: New song with no interactions?
4. Evaluate: Precision vs. discovery trade-off

**Learning Outcomes:** Design multi-modal systems, balance signals, handle cold start

---

## Problem 3: Amazon Item-to-Item CF
**Difficulty:** Medium

**"Customers who bought this also bought":**

**Algorithm:**
1. For each item, find similar items (cosine similarity on co-purchase vectors)
2. Precompute and store
3. At serving: Look up similar items, personalize order

**Questions:**
1. Why item-item instead of user-user?
2. How often do you recompute similarities?
3. How do you personalize the "also bought" list?
4. Handle new products?

**Learning Outcomes:** Design item-item systems, optimize precomputation, personalize generic recs

---

## Problem 4: YouTube Watch Time Optimization
**Difficulty:** Hard

**Objective:** Maximize total watch time (not just clicks)

**Challenge:** Clickbait gets clicks but short watch time

**Approach:**
1. Train to predict watch time (regression)
2. Rank by predicted watch time
3. Balance with CTR (clicks matter too)

**Questions:**
1. How do you collect watch time labels?
2. Handle censored data (user stopped for external reason)?
3. Design loss function
4. Evaluate: Watch time vs. user satisfaction

**Learning Outcomes:** Optimize for engagement quality, handle censored data, balance metrics

---

## Problem 5: E-Commerce Conversion Optimization
**Difficulty:** Hard

**Funnel:** View → Add to Cart → Purchase

**Multi-Objective:**
- Maximize clicks (traffic)
- Maximize add-to-cart
- Maximize purchases (revenue)

**Design:**
1. Multi-task model predicting all three
2. Scoring function combining objectives
3. A/B test: Revenue-optimized vs. click-optimized

**Learning Outcomes:** Optimize conversion funnels, balance short and long-term metrics, drive revenue

---

## Programming Exercises

### Exercise 1: Implement Netflix Row Ranker

```python
def generate_rows(user_id, candidate_rows):
    # Score each row by expected engagement
    row_scores = {}
    for row in candidate_rows:
        items = get_row_items(row, user_id)
        expected_engagement = predict_row_engagement(user_id, items)
        row_scores[row] = expected_engagement

    # Rank rows
    ranked_rows = sorted(row_scores, key=row_scores.get, reverse=True)

    # Diversify (avoid similar rows back-to-back)
    diversified = diversify_rows(ranked_rows)

    return diversified[:10]  # Top 10 rows
```

---

### Exercise 2: Multi-Modal Spotify Recommender

```python
def recommend_songs(user_id):
    # CF score
    cf_scores = collaborative_filtering_scores(user_id)

    # Audio similarity
    liked_songs = get_user_likes(user_id)
    audio_profiles = [get_audio_features(song) for song in liked_songs]
    user_audio_profile = np.mean(audio_profiles, axis=0)
    audio_scores = {song: cosine_similarity(user_audio_profile, get_audio_features(song))
                    for song in all_songs}

    # Combine
    alpha, beta = 0.7, 0.3
    final_scores = {song: alpha * cf_scores.get(song, 0) + beta * audio_scores.get(song, 0)
                    for song in all_songs}

    return sorted(final_scores, key=final_scores.get, reverse=True)[:30]
```

---

### Exercise 3: Predict Watch Time

```python
class WatchTimePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, features):
        return self.network(features)  # Predict watch time in seconds

# Training with weighted loss (longer videos have more variance)
def weighted_mse_loss(predictions, targets, video_durations):
    weights = 1.0 / video_durations  # Downweight long videos
    return torch.mean(weights * (predictions - targets) ** 2)
```

---

## Discussion Questions

1. **Binge Watching:** Is optimizing for watch time ethical? (Netflix addictive design)
2. **Revenue vs. Engagement:** Amazon could optimize clicks or purchases. Which?
3. **Discovery vs. Familiarity:** Spotify balance known favorites vs. new music
4. **Autoplay:** YouTube autoplay keeps users watching. Good or manipulative?
5. **Long-tail:** How do platforms balance popular vs. niche content?

---

## References
1. Gomez-Uribe, C. A., & Hunt, N. (2016). "The Netflix recommender system". ACM TIST.
2. Covington, P., et al. (2016). "Deep neural networks for YouTube recommendations". RecSys.
3. Linden, G., et al. (2003). "Amazon.com recommendations: Item-to-item collaborative filtering". IEEE Internet Computing.

---

*Return to [Week 15 Main Page](README.md)*
