# Week 1: Applications Across Domains

## Learning Objectives

- Identify major application domains for recommendation systems
- Understand domain-specific challenges and requirements
- Recognize how recommendations drive business value across industries

---

## 1. E-Commerce

### Amazon

**Use Cases**:
- **Product recommendations**: "Customers who bought this also bought"
- **Personalized homepage**: Different for each user
- **Search ranking**: Personalized search results
- **Email campaigns**: Product suggestions in emails

**Business Impact**:
- **35% of revenue** attributed to recommendations (McKinsey estimate)
- Q1 2024: \$143B in net sales
- Recommendations increase average order value

**Technical Approach**:
- Item-to-item collaborative filtering (Linden et al., 2003)
- Deep learning models for personalization
- Real-time session-based recommendations

**Metrics**:
- Click-through rate (CTR)
- Conversion rate
- Revenue per user
- Cart abandonment reduction

---

### eBay

**Use Cases**:
- **Similar items**: Alternatives to current listing
- **Personalized best matches**: Ranked search results
- **Related searches**: Query suggestions
- **Dynamic pricing**: Price recommendations for sellers

**Challenges**:
- **Inventory volatility**: Items sell out quickly
- **Two-sided marketplace**: Buyers and sellers
- **Diversity**: Extremely heterogeneous catalog

**Approach**:
- Collaborative filtering + content-based hybrid
- Real-time inventory tracking
- Auction dynamics integration

---

## 2. Streaming Platforms

### Netflix

**Scale**:
- 260M+ subscribers (2024)
- 15K+ titles
- 190+ countries
- **80%+ of viewing** from recommendations

**Recommendation Types**:
1. **Personalized rows**: Different genres/themes per user
2. **Top picks**: Best matches for user
3. **Because you watched X**: Similar content
4. **Trending now**: Popular + personalized
5. **Artwork personalization**: Different thumbnails per user

**Technical Stack**:
- Two-stage: Candidate generation → ranking
- Multi-task learning (engagement, completion, satisfaction)
- Contextual bandits for artwork selection
- Deep learning ensembles

**Metrics**:
- **Watch time**: Primary metric
- Completion rate
- Retention
- Long-term engagement

**Reference**: Gomez-Uribe & Hunt (2016). "The Netflix recommender system". *ACM TIST*.

---

### Spotify

**Scale**:
- 600M+ users (2024)
- 100M+ songs
- 5B+ playlists
- **30%+ of streams** from recommendations

**Flagship Features**:

#### **Discover Weekly** (2015)
- **Personalized**: 30-song playlist updated every Monday
- **Approach**: Collaborative filtering + audio analysis + NLP
- **Impact**: 40M users → 100M+ users

#### **Daily Mix** (2016)
- **Personalized**: 6 playlists based on listening clusters
- **Approach**: K-means clustering on user taste profiles
- **Goal**: Familiar favorites + occasional discovery

#### **Release Radar** (2018)
- **Personalized**: New releases from followed artists + similar artists
- **Frequency**: Updated every Friday

#### **Radio**
- **Session-based**: Sequential recommendations
- **Approach**: Seed song → similar songs via embeddings

**Technical Approach**:
- **Collaborative filtering**: 2B user-generated playlists
- **Audio analysis**: CNN for tempo, energy, danceability, valence
- **NLP**: Playlist names, metadata, lyrics
- **Embeddings**: Song2Vec, Artist2Vec

**Metrics**:
- Stream count
- Skip rate (negative signal)
- Save/add to playlist
- Listen-through rate

---

### YouTube

**Scale**:
- 2.7B+ users
- 800M+ videos
- 1B+ hours watched daily
- **70%+ of watch time** from recommendations

**Recommendation Surfaces**:
1. **Homepage**: Personalized feed
2. **Watch next**: Autoplay and sidebar
3. **Subscriptions**: New uploads from subscriptions
4. **Shorts**: TikTok-like short-form video feed

**Architecture** (Covington et al., 2016):
- **Stage 1**: Candidate generation (billions of videos → hundreds)
- **Stage 2**: Ranking (hundreds → dozens, ordered by watch time)

**Challenges**:
- **Scale**: Billions of users, millions of creators
- **Freshness**: New videos uploaded every second
- **Responsibility**: Misinformation, harmful content
- **Clickbait**: Misleading thumbnails

**Metrics**:
- **Watch time**: Primary optimization goal
- Click-through rate (CTR)
- Likes, comments, shares
- Subscription rate

---

## 3. Social Networks

### Facebook News Feed

**Goal**: Show most relevant posts, links, photos, videos

**Evolution**:
- **EdgeRank (2010)**: Affinity × Weight × Time Decay
- **Machine Learning (2013+)**: Thousands of signals
- **Deep Learning (2018+)**: Multi-task neural networks

**Ranking Signals**:
- **User interactions**: Likes, comments, shares, clicks, time spent
- **Post type**: Video, photo, link, text
- **Recency**: Newer content prioritized
- **Relationship**: Close friends vs. acquaintances
- **Content quality**: Clickbait detection, misinformation scores

**Multi-Objective Optimization**:
- **Engagement**: Likes, comments, shares
- **Meaningful interactions**: Quality over quantity
- **Well-being**: Reduce polarization, addictive content
- **Business**: Ad revenue

**Challenges**:
- Filter bubbles and echo chambers
- Misinformation spread
- Mental health concerns
- Regulatory scrutiny

---

### Instagram

**Recommendation Surfaces**:
1. **Feed**: Posts from followed accounts + suggested posts
2. **Explore**: Discover new content and creators
3. **Reels**: Short-form videos (competing with TikTok)
4. **Stories**: Ephemeral content from followed accounts

**Explore Algorithm (2025)**:
- **Step 1**: Seed from user's likes/saves
- **Step 2**: Find similar content via embeddings
- **Step 3**: Rank by engagement likelihood
- **Step 4**: Diversify topics

**Top Ranking Signals** (confirmed by Instagram, Jan 2025):
1. **Watch time**: Especially first 3 seconds
2. **Likes per reach**: % of viewers who like
3. **Sends per reach**: DM sharing (most powerful)

**Reels Algorithm**:
- **Watch-until-end**: Primary signal
- **Audio**: Trending sounds boost visibility
- **Effects**: Trending AR effects
- **Original content**: No watermarks from other apps

---

### TikTok For You Page

**Innovation**: Interest graph > social graph

**Ranking Signals**:
1. **User interactions**: Watch time, likes, shares, comments, skips
2. **Video information**: Captions, hashtags, sounds, effects
3. **User information**: Language, country, device type

**Key Characteristics**:
- **Personalized from day one**: Initial interests + rapid learning
- **Viral potential**: Zero-follower videos can go viral
- **Engagement optimization**: Watch time, completion rate
- **Diversification**: Prevents content fatigue

**Algorithm Mechanics** (confirmed by TikTok):
- Not influenced by follower count or previous video performance
- Each video is tested with small audience first
- High engagement → broader distribution
- Fresh content prioritized

**Challenges**:
- **Addictive design**: Infinite scroll + perfect personalization
- **Mental health**: Comparison, FOMO
- **Echo chambers**: Reinforcing beliefs
- **Content moderation**: Harmful content

---

## 4. News and Content Aggregation

### Google News

**Approach**:
- **Personalization**: Based on search history, location, interests
- **Editorial curation**: Human oversight + algorithms
- **Freshness**: Real-time breaking news
- **Source diversity**: Multiple viewpoints

**Ranking Signals**:
- **Recency**: Breaking news prioritized
- **Source authority**: Credible publishers ranked higher
- **User interests**: Personalization
- **Engagement**: Clicks, time spent
- **Fact-checking**: Verified vs. misinformation

**Challenges**:
- Filter bubbles (showing diverse viewpoints)
- Misinformation and fake news
- Balancing personalization with serendipity

---

### Reddit

**Use Cases**:
- **Home feed**: Personalized posts from subscribed subreddits
- **Popular**: Trending across all of Reddit
- **Recommended communities**: Subreddit discovery

**Ranking**:
- **Upvotes - downvotes**: Community voting
- **Recency**: Time decay function
- **Engagement**: Comments, awards
- **User history**: Subscribed subreddits, interactions

**Unique Aspect**: Community-driven curation + algorithmic boost

---

## 5. Dating Apps

### Tinder

**Challenge**: Two-sided matching (mutual interest required)

**Algorithm**:
- **Elo score** (historically): Chess-like rating system
- **Current**: Machine learning on swipe patterns
- **Signals**: Swipes, messages, profile completeness

**Approach**:
1. Show profiles likely to swipe right on you
2. Show profiles you're likely to swipe right on
3. Prioritize active users
4. Promote profile completeness

**Business Model Tension**:
- Good matches → users leave app (bad for business)
- Poor matches → user frustration (bad for retention)
- **Balance**: Successful matches at sustainable rate

---

### Hinge

**Tagline**: "Designed to be deleted"

**Approach**:
- **Most Compatible**: Daily personalized recommendation
- **Standouts**: 10 profiles predicted as best matches
- **Machine learning**: Learns from "We Met" feedback

**Differentiator**: Optimizes for dates, not just matches

---

## 6. Music Discovery

### Spotify (covered above in streaming)

### Apple Music

**Features**:
- **For You**: Personalized playlists
- **Listen Now**: Recent + recommended
- **New Music Mix**: Discovery
- **Favorites Mix**: Best-loved tracks

**Approach**:
- Human curation + algorithmic personalization
- Editorial playlists as seeds
- Collaborative filtering on listening behavior

---

## 7. Professional Networks

### LinkedIn

**Recommendation Types**:
1. **Feed**: Posts, articles, updates
2. **Jobs**: Personalized job recommendations
3. **People You May Know (PYMK)**: Connection suggestions
4. **Learning**: Course recommendations
5. **Content**: Articles, news

**Job Recommendations**:
- **Signals**: Skills, experience, education, location, salary
- **Graph**: Professional network structure
- **Engagement**: Applications, saves, views

**PYMK Algorithm**:
- Mutual connections
- Same company/school
- Similar industries/skills
- Profile views

**Challenge**: Balancing organic content with sponsored ads

---

## 8. Travel and Marketplace

### Airbnb

**Challenges**:
- **Two-sided marketplace**: Guest satisfaction + host earnings
- **Location-based**: Geographic constraints
- **Availability**: Real-time booking dynamics
- **Price sensitivity**: Budget constraints

**Ranking Signals**:
- **Search query**: Location, dates, guests, amenities
- **Guest preferences**: Previous bookings, saves
- **Listing quality**: Photos, description, reviews
- **Host**: Superhost status, response rate
- **Price**: Budget alignment
- **Availability**: Booking probability

**Personalization**:
- Search ranking
- Similar listings
- Experiences recommendations
- Destination suggestions

---

### Uber Eats

**Use Cases**:
- Restaurant discovery
- Dish recommendations
- Reorder shortcuts
- Promotions

**Ranking Signals**:
- **Cuisine preferences**: Previous orders
- **Time of day**: Breakfast vs. dinner
- **Location**: Delivery time, distance
- **Weather**: Cold day → soup, hot day → ice cream
- **Ratings**: Restaurant and dish ratings

**Challenge**: Balancing discovery (new restaurants) with favorites

---

## 9. Education

### Coursera

**Recommendations**:
- **Courses**: Based on career goals, skills
- **Learning paths**: Structured sequences
- **Specializations**: Multi-course programs
- **Projects**: Hands-on experience

**Personalization**:
- Career goals
- Current skills (self-reported + inferred)
- Industry/role
- Learning progress and performance

---

### Khan Academy

**Adaptive Learning**:
- Mastery-based progression
- Difficulty adaptation
- Prerequisite knowledge modeling
- Personalized practice

**Approach**: Educational graph + student performance → next best exercise

---

## 10. Developer Tools

### GitHub

**Recommendations**:
- **Repositories**: Based on stars, contributions
- **Trending**: Daily/weekly/monthly trends by language
- **Topics**: Interest-based discovery
- **Dependency suggestions**: Package recommendations

---

### Cursor AI (2024)

**Innovation**: Online RL for code completion

**Approach**:
- 400M+ requests/day
- Real-time RL from user acceptance/rejection
- 21% fewer suggestions (noise reduction)
- 28% higher accept rate

**Metric**: Reward on acceptance, penalty on rejection

**Impact**: First large-scale online RL in production for code

---

## Domain-Specific Considerations

| Domain | Primary Metric | Key Challenge | Unique Aspect |
|--------|---------------|---------------|---------------|
| E-commerce | Revenue, conversion | Cold start for new products | Inventory dynamics |
| Streaming | Watch time | Content discovery vs. familiarity | Session length optimization |
| Social | Engagement, time spent | Filter bubbles, mental health | Multi-objective (user + platform) |
| News | Click-through, diversity | Misinformation, bias | Freshness + accuracy |
| Dating | Matches, dates | Two-sided matching | Success = user leaves |
| Travel | Bookings | Supply-demand imbalance | Geographic constraints |
| Education | Learning outcomes | Knowledge prerequisites | Adaptive difficulty |

---

## Common Patterns Across Domains

1. **Two-stage architecture**: Fast candidate generation → expensive ranking
2. **Multi-objective optimization**: Balancing multiple goals
3. **Personalization vs. exploration**: Familiar vs. novel
4. **Real-time adaptation**: Session-based, contextual
5. **A/B testing**: Continuous experimentation
6. **Fairness**: Provider (creator) and consumer (user) fairness

---

## Summary

Recommendation systems are ubiquitous across industries:
- **E-commerce**: Drive 35% of revenue (Amazon)
- **Streaming**: Account for 80% of consumption (Netflix)
- **Social**: Curate infinite feeds (Facebook, TikTok)
- **Dating**: Enable two-sided matching (Tinder, Hinge)
- **Professional**: Power job search and networking (LinkedIn)

Each domain has unique challenges, but common patterns emerge: personalization, scalability, diversity, and business alignment.

**Next**: See **pipeline.md** for the end-to-end recommendation pipeline

---

## References

1. Linden, G., Smith, B., & York, J. (2003). "Amazon.com recommendations". *IEEE Internet Computing*.
2. Gomez-Uribe, C. A., & Hunt, N. (2016). "The Netflix recommender system". *ACM TIST*.
3. Covington, P., et al. (2016). "Deep neural networks for YouTube recommendations". *RecSys*.
4. Cialdini, T. (2021). "How Spotify's algorithm knows exactly what you want to listen to". *Business Insider*.
