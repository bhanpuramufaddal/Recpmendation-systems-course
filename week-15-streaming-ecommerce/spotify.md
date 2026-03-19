# System Design: Spotify Music Recommendations

## Problem Statement & Requirements

### Interview Prompt

> "Design a music recommendation system for Spotify with 600M+ users and 100M+ tracks, powering personalized playlists like Discover Weekly."

### Functional Requirements

1. **Discover Weekly**: 30-song personalized playlist, updated every Monday
2. **Daily Mix**: Multiple genre-based personalized playlists
3. **Home page**: Personalized feed of playlists, albums, and podcasts
4. **Radio**: Endless stream seeded by artist/track/playlist
5. **Release Radar**: New music from followed artists
6. **Made For You**: Real-time personalized recommendations

### Non-Functional Requirements

1. **Latency**: Home page load < 500ms
2. **Freshness**: Discover Weekly generated Sunday night
3. **Scale**: 600M users, 100M tracks, 5B playlists
4. **Availability**: 99.9% uptime
5. **Personalization**: Meaningful recommendations from first session

### Scope

**In scope**:
- Music recommendation algorithm
- Playlist generation
- Candidate retrieval and ranking
- Cold start for users and tracks

**Out of scope**:
- Audio streaming infrastructure
- Podcast recommendations (different system)
- Ads and monetization

---

## Scale Estimation (Back-of-Envelope)

### Users & Traffic

```
Users:
- Monthly Active Users (MAU): 600M
- Daily Active Users (DAU): 200M
- Premium subscribers: 250M
- Free users: 350M

Listening behavior:
- Average sessions per user per day: 2
- Average session duration: 30 minutes
- Average tracks per session: 10
- Total track plays per day: 200M × 2 × 10 = 4B plays/day

Recommendation requests:
- Home page loads per day: 200M × 3 = 600M
- Discover Weekly generation: 600M users × 1/week = 86M/day (batch)
- Radio requests: 100M/day

QPS:
- Home page: 600M / 86,400 = 7,000 QPS
- Peak QPS: 20,000 QPS
```

### Content Scale

```
Tracks:
- Total tracks: 100M+
- New tracks per day: 100,000
- Active tracks (played in last 30 days): 30M

Playlists:
- User-created playlists: 5B
- Editorial playlists: 100K
- Algorithmic playlists: 600M × 10 = 6B (Daily Mix, Discover Weekly, etc.)

Artists:
- Total artists: 11M
- Active artists: 3M
```

### Storage

```
Track Embeddings:
- Tracks: 100M
- Embedding dimension: 128
- Storage: 100M × 128 × 4 bytes = 50GB

User Embeddings:
- Users: 600M
- Embedding dimension: 128
- Storage: 600M × 128 × 4 bytes = 300GB

Audio Features:
- Tracks: 100M
- Features per track: 20 (tempo, energy, etc.)
- Storage: 100M × 20 × 4 bytes = 8GB

Playlist Co-occurrence Matrix:
- Active tracks: 30M
- Sparse storage: ~100GB
```

### Latency Budget (Home Page)

```
Total budget: 500ms

Component breakdown:
- User feature lookup: 30ms
- Candidate retrieval (multiple sources): 100ms
- Track feature lookup: 50ms
- Ranking model inference: 100ms
- Blending and diversity: 50ms
- Playlist metadata: 50ms
- Network overhead: 120ms
```

---

## High-Level Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           Recommendation Surfaces        │
                    │  ┌─────────┐ ┌─────────┐ ┌───────────┐  │
                    │  │ Home    │ │ Discover│ │ Radio     │  │
                    │  │ Page    │ │ Weekly  │ │           │  │
                    │  └─────────┘ └─────────┘ └───────────┘  │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │          Candidate Generation           │
                    │                                         │
                    │  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
                    │  │ Collab   │ │ Audio    │ │Playlist │ │
                    │  │ Filter   │ │ Based    │ │ Embed   │ │
                    │  └──────────┘ └──────────┘ └─────────┘ │
                    │                                         │
                    │  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
                    │  │ Artist   │ │ Editorial│ │Trending │ │
                    │  │ Graph    │ │ Curated  │ │         │ │
                    │  └──────────┘ └──────────┘ └─────────┘ │
                    └──────────────────┬──────────────────────┘
                                       │ ~5000 candidates
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │              Ranker                      │
                    │  P(stream), P(skip), P(save)            │
                    └──────────────────┬──────────────────────┘
                                       │ ~500 ranked
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │           Blender / Sequencer           │
                    │  - Familiar vs Discovery balance        │
                    │  - Genre/mood coherence                 │
                    │  - Skip prevention                      │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │         Final Recommendations            │
                    └─────────────────────────────────────────┘
```

---

## Data Model

### User Profile

```python
class UserProfile:
    user_id: str

    # Account
    subscription_type: str        # "free", "premium", "family"
    account_age_days: int

    # Listening history
    top_artists: List[str]        # Top 50 artists (all time)
    top_tracks: List[str]         # Top 100 tracks (all time)
    recent_tracks: List[str]      # Last 500 tracks played
    recent_artists: List[str]     # Last 100 artists

    # Preferences (learned)
    genre_affinity: Dict[str, float]    # {"rock": 0.4, "pop": 0.3, ...}
    audio_preferences: Dict[str, float]  # {"energy": 0.7, "tempo": 120, ...}
    listening_patterns: Dict[str, Any]   # Time of day, device, etc.

    # Embeddings
    taste_embedding: List[float]   # 128-dim (learned from listening)
    social_embedding: List[float]  # 128-dim (from followed users/playlists)

    # Explicit signals
    liked_tracks: List[str]
    saved_albums: List[str]
    followed_artists: List[str]
    created_playlists: List[str]
```

### Track Features

```python
class Track:
    track_id: str
    name: str
    artist_id: str
    album_id: str

    # Audio features (Spotify Audio Analysis API)
    duration_ms: int
    tempo: float              # BPM (0-250)
    energy: float             # 0-1 (intensity)
    danceability: float       # 0-1 (rhythm stability)
    valence: float            # 0-1 (happiness)
    acousticness: float       # 0-1 (acoustic vs electronic)
    instrumentalness: float   # 0-1 (no vocals)
    speechiness: float        # 0-1 (spoken word content)
    loudness: float           # dB (-60 to 0)
    key: int                  # 0-11 (musical key)
    mode: int                 # 0=minor, 1=major

    # ML-derived
    audio_embedding: List[float]    # 128-dim (CNN on spectrogram)
    content_embedding: List[float]  # 128-dim (combined features)

    # Metadata
    genres: List[str]
    release_date: datetime
    explicit: bool
    popularity: int           # 0-100 (Spotify popularity score)

    # Engagement (aggregated)
    total_streams: int
    skip_rate: float          # % listeners who skip
    save_rate: float          # % who save to library
    playlist_add_rate: float  # % who add to playlist
```

---

## Candidate Generation

### Source 1: Collaborative Filtering

**Course Connection**: Week 3 (Matrix Factorization)

```python
class CollaborativeFilteringSource:
    """
    User-track matrix factorization using ALS.
    """
    def __init__(self, embedding_dim=128):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=embedding_dim,
            regularization=0.1,
            iterations=50
        )

    def train(self, user_track_matrix: csr_matrix):
        """
        Train on implicit feedback (stream counts).
        """
        # Weight: c_ui = 1 + alpha * log(1 + plays)
        self.model.fit(user_track_matrix)

    def get_candidates(self, user_id: str, k: int = 1000) -> List[str]:
        """
        Get top-k similar tracks for user.
        """
        user_idx = self.user_to_idx[user_id]
        track_ids, scores = self.model.recommend(
            user_idx,
            user_track_matrix[user_idx],
            N=k,
            filter_already_liked_items=True
        )
        return [self.idx_to_track[i] for i in track_ids]
```

### Source 2: Audio-Based Similarity (Content-Based)

**Course Connection**: Week 4 (Content-Based Filtering)

```python
class AudioBasedSource:
    """
    Find similar tracks using audio features from CNN.
    """
    def __init__(self):
        self.audio_model = SpectrogramCNN()  # Pre-trained
        self.ann_index = faiss.IndexFlatIP(128)  # 128-dim embeddings

    def extract_embedding(self, audio_file: str) -> np.ndarray:
        """
        Extract 128-dim embedding from audio spectrogram.
        """
        # Load audio, compute mel spectrogram
        y, sr = librosa.load(audio_file, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel = librosa.power_to_db(mel_spec)

        # Pass through CNN
        with torch.no_grad():
            embedding = self.audio_model(log_mel)

        return embedding.numpy()

    def get_similar_tracks(self, seed_track_ids: List[str], k: int = 500) -> List[str]:
        """
        Find tracks with similar audio characteristics.
        """
        # Average seed embeddings
        seed_embeddings = [self.track_embeddings[tid] for tid in seed_track_ids]
        query = np.mean(seed_embeddings, axis=0)

        # ANN search
        distances, indices = self.ann_index.search(query.reshape(1, -1), k)
        return [self.idx_to_track[i] for i in indices[0]]
```

### Source 3: Playlist Co-occurrence (Playlist2Vec)

**Course Connection**: Week 9 (Embeddings)

```python
class PlaylistEmbeddingSource:
    """
    Word2Vec-style embeddings from playlist co-occurrence.

    Intuition: Tracks that appear together in playlists are similar.
    """
    def __init__(self, embedding_dim=128, window_size=10):
        self.model = Word2Vec(
            vector_size=embedding_dim,
            window=window_size,
            min_count=5,
            workers=8,
            sg=1  # Skip-gram
        )

    def train(self, playlists: List[List[str]]):
        """
        Train on playlist sequences (shuffled for each epoch).

        playlist = [track_a, track_b, track_c, ...]
        Context for track_a: [track_b, track_c] (within window)
        """
        self.model.train(playlists, total_examples=len(playlists), epochs=10)

    def get_similar_tracks(self, seed_tracks: List[str], k: int = 500) -> List[str]:
        """
        Find tracks that co-occur in playlists with seed tracks.
        """
        # Average seed embeddings
        seed_embeddings = [self.model.wv[t] for t in seed_tracks if t in self.model.wv]
        if not seed_embeddings:
            return []

        query = np.mean(seed_embeddings, axis=0)

        # Most similar tracks
        similar = self.model.wv.most_similar(positive=[query], topn=k)
        return [track_id for track_id, score in similar]
```

### Source 4: Artist Graph

**Course Connection**: Week 7 (Graph-Based Methods)

```python
class ArtistGraphSource:
    """
    Graph-based recommendations using artist similarity.

    Graph edges:
    - Collaboration (featured artists)
    - Similar listener base (co-listened artists)
    - Genre similarity
    """
    def __init__(self):
        self.artist_graph = nx.Graph()
        self.artist_embeddings = {}  # Node2Vec embeddings

    def build_graph(self, collaborations: List, co_listens: List):
        # Add collaboration edges
        for artist_a, artist_b, song in collaborations:
            self.artist_graph.add_edge(artist_a, artist_b, weight=1.0)

        # Add co-listen edges (weighted by similarity)
        for artist_a, artist_b, similarity in co_listens:
            if similarity > 0.3:  # Threshold
                self.artist_graph.add_edge(artist_a, artist_b, weight=similarity)

    def get_similar_artists(self, seed_artists: List[str], k: int = 50) -> List[str]:
        """
        Find related artists via graph traversal.
        """
        similar_artists = set()

        for artist in seed_artists:
            # 2-hop neighbors
            for neighbor in self.artist_graph.neighbors(artist):
                similar_artists.add(neighbor)
                for second_neighbor in self.artist_graph.neighbors(neighbor):
                    similar_artists.add(second_neighbor)

        # Rank by embedding similarity
        seed_embedding = np.mean([self.artist_embeddings[a] for a in seed_artists], axis=0)
        scored = [
            (a, cosine_similarity(seed_embedding, self.artist_embeddings[a]))
            for a in similar_artists if a not in seed_artists
        ]
        scored.sort(key=lambda x: -x[1])

        return [a for a, _ in scored[:k]]

    def get_tracks_from_artists(self, artists: List[str], k: int = 500) -> List[str]:
        """
        Get top tracks from similar artists.
        """
        tracks = []
        for artist in artists:
            artist_top_tracks = self.get_artist_top_tracks(artist, limit=10)
            tracks.extend(artist_top_tracks)
        return tracks[:k]
```

### Candidate Merging

```python
def generate_candidates(user: UserProfile) -> List[Track]:
    """
    Merge candidates from all sources with deduplication.
    """
    candidates = {}  # track_id -> source weights

    # Collaborative filtering (primary)
    cf_tracks = cf_source.get_candidates(user.user_id, k=2000)
    for track in cf_tracks:
        candidates[track] = candidates.get(track, {})
        candidates[track]['cf'] = 1.0

    # Audio-based (for diversity)
    seed_tracks = user.recent_tracks[:50]
    audio_tracks = audio_source.get_similar_tracks(seed_tracks, k=1000)
    for track in audio_tracks:
        candidates[track] = candidates.get(track, {})
        candidates[track]['audio'] = 0.8

    # Playlist embeddings
    playlist_tracks = playlist_source.get_similar_tracks(user.liked_tracks[:100], k=1000)
    for track in playlist_tracks:
        candidates[track] = candidates.get(track, {})
        candidates[track]['playlist'] = 0.9

    # Artist graph
    similar_artists = artist_source.get_similar_artists(user.top_artists[:20])
    artist_tracks = artist_source.get_tracks_from_artists(similar_artists, k=500)
    for track in artist_tracks:
        candidates[track] = candidates.get(track, {})
        candidates[track]['artist'] = 0.7

    # Combine sources
    scored_candidates = []
    for track_id, sources in candidates.items():
        combined_score = sum(sources.values()) / len(sources)
        scored_candidates.append((track_id, combined_score, sources))

    # Sort and return top candidates
    scored_candidates.sort(key=lambda x: -x[1])
    return [c[0] for c in scored_candidates[:5000]]
```

---

## Ranking Model

### Multi-Objective Prediction

**Course Connection**: Week 8 (Multi-Task Learning)

```python
class SpotifyRankingModel(nn.Module):
    """
    Multi-task model predicting stream, skip, and save probabilities.
    """
    def __init__(self, user_dim=128, track_dim=128, context_dim=32):
        super().__init__()

        input_dim = user_dim + track_dim + context_dim

        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Task-specific heads
        self.stream_head = nn.Linear(128, 1)   # P(stream > 30s)
        self.skip_head = nn.Linear(128, 1)     # P(skip < 30s)
        self.save_head = nn.Linear(128, 1)     # P(save to library)
        self.add_playlist_head = nn.Linear(128, 1)  # P(add to playlist)

    def forward(self, user_emb, track_emb, context):
        x = torch.cat([user_emb, track_emb, context], dim=1)
        shared = self.shared(x)

        return {
            'stream': torch.sigmoid(self.stream_head(shared)),
            'skip': torch.sigmoid(self.skip_head(shared)),
            'save': torch.sigmoid(self.save_head(shared)),
            'add_playlist': torch.sigmoid(self.add_playlist_head(shared))
        }
```

### Ranking Score

```python
def compute_ranking_score(predictions: Dict[str, float],
                          track: Track,
                          user: UserProfile) -> float:
    """
    Combine predictions into ranking score.

    Key insight: Penalize skips heavily (user frustration signal)
    """
    # Base engagement score
    engagement_score = (
        0.4 * predictions['stream'] +
        0.2 * predictions['save'] +
        0.1 * predictions['add_playlist'] -
        0.5 * predictions['skip']  # Heavy skip penalty
    )

    # Popularity adjustment (boost long-tail for discovery)
    popularity_factor = 1.0 - 0.3 * (track.popularity / 100)

    # Recency boost for new releases
    if (datetime.now() - track.release_date).days < 30:
        recency_boost = 1.2
    else:
        recency_boost = 1.0

    # Artist familiarity
    if track.artist_id in user.followed_artists:
        familiarity_boost = 1.1
    else:
        familiarity_boost = 1.0

    return engagement_score * popularity_factor * recency_boost * familiarity_boost
```

---

## The Blender: Familiarity vs Discovery

### The 60/40 Balance

**Key insight**: Users want both comfort and novelty.

```python
class PlaylistBlender:
    """
    Balance familiar and discovery content in recommendations.

    Target: 60% familiar, 40% discovery
    """
    def __init__(self, familiar_ratio=0.6):
        self.familiar_ratio = familiar_ratio

    def blend_playlist(self, user: UserProfile,
                       ranked_tracks: List[Track],
                       playlist_size: int = 30) -> List[Track]:
        """
        Create balanced playlist.
        """
        # Classify tracks as familiar or discovery
        familiar = []
        discovery = []

        for track in ranked_tracks:
            if self.is_familiar(track, user):
                familiar.append(track)
            else:
                discovery.append(track)

        # Calculate quotas
        n_familiar = int(playlist_size * self.familiar_ratio)
        n_discovery = playlist_size - n_familiar

        # Select from each bucket
        selected_familiar = familiar[:n_familiar]
        selected_discovery = discovery[:n_discovery]

        # Interleave for variety
        playlist = self.interleave(selected_familiar, selected_discovery)

        return playlist

    def is_familiar(self, track: Track, user: UserProfile) -> bool:
        """
        Track is familiar if:
        - User has played before
        - Artist is in top 50
        - Genre is user's primary genre
        """
        if track.track_id in user.recent_tracks:
            return True
        if track.artist_id in user.top_artists[:50]:
            return True
        return False

    def interleave(self, familiar: List, discovery: List) -> List:
        """
        Interleave to avoid clustering same type.
        """
        result = []
        f_idx, d_idx = 0, 0

        while f_idx < len(familiar) or d_idx < len(discovery):
            # Pattern: F, F, D, F, D, D, F, ...
            if f_idx < len(familiar):
                result.append(familiar[f_idx])
                f_idx += 1
            if d_idx < len(discovery):
                result.append(discovery[d_idx])
                d_idx += 1

        return result
```

### Exploration with Thompson Sampling

**Course Connection**: Week 10 (Contextual Bandits)

```python
class ExplorationManager:
    """
    Exploration for discovering user's latent interests.
    """
    def __init__(self, n_genres=50):
        # Beta distribution parameters per genre
        self.alpha = np.ones(n_genres)  # Successes
        self.beta = np.ones(n_genres)   # Failures

    def select_exploration_genre(self) -> int:
        """
        Thompson Sampling: Sample from posterior, pick highest.
        """
        sampled_rates = np.random.beta(self.alpha, self.beta)
        return np.argmax(sampled_rates)

    def update(self, genre_idx: int, streamed: bool):
        """
        Update posterior based on user response.
        """
        if streamed:
            self.alpha[genre_idx] += 1
        else:
            self.beta[genre_idx] += 1

    def get_exploration_tracks(self, user: UserProfile,
                                candidates: List[Track],
                                n: int = 3) -> List[Track]:
        """
        Get exploration tracks from under-explored genres.
        """
        exploration_genre = self.select_exploration_genre()
        genre_tracks = [t for t in candidates if exploration_genre in t.genres]

        # Return top tracks from exploration genre
        return genre_tracks[:n]
```

---

## Discover Weekly Generation

### Batch Pipeline

```python
class DiscoverWeeklyPipeline:
    """
    Weekly batch job to generate Discover Weekly for all users.

    Runs: Sunday night, completes by Monday 6 AM
    """
    def __init__(self):
        self.cf_model = CollaborativeFilteringModel()
        self.audio_model = AudioSimilarityModel()
        self.ranker = SpotifyRankingModel()
        self.blender = PlaylistBlender()

    def generate_for_user(self, user: UserProfile) -> List[Track]:
        """
        Generate 30-track Discover Weekly playlist.
        """
        # Step 1: Candidate generation (parallel)
        cf_candidates = self.cf_model.get_candidates(user, k=2000)
        audio_candidates = self.audio_model.get_similar(user.recent_tracks, k=1000)
        playlist_candidates = self.playlist_model.get_similar(user, k=1000)

        candidates = list(set(cf_candidates + audio_candidates + playlist_candidates))

        # Step 2: Filter already-played tracks
        candidates = [t for t in candidates if t not in user.recent_tracks]

        # Step 3: Rank by predicted engagement
        scores = self.ranker.score_batch(user, candidates)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])

        # Step 4: Blend familiar/discovery
        playlist = self.blender.blend_playlist(user, [t for t, _ in ranked], 30)

        # Step 5: Sequence for listening flow
        playlist = self.sequence_for_flow(playlist)

        return playlist

    def sequence_for_flow(self, tracks: List[Track]) -> List[Track]:
        """
        Order tracks for smooth listening experience.

        Considerations:
        - Tempo transitions (not jarring)
        - Energy arc (build up, then chill)
        - Genre clustering (some, not too much)
        """
        # Sort by energy, then add variety
        tracks_by_energy = sorted(tracks, key=lambda t: t.energy)

        # Start medium, build up, cool down
        n = len(tracks)
        ordered = []
        ordered.extend(tracks_by_energy[n//3:2*n//3])  # Medium energy start
        ordered.extend(tracks_by_energy[2*n//3:])      # High energy peak
        ordered.extend(tracks_by_energy[:n//3])        # Cool down

        return ordered

    def run_batch(self, users: List[UserProfile]):
        """
        Generate playlists for all users.

        Uses Spark for distributed processing.
        """
        # Partition users across workers
        # Each worker generates playlists for subset
        # Write to playlist database
        pass
```

---

## Cold Start Strategies

### New Users

```python
class NewUserColdStart:
    """
    Handle users with no listening history.
    """
    def onboarding_recommendations(self, user: UserProfile) -> List[Track]:
        """
        First session recommendations.
        """
        # Step 1: Collect explicit preferences
        selected_artists = user.onboarding_artists  # "Select artists you like"
        selected_genres = user.onboarding_genres    # "Select genres"

        # Step 2: Bootstrap from similar users (demographics)
        similar_users = self.find_similar_by_demographics(user)
        popular_among_similar = self.get_popular_tracks(similar_users)

        # Step 3: Mix explicit preferences + demographics
        artist_tracks = self.get_top_tracks_by_artists(selected_artists)
        genre_tracks = self.get_popular_by_genre(selected_genres)

        candidates = artist_tracks + genre_tracks + popular_among_similar

        # Step 4: Rank by global popularity (no personalization yet)
        ranked = sorted(candidates, key=lambda t: -t.popularity)

        return ranked[:50]

    def rapid_personalization(self, user: UserProfile,
                               plays: List[Play],
                               skips: List[Skip]):
        """
        Update user model after first 10-20 interactions.
        """
        # Immediate signals
        for play in plays:
            user.genre_affinity[play.track.genre] += 0.1
            user.audio_preferences = self.update_audio_prefs(
                user.audio_preferences, play.track
            )

        for skip in skips:
            user.genre_affinity[skip.track.genre] -= 0.05

        # After ~20 plays, can use collaborative filtering
        if len(plays) >= 20:
            user.taste_embedding = self.cf_model.embed_user(plays)
```

### New Tracks

```python
class NewTrackColdStart:
    """
    Handle tracks with no play history.
    """
    def __init__(self):
        self.audio_model = AudioEmbeddingModel()

    def embed_new_track(self, track: Track) -> np.ndarray:
        """
        Create embedding for new track using audio features only.
        """
        # Audio embedding from spectrogram
        audio_embedding = self.audio_model.embed(track.audio_file)

        # Metadata features
        metadata = [
            track.tempo / 200,  # Normalize
            track.energy,
            track.danceability,
            track.valence,
            track.acousticness
        ]

        # Artist embedding (if known artist)
        if track.artist_id in self.artist_embeddings:
            artist_emb = self.artist_embeddings[track.artist_id]
        else:
            artist_emb = np.zeros(64)

        # Combine
        combined = np.concatenate([audio_embedding, metadata, artist_emb])
        return combined

    def find_similar_for_new_track(self, track: Track, k: int = 100) -> List[str]:
        """
        Find similar existing tracks for cold-start recommendations.
        """
        track_embedding = self.embed_new_track(track)
        similar = self.ann_index.search(track_embedding, k)
        return similar
```

---

## Artist Fairness

### The Long-Tail Problem

```
Distribution:
- Top 1% of artists: 50% of streams
- Bottom 80% of artists: 5% of streams

Challenge: New/small artists struggle to gain traction
```

### Fairness Mechanisms

```python
class ArtistFairnessManager:
    """
    Ensure fair exposure for emerging artists.
    """
    def __init__(self, exploration_rate=0.1):
        self.exploration_rate = exploration_rate

    def inject_emerging_artists(self, recommendations: List[Track],
                                 user: UserProfile) -> List[Track]:
        """
        Replace some recommendations with emerging artist tracks.
        """
        n_explore = int(len(recommendations) * self.exploration_rate)

        # Find emerging artists matching user's taste
        emerging_tracks = self.get_emerging_artist_tracks(user)

        # Replace lowest-ranked tracks
        recommendations[-n_explore:] = emerging_tracks[:n_explore]

        return recommendations

    def get_emerging_artist_tracks(self, user: UserProfile) -> List[Track]:
        """
        Find tracks from artists with:
        - < 10K monthly listeners
        - High audio similarity to user's preferences
        - Good engagement rate (skip rate < 30%)
        """
        emerging = self.emerging_artist_index.search(
            user.taste_embedding,
            filters={
                'monthly_listeners': '<10000',
                'skip_rate': '<0.3',
                'release_date': '>2024-01-01'
            }
        )
        return emerging
```

---

## Training Pipeline

### Data Collection

```python
class SpotifyTrainingData:
    """
    Collect implicit feedback from streaming sessions.
    """
    def process_stream_event(self, event: StreamEvent) -> TrainingExample:
        """
        Convert streaming event to training example.

        Positive: Streamed > 30 seconds
        Strong positive: Streamed to completion, saved, added to playlist
        Negative: Skipped < 30 seconds
        """
        return TrainingExample(
            user_id=event.user_id,
            track_id=event.track_id,
            context=event.context,  # playlist, radio, search, etc.
            labels={
                'stream': event.duration_ms > 30000,
                'skip': event.duration_ms < 30000 and not event.completed,
                'save': event.saved,
                'add_playlist': event.added_to_playlist
            },
            timestamp=event.timestamp
        )
```

### Training Schedule

```
Model updates:
- Collaborative filtering (ALS): Daily (batch)
- Audio embeddings: Weekly (batch, GPU-intensive)
- Ranking model: Daily (incremental updates)
- Playlist embeddings: Weekly (batch)

Discover Weekly generation:
- Full regeneration: Sunday night
- Pre-computation: Saturday (candidate generation)
- Serving: Monday 12:00 AM UTC
```

---

## Serving Infrastructure

### Architecture

**Course Connection**: Week 13 (Production Systems)

```
┌─────────────────────────────────────────────────────────────┐
│                    BATCH LAYER                               │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Discover     │  │ User         │  │ Track            │   │
│  │ Weekly Gen   │  │ Embedding    │  │ Embedding        │   │
│  │ (Weekly)     │  │ (Daily)      │  │ (Daily)          │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ Playlist     │  │ CF Model     │                         │
│  │ Embeddings   │  │ Training     │                         │
│  │ (Weekly)     │  │ (Daily)      │                         │
│  └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    REAL-TIME LAYER                           │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Home Page    │  │ Radio        │  │ Search           │   │
│  │ Service      │  │ Service      │  │ Ranking          │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Feature      │  │ ANN          │  │ Ranking          │   │
│  │ Store        │  │ Index        │  │ Model            │   │
│  │ (Redis)      │  │ (FAISS)      │  │ (TF Serving)     │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Latency Optimization

```python
class SpotifyServingOptimizations:
    # 1. Pre-compute heavy operations
    precomputed = {
        'discover_weekly': 'weekly batch',
        'daily_mix': 'daily batch',
        'user_embeddings': 'daily batch',
    }

    # 2. Multi-level caching
    cache_layers = {
        'user_embedding': ('redis', 3600),       # 1 hour
        'track_features': ('memcached', 86400),  # 24 hours
        'home_page': ('cdn', 60),                # 1 minute
        'playlist_tracks': ('redis', 300),       # 5 minutes
    }

    # 3. ANN index sharding
    ann_config = {
        'index_type': 'HNSW',
        'shards': 8,
        'replicas_per_shard': 3,
        'ef_search': 64,  # Quality/speed tradeoff
    }
```

---

## Metrics & Evaluation

### Online Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Stream rate** | % tracks played > 30s | > 70% |
| **Skip rate** | % tracks skipped < 30s | < 25% |
| **Save rate** | % tracks saved to library | > 5% |
| **Discover Weekly streams** | Tracks played from DW | > 15/30 |
| **Time spent** | Minutes per session | > 30 min |
| **DAU** | Daily active users | Growth |
| **Monthly retention** | 30-day return rate | > 80% |

### A/B Testing

```
Spotify experimentation:
- Platform: Internal A/B framework
- Typical experiment size: 1-5% of users
- Duration: 2-4 weeks
- Key metrics: Stream rate, skip rate, save rate, session length
```

---

## Course Concepts Applied

| Concept | Week | Application in Spotify |
|---------|------|------------------------|
| **Memory-Based CF** | 2 | User-user similarity for taste clustering |
| **Matrix Factorization** | 3 | ALS for user-track embeddings |
| **Content-Based** | 4 | Audio CNN embeddings for cold start |
| **Neural CF** | 5 | Deep ranking model |
| **Sequential** | 6 | Session-based radio recommendations |
| **Graph-Based** | 7 | Artist collaboration/similarity graph |
| **Two-Tower** | 8 | User tower + Track tower retrieval |
| **Multi-Task** | 8 | Stream/skip/save joint prediction |
| **Embeddings** | 9 | Playlist2Vec, audio embeddings |
| **Bandits** | 10 | Exploration for genre discovery |
| **Evaluation** | 11 | Skip rate, save rate as quality metrics |
| **Fairness** | 12 | Emerging artist exposure |
| **Production** | 13 | Batch DW generation, real-time radio |

---

## Summary

**Spotify's recommendation system** combines multiple techniques:

1. **Scale**: 600M users, 100M tracks, 4B plays/day
2. **Hybrid approach**: CF + Content (audio) + NLP (playlists)
3. **Discover Weekly**: Weekly batch generation with 60/40 familiar/discovery
4. **Playlist2Vec**: Word2Vec on playlist sequences
5. **Audio features**: CNN on spectrograms for similarity
6. **Multi-task ranking**: Stream, skip, save prediction
7. **Artist fairness**: Emerging artist exploration

**Key innovations**:
- Playlist2Vec for co-occurrence learning
- Audio-based cold start (every track can be recommended day 1)
- Skip penalty in ranking (user frustration signal)
- Familiar/discovery balance (60/40 rule)

**Impact**: Discover Weekly has 40M+ weekly listeners, driving 30%+ of discovery streams.

---

## References

1. **Jacobson, K., et al. (2016)**. "Music Personalization at Spotify". *RecSys*.
   - Overview of Spotify's approach

2. **Chen, C.-W., et al. (2012)**. "Playlist Prediction via Metric Embedding". *KDD*.
   - Playlist2Vec foundation

3. **van den Oord, A., et al. (2013)**. "Deep Content-Based Music Recommendation". *NIPS*.
   - Audio CNN for music similarity

4. **Spotify Engineering Blog**. "Discover Weekly: How Spotify Knows What You'll Love".
   - Official system description

5. **Mehrotra, R., et al. (2018)**. "Explore, Exploit, and Explain: Personalizing Explainable Recommendations with Bandits". *RecSys*.
   - Exploration in Spotify
