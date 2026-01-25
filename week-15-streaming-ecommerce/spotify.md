# Week 15: Spotify Music Recommendations

## Overview

**Spotify**: 500M+ users, 100M+ tracks.

**Recommendation surfaces**:
1. **Discover Weekly**: Personalized 30-song playlist (updated Monday)
2. **Daily Mix**: Multiple playlists by genre/mood
3. **Release Radar**: New music from followed artists
4. **Radio**: Endless stream based on seed
5. **Home**: Personalized homepage with playlists, albums

**Challenge**: Cold start (new users), discovery vs. familiarity balance.

---

## Discover Weekly

### Algorithm

**Hybrid approach**: Collaborative filtering + NLP + audio analysis.

**Step 1: Collaborative Filtering**
- Matrix factorization on listening data
- Users who listen to Artist A also like Artist B

**Step 2: NLP on Playlists**
- Scrape user-created playlists, blog posts
- Word2Vec on playlist track sequences
- Tracks appearing together → similar embeddings

**Step 3: Audio Analysis**
- CNN on audio spectrograms
- Extract features: tempo, energy, danceability, valence
- Find tracks with similar audio profiles

**Combination**:
$$\text{Score} = 0.4 \cdot \text{CF} + 0.3 \cdot \text{NLP} + 0.3 \cdot \text{Audio}$$

---

### Personalization

**For each user**:
1. Identify top genres/artists (from history)
2. Find similar tracks (via 3 methods above)
3. Filter out already-listened tracks
4. Rank by predicted preference
5. Add diversity (not all same genre)
6. Generate 30-track playlist

**Update**: Every Monday (gives users time to listen).

---

## Daily Mix

### Clustering Approach

**Goal**: Create multiple playlists, each cohesive by genre/mood.

**Algorithm**:
1. **Cluster** user's listening history (k-means on track embeddings)
2. **For each cluster**: Create playlist
3. **Add similar tracks** to each cluster (expand beyond history)

**Example**:
```
User listens to: Rock, Hip-Hop, Classical

Daily Mix 1: Rock (user's rock songs + similar rock)
Daily Mix 2: Hip-Hop (user's hip-hop + similar)
Daily Mix 3: Classical (user's classical + similar)
```

**Benefits**: Familiar (starts with known songs) + discovery (adds similar).

---

## Audio Feature Extraction

### CNN on Spectrograms

**Input**: Audio waveform → Spectrogram (time-frequency representation).

**CNN**: Pretrained on music classification.

**Output**: 128-dim embedding per track.

**Features**:
- **Tempo**: BPM (beats per minute)
- **Energy**: Intensity (0-1)
- **Danceability**: Rhythm stability (0-1)
- **Valence**: Musical positivity (0-1, happy vs. sad)
- **Acousticness**: Acoustic vs. electronic (0-1)

```python
def extract_audio_features(track_file):
    """
    Extract Spotify-style audio features.
    """
    y, sr = librosa.load(track_file)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Energy (RMS)
    rms = librosa.feature.rms(y=y).mean()

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

    # MFCC (timbre)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)

    return {
        'tempo': tempo,
        'energy': rms,
        'spectral_centroid': spectral_centroid,
        'mfcc': mfcc
    }
```

---

## NLP on Playlists

### Playlist2Vec

**Idea**: Treat playlists as "sentences", tracks as "words".

**Word2Vec**: Skip-gram on playlist sequences.

**Example**:
```
Playlist 1: [Track A, Track B, Track C]
Playlist 2: [Track A, Track D, Track E]

Context for Track A: [B, C, D, E]
Learn embedding such that A is close to {B, C, D, E}
```

**Result**: Tracks that co-occur in playlists have similar embeddings.

---

## Balancing Familiarity and Discovery

### Exploitation vs. Exploration

**User expectation**:
- 60% familiar (artists/genres already liked)
- 40% discovery (new artists, similar style)

**Implementation**:
```python
def generate_recommendations(user, k=30):
    recs = []

    # 60% familiar
    familiar = get_familiar_tracks(user, k=18)
    recs.extend(familiar)

    # 40% discovery
    discovery = get_discovery_tracks(user, k=12)
    recs.extend(discovery)

    # Shuffle to avoid all familiar first
    random.shuffle(recs)

    return recs[:k]
```

---

## Artist Fairness

### Problem

**Long-tail artists**: 99% of artists get <1% of streams.

**Head artists**: Top 1% get 50% of streams.

**Impact**: Hard for new artists to gain traction.

---

### Mitigation

**1. Explore new artists**: Include emerging artists in Discover Weekly.

**2. Personalized niches**: Recommend obscure artists to fans of similar established artists.

**3. Editorial playlists**: Human curators boost new artists.

**4. Algorithmic fairness**: Ensure small artists get proportional exposure.

---

## Podcast Recommendations

### Different Modality

**Podcasts ≠ Music**:
- **Episodic**: Must listen in order (often)
- **Long-form**: 30-60 min episodes
- **Content-driven**: Topic matters more than audio features

**Approach**:
- **NLP on descriptions**: BERT embeddings of episode summaries
- **Collaborative filtering**: Users who listened to Podcast A also like Podcast B
- **Cross-modal**: Music listeners who like Genre X also like Podcast Y

---

## Summary

**Key Takeaways**:
1. **Discover Weekly**: CF + NLP + Audio (hybrid)
2. **Daily Mix**: Clustering user history into cohesive playlists
3. **Audio features**: Tempo, energy, danceability (CNN on spectrograms)
4. **Playlist2Vec**: Word2Vec on playlist sequences
5. **Balance**: 60% familiar, 40% discovery
6. **Artist fairness**: Boost emerging artists

**Impact**: Discover Weekly has 40M+ weekly listeners.

---

## References

1. **Eriksson, O. (2014)**. "Spotify's Discover Weekly: How Machine Learning Finds Your New Music". *Spotify Blog*.
2. **Jacobson, K., et al. (2016)**. "Music Personalization at Spotify". *RecSys*.
3. **Chen, C.-W., et al. (2012)**. "Playlist Prediction via Metric Embedding". *KDD*.
