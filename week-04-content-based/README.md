# Week 4: Content-Based Recommendation

## Overview

Content-based filtering recommends items similar to what a user has liked before, based on item features rather than collaborative signals. This approach excels when you have rich content (text, images, audio) and handles cold-start items well.

##  Learning Objectives

- Master content-based filtering foundations and techniques
- Learn feature extraction from text (TF-IDF), images (CNN), and audio
- Implement profile learning methods
- Design hybrid systems combining content-based and collaborative filtering

## Topics Covered

### [1. Foundations](foundations.md)
- Item profile construction
- User profile learning
- Matching profiles for recommendations

### [2. Feature Representation](feature-representation.md)
- **Text**: TF-IDF, n-grams, word embeddings
- **Images**: CNN embeddings (ResNet, VGG)
- **Audio**: Spectrograms, MFCCs
- **Metadata**: Structured attributes

### [3. Profile Learning](profile-learning.md)
- Weighted averaging
- Rocchio algorithm
- Naive Bayes classifiers
- Logistic regression

### [4. Hybrid Strategies](hybrid-strategies.md)
- Weighted hybrid
- Switching hybrid
- Feature combination
- Meta-level hybrid

### [5. Code Examples](code-examples.md)
- TF-IDF-based movie recommender
- CNN image similarity
- Hybrid CF + content-based

### [6. Practice Problems](practice-problems.md)

## Key Formulas

**TF-IDF**:
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\frac{N}{\text{DF}(t)}$$

**Cosine Similarity**:
$$\text{sim}(d_1, d_2) = \frac{\mathbf{d}_1 \cdot \mathbf{d}_2}{||\mathbf{d}_1|| \cdot ||\mathbf{d}_2||}$$

**Rocchio User Profile**:
$$\mathbf{profile}_u = \frac{1}{|I_u^+|} \sum_{i \in I_u^+} \mathbf{item}_i - \frac{1}{|I_u^-|} \sum_{j \in I_u^-} \mathbf{item}_j$$

## Advantages vs. Limitations

**Advantages**:
- No cold-start for items (use content features)
- No need for other users' data
- Transparent recommendations
- Domain knowledge integration

**Limitations**:
- Overspecialization (filter bubble)
- Limited serendipity
- Feature engineering burden
- Can't capture implicit quality signals

## Datasets

- **Movie**Lens with metadata
- **Amazon Product Data** (text reviews, images)
- **Spotify Million Playlist** (audio features)
- **arXiv papers** (abstracts, citations)

## Required Reading

1. **Pazzani, M. J., & Billsus, D. (2007)**. "Content-based recommendation systems". *The Adaptive Web*.
2. **Lops, P., et al. (2011)**. "Content-based recommender systems". *Recommender Systems Handbook*.

*Return to [Main Course Page](../README.md)*
