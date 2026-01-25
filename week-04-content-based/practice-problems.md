# Week 4: Content-Based Recommendation - Practice Problems

## Overview
These problems test your understanding of content-based filtering, feature extraction from text/images/audio, profile learning methods, and hybrid recommendation strategies. Focus on TF-IDF, embeddings, and combining content with collaborative signals.

---

## Problem 1: TF-IDF Calculation by Hand
**Difficulty:** Easy
**Topics:** TF-IDF, text representation

Given 4 movie descriptions:

| Movie | Description |
|-------|-------------|
| M1    | "action hero saves world" |
| M2    | "hero fights villain" |
| M3    | "romantic comedy love story" |
| M4    | "action packed thriller" |

Calculate the TF-IDF vector for M1.

**Formulas:**
- TF(term, doc) = count of term in doc / total terms in doc
- IDF(term) = log(N / DF(term)), where N = total docs, DF = docs containing term
- TF-IDF(term, doc) = TF(term, doc) × IDF(term)

**Hints:**
- "action" appears in M1 and M4 (DF = 2)
- "hero" appears in M1 and M2 (DF = 2)
- "saves" appears only in M1 (DF = 1)

**Learning Outcomes:**
- Compute TF-IDF manually
- Understand inverse document frequency
- Recognize term importance

---

## Problem 2: Cosine Similarity for Content-Based Recommendation
**Difficulty:** Easy
**Topics:** Cosine similarity, item similarity

After computing TF-IDF vectors:
- M1: [0.4, 0.3, 0.7, 0.5]
- M2: [0.0, 0.6, 0.5, 0.0]
- M3: [0.0, 0.0, 0.0, 0.8]

User Alice watched and liked M1. Which movie should you recommend next?

**Formula:**
$$\text{sim}(i, j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{||\mathbf{v}_i|| \cdot ||\mathbf{v}_j||}$$

**Tasks:**
1. Calculate cosine similarity: sim(M1, M2) and sim(M1, M3)
2. Recommend the most similar movie
3. Explain why the chosen movie is more similar

**Learning Outcomes:**
- Apply cosine similarity to content vectors
- Make content-based recommendations
- Interpret similarity scores

---

## Problem 3: Rocchio Algorithm for User Profiling
**Difficulty:** Medium
**Topics:** User profile learning, Rocchio algorithm

The Rocchio algorithm builds user profiles from positive and negative examples:

$$\mathbf{profile}_u = \alpha \frac{1}{|I_u^+|} \sum_{i \in I_u^+} \mathbf{v}_i - \beta \frac{1}{|I_u^-|} \sum_{j \in I_u^-} \mathbf{v}_j$$

**Given:**
- User liked movies: M1[1.0, 0.5, 0.3], M2[0.8, 0.6, 0.2]
- User disliked movies: M3[0.2, 0.9, 0.8]
- α = 1.0, β = 0.5

**Tasks:**
1. Compute the user profile vector
2. Explain what each term (positive and negative) contributes
3. Why might you set β < α?
4. How would you use this profile to recommend new items?

**Learning Outcomes:**
- Implement Rocchio algorithm
- Understand positive and negative feedback
- Build interpretable user profiles

---

## Problem 4: Content-Based vs. Collaborative Filtering
**Difficulty:** Medium
**Topics:** Comparison, cold start, limitations

Compare content-based and collaborative filtering for these scenarios:

**Scenario A:** A new movie is released today. You have its plot summary, genre, director, and cast.

**Scenario B:** A user with 500 ratings wants personalized recommendations.

**Scenario C:** A niche documentary that only 3 users have watched (all loved it).

For each scenario:
1. Which approach (content-based or collaborative) would work better?
2. Why?
3. What are the failure modes of the worse approach?

**Learning Outcomes:**
- Recognize when to use each approach
- Understand complementary strengths
- Think about hybrid strategies

---

## Problem 5: Feature Engineering for Movies
**Difficulty:** Medium
**Topics:** Feature extraction, representation learning

Design a feature representation for movies that combines:
- **Text:** Plot summary (TF-IDF or embeddings)
- **Categorical:** Genre, director, actors
- **Numerical:** Year, runtime, budget

**Questions:**
1. How would you encode multi-valued categorical features (e.g., a movie has multiple genres)?
2. Should you normalize numerical features? Why?
3. How would you combine heterogeneous features into a single vector?
4. What dimensionality would you target for the final representation?

**Hints:**
- One-hot encoding for categoricals
- Multi-hot for multi-valued categoricals
- Standardization (z-score) for numerical
- Concatenation or weighted combination

**Learning Outcomes:**
- Design feature representations
- Handle mixed data types
- Prepare data for machine learning

---

## Problem 6: Word Embeddings vs. TF-IDF
**Difficulty:** Medium
**Topics:** Word2Vec, embeddings, semantic similarity

Compare TF-IDF and Word2Vec embeddings:

**TF-IDF vectors:**
- "action adventure movie" → [0.4, 0.3, 0.2, 0.0, ...]
- "thriller suspense film" → [0.0, 0.0, 0.0, 0.5, ...]

**Word2Vec (averaged):**
- "action adventure movie" → [0.2, -0.5, 0.8, ...]
- "thriller suspense film" → [0.3, -0.4, 0.7, ...]

1. Which would have higher cosine similarity: TF-IDF or Word2Vec?
2. Why might Word2Vec capture semantics better?
3. What are the disadvantages of Word2Vec for recommendations?
4. When would you prefer TF-IDF?

**Learning Outcomes:**
- Compare sparse vs. dense representations
- Understand semantic vs. lexical similarity
- Choose appropriate text representations

---

## Problem 7: CNN Features for Visual Recommendations
**Difficulty:** Hard
**Topics:** Computer vision, CNN embeddings, image similarity

You extract features from product images using a pre-trained ResNet-50 (2048-dimensional vectors).

**Given:**
- Product A (red dress): [0.8, 0.1, 0.6, 0.2, ...] (2048-dim)
- Product B (blue dress): [0.7, 0.2, 0.5, 0.3, ...]
- Product C (red jacket): [0.9, 0.0, 0.7, 0.1, ...]

User likes Product A.

**Tasks:**
1. Which product is more similar visually: B or C?
2. What aspects of visual similarity does ResNet capture (color, shape, texture)?
3. How would you combine visual features with textual descriptions?
4. What are the limitations of pure visual similarity?

**Hints:**
- ResNet learns hierarchical features (edges → textures → objects)
- Color and shape are often correlated in embeddings
- Late fusion: compute separate similarities and combine
- Visual similarity ≠ user preference (may want variety)

**Learning Outcomes:**
- Work with CNN embeddings
- Understand visual similarity
- Design multi-modal systems

---

## Problem 8: Hybrid Recommendation System Design
**Difficulty:** Hard
**Topics:** Hybrid systems, feature combination, ensemble methods

Design a hybrid system that combines:
1. Collaborative filtering (CF)
2. Content-based filtering (CB)
3. Popularity baseline

**Approaches to consider:**
- **Weighted hybrid:** score = α × CF + β × CB + γ × popularity
- **Switching hybrid:** Use CF if user has >50 ratings, else CB
- **Feature combination:** Use content features in a CF model (Factorization Machines)
- **Meta-level:** Use CB to learn initial user profile, then CF

**Questions:**
1. Which approach would you choose for a production system?
2. How would you determine the weights (α, β, γ)?
3. What are the trade-offs of each approach?
4. How does this help with cold start?

**Learning Outcomes:**
- Design hybrid systems
- Balance multiple signals
- Solve real-world recommendation challenges

---

## Programming Exercises

### Exercise 1: TF-IDF Movie Recommender
**Dataset:** MovieLens with plot summaries (from IMDB or TMDB API)
**Task:** Build a pure content-based recommender using TF-IDF

**Implementation:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie descriptions
movies = load_movie_data()  # columns: movie_id, title, description

# Compute TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['description'])

# Find similar movies
def recommend(movie_id, top_n=10):
    idx = movie_id_to_index[movie_id]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n-1:-1][::-1]
    return movies.iloc[top_indices]
```

**Evaluation:**
1. For each user, take their top-rated movie
2. Generate 10 recommendations based on that movie
3. Measure how many recommendations they actually rated ≥4 (Precision)

**Expected:** Precision@10 ≈ 0.15-0.25

---

### Exercise 2: Content-Based Filtering with User Profiles
**Dataset:** MovieLens + IMDB metadata (genre, director, actors)
**Task:** Learn user profiles from their rating history

**Steps:**
1. Represent each movie as a feature vector (genre, director, actors, year)
2. For each user, compute profile = weighted average of liked movie vectors
3. Recommend items most similar to user profile

**Implementation:**
```python
def build_user_profile(user_id, ratings, movie_features):
    user_ratings = ratings[ratings['user_id'] == user_id]
    liked_movies = user_ratings[user_ratings['rating'] >= 4]

    # Weighted average of liked movie features
    weights = liked_movies['rating'].values
    features = movie_features.loc[liked_movies['movie_id']]
    profile = np.average(features, axis=0, weights=weights)
    return profile

def recommend_for_user(user_profile, movie_features, top_n=10):
    similarities = cosine_similarity([user_profile], movie_features)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return top_indices
```

**Evaluation:**
- Split data: 80% train, 20% test
- Build profiles on train set
- Recommend on test set
- Measure Precision@10, Recall@10, NDCG@10

---

### Exercise 3: Image-Based Product Recommendations
**Dataset:** Amazon product images (or Fashion-MNIST)
**Task:** Recommend visually similar products using CNN features

**Steps:**
1. Use pre-trained ResNet-50 to extract features
2. Store feature vectors in a database (FAISS for fast search)
3. For a query image, find k nearest neighbors

**Implementation:**
```python
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Load pre-trained ResNet
resnet = models.resnet50(pretrained=True)
resnet.eval()
# Remove final classification layer
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

def extract_features(image_path):
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(img_tensor)
    return features.squeeze().numpy()

# Build index
import faiss
features = np.array([extract_features(img) for img in image_paths])
index = faiss.IndexFlatIP(features.shape[1])  # Inner product (cosine)
index.add(features)

# Search
def find_similar(query_image, k=10):
    query_features = extract_features(query_image)
    distances, indices = index.search(query_features.reshape(1, -1), k)
    return indices[0]
```

**Evaluation:**
- Qualitative: Visualize top-10 similar images
- Quantitative: If you have labels (e.g., "dress", "shoe"), measure same-category precision

---

### Exercise 4: Word2Vec for Movie Descriptions
**Dataset:** MovieLens + plot summaries
**Task:** Use Word2Vec embeddings instead of TF-IDF

**Steps:**
1. Train Word2Vec on movie descriptions (or use pre-trained GloVe)
2. Represent each movie as average of word embeddings
3. Compute cosine similarity
4. Compare with TF-IDF results

**Implementation:**
```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Tokenize descriptions
tokenized_docs = [word_tokenize(desc.lower()) for desc in movies['description']]

# Train Word2Vec
model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=2, workers=4)

def document_vector(doc, model):
    # Average word vectors
    vectors = [model.wv[word] for word in doc if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

movie_vectors = np.array([document_vector(doc, model) for doc in tokenized_docs])
```

**Comparison:**
| Method    | Precision@10 | Coverage | Runtime |
|-----------|--------------|----------|---------|
| TF-IDF    | 0.22         | 95%      | Fast    |
| Word2Vec  | 0.26         | 98%      | Medium  |

---

### Exercise 5: Hybrid Weighted Combination
**Dataset:** MovieLens 100K
**Task:** Combine collaborative filtering and content-based scores

**Implementation:**
```python
# Get CF scores (from matrix factorization)
cf_scores = mf_model.predict(user_id, all_items)

# Get content-based scores
user_profile = build_user_profile(user_id, ratings, movie_features)
cb_scores = cosine_similarity([user_profile], movie_features)[0]

# Normalize scores to [0, 1]
cf_scores_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min())
cb_scores_norm = (cb_scores - cb_scores.min()) / (cb_scores.max() - cb_scores.min())

# Weighted combination
alpha = 0.7  # Weight for CF
beta = 0.3   # Weight for CB
hybrid_scores = alpha * cf_scores_norm + beta * cb_scores_norm

# Recommend top-N
top_indices = hybrid_scores.argsort()[-10:][::-1]
```

**Hyperparameter Tuning:**
- Grid search over α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
- Measure NDCG@10 on validation set
- Find optimal α

**Expected Result:** Hybrid outperforms both CF and CB individually

---

### Exercise 6: Factorization Machines for Hybrid Recommendations
**Dataset:** MovieLens with metadata
**Task:** Use Factorization Machines to incorporate content features

**Key Idea:** FM models feature interactions

$$\hat{y} = w_0 + \sum_i w_i x_i + \sum_i \sum_{j>i} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

**Features:**
- User ID (one-hot)
- Item ID (one-hot)
- Genre (multi-hot)
- Director (one-hot)
- Year (numerical)

**Implementation:**
```python
from xlearn import FMModel

# Prepare data in LibFM format
# user_id:1 item_id:1 genre_action:1 genre_scifi:1 year:1999

fm_model = FMModel()
fm_model.fit("train.libfm", "model.out")
predictions = fm_model.predict("test.libfm")
```

**Evaluation:**
- Compare FM with pure MF
- Analyze which features are most important
- Test on cold-start items (new movies with metadata but no ratings)

---

### Exercise 7: BERT Embeddings for Text
**Dataset:** Movie plot summaries
**Task:** Use BERT to create semantic embeddings

**Implementation:**
```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use [CLS] token embedding
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

movie_bert_embeddings = np.array([get_bert_embedding(desc) for desc in descriptions])
```

**Comparison:**
| Method | Precision@10 | Semantic Quality |
|--------|--------------|------------------|
| TF-IDF | 0.22         | Low              |
| Word2Vec | 0.26       | Medium           |
| BERT   | 0.31         | High             |

---

## Discussion Questions

1. **Filter Bubble:** Content-based systems tend to recommend items very similar to past likes. How can you increase diversity without sacrificing relevance?

2. **Cold Start:** Content-based CF handles item cold start well. But what about user cold start? How would you onboard a new user?

3. **Overspecialization:** If a user only watches action movies and your system only recommends action, they may never discover they like documentaries. How do you introduce serendipity?

4. **Feature Quality:** How do you evaluate if your content features are good? What metrics or visualizations would you use?

5. **Multimodal Fusion:** How would you combine text, image, and audio features for music or video recommendations? Early fusion vs. late fusion?

6. **Temporal Dynamics:** Movie genres go in and out of fashion. How would you incorporate temporal trends into content-based recommendations?

7. **Explainability:** Content-based recommendations are inherently more explainable. How would you surface explanations to users?

8. **Implicit Feedback:** How do you adapt content-based methods for implicit feedback (clicks, views) instead of ratings?

---

## Challenge Problem: Multi-Modal Product Search

**Difficulty:** Hard
**Topics:** Multi-modal learning, image + text fusion

**Task:** Build a product recommendation system that accepts:
- Text query: "red summer dress"
- Reference image: (upload)

And returns relevant products by combining both signals.

**Approach:**
1. Extract text features (BERT or TF-IDF)
2. Extract image features (ResNet or CLIP)
3. Fuse features:
   - **Early fusion:** Concatenate text and image vectors
   - **Late fusion:** Compute separate similarities and combine
4. Rank products by fused similarity

**Advanced:** Use CLIP (Contrastive Language-Image Pre-training)
- CLIP learns joint text-image embeddings
- Text and image in same space → direct comparison

**Implementation (CLIP):**
```python
import clip
import torch

model, preprocess = clip.load("ViT-B/32")

def encode_text(text):
    text_tokens = clip.tokenize([text])
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    return text_features

def encode_image(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features

# Hybrid query
text_feat = encode_text("red summer dress")
image_feat = encode_image("reference.jpg")
query_feat = 0.5 * text_feat + 0.5 * image_feat

# Search
similarities = torch.nn.functional.cosine_similarity(query_feat, product_features)
```

**Evaluation:**
- User study: Show results to users, measure relevance
- A/B test: Text-only vs. image-only vs. multi-modal

---

## References

### Papers
1. Pazzani, M. J., & Billsus, D. (2007). "Content-based recommendation systems". The Adaptive Web.
2. Lops, P., et al. (2011). "Content-based recommender systems". Recommender Systems Handbook.
3. Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision". CLIP paper.

### Libraries
- **scikit-learn:** TF-IDF, cosine similarity
- **Gensim:** Word2Vec, Doc2Vec
- **Transformers (Hugging Face):** BERT, RoBERTa
- **PyTorch/TensorFlow:** CNN feature extraction
- **CLIP:** Multi-modal embeddings

### Datasets
- MovieLens + IMDB metadata: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
- Amazon Product Data: http://jmcauley.ucsd.edu/data/amazon/
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist

---

*Return to [Week 4 Main Page](README.md)*
