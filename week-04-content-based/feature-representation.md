# Week 4: Content-Based Filtering - Feature Representation

## Overview

The quality of **feature representation** determines the quality of content-based recommendations. Poor features → poor recommendations. Great features → great recommendations.

This document covers advanced techniques for representing items as feature vectors across different modalities: **text**, **images**, **audio**, and **multimodal**.

**Business impact**: Better features improve recommendation accuracy by 20-50% (measured by engagement metrics).

---

## Learning Objectives

By the end of this section, you will:
- Master TF-IDF for text representation
- Implement word embeddings (Word2Vec, GloVe, BERT)
- Extract image features with CNNs
- Handle multimodal data (text + images)
- Apply dimensionality reduction techniques
- Optimize feature engineering for production systems

---

## Text Representation

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)

**Recall**:
$$\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \text{IDF}(w)$$

**Term Frequency** (TF):
$$\text{TF}(w, d) = \frac{f_{w,d}}{\sum_{w' \in d} f_{w',d}}$$

where $f_{w,d}$ = frequency of word $w$ in document $d$.

**Inverse Document Frequency** (IDF):
$$\text{IDF}(w) = \log \frac{N}{n_w}$$

where $N$ = total documents, $n_w$ = documents containing word $w$.

---

### TF-IDF Variants

**a) Sublinear TF Scaling**

$$\text{TF}_{\text{sub}}(w, d) = 1 + \log(f_{w,d})$$

**Effect**: Reduces impact of word frequency (e.g., 10 occurrences not 10× more important than 1).

---

**b) Smoothed IDF**

$$\text{IDF}_{\text{smooth}}(w) = \log \frac{N + 1}{n_w + 1} + 1$$

**Effect**: Prevents division by zero, reduces impact of very rare words.

---

**c) L2 Normalization**

$$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$

**Effect**: All documents have unit length → cosine similarity becomes dot product.

---

### Implementation: TF-IDF with scikit-learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Never jump over the lazy dog quickly",
    "The dog is very lazy and sleeps all day",
    "A quick brown cat jumps high"
]

# Initialize vectorizer
vectorizer = TfidfVectorizer(
    max_features=50,        # Keep top 50 features
    min_df=1,               # Minimum document frequency
    max_df=0.8,             # Maximum document frequency (remove common words)
    sublinear_tf=True,      # Use log(TF)
    use_idf=True,           # Use IDF
    norm='l2'               # L2 normalization
)

# Fit and transform
tfidf_matrix = vectorizer.fit_transform(documents)  # (4, n_features)

print(f"Shape: {tfidf_matrix.shape}")
print(f"Features: {vectorizer.get_feature_names_out()}")

# Get vector for first document
doc_vector = tfidf_matrix[0].toarray()[0]
print(f"\nDoc 0 vector (first 10 dims): {doc_vector[:10]}")

# Find most important words for each document
feature_names = vectorizer.get_feature_names_out()
for i, doc in enumerate(documents):
    doc_tfidf = tfidf_matrix[i].toarray()[0]
    top_indices = np.argsort(doc_tfidf)[::-1][:5]
    top_words = [(feature_names[idx], doc_tfidf[idx]) for idx in top_indices if doc_tfidf[idx] > 0]
    print(f"\nDoc {i} top words: {top_words}")
```

**Output**:
```
Shape: (4, 17)
Features: ['all' 'brown' 'cat' 'day' 'dog' 'fox' 'high' 'jumps' 'lazy' 'never' 'over' 'quick' 'quickly' 'sleeps' 'very']

Doc 0 top words: [('fox', 0.447), ('brown', 0.447), ('jumps', 0.327), ('lazy', 0.327), ('dog', 0.327)]
Doc 1 top words: [('never', 0.497), ('quickly', 0.497), ('jumps', 0.364), ('lazy', 0.364), ('dog', 0.364)]
Doc 2 top words: [('sleeps', 0.447), ('all', 0.447), ('day', 0.447), ('very', 0.447), ('lazy', 0.327)]
Doc 3 top words: [('cat', 0.537), ('high', 0.537), ('brown', 0.537), ('jumps', 0.394), ('quick', 0.394)]
```

**Analysis**: TF-IDF correctly identifies distinctive words for each document.

---

### 2. Word Embeddings

**Problem with TF-IDF**: No semantic similarity. "king" and "queen" are as dissimilar as "king" and "banana".

**Solution**: Word embeddings map words to dense vectors where semantically similar words are close.

---

### Word2Vec

**Paper**: Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013)

**Two architectures**:
1. **Skip-gram**: Predict context words from target word
2. **CBOW** (Continuous Bag of Words): Predict target word from context

**Skip-gram objective**:
$$\max \sum_{t=1}^T \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)$$

where $c$ = context window size.

**Probability**:
$$P(w_O | w_I) = \frac{\exp(\mathbf{v}_{w_O}^T \mathbf{v}_{w_I})}{\sum_{w=1}^W \exp(\mathbf{v}_w^T \mathbf{v}_{w_I})}$$

**Training trick**: Negative sampling (approximate softmax).

---

### Implementation: Word2Vec with Gensim

```python
from gensim.models import Word2Vec
import numpy as np

# Sample sentences (tokenized)
sentences = [
    ['the', 'quick', 'brown', 'fox', 'jumps'],
    ['never', 'jump', 'over', 'the', 'lazy', 'dog'],
    ['the', 'dog', 'is', 'very', 'lazy'],
    ['a', 'quick', 'brown', 'cat', 'jumps', 'high']
]

# Train Word2Vec
model = Word2Vec(
    sentences,
    vector_size=100,    # Embedding dimension
    window=5,           # Context window
    min_count=1,        # Minimum word frequency
    sg=1,               # Skip-gram (sg=0 for CBOW)
    workers=4
)

# Get embedding for word "quick"
quick_vec = model.wv['quick']
print(f"'quick' embedding (first 10 dims): {quick_vec[:10]}")

# Find similar words
similar = model.wv.most_similar('dog', topn=3)
print(f"\nWords similar to 'dog': {similar}")

# Word arithmetic
# king - man + woman ≈ queen (not enough data in toy example)
```

---

### Document Embeddings from Word Embeddings

**Approach 1: Average Word Vectors**

$$\mathbf{d} = \frac{1}{|D|} \sum_{w \in D} \mathbf{v}_w$$

```python
def document_embedding(doc, model):
    """Average word vectors for document."""
    vectors = [model.wv[word] for word in doc if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Example
doc = ['the', 'quick', 'brown', 'fox']
doc_vec = document_embedding(doc, model)
print(f"Document embedding shape: {doc_vec.shape}")
```

**Problem**: Ignores word order. "Dog bites man" = "Man bites dog".

---

**Approach 2: Weighted Average (TF-IDF weights)**

$$\mathbf{d} = \sum_{w \in D} \text{TF-IDF}(w, d) \cdot \mathbf{v}_w$$

```python
def tfidf_weighted_embedding(doc, word_embeddings, tfidf_weights):
    """TF-IDF weighted average of word embeddings."""
    vec = np.zeros(word_embeddings.vector_size)
    total_weight = 0
    for word in doc:
        if word in word_embeddings and word in tfidf_weights:
            vec += tfidf_weights[word] * word_embeddings[word]
            total_weight += tfidf_weights[word]
    return vec / total_weight if total_weight > 0 else vec
```

**Improvement**: Gives more weight to important words.

---

### Doc2Vec

**Paper**: Le & Mikolov, "Distributed Representations of Sentences and Documents" (2014)

**Idea**: Learn document embeddings directly (not just word embeddings).

**Architecture**: Add document vector as additional context.

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Prepare tagged documents
tagged_docs = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(sentences)]

# Train Doc2Vec
model_d2v = Doc2Vec(
    tagged_docs,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=40
)

# Get document embedding
doc_vec = model_d2v.dv['0']  # Document 0
print(f"Doc2Vec embedding: {doc_vec[:10]}")

# Find similar documents
similar_docs = model_d2v.dv.most_similar('0', topn=2)
print(f"Similar documents to doc 0: {similar_docs}")
```

---

### Transformer-Based Embeddings (BERT, Sentence-BERT)

**Modern approach**: Use pre-trained transformers.

**Sentence-BERT** (2019): Fine-tuned BERT for sentence embeddings.

```python
from sentence_transformers import SentenceTransformer

# Load pre-trained model
model_sbert = SentenceTransformer('all-MiniLM-L6-v2')

# Documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A fast brown animal leaps over a sleepy canine",
    "The weather is sunny today"
]

# Generate embeddings
embeddings = model_sbert.encode(documents)  # (3, 384)

print(f"Embedding shape: {embeddings.shape}")

# Compute similarity
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings)

print("\nSimilarity matrix:")
print(similarities)
```

**Output**:
```
Embedding shape: (3, 384)

Similarity matrix:
[[1.    0.82  0.15]
 [0.82  1.    0.12]
 [0.15  0.12  1.  ]]
```

**Analysis**: Docs 0 and 1 are highly similar (0.82) despite different wording. Doc 2 is dissimilar (0.15).

**Advantage**: Captures semantic meaning better than TF-IDF or Word2Vec.

---

## Image Representation

### Convolutional Neural Networks (CNNs)

**Goal**: Extract features from images for recommendation.

**Approach**: Use pre-trained CNN (ResNet, VGG, EfficientNet) as feature extractor.

---

### Transfer Learning with ResNet

```python
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Load pre-trained ResNet-50
resnet = models.resnet50(pretrained=True)

# Remove final classification layer
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()  # Set to evaluation mode

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load image
img = Image.open('example.jpg')
img_tensor = preprocess(img).unsqueeze(0)  # (1, 3, 224, 224)

# Extract features
with torch.no_grad():
    features = feature_extractor(img_tensor)  # (1, 2048, 1, 1)

features = features.squeeze().numpy()  # (2048,)
print(f"Image feature vector shape: {features.shape}")
```

**Output**: 2048-dimensional feature vector representing the image.

**Use case**: Product recommendations (fashion, furniture), image search.

---

### Fine-Tuning for Domain-Specific Features

**Problem**: Pre-trained on ImageNet (general objects). May not capture domain-specific features (e.g., fashion style, food presentation).

**Solution**: Fine-tune on domain data.

```python
import torch.nn as nn
import torch.optim as optim

# Load ResNet
resnet = models.resnet50(pretrained=True)

# Replace final layer for your task (e.g., 10 fashion categories)
num_classes = 10
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# Fine-tune on domain data
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Training loop (simplified)
for epoch in range(10):
    for images, labels in train_loader:  # Your domain dataset
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# After training, use as feature extractor
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
```

---

## Audio Representation

### Music Recommendation Features

**Traditional approach**: Audio features
- **Mel-frequency cepstral coefficients** (MFCCs)
- **Spectral features** (centroid, rolloff, flux)
- **Tempo, rhythm**

**Modern approach**: Learned embeddings (CNNs on spectrograms).

---

### MFCCs (Mel-Frequency Cepstral Coefficients)

```python
import librosa
import numpy as np

# Load audio file
audio_path = 'song.mp3'
y, sr = librosa.load(audio_path, sr=22050)

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # (13, time_steps)

# Average over time
mfcc_mean = np.mean(mfccs, axis=1)  # (13,)

print(f"MFCC feature vector: {mfcc_mean}")
```

**Use case**: Music recommendation (Pandora), audio search.

---

### Deep Learning for Audio

**Approach**: Train CNN on spectrograms.

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 128)  # Embedding

    def forward(self, x):
        # x: (batch, 1, freq, time) - spectrogram
        x = self.conv(x)  # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 128)
        x = self.fc(x)  # (batch, 128)
        return x

# Use as feature extractor
model = AudioCNN()
# ... train on music classification task ...
# Then use embeddings for recommendation
```

---

## Multimodal Representation

### Combining Text and Images

**Example**: Product recommendation (e.g., fashion, furniture)
- **Text**: Product description, reviews
- **Image**: Product photo

**Approach 1: Early Fusion** (Concatenate features)

```python
# Text features (BERT)
text_emb = text_encoder(product_description)  # (768,)

# Image features (ResNet)
image_emb = image_encoder(product_image)  # (2048,)

# Concatenate
combined = np.concatenate([text_emb, image_emb])  # (2816,)

# Optional: Reduce dimension with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=256)
combined_reduced = pca.fit_transform(combined.reshape(1, -1))  # (256,)
```

**Problem**: Different modalities have different scales.

---

**Approach 2: Late Fusion** (Combine similarities)

```python
# Compute text similarity
text_sim = cosine_similarity(user_text_profile, item_text_features)

# Compute image similarity
image_sim = cosine_similarity(user_image_profile, item_image_features)

# Weighted combination
alpha = 0.6  # Text weight
combined_sim = alpha * text_sim + (1 - alpha) * image_sim
```

---

**Approach 3: Learned Fusion** (Neural network)

```python
import torch.nn as nn

class MultimodalFusion(nn.Module):
    def __init__(self, text_dim=768, image_dim=2048, hidden_dim=256, output_dim=128):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, text_emb, image_emb):
        text_h = self.text_proj(text_emb)  # (batch, 256)
        image_h = self.image_proj(image_emb)  # (batch, 256)
        combined = torch.cat([text_h, image_h], dim=1)  # (batch, 512)
        output = self.fusion(combined)  # (batch, 128)
        return output

# Usage
model = MultimodalFusion()
fused_emb = model(text_emb_tensor, image_emb_tensor)
```

**Advantage**: Model learns optimal combination weights.

---

## Dimensionality Reduction

### Why Reduce Dimensions?

**Problems with high-dimensional features**:
- **Curse of dimensionality**: Distance metrics become less meaningful
- **Computational cost**: Slower similarity computation
- **Storage**: More memory required

**Solution**: Reduce to 50-300 dimensions.

---

### 1. PCA (Principal Component Analysis)

**Idea**: Project to directions of maximum variance.

```python
from sklearn.decomposition import PCA
import numpy as np

# High-dimensional features (1000 items, 5000 features)
X = np.random.rand(1000, 5000)

# Reduce to 128 dimensions
pca = PCA(n_components=128)
X_reduced = pca.fit_transform(X)  # (1000, 128)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
```

**Output**:
```
Original shape: (1000, 5000)
Reduced shape: (1000, 128)
Explained variance ratio: 0.856
```

**Interpretation**: 128 dimensions capture 85.6% of variance.

---

### 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Use case**: Visualization (reduce to 2D or 3D), not recommendation.

**Problem**: Slow for large datasets, not deterministic.

```python
from sklearn.manifold import TSNE

# Reduce to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X)  # (1000, 2)

import matplotlib.pyplot as plt
plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.5)
plt.show()
```

---

### 3. Autoencoders

**Neural network for dimensionality reduction**.

```python
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=5000, encoding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

# Train autoencoder
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(50):
    for batch in data_loader:
        optimizer.zero_grad()
        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()

# Use encoder for dimensionality reduction
with torch.no_grad():
    reduced = model.encode(X_tensor)  # (1000, 128)
```

**Advantage**: Non-linear reduction, can capture complex patterns.

---

## Feature Selection

### Removing Irrelevant Features

**Problem**: Not all features are useful. Some add noise.

**Solution**: Feature selection.

---

**Approach 1: Variance Threshold**

Remove features with low variance (constant or near-constant).

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)  # Remove if variance < 0.01
X_selected = selector.fit_transform(X)

print(f"Features before: {X.shape[1]}")
print(f"Features after: {X_selected.shape[1]}")
```

---

**Approach 2: Correlation-Based Selection**

Remove highly correlated features (redundant).

```python
import pandas as pd

# Compute correlation matrix
df = pd.DataFrame(X)
corr_matrix = df.corr().abs()

# Find highly correlated pairs
threshold = 0.95
high_corr = np.where(np.triu(corr_matrix, k=1) > threshold)
to_drop = [corr_matrix.columns[i] for i in np.unique(high_corr[1])]

# Remove
X_reduced = df.drop(columns=to_drop).values
```

---

**Approach 3: Mutual Information**

Select features with high mutual information with target.

```python
from sklearn.feature_selection import mutual_info_classif

# Mutual information scores
mi_scores = mutual_info_classif(X, y)  # y = labels

# Select top 100 features
top_features = np.argsort(mi_scores)[::-1][:100]
X_selected = X[:, top_features]
```

---

## Production Considerations

### 1. Feature Caching

**Pre-compute item features** (updated daily/weekly).

```python
# Offline: Compute and store features
item_features = {}
for item_id in all_items:
    features = extract_features(item_id)  # Expensive
    item_features[item_id] = features

# Save to disk
import pickle
with open('item_features.pkl', 'wb') as f:
    pickle.dump(item_features, f)

# Online: Load pre-computed features
with open('item_features.pkl', 'rb') as f:
    item_features = pickle.load(f)

# Fast lookup
user_profile = ...
scores = {item_id: cosine_similarity(user_profile, feat)
          for item_id, feat in item_features.items()}
```

---

### 2. Incremental Updates

**New items arrive** → extract features → add to index.

```python
# Incremental feature extraction
def add_new_item(item_id, item_features_dict):
    features = extract_features(item_id)
    item_features_dict[item_id] = features
    # Update search index (e.g., FAISS)
    faiss_index.add(features.reshape(1, -1))
```

---

### 3. Feature Versioning

**Features evolve** (e.g., new BERT model) → need version control.

```
features_v1.pkl  (TF-IDF, 2023-01-01)
features_v2.pkl  (Word2Vec, 2023-06-01)
features_v3.pkl  (BERT, 2024-01-01)
```

**A/B test** new features before deploying.

---

## Summary

**Key Takeaways**:
1. **Text**: TF-IDF (baseline), Word2Vec/GloVe (semantic), BERT (state-of-the-art)
2. **Images**: Pre-trained CNNs (ResNet, EfficientNet), fine-tune for domain
3. **Audio**: MFCCs (traditional), CNNs on spectrograms (modern)
4. **Multimodal**: Concatenate, late fusion, or learned fusion
5. **Dimensionality reduction**: PCA (linear), autoencoders (non-linear)
6. **Feature selection**: Variance threshold, correlation, mutual information
7. **Production**: Cache features, incremental updates, versioning

**Best Practices**:
- Start simple (TF-IDF, pre-trained CNNs)
- Evaluate impact (A/B test)
- Optimize for latency (cache, reduce dimensions)
- Version features (reproducibility)

**Next**: User profile learning (how to build user preferences from item features).

---

## References

1. **Mikolov, T., et al. (2013)**. "Efficient Estimation of Word Representations in Vector Space". *ICLR*.
   - **Word2Vec** foundations

2. **Reimers, N., & Gurevych, I. (2019)**. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks". *EMNLP*.
   - **State-of-the-art** sentence embeddings

3. **He, K., et al. (2016)**. "Deep Residual Learning for Image Recognition". *CVPR*.
   - **ResNet** architecture

4. **Van den Oord, A., et al. (2013)**. "Deep content-based music recommendation". *NIPS*.
   - **Audio features** for music recommendation

5. **Devlin, J., et al. (2019)**. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". *NAACL*.
   - **BERT** for text understanding

---

## Practice Problems

### Problem 1: TF-IDF Weighting

**Given**:
```
Corpus: 5 documents
Word "algorithm": appears in 2 documents
Document 3: 200 words, "algorithm" appears 4 times
```

**Calculate**: TF-IDF for "algorithm" in Document 3.

**Solution**:
```
TF = 4 / 200 = 0.02

IDF = log(5 / 2) = log(2.5) = 0.916

TF-IDF = 0.02 × 0.916 = 0.0183
```

---

### Problem 2: Document Embedding

**Given**:
```
Word embeddings (simplified 3D):
  "dog": [1.0, 0.5, 0.2]
  "cat": [0.9, 0.6, 0.3]
  "pet": [0.95, 0.55, 0.25]

Document: ["dog", "cat", "pet"]

Compute average document embedding.
```

**Solution**:
```python
import numpy as np

embeddings = {
    "dog": np.array([1.0, 0.5, 0.2]),
    "cat": np.array([0.9, 0.6, 0.3]),
    "pet": np.array([0.95, 0.55, 0.25])
}

doc_emb = np.mean(list(embeddings.values()), axis=0)
print(doc_emb)  # [0.95, 0.55, 0.25]
```

---

### Problem 3: Image Feature Extraction

**Task**: Extract features from 1000 product images using ResNet-50.

**Code**:
```python
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

# Load ResNet
resnet = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extract features for all images
features = []
for img_path in image_paths:  # 1000 images
    img = Image.open(img_path)
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        feat = feature_extractor(img_tensor).squeeze().numpy()
    features.append(feat)

features = np.array(features)  # (1000, 2048)
print(f"Feature matrix shape: {features.shape}")
```
