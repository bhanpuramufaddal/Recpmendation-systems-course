# Week 9: Multi-Modal Recommendations

## Overview

**Multi-modal recommendations** leverage multiple data modalities (text, images, audio, video) to understand items and users more comprehensively.

**Example modalities**:
- **E-commerce**: Product images + titles + descriptions + reviews
- **Music**: Audio features + lyrics + album art + metadata
- **Movies**: Trailers + posters + plot summaries + cast
- **Fashion**: Images + material + brand + user photos

**Key advantages**:
- **Richer representations**: Capture aspects text alone misses (visual style, audio mood)
- **Cold start**: New items with images but no interactions
- **Cross-modal retrieval**: Search products with images, find similar via text

**Foundation**: **CLIP** (Contrastive Language-Image Pre-training) - aligns vision and language.

This document covers multi-modal recommendation systems.

---

## Learning Objectives

By the end of this section, you will:
- Understand vision-language models (CLIP, ALIGN)
- Implement multi-modal fusion strategies
- Apply cross-modal retrieval for recommendations
- Build multi-modal recommendation systems
- Handle missing modalities gracefully

---

## CLIP: Contrastive Language-Image Pre-training

### Core Idea

**Goal**: Learn joint embedding space where **images and text** describing the same concept are close.

**Training data**: 400M (image, caption) pairs from the internet.

**Architecture**:
```
Image ─→ Image Encoder ─→ Image Embedding (d-dim)
                                ↓
                         Cosine Similarity
                                ↓
Text ──→ Text Encoder ──→ Text Embedding (d-dim)
```

**Objective**: Contrastive loss (similar to Word2Vec's negative sampling).

---

### CLIP Training

**Batch**: $N$ (image, text) pairs.

**Positive pairs**: $(image_i, text_i)$ - correctly matched.

**Negative pairs**: $(image_i, text_j)$ for $i \neq j$ - mismatched.

**Loss** (symmetric):
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ \log \frac{\exp(\mathbf{v}_i^T \mathbf{t}_i / \tau)}{\sum_{j=1}^N \exp(\mathbf{v}_i^T \mathbf{t}_j / \tau)} + \log \frac{\exp(\mathbf{t}_i^T \mathbf{v}_i / \tau)}{\sum_{j=1}^N \exp(\mathbf{t}_i^T \mathbf{v}_j / \tau)} \right]$$

where:
- $\mathbf{v}_i$ = image embedding
- $\mathbf{t}_i$ = text embedding
- $\tau$ = temperature (typically 0.07)

**Intuition**: Pull matching (image, text) together, push apart non-matching.

---

### Using Pre-trained CLIP

```python
import torch
import clip
from PIL import Image

# Load pre-trained CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Encode image
image = Image.open("product.jpg")
image_input = preprocess(image).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize

print(f"Image embedding shape: {image_features.shape}")  # (1, 512)

# Encode text
text_inputs = clip.tokenize(["a red dress", "a blue shirt", "running shoes"]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

print(f"Text embeddings shape: {text_features.shape}")  # (3, 512)

# Compute similarities
similarities = (image_features @ text_features.T).squeeze()
print(f"Similarities: {similarities}")

# Find best match
best_match_idx = similarities.argmax()
labels = ["a red dress", "a blue shirt", "running shoes"]
print(f"Best match: {labels[best_match_idx]} (score: {similarities[best_match_idx]:.3f})")
```

---

## Multi-Modal Item Representations

### Fusion Strategies

**Problem**: How to combine embeddings from different modalities?

**Strategies**:

**1. Early Fusion** (concatenation):
$$\mathbf{h} = [\mathbf{v}_{\text{image}}; \mathbf{v}_{\text{text}}; \mathbf{v}_{\text{audio}}]$$

**2. Late Fusion** (separate scores, then combine):
$$\text{score} = \alpha \cdot \text{score}_{\text{image}} + \beta \cdot \text{score}_{\text{text}} + \gamma \cdot \text{score}_{\text{audio}}$$

**3. Cross-Attention** (learn interactions):
$$\mathbf{h}_{\text{image}} = \text{Attention}(\mathbf{Q}_{\text{image}}, \mathbf{K}_{\text{text}}, \mathbf{V}_{\text{text}})$$

**4. Gated Fusion** (learned weighting):
$$\mathbf{h} = \mathbf{g} \odot \mathbf{v}_{\text{image}} + (1 - \mathbf{g}) \odot \mathbf{v}_{\text{text}}$$

where $\mathbf{g} = \sigma(\mathbf{W} [\mathbf{v}_{\text{image}}; \mathbf{v}_{\text{text}}])$

---

### Implementation: Multi-Modal Item Encoder

```python
import torch.nn as nn
import torch.nn.functional as F

class MultiModalItemEncoder(nn.Module):
    def __init__(self, image_dim=512, text_dim=512, fusion_dim=256, fusion_type='concat'):
        """
        Multi-modal encoder for items with image and text.

        fusion_type: 'concat', 'gated', 'attention'
        """
        super().__init__()

        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            # Early fusion: concatenate and project
            self.fusion = nn.Sequential(
                nn.Linear(image_dim + text_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim)
            )

        elif fusion_type == 'gated':
            # Gated fusion
            self.image_proj = nn.Linear(image_dim, fusion_dim)
            self.text_proj = nn.Linear(text_dim, fusion_dim)
            self.gate = nn.Linear(image_dim + text_dim, fusion_dim)

        elif fusion_type == 'attention':
            # Cross-attention
            self.image_proj = nn.Linear(image_dim, fusion_dim)
            self.text_proj = nn.Linear(text_dim, fusion_dim)
            self.attention = nn.MultiheadAttention(fusion_dim, num_heads=4, batch_first=True)

    def forward(self, image_emb, text_emb):
        """
        image_emb: (batch, image_dim)
        text_emb: (batch, text_dim)
        """
        if self.fusion_type == 'concat':
            # Concatenate
            combined = torch.cat([image_emb, text_emb], dim=1)
            fused = self.fusion(combined)

        elif self.fusion_type == 'gated':
            # Project modalities
            v_img = self.image_proj(image_emb)
            v_text = self.text_proj(text_emb)

            # Compute gate
            gate_input = torch.cat([image_emb, text_emb], dim=1)
            g = torch.sigmoid(self.gate(gate_input))

            # Gated fusion
            fused = g * v_img + (1 - g) * v_text

        elif self.fusion_type == 'attention':
            # Project
            v_img = self.image_proj(image_emb).unsqueeze(1)  # (batch, 1, dim)
            v_text = self.text_proj(text_emb).unsqueeze(1)  # (batch, 1, dim)

            # Cross-attention (image attends to text)
            fused, _ = self.attention(v_img, v_text, v_text)
            fused = fused.squeeze(1)  # (batch, dim)

        # L2 normalize
        fused = F.normalize(fused, dim=1)

        return fused


# Example
encoder = MultiModalItemEncoder(image_dim=512, text_dim=512, fusion_dim=256, fusion_type='gated')

# Sample embeddings
batch_size = 32
image_embs = torch.randn(batch_size, 512)
text_embs = torch.randn(batch_size, 512)

# Fuse
item_embs = encoder(image_embs, text_embs)
print(f"Fused embeddings shape: {item_embs.shape}")  # (32, 256)
```

---

## Multi-Modal Recommendation System

### Architecture

```
User History (item IDs)
        ↓
  [Image Encoder, Text Encoder]
        ↓
  Multi-Modal Item Embeddings
        ↓
  Aggregate (average, attention)
        ↓
  User Embedding
        ↓
  Dot Product with Candidate Items
        ↓
  Scores
```

---

### End-to-End Multi-Modal RecSys

```python
class MultiModalRecommender(nn.Module):
    def __init__(self, num_items, image_encoder, text_encoder, fusion_dim=256):
        """
        num_items: Total number of items
        image_encoder: Pre-trained image encoder (e.g., CLIP)
        text_encoder: Pre-trained text encoder (e.g., CLIP)
        """
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # Multi-modal fusion
        image_dim = image_encoder.output_dim
        text_dim = text_encoder.output_dim
        self.item_encoder = MultiModalItemEncoder(image_dim, text_dim, fusion_dim, fusion_type='gated')

        # User encoder (aggregate item embeddings)
        self.user_encoder = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )

    def encode_item(self, images, texts):
        """
        Encode items from images and texts.

        images: (batch, C, H, W)
        texts: (batch, seq_len) - tokenized
        """
        with torch.no_grad():
            image_embs = self.image_encoder(images)
            text_embs = self.text_encoder(texts)

        # Fuse modalities
        item_embs = self.item_encoder(image_embs, text_embs)
        return item_embs

    def encode_user(self, user_history_embs):
        """
        Encode user from history of item embeddings.

        user_history_embs: (batch, num_history, fusion_dim)
        """
        # Average pooling (could use attention)
        user_agg = user_history_embs.mean(dim=1)  # (batch, fusion_dim)

        # Project
        user_emb = self.user_encoder(user_agg)
        user_emb = F.normalize(user_emb, dim=1)

        return user_emb

    def forward(self, user_history_embs, candidate_items_embs):
        """
        Score candidates for user.

        user_history_embs: (batch, num_history, fusion_dim)
        candidate_items_embs: (batch, num_candidates, fusion_dim)
        """
        # User embedding
        user_emb = self.encode_user(user_history_embs)  # (batch, fusion_dim)

        # Scores (dot product)
        scores = torch.bmm(candidate_items_embs, user_emb.unsqueeze(2)).squeeze()  # (batch, num_candidates)

        return scores


# Note: Simplified example - in practice, you'd load CLIP or similar pre-trained models
```

---

## Cross-Modal Retrieval

### Text-to-Image Search

**Use case**: User searches "red running shoes" → retrieve product images.

**Method**:
1. Encode search query (text) → embedding
2. Compute similarity to all product images
3. Return top-K most similar

```python
def text_to_image_search(text_query, image_embeddings, model, preprocess, top_k=10):
    """
    Search images using text query.

    text_query: String (e.g., "red running shoes")
    image_embeddings: Pre-computed image embeddings (num_images, dim)
    """
    import clip

    # Encode text query
    text_input = clip.tokenize([text_query]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text_input)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)

    # Compute similarities
    similarities = (text_emb @ image_embeddings.T).squeeze()

    # Top-K
    top_indices = torch.argsort(similarities, descending=True)[:top_k]
    top_scores = similarities[top_indices]

    return top_indices.cpu().numpy(), top_scores.cpu().numpy()


# Example
# Assume we have 10000 product images pre-encoded
# image_embeddings = torch.randn(10000, 512)  # Replace with actual CLIP embeddings

# query = "red dress"
# indices, scores = text_to_image_search(query, image_embeddings, model, preprocess, top_k=5)
# print(f"Top 5 results for '{query}':")
# for idx, score in zip(indices, scores):
#     print(f"  Image {idx}: score = {score:.3f}")
```

---

### Image-to-Text Search

**Use case**: User uploads photo → find similar products with descriptions.

**Example** (fashion):
```
User uploads: [photo of blue jeans]
System returns:
  1. "Slim-fit denim jeans in blue wash" (score: 0.92)
  2. "Classic blue jeans with stretch fabric" (score: 0.87)
  3. "Dark blue skinny jeans" (score: 0.85)
```

```python
def image_to_text_search(image_path, text_embeddings, item_descriptions, model, preprocess, top_k=10):
    """
    Search text descriptions using image query.
    """
    # Load and encode image
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_emb = model.encode_image(image_input)
        image_emb /= image_emb.norm(dim=-1, keepdim=True)

    # Compute similarities
    similarities = (image_emb @ text_embeddings.T).squeeze()

    # Top-K
    top_indices = torch.argsort(similarities, descending=True)[:top_k]

    results = [(item_descriptions[idx], similarities[idx].item()) for idx in top_indices]

    return results


# Example
# descriptions = ["red dress", "blue shirt", "running shoes", ...]
# text_embeddings = encode_texts(descriptions, model)  # (num_items, 512)

# results = image_to_text_search("user_photo.jpg", text_embeddings, descriptions, model, preprocess)
# for desc, score in results[:5]:
#     print(f"{desc}: {score:.3f}")
```

---

## Handling Missing Modalities

### Problem

**Real-world**: Not all items have all modalities.

**Examples**:
- E-commerce: 80% have images, 60% have detailed descriptions, 20% have videos
- Music: All have audio, 70% have lyrics, 30% have music videos

**Challenge**: How to recommend items with incomplete modalities?

---

### Strategies

**1. Modality Dropout** (training):
- Randomly drop modalities during training → model learns to handle missing data

```python
def modality_dropout(image_emb, text_emb, dropout_prob=0.3):
    """
    Randomly drop modalities during training.
    """
    if training and torch.rand(1).item() < dropout_prob:
        # Drop image
        image_emb = torch.zeros_like(image_emb)

    if training and torch.rand(1).item() < dropout_prob:
        # Drop text
        text_emb = torch.zeros_like(text_emb)

    return image_emb, text_emb
```

**2. Imputation** (inference):
- Predict missing modality from available ones

```python
class ModalityImputer(nn.Module):
    def __init__(self, text_dim=512, image_dim=512):
        super().__init__()
        self.text_to_image = nn.Sequential(
            nn.Linear(text_dim, image_dim),
            nn.ReLU(),
            nn.Linear(image_dim, image_dim)
        )

    def forward(self, text_emb, has_image):
        """
        Impute image embedding from text if missing.

        has_image: (batch,) boolean tensor
        """
        imputed_image = self.text_to_image(text_emb)

        # Use imputed only where image is missing
        return imputed_image
```

**3. Weighted Fusion** (inference):
- Adjust fusion weights based on available modalities

```python
def adaptive_fusion(image_emb, text_emb, has_image, has_text):
    """
    Fuse modalities, giving more weight to available ones.
    """
    weight_image = has_image.float().unsqueeze(1)
    weight_text = has_text.float().unsqueeze(1)

    # Normalize weights
    total_weight = weight_image + weight_text + 1e-8
    weight_image /= total_weight
    weight_text /= total_weight

    fused = weight_image * image_emb + weight_text * text_emb

    return fused
```

---

## Case Study: Pinterest's Multi-Modal System

### Overview

**Pinterest**: Visual discovery platform (2B+ pins, 450M+ users).

**Modalities**: Images + text (titles, descriptions) + engagement (clicks, saves).

**Challenge**: Recommend visually similar pins with relevant content.

---

### Architecture

**1. Visual Encoder**: ResNet-50 pre-trained on ImageNet, fine-tuned on pins.

**2. Text Encoder**: BERT fine-tuned on pin descriptions.

**3. Fusion**: Concatenate image + text embeddings → MLP.

**4. Training**: Triplet loss (user-positive pin-negative pin).

**5. Serving**: ANN index (HNSW) on fused embeddings → <50ms retrieval.

---

### Key Insights

**Insight 1**: Multi-modal **outperforms** single-modal by 12% in engagement.

**Insight 2**: **Image dominates** for fashion/home decor, **text dominates** for recipes/DIY.

**Insight 3**: **Cross-modal search** (text → image) drives 25% of searches.

---

## Production Considerations

### 1. Scalability

**Challenge**: Encoding images/text for billions of items.

**Solutions**:
- **Batch processing**: Encode items offline (daily)
- **Caching**: Store embeddings in fast key-value store (Redis, Memcached)
- **Quantization**: Reduce embedding precision (float32 → int8)

---

### 2. Model Updates

**Challenge**: Retrain multi-modal models without disrupting service.

**Solutions**:
- **Incremental fine-tuning**: Add new items without full retraining
- **Online learning**: Update embeddings as users interact
- **A/B testing**: Test new models on small traffic before rollout

---

### 3. Modality Imbalance

**Challenge**: One modality dominates (e.g., images always weighted higher).

**Solutions**:
- **Balanced sampling**: Ensure equal representation during training
- **Modality-specific losses**: Separate losses for each modality
- **Calibration**: Adjust fusion weights based on validation performance

---

## Summary

**Key Takeaways**:
1. **Multi-modal**: Leverage images, text, audio for richer item understanding
2. **CLIP**: Foundation model for vision-language alignment
3. **Fusion**: Concat, gated, attention strategies for combining modalities
4. **Cross-modal retrieval**: Text → image, image → text search
5. **Missing modalities**: Dropout, imputation, weighted fusion

**Best Practices**:
- **Pre-trained models**: Use CLIP, ALIGN for strong baselines
- **Modality dropout**: Train with missing modalities for robustness
- **Balanced fusion**: Avoid single modality dominating
- **Offline encoding**: Pre-compute embeddings for fast serving

**When to use**:
- **Rich content**: Items with images + text (e-commerce, media)
- **Visual discovery**: Users browse visually (Pinterest, Instagram)
- **Cross-modal search**: Enable flexible search (text query → image results)

**Next**: Context-aware recommendations and bandits.

---

## References

1. **Radford, A., et al. (2021)**. "Learning Transferable Visual Models From Natural Language Supervision". *ICML*.
   - **CLIP** model

2. **Jia, C., et al. (2021)**. "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision". *ICML*.
   - **ALIGN** (similar to CLIP)

3. **Ying, R., et al. (2018)**. "Graph Convolutional Neural Networks for Web-Scale Recommender Systems". *KDD*.
   - **PinSage** (Pinterest's multi-modal GNN)

4. **Li, L., et al. (2020)**. "OSCAR: Object-Semantics Aligned Pre-training for Vision-Language Tasks". *ECCV*.
   - **Vision-language pre-training**

5. **Chen, Y.-C., et al. (2020)**. "UNITER: Universal Image-Text Representation Learning". *ECCV*.
   - **Multi-modal Transformer**

---

## Practice Problems

### Problem 1: CLIP Similarity

**Given CLIP embeddings** (normalized):
- Image: $\mathbf{v} = [0.6, 0.8]$
- Text 1: $\mathbf{t}_1 = [0.8, 0.6]$
- Text 2: $\mathbf{t}_2 = [-0.6, 0.8]$

**Compute**: Which text matches the image better?

**Solution**:
```python
import numpy as np

v = np.array([0.6, 0.8])
t1 = np.array([0.8, 0.6])
t2 = np.array([-0.6, 0.8])

sim1 = np.dot(v, t1)  # 0.6*0.8 + 0.8*0.6 = 0.96
sim2 = np.dot(v, t2)  # 0.6*(-0.6) + 0.8*0.8 = 0.28

print(f"Similarity with Text 1: {sim1:.3f}")
print(f"Similarity with Text 2: {sim2:.3f}")
print(f"Best match: Text 1")
```

---

### Problem 2: Gated Fusion

**Given**:
- Image embedding: $\mathbf{v}_{\text{img}} = [1, 0]$
- Text embedding: $\mathbf{v}_{\text{text}} = [0, 1]$
- Gate: $\mathbf{g} = [0.7, 0.3]$

**Compute**: Fused embedding.

**Solution**:
```python
v_img = np.array([1, 0])
v_text = np.array([0, 1])
g = np.array([0.7, 0.3])

fused = g * v_img + (1 - g) * v_text
print(f"Fused embedding: {fused}")
# Output: [0.7, 0.7]

# Explanation:
# Dimension 1: 0.7*1 + 0.3*0 = 0.7 (more from image)
# Dimension 2: 0.3*0 + 0.7*1 = 0.7 (more from text)
```

---

### Problem 3: Modality Dropout

**Scenario**: Training with 30% modality dropout.

**Question**: What's the probability both modalities are dropped for an item?

**Solution**:
```python
p_drop_image = 0.3
p_drop_text = 0.3

# Assuming independent dropout
p_both_dropped = p_drop_image * p_drop_text
print(f"Probability both dropped: {p_both_dropped:.3f}")
# Output: 0.09 (9%)

# Probability at least one available
p_at_least_one = 1 - p_both_dropped
print(f"Probability at least one available: {p_at_least_one:.3f}")
# Output: 0.91 (91%)
```
