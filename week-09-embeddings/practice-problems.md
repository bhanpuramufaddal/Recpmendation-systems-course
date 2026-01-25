# Week 9: Embeddings and Pre-training - Practice Problems

## Overview
Master embedding learning, Item2Vec, pre-training strategies, LLMs for recommendations, and multi-modal models like CLIP.

---

## Problem 1: Item2Vec Understanding
**Difficulty:** Medium

Item2Vec adapts Word2Vec to items. Given user sessions:
- Session 1: [A, B, C, D]
- Session 2: [B, C, E]
- Session 3: [A, C, F]

**Tasks:**
1. What is the "sentence" analogy for recommendations?
2. Generate skip-gram training pairs (window=2) for Session 1
3. Why is negative sampling needed?
4. Compare Item2Vec with matrix factorization

**Learning Outcomes:** Understand Word2Vec adaptation, generate training data, recognize connections to MF

---

## Problem 2: Contrastive Learning
**Difficulty:** Hard

Contrastive learning (SimCLR) learns embeddings by pulling positive pairs together, pushing negatives apart.

**Loss:** $L = -\log \frac{\exp(sim(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \exp(sim(z_i, z_k) / \tau)}$

**Questions:**
1. What are positive pairs for items? (same user's items, augmented views)
2. What is the temperature parameter τ?
3. How does this differ from BPR?
4. Design data augmentations for product images

**Learning Outcomes:** Implement contrastive learning, design augmentations, tune temperature

---

## Problem 3: LLMs for Zero-Shot Recommendations
**Difficulty:** Hard

Use GPT-4 to recommend movies via prompting:

**Prompt:** "User watched and loved: [Inception, The Matrix, Interstellar]. Recommend 5 similar movies."

**Questions:**
1. What are advantages over traditional RecSys?
2. What are limitations (cold start, personalization, scale)?
3. How would you incorporate user history efficiently (token limit)?
4. Design a hybrid: LLM for cold start, MF for warm users

**Learning Outcomes:** Apply LLMs to recommendations, recognize trade-offs, design hybrid systems

---

## Problem 4: CLIP for Multi-Modal Recommendations
**Difficulty:** Hard

CLIP learns joint text-image embeddings. Design a fashion recommender:

**Query:** "red summer dress, floral pattern"
**Database:** Product images

**Tasks:**
1. Encode text query with CLIP text encoder
2. Encode images with CLIP image encoder
3. Compute cosine similarity and rank
4. How do you handle multiple modalities (text + ref image)?

**Learning Outcomes:** Use CLIP for retrieval, combine multi-modal queries, design fusion strategies

---

## Problem 5: Pre-training vs. Training from Scratch
**Difficulty:** Medium

**Scenario:** Recommend scientific papers (small dataset, 10K papers, 1K users)

**Options:**
1. Train embeddings from scratch on interaction data
2. Pre-train on paper abstracts (BERT), fine-tune on interactions
3. Use pre-trained SciBERT, freeze encoder, train only ranking head

**Questions:**
1. Which would you choose and why?
2. What if you had 1M users instead?
3. How much does pre-training help on cold-start items?
4. Design the fine-tuning strategy

**Learning Outcomes:** Choose pre-training strategies, adapt to dataset size, design fine-tuning

---

## Programming Exercises

### Exercise 1: Implement Item2Vec

```python
from gensim.models import Word2Vec

# Prepare sessions
sessions = [['item_1', 'item_5', 'item_3'],
            ['item_2', 'item_5', 'item_6'],
            ...]  # User sessions as "sentences"

# Train Item2Vec
model = Word2Vec(sentences=sessions,
                 vector_size=100,
                 window=5,
                 min_count=2,
                 sg=1,  # Skip-gram
                 negative=5,
                 workers=4)

# Find similar items
similar = model.wv.most_similar('item_5', topn=10)
```

**Evaluation:** Measure if similar items share categories/tags

---

### Exercise 2: Fine-tune BERT for Movie Recommendations

```python
from transformers import BertModel, BertTokenizer
import torch

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')

# Encode movie descriptions
def encode_movie(description):
    inputs = tokenizer(description, return_tensors='pt', truncation=True, max_length=512)
    outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # [CLS] token

# Build movie embeddings
movie_embeddings = {movie_id: encode_movie(desc) for movie_id, desc in movies.items()}

# Recommend by similarity
user_liked_embs = [movie_embeddings[m] for m in user_liked_movies]
user_profile = torch.mean(torch.stack(user_liked_embs), dim=0)
similarities = {m: torch.cosine_similarity(user_profile, emb) for m, emb in movie_embeddings.items()}
```

---

### Exercise 3: Use CLIP for Product Search

```python
import clip
from PIL import Image

model, preprocess = clip.load("ViT-B/32", device="cuda")

# Encode text query
text = clip.tokenize(["red summer dress"]).to("cuda")
with torch.no_grad():
    text_features = model.encode_text(text)

# Encode product images
image_features = []
for img_path in product_images:
    image = preprocess(Image.open(img_path)).unsqueeze(0).to("cuda")
    with torch.no_grad():
        image_features.append(model.encode_image(image))

image_features = torch.cat(image_features)

# Compute similarities
similarities = (text_features @ image_features.T).softmax(dim=-1)
top_products = similarities.topk(10).indices
```

---

### Exercise 4: Contrastive Learning for Items

```python
class ContrastiveRecommender(nn.Module):
    def __init__(self, n_items, embedding_dim=128):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, embedding_dim)

    def forward(self, anchor, positive, negatives):
        anchor_emb = self.item_emb(anchor)
        pos_emb = self.item_emb(positive)
        neg_embs = self.item_emb(negatives)

        # Positive similarity
        pos_sim = F.cosine_similarity(anchor_emb, pos_emb)

        # Negative similarities
        neg_sims = F.cosine_similarity(anchor_emb.unsqueeze(1), neg_embs, dim=-1)

        # Contrastive loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1) / temperature
        labels = torch.zeros(len(anchor), dtype=torch.long)  # Positive is at index 0
        loss = F.cross_entropy(logits, labels)
        return loss
```

---

## Discussion Questions

1. **Embedding Quality:** How do you evaluate if learned embeddings are good without a downstream task?
2. **Cross-Domain Transfer:** Can movie embeddings transfer to book recommendations?
3. **LLM Hallucination:** LLMs can recommend non-existent items. How do you constrain them?
4. **Multi-Modal Fusion:** Early vs. late fusion for text+image+audio?
5. **Privacy:** Can embeddings leak user information? How do you protect privacy?

---

## References
1. Barkan, O., & Koenigstein, N. (2016). "Item2Vec: Neural item embedding for collaborative filtering". IEEE.
2. Radford, A., et al. (2021). "Learning transferable visual models from natural language supervision". CLIP.
3. Chen, T., et al. (2020). "A simple framework for contrastive learning of visual representations". SimCLR.

---

*Return to [Week 9 Main Page](README.md)*
