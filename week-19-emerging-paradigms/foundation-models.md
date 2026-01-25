# Week 19: Foundation Models for Recommendations

## Overview

**Foundation models**: Large pre-trained models that can be adapted to many recommendation tasks.

**Vision**: One model for all recommendation tasks (like GPT for text).

**Key characteristics**:
1. **Pre-training**: Train on massive interaction data
2. **Task-agnostic**: No task-specific architecture
3. **Prompt-based**: Use prompts to specify task
4. **Few-shot adaptation**: Adapt quickly with minimal data

**Examples**:
- GPT-4 for text recommendations
- CLIP for multimodal recommendations
- UniRec: Universal recommender

---

## Motivation

### Current State

**Problem**: Each recommendation domain has separate model.
- Movies: Specialized movie recommender
- Music: Specialized music recommender
- E-commerce: Specialized product recommender

**Limitations**:
- No knowledge transfer across domains
- Cold start in new domains
- Redundant engineering

---

### Foundation Model Vision

**Goal**: Single model that works across all domains.

**Benefits**:
- **Transfer learning**: Knowledge from movies helps books
- **Zero-shot**: Recommend in new domain without training
- **Efficiency**: Train once, use everywhere

**Analogy**: GPT-3 for language → Universal Recommender for items.

---

## Pre-training Strategies

### Masked Item Prediction

**Idea**: Mask random items, predict them (like BERT).

**Process**:
1. User history: `[Item1, Item2, [MASK], Item4, Item5]`
2. Model predicts: `[MASK] = Item3`

**Objective**:
$$\mathcal{L} = -\sum_{i \in \text{masked}} \log P(\text{Item}_i | \text{context})$$

---

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FoundationRecommender(nn.Module):
    def __init__(self, n_items, d_model=512, n_heads=8, n_layers=6):
        super().__init__()

        # Item embedding
        self.item_embedding = nn.Embedding(n_items + 2, d_model)  # +2 for [MASK], [PAD]

        # Positional encoding
        self.pos_embedding = nn.Embedding(1000, d_model)  # Max sequence length

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, n_items)

        self.mask_token_id = n_items  # Special token for masking
        self.pad_token_id = n_items + 1

    def forward(self, item_sequence, masked_positions=None):
        """
        Forward pass for pre-training or inference.

        Args:
            item_sequence: [batch_size, seq_len] item IDs
            masked_positions: [batch_size, n_masked] positions of masked items

        Returns:
            logits: [batch_size, seq_len, n_items] prediction logits
        """
        batch_size, seq_len = item_sequence.size()

        # Embed items
        item_emb = self.item_embedding(item_sequence)  # [batch, seq_len, d_model]

        # Add positional embeddings
        positions = torch.arange(seq_len, device=item_sequence.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)

        x = item_emb + pos_emb

        # Transformer encoding
        encoded = self.transformer(x)  # [batch, seq_len, d_model]

        # Project to item space
        logits = self.output_proj(encoded)  # [batch, seq_len, n_items]

        return logits

    def pre_train_step(self, item_sequence, mask_prob=0.15):
        """
        Pre-training step with random masking.

        Args:
            item_sequence: [batch_size, seq_len]
            mask_prob: Probability of masking each item

        Returns:
            loss: Masked item prediction loss
        """
        batch_size, seq_len = item_sequence.size()

        # Create random mask
        mask = torch.rand(batch_size, seq_len) < mask_prob

        # Save original items for target
        target_items = item_sequence.clone()

        # Replace masked positions with [MASK] token
        masked_sequence = item_sequence.clone()
        masked_sequence[mask] = self.mask_token_id

        # Forward pass
        logits = self.forward(masked_sequence)

        # Compute loss only on masked positions
        loss = F.cross_entropy(
            logits[mask],
            target_items[mask],
            ignore_index=self.pad_token_id
        )

        return loss


# Pre-training
model = FoundationRecommender(n_items=100000, d_model=512, n_heads=8, n_layers=6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(100):
    for item_sequences in pretrain_loader:
        # item_sequences: [batch_size, seq_len] from millions of users

        loss = model.pre_train_step(item_sequences, mask_prob=0.15)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Pre-train Loss = {loss:.4f}")
```

---

### Contrastive Pre-training

**Idea**: Similar users should have similar representations.

**Approach**:
1. Sample user pairs (similar vs. random)
2. Maximize similarity for similar users
3. Minimize similarity for random users

**Loss (InfoNCE)**:
$$\mathcal{L} = -\log \frac{\exp(\text{sim}(u, u^+) / \tau)}{\exp(\text{sim}(u, u^+) / \tau) + \sum_{u^-} \exp(\text{sim}(u, u^-) / \tau)}$$

---

### Implementation

```python
class ContrastiveFoundationModel(nn.Module):
    def __init__(self, base_model, projection_dim=128):
        super().__init__()
        self.base_model = base_model

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def encode(self, item_sequence):
        """
        Encode user sequence to representation.

        Returns:
            user_emb: [batch_size, projection_dim]
        """
        # Get transformer encoding
        logits = self.base_model(item_sequence)

        # Pool sequence (take mean)
        sequence_emb = logits.mean(dim=1)  # [batch, d_model]

        # Project to contrastive space
        user_emb = self.projection(sequence_emb)

        # L2 normalize
        user_emb = F.normalize(user_emb, dim=1)

        return user_emb

    def contrastive_loss(self, user_sequences, positive_sequences, negative_sequences, temperature=0.07):
        """
        Compute contrastive loss.

        Args:
            user_sequences: [batch_size, seq_len] anchor users
            positive_sequences: [batch_size, seq_len] similar users
            negative_sequences: [batch_size * K, seq_len] dissimilar users

        Returns:
            loss: Contrastive loss
        """
        # Encode sequences
        user_emb = self.encode(user_sequences)  # [batch, proj_dim]
        pos_emb = self.encode(positive_sequences)  # [batch, proj_dim]
        neg_emb = self.encode(negative_sequences)  # [batch * K, proj_dim]

        # Positive similarity
        pos_sim = (user_emb * pos_emb).sum(dim=1) / temperature  # [batch]

        # Negative similarities
        neg_sim = torch.matmul(user_emb, neg_emb.T) / temperature  # [batch, batch * K]

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch, 1 + batch * K]
        labels = torch.zeros(len(user_emb), dtype=torch.long, device=user_emb.device)

        loss = F.cross_entropy(logits, labels)

        return loss


# Pre-training with contrastive learning
base_model = FoundationRecommender(n_items=100000, d_model=512)
model = ContrastiveFoundationModel(base_model, projection_dim=128)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(100):
    for user_seq, pos_seq, neg_seqs in contrastive_loader:
        loss = model.contrastive_loss(user_seq, pos_seq, neg_seqs, temperature=0.07)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Contrastive Loss = {loss:.4f}")
```

---

## Prompt-Based Recommendations

### LLM as Recommender

**Idea**: Use large language models (GPT-4) for recommendations via prompting.

**Approach**:
1. Convert user history to text
2. Prompt LLM: "Given user liked X, Y, Z, recommend items"
3. Parse LLM output

---

### Implementation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMRecommender:
    def __init__(self, model_name="gpt-3.5-turbo"):
        """
        Args:
            model_name: Hugging Face model ID or OpenAI model name
        """
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Example
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    def create_prompt(self, user_history, item_catalog, n_recommendations=10):
        """
        Create prompt for LLM.

        Args:
            user_history: List of items user liked
            item_catalog: Dict mapping item IDs to names
            n_recommendations: Number of recommendations to generate

        Returns:
            prompt: Text prompt for LLM
        """
        # Convert history to text
        history_text = ", ".join([item_catalog[item_id] for item_id in user_history])

        prompt = f"""You are a recommendation system. Given a user's interaction history, recommend {n_recommendations} items they might like.

User's history: {history_text}

Recommended items:"""

        return prompt

    def recommend(self, user_history, item_catalog, n_recommendations=10):
        """
        Generate recommendations using LLM.

        Returns:
            recommendations: List of recommended item IDs
        """
        # Create prompt
        prompt = self.create_prompt(user_history, item_catalog, n_recommendations)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse recommendations (simplified)
        recommendations = self._parse_recommendations(generated_text, item_catalog)

        return recommendations

    def _parse_recommendations(self, generated_text, item_catalog):
        """
        Parse LLM output to extract recommended items.
        """
        # Simplified parsing: extract item names from generated text
        # In practice, use more robust parsing (regex, structured output)

        recommendations = []

        for item_id, item_name in item_catalog.items():
            if item_name.lower() in generated_text.lower():
                recommendations.append(item_id)

            if len(recommendations) >= 10:
                break

        return recommendations


# Example usage
llm_recommender = LLMRecommender(model_name="gpt2")

user_history = [10, 25, 47]  # Item IDs
item_catalog = {
    10: "The Matrix",
    25: "Inception",
    47: "Interstellar",
    100: "Blade Runner",
    200: "The Prestige",
    # ... more items
}

recommendations = llm_recommender.recommend(user_history, item_catalog, n_recommendations=5)
print(f"Recommendations: {recommendations}")
```

---

### Prompt Engineering for RecSys

**Strategies**:

**1. Few-shot prompting**: Include examples in prompt
```
User A liked: X, Y → Recommended: Z
User B liked: A, B → Recommended: C
User C liked: {user_history} → Recommended:
```

**2. Chain-of-thought**: Ask model to explain reasoning
```
User liked {history}.
Think step-by-step about what genres/themes they enjoy.
Then recommend items based on those preferences.
```

**3. Constrained generation**: Force output format
```
Recommend exactly 5 items in JSON format:
{"recommendations": ["Item1", "Item2", ...]}
```

---

## Zero-Shot and Few-Shot Adaptation

### Zero-Shot Transfer

**Goal**: Recommend in new domain without training.

**Approach**:
1. Pre-train on domains A, B, C
2. Test on domain D (never seen)
3. Use prompts to specify domain: "Recommend movies similar to..."

---

### Few-Shot Fine-Tuning

**Approach**: Adapt with minimal domain-specific data.

```python
def few_shot_adapt(foundation_model, few_shot_data, n_epochs=5):
    """
    Adapt foundation model to new domain with few examples.

    Args:
        foundation_model: Pre-trained model
        few_shot_data: Small dataset from new domain (e.g., 100 users)
        n_epochs: Number of fine-tuning epochs

    Returns:
        adapted_model: Fine-tuned model
    """
    # Freeze most layers, only fine-tune last layers
    for param in foundation_model.parameters():
        param.requires_grad = False

    # Unfreeze last 2 layers
    for param in foundation_model.transformer.layers[-2:].parameters():
        param.requires_grad = True

    for param in foundation_model.output_proj.parameters():
        param.requires_grad = True

    # Fine-tune
    optimizer = torch.optim.Adam(foundation_model.parameters(), lr=0.0001)

    for epoch in range(n_epochs):
        for item_sequences in few_shot_data:
            loss = foundation_model.pre_train_step(item_sequences)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return foundation_model


# Example
foundation_model = load_pretrained_model()  # Pre-trained on movies, music, books

# Adapt to new domain (e.g., podcasts) with only 100 users
podcast_data = load_few_shot_data(domain='podcasts', n_users=100)

adapted_model = few_shot_adapt(foundation_model, podcast_data, n_epochs=5)

# Evaluate on podcast test set
podcast_test = load_test_data(domain='podcasts')
ndcg = evaluate(adapted_model, podcast_test)

print(f"Zero-shot NDCG: {ndcg_zero_shot:.4f}")  # Without adaptation
print(f"Few-shot NDCG: {ndcg:.4f}")  # With 100-user adaptation
```

---

## Unified Recommendation Interface

### Task-Agnostic API

**Goal**: Single API for all recommendation tasks.

**Tasks**:
- Rating prediction
- Top-N recommendation
- Sequential recommendation
- Session-based recommendation
- Cross-domain recommendation

**Unified interface**:
```python
model.recommend(
    user_history=[10, 25, 47],
    task="top_n",
    domain="movies",
    k=10
)
```

---

### Implementation

```python
class UniversalRecommender:
    def __init__(self, foundation_model):
        self.model = foundation_model

    def recommend(self, user_history, task="top_n", domain=None, k=10, **kwargs):
        """
        Universal recommendation interface.

        Args:
            user_history: List of item IDs user interacted with
            task: Recommendation task type
            domain: Domain name (optional, for multi-domain models)
            k: Number of recommendations
            **kwargs: Task-specific parameters

        Returns:
            recommendations: Task-specific output
        """
        if task == "top_n":
            return self._top_n_recommendation(user_history, k)

        elif task == "rating_prediction":
            item_id = kwargs.get('item_id')
            return self._predict_rating(user_history, item_id)

        elif task == "sequential":
            return self._sequential_recommendation(user_history, k)

        elif task == "session":
            session_items = kwargs.get('session_items')
            return self._session_recommendation(session_items, k)

        else:
            raise ValueError(f"Unknown task: {task}")

    def _top_n_recommendation(self, user_history, k):
        """Standard top-N recommendation."""
        user_seq = torch.tensor([user_history])

        with torch.no_grad():
            logits = self.model(user_seq)

            # Take last position logits
            last_logits = logits[:, -1, :]

            # Mask already-interacted items
            last_logits[:, user_history] = -float('inf')

            # Top-K
            top_k_items = torch.topk(last_logits, k).indices.squeeze()

        return top_k_items.tolist()

    def _predict_rating(self, user_history, item_id):
        """Predict rating for specific item."""
        user_seq = torch.tensor([user_history + [item_id]])

        with torch.no_grad():
            logits = self.model(user_seq)

            # Get score for item
            score = logits[:, -1, item_id].item()

        # Convert logit to rating (1-5 scale)
        rating = torch.sigmoid(torch.tensor(score)).item() * 4 + 1

        return rating

    def _sequential_recommendation(self, user_history, k):
        """Predict next items in sequence."""
        return self._top_n_recommendation(user_history, k)

    def _session_recommendation(self, session_items, k):
        """Session-based recommendation (within-session)."""
        return self._top_n_recommendation(session_items, k)


# Usage
foundation_model = load_pretrained_model()
recommender = UniversalRecommender(foundation_model)

# Top-N recommendation
top_n = recommender.recommend(
    user_history=[10, 25, 47],
    task="top_n",
    k=10
)
print(f"Top-10 recommendations: {top_n}")

# Rating prediction
rating = recommender.recommend(
    user_history=[10, 25, 47],
    task="rating_prediction",
    item_id=100
)
print(f"Predicted rating for item 100: {rating:.2f}")

# Session-based
session_recs = recommender.recommend(
    task="session",
    session_items=[10, 25],
    k=5
)
print(f"Session recommendations: {session_recs}")
```

---

## Summary

**Key Takeaways**:
1. **Pre-training**: Masked item prediction, contrastive learning
2. **LLM prompting**: Use GPT-4 for recommendations via text prompts
3. **Zero-shot**: Recommend in new domains without training
4. **Few-shot**: Adapt with minimal domain data (100 users)
5. **Unified API**: Single interface for all recommendation tasks

**Vision**: One foundation model for all recommendations (like GPT for text).

**Current state**: Still research, not yet production-ready at scale.

**Best practices**:
- Pre-train on diverse, large-scale data
- Use prompting for quick prototyping
- Fine-tune for production performance

---

## Practice Problems

**Problem 1**: Pre-train foundation model on MovieLens + Amazon + Spotify. Evaluate zero-shot on books domain.

**Problem 2**: Compare LLM prompting (GPT-4) vs. specialized model on movie recommendations. Which is more accurate?

**Problem 3**: Design few-shot adaptation strategy. How many examples needed to match fully-trained model performance?

**Problem 4**: Implement unified API for 5 different recommendation tasks. Measure latency and accuracy.

---

## References

1. **Brown, T., et al. (2020)**. "Language Models are Few-Shot Learners". *NeurIPS* (GPT-3).

2. **Devlin, J., et al. (2019)**. "BERT: Pre-training of Deep Bidirectional Transformers". *NAACL*.

3. **Radford, A., et al. (2021)**. "Learning Transferable Visual Models From Natural Language Supervision". *ICML* (CLIP).

4. **Hou, Y., et al. (2022)**. "Towards Universal Sequence Representation Learning for Recommender Systems". *KDD*.

5. **Liu, J., et al. (2023)**. "Is ChatGPT a Good Recommender? A Preliminary Study". *arXiv*.
