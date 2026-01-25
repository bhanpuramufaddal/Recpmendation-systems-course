# Week 9: Large Language Models for Recommendations

## Overview

**Large Language Models (LLMs)** like GPT-4, Claude, and LLaMA have transformed natural language understanding. Their application to recommendations enables:
- **Zero-shot recommendations**: Recommend without training data
- **Semantic understanding**: Rich item/user descriptions
- **Conversational recommendations**: Natural dialogue interactions
- **Explainability**: Generate natural language explanations

**Key paradigms**:
1. **LLMs as feature extractors**: Encode text (titles, reviews) into embeddings
2. **LLMs as rankers**: Directly rank items via prompting
3. **LLMs as generators**: Generate personalized content/explanations
4. **Fine-tuned LLMs**: Adapt LLMs to recommendation tasks

This document covers how to leverage LLMs for recommendation systems.

---

## Learning Objectives

By the end of this section, you will:
- Use LLMs as feature extractors for items/users
- Implement prompt engineering for recommendations
- Apply zero-shot and few-shot recommendation
- Fine-tune LLMs on recommendation data
- Build conversational recommendation systems

---

## LLMs as Feature Extractors

### Motivation

**Traditional features**: Manual feature engineering (TF-IDF, one-hot, metadata).

**LLM embeddings**: Automatic semantic feature extraction from text.

**Example** (Movie recommendations):
```
Title: "The Shawshank Redemption"
Traditional: [genre=drama, year=1994, director=Frank_Darabont]
LLM embedding: 768-dim vector capturing:
  - Themes: hope, friendship, redemption
  - Tone: inspirational, dramatic
  - Plot: prison escape, wrongful conviction
```

---

### Using Pre-trained LLM Embeddings

**Popular models**:
- **Sentence-BERT**: 384-1024 dim, optimized for sentence similarity
- **OpenAI text-embedding-ada-002**: 1536 dim, high quality
- **Instructor**: Task-specific instructions for embeddings

**Example: Sentence-BERT**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings

# Movie descriptions
movies = [
    "A wrongfully convicted banker befriends a fellow prisoner while maintaining his innocence.",  # Shawshank
    "A computer hacker learns the truth about his reality and his role in the war against its controllers.",  # Matrix
    "Two detectives hunt a serial killer who uses the seven deadly sins as his motives.",  # Se7en
]

# Encode to embeddings
embeddings = model.encode(movies)
print(f"Embeddings shape: {embeddings.shape}")  # (3, 384)

# Compute similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings)

print("Similarities:")
print(similarities)
```

---

### Integrating LLM Embeddings into RecSys

**Scenario**: E-commerce with product descriptions.

**Pipeline**:
1. **Encode products**: Use LLM to embed titles/descriptions
2. **User representation**: Average embeddings of user's past products
3. **Recommendation**: Nearest neighbors in embedding space

```python
class LLMBasedRecommender:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name)
        self.item_embeddings = None
        self.item_ids = None

    def fit(self, item_texts, item_ids):
        """
        Encode all items.

        item_texts: List of text descriptions
        item_ids: List of item IDs
        """
        self.item_embeddings = self.encoder.encode(item_texts, show_progress_bar=True)
        self.item_ids = np.array(item_ids)

    def get_user_embedding(self, user_history):
        """
        Compute user embedding from interaction history.

        user_history: List of item IDs user interacted with
        """
        # Get embeddings of interacted items
        indices = [np.where(self.item_ids == item_id)[0][0] for item_id in user_history]
        user_items_emb = self.item_embeddings[indices]

        # Average (could also use weighted average by recency)
        user_emb = user_items_emb.mean(axis=0)
        return user_emb

    def recommend(self, user_history, top_k=10, exclude_seen=True):
        """
        Recommend items for user.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # User embedding
        user_emb = self.get_user_embedding(user_history)

        # Similarities to all items
        similarities = cosine_similarity([user_emb], self.item_embeddings)[0]

        # Exclude seen items
        if exclude_seen:
            for item_id in user_history:
                idx = np.where(self.item_ids == item_id)[0][0]
                similarities[idx] = -np.inf

        # Top-K
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        recommendations = [(self.item_ids[idx], similarities[idx]) for idx in top_indices]

        return recommendations


# Example usage
item_texts = [
    "High-quality noise-cancelling headphones with 30-hour battery life",  # Item 1
    "Wireless Bluetooth earbuds with charging case",  # Item 2
    "Over-ear gaming headset with surround sound",  # Item 3
    "Portable Bluetooth speaker with waterproof design",  # Item 4
    "USB-C charging cable 6ft braided",  # Item 5
]
item_ids = [101, 102, 103, 104, 105]

recommender = LLMBasedRecommender()
recommender.fit(item_texts, item_ids)

# User purchased headphones (item 101)
user_history = [101]
recommendations = recommender.recommend(user_history, top_k=3)

print("Recommendations:")
for item_id, score in recommendations:
    print(f"  Item {item_id}: {score:.3f}")
```

---

## Prompt Engineering for Recommendations

### Zero-Shot Recommendation

**Idea**: Use LLM's world knowledge to recommend without training.

**Prompt structure**:
```
You are a recommendation system. Given a user's preferences, recommend items.

User Profile:
- Likes: [past items or preferences]
- Dislikes: [optional]

Available Items:
1. [Item 1 description]
2. [Item 2 description]
...

Task: Rank the items from most to least relevant for this user. Provide your ranking as a numbered list.
```

**Example: Movie Recommendations**

```python
import openai

def zero_shot_recommend(user_profile, items, api_key):
    """
    Zero-shot recommendation using GPT.

    user_profile: Dict with 'likes', 'dislikes'
    items: List of (id, title, description)
    """
    # Build prompt
    prompt = "You are a movie recommendation system.\n\n"
    prompt += "User Profile:\n"
    prompt += f"Likes: {', '.join(user_profile['likes'])}\n"
    if 'dislikes' in user_profile:
        prompt += f"Dislikes: {', '.join(user_profile['dislikes'])}\n"

    prompt += "\nAvailable Movies:\n"
    for i, (item_id, title, desc) in enumerate(items, 1):
        prompt += f"{i}. {title}: {desc}\n"

    prompt += "\nTask: Rank these movies from most to least relevant for this user. "
    prompt += "Provide ONLY a comma-separated list of movie numbers (e.g., 3,1,4,2)."

    # Call OpenAI API
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful recommendation assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    # Parse response
    ranking_str = response.choices[0].message.content.strip()
    ranking = [int(x.strip()) for x in ranking_str.split(',')]

    # Map back to item IDs
    ranked_items = [items[i-1][0] for i in ranking]

    return ranked_items


# Example
user_profile = {
    'likes': ['The Shawshank Redemption', 'The Dark Knight', 'Inception'],
    'dislikes': ['Romantic comedies', 'Horror films']
}

items = [
    (1, "The Godfather", "Crime drama about an Italian-American mafia family"),
    (2, "Titanic", "Romance and disaster film set on the ill-fated ship"),
    (3, "The Prestige", "Thriller about rival magicians in Victorian London"),
    (4, "The Conjuring", "Horror film about paranormal investigators"),
]

# Note: Requires OpenAI API key
# ranked_items = zero_shot_recommend(user_profile, items, api_key="YOUR_KEY")
# print(f"Ranked items: {ranked_items}")
```

---

### Few-Shot Recommendation

**Idea**: Provide examples of user-item pairs → LLM learns pattern.

**Prompt structure**:
```
Here are examples of users and their preferred items:

User 1 likes: [item A, item B] → Recommended: [item C]
User 2 likes: [item D, item E] → Recommended: [item F]

Now, for User 3 who likes: [item G, item H] → Recommend: ?
```

**Example**:
```python
def few_shot_recommend(examples, query_user, candidate_items, api_key):
    """
    Few-shot recommendation.

    examples: List of (user_likes, recommended_item)
    query_user: Dict {'likes': [...]}
    candidate_items: List of items to rank
    """
    prompt = "Given a user's liked items, recommend the most relevant item.\n\n"

    # Add examples
    prompt += "Examples:\n"
    for i, (likes, recommended) in enumerate(examples, 1):
        prompt += f"{i}. User likes: {', '.join(likes)} → Recommended: {recommended}\n"

    # Query
    prompt += f"\nNow, user likes: {', '.join(query_user['likes'])}\n"
    prompt += f"Candidate items: {', '.join(candidate_items)}\n"
    prompt += "Recommended: "

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=50
    )

    recommendation = response.choices[0].message.content.strip()
    return recommendation


# Example
examples = [
    (['The Matrix', 'Inception'], 'Interstellar'),
    (['The Godfather', 'Goodfellas'], 'The Irishman'),
]

query_user = {'likes': ['The Dark Knight', 'Joker']}
candidates = ['The Batman', 'Shutter Island', 'The Prestige', 'Gone Girl']

# rec = few_shot_recommend(examples, query_user, candidates, api_key="YOUR_KEY")
```

---

### Chain-of-Thought Prompting

**Idea**: Ask LLM to explain reasoning → better recommendations.

**Prompt**:
```
Think step-by-step:
1. What are the common themes in the user's liked movies?
2. Which candidate movies share these themes?
3. Rank candidates by relevance.

Provide your reasoning and final ranking.
```

**Benefits**:
- **Explainability**: LLM explains why it recommended items
- **Accuracy**: Reasoning improves recommendation quality

---

## Fine-Tuning LLMs for Recommendations

### When to Fine-Tune?

**Zero-shot/Few-shot**: Good for general recommendations, limited data.

**Fine-tuning**: Better when:
- Domain-specific (e.g., medical products, niche hobbies)
- Large training data available
- Need consistent performance

---

### Task Formulation

**Ranking as text generation**:
```
Input: "User liked: Item A, Item B. Rank: Item X, Item Y, Item Z"
Output: "Item Y, Item X, Item Z"
```

**Prediction as text generation**:
```
Input: "User profile: [description]. Recommend top 3 items."
Output: "1. Item A, 2. Item C, 3. Item F"
```

---

### Fine-Tuning with LoRA

**Problem**: Full fine-tuning of LLMs (billions of parameters) is expensive.

**Solution**: **LoRA** (Low-Rank Adaptation) - update small adapter matrices.

**Idea**:
$$W = W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$, and $r \ll d$ (rank).

**Benefits**:
- **Parameter efficiency**: Only train $r(d+k)$ params instead of $dk$
- **Fast**: Adapters are small → quick training
- **Modular**: Swap adapters for different tasks

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Load base model
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Which layers to adapt
)

# Add LoRA adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4M / 6.7B (0.06%)

# Training data format
def format_recommendation_prompt(user_history, target_item):
    """
    Create training example.
    """
    prompt = f"User previously liked: {', '.join(user_history)}\nRecommended item: "
    completion = target_item
    return prompt + completion

# Example training samples
train_data = [
    {
        'user_history': ['The Matrix', 'Inception'],
        'target_item': 'Interstellar'
    },
    {
        'user_history': ['The Godfather', 'Scarface'],
        'target_item': 'Goodfellas'
    },
]

# Prepare dataset
formatted_data = [format_recommendation_prompt(ex['user_history'], ex['target_item'])
                  for ex in train_data]

# Tokenize
encoded = tokenizer(formatted_data, padding=True, truncation=True, return_tensors='pt')

# Fine-tuning loop (simplified)
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./lora_recommendation',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=1e-4,
    logging_steps=10,
)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=encoded,
# )
# trainer.train()
```

---

## LLMs for Conversational Recommendations

### Multi-Turn Dialogue

**Scenario**: User refines preferences through conversation.

**Example dialogue**:
```
User: I want a movie recommendation.
System: What genres do you enjoy?
User: I like sci-fi and thrillers.
System: How about "Blade Runner 2049"? It's a sci-fi thriller about...
User: Too slow. Something more action-packed.
System: Try "Edge of Tomorrow"! It has sci-fi elements with intense action sequences.
User: Perfect!
```

---

### Implementing Conversational RecSys

```python
class ConversationalRecommender:
    def __init__(self, items_catalog, api_key):
        """
        items_catalog: List of (id, title, description, genres)
        """
        self.catalog = items_catalog
        self.conversation_history = []
        openai.api_key = api_key

    def get_response(self, user_message):
        """
        Generate system response and recommendation.
        """
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        # System prompt
        system_prompt = (
            "You are a helpful movie recommendation assistant. "
            "Ask clarifying questions to understand user preferences, "
            "then recommend movies from the catalog. Be conversational and engaging."
        )

        # Add catalog context
        catalog_str = "\n".join([
            f"- {title} ({', '.join(genres)}): {desc}"
            for _, title, desc, genres in self.catalog
        ])
        system_prompt += f"\n\nAvailable movies:\n{catalog_str}"

        # Call GPT
        messages = [{"role": "system", "content": system_prompt}] + self.conversation_history

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )

        assistant_message = response.choices[0].message.content

        # Add to history
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message


# Example usage
catalog = [
    (1, "Blade Runner 2049", "Sci-fi thriller about synthetic humans", ["sci-fi", "thriller"]),
    (2, "Edge of Tomorrow", "Action sci-fi with time loops", ["sci-fi", "action"]),
    (3, "The Godfather", "Crime drama about mafia family", ["crime", "drama"]),
]

# recommender = ConversationalRecommender(catalog, api_key="YOUR_KEY")

# Simulated conversation
# response1 = recommender.get_response("I want a movie recommendation")
# print(f"System: {response1}")

# response2 = recommender.get_response("I like sci-fi and thrillers")
# print(f"System: {response2}")
```

---

### Critique-Based Refinement

**Idea**: User critiques recommendations → system refines.

**Example**:
```
System: I recommend "Inception" (sci-fi thriller)
User: Too complex, I want something simpler.
System: How about "Edge of Tomorrow"? It's more straightforward with action.
```

**Implementation**:
```python
def refine_recommendation(original_rec, critique, candidate_items, api_key):
    """
    Refine recommendation based on user critique.
    """
    prompt = f"Original recommendation: {original_rec}\n"
    prompt += f"User feedback: {critique}\n\n"
    prompt += f"Candidate items: {', '.join(candidate_items)}\n"
    prompt += "Based on the feedback, which item should I recommend instead?"

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    refined_rec = response.choices[0].message.content.strip()
    return refined_rec
```

---

## Explainability with LLMs

### Generating Explanations

**Why explain?**
- **Trust**: Users trust recommendations they understand
- **Engagement**: Explanations increase click-through rates
- **Debugging**: Identify recommendation failures

**Explanation types**:
1. **Content-based**: "Because you liked X, which is similar to Y"
2. **Collaborative**: "Users like you also enjoyed Z"
3. **Hybrid**: "Based on your interest in genre A and ratings from similar users"

---

### Explanation Generation

```python
def generate_explanation(user_history, recommended_item, item_details, api_key):
    """
    Generate natural language explanation for recommendation.

    user_history: List of items user liked
    recommended_item: Item being recommended
    item_details: Dict with item metadata
    """
    prompt = f"User's viewing history: {', '.join(user_history)}\n\n"
    prompt += f"Recommended item: {recommended_item}\n"
    prompt += f"Item details: {item_details}\n\n"
    prompt += "Explain in 1-2 sentences why this item is recommended for this user."

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=100
    )

    explanation = response.choices[0].message.content.strip()
    return explanation


# Example
user_history = ["The Matrix", "Inception", "Interstellar"]
recommended = "Blade Runner 2049"
details = {
    "genre": "sci-fi thriller",
    "themes": "artificial intelligence, dystopia, identity",
    "director": "Denis Villeneuve"
}

# explanation = generate_explanation(user_history, recommended, details, api_key="YOUR_KEY")
# print(f"Explanation: {explanation}")
# Output: "Blade Runner 2049 is recommended because you enjoy thought-provoking
#          sci-fi films like The Matrix and Inception that explore themes of
#          reality and artificial intelligence."
```

---

## Challenges and Limitations

### 1. Cost

**Problem**: API calls expensive ($0.01-0.10 per 1K tokens).

**Example**: 1M recommendations × 500 tokens = $500-5000!

**Solutions**:
- Use LLMs for **cold start** or **high-value** users only
- Cache frequent queries
- Use smaller models (e.g., Llama-7B instead of GPT-4)

---

### 2. Latency

**Problem**: LLM inference slow (100ms-2s per request).

**Solutions**:
- **Async processing**: Generate recommendations in background
- **Pre-compute**: Use LLM to generate explanations offline
- **Hybrid**: Use fast model (collaborative filtering) + LLM for explanations

---

### 3. Hallucination

**Problem**: LLMs may generate non-existent items or false claims.

**Example**:
```
User: Recommend sci-fi movies
LLM: "Try 'The Quantum Paradox' (2023), a mind-bending film about..." [doesn't exist!]
```

**Solutions**:
- **Constrain output**: Force selection from catalog
- **Verification**: Check LLM output against database
- **Prompt engineering**: "Only recommend items from this list: ..."

---

### 4. Bias

**Problem**: LLMs inherit biases from training data.

**Example**: May favor popular/Western items, stereotypes.

**Solutions**:
- **Diverse prompts**: Include diversity in instructions
- **Bias detection**: Monitor demographic fairness
- **Human-in-the-loop**: Review LLM recommendations

---

## Summary

**Key Takeaways**:
1. **LLM embeddings**: Rich semantic features from text (Sentence-BERT, OpenAI)
2. **Prompt engineering**: Zero-shot, few-shot, chain-of-thought
3. **Fine-tuning**: LoRA for parameter-efficient adaptation
4. **Conversational**: Multi-turn dialogue, critique-based refinement
5. **Explainability**: Natural language explanations

**Best Practices**:
- **Hybrid approach**: LLMs + traditional RecSys (speed + accuracy)
- **Cost management**: Cache, batch, use smaller models
- **Validation**: Verify LLM outputs against catalog
- **Explainability**: Always provide reasoning

**When to use**:
- **Cold start**: New users/items with rich text
- **Conversational**: Chatbot interfaces
- **Explainability**: High-stakes recommendations (medical, finance)
- **Niche domains**: Where world knowledge helps (books, travel)

**Next**: Multi-modal recommendations with vision-language models.

---

## References

1. **Geng, S., et al. (2022)**. "Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm". *RecSys*.
   - **P5 model**: Unify RecSys tasks as text generation

2. **Hou, Y., et al. (2023)**. "Large Language Models are Zero-Shot Rankers for Recommender Systems". *arXiv*.
   - **Zero-shot ranking** with LLMs

3. **Bao, K., et al. (2023)**. "TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Models with Recommendation". *RecSys*.
   - **LoRA fine-tuning** for RecSys

4. **Liu, J., et al. (2023)**. "Is ChatGPT a Good Recommender? A Preliminary Study". *arXiv*.
   - **ChatGPT evaluation** on RecSys tasks

5. **Zhang, Y., et al. (2023)**. "Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach". *arXiv*.
   - **Instruction-following** paradigm

---

## Practice Problems

### Problem 1: Prompt Engineering

**Design a prompt** for zero-shot movie recommendation given:
- User likes: "The Matrix", "Inception"
- Candidates: "Interstellar", "The Notebook", "The Conjuring"

**Solution**:
```python
prompt = """You are a movie recommendation expert.

User Profile:
- Previously liked: The Matrix, Inception

Available Movies:
1. Interstellar - Epic space exploration and time dilation
2. The Notebook - Romantic drama spanning decades
3. The Conjuring - Supernatural horror investigation

Task: Rank these 3 movies from most to least suitable for this user.
Provide your ranking as: [number], [number], [number]
Also briefly explain your reasoning.
"""

# Expected output: "1, 3, 2" or "Interstellar, The Conjuring, The Notebook"
# Reasoning: User likes complex sci-fi (Matrix, Inception) → Interstellar fits best
```

---

### Problem 2: Embedding Similarity

**Given LLM embeddings** (simplified to 3-dim):
- Item A: [0.8, 0.1, 0.3]
- Item B: [0.7, 0.2, 0.2]
- Item C: [0.2, 0.9, 0.1]

**User liked** Item A. **Compute**: Most similar item.

**Solution**:
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

A = np.array([[0.8, 0.1, 0.3]])
B = np.array([[0.7, 0.2, 0.2]])
C = np.array([[0.2, 0.9, 0.1]])

sim_AB = cosine_similarity(A, B)[0, 0]
sim_AC = cosine_similarity(A, C)[0, 0]

print(f"Similarity A-B: {sim_AB:.3f}")
print(f"Similarity A-C: {sim_AC:.3f}")

# Output:
# Similarity A-B: 0.988
# Similarity A-C: 0.333
# Most similar: Item B
```

---

### Problem 3: LoRA Parameters

**Given**:
- Original weight matrix: $W \in \mathbb{R}^{4096 \times 4096}$
- LoRA rank: $r = 8$

**Compute**: Number of trainable parameters with LoRA vs. full fine-tuning.

**Solution**:
```python
d = 4096
r = 8

# Full fine-tuning
full_params = d * d
print(f"Full fine-tuning: {full_params:,} parameters")  # 16,777,216

# LoRA (B: d×r, A: r×d)
lora_params = d * r + r * d
print(f"LoRA: {lora_params:,} parameters")  # 65,536

# Reduction
reduction = (1 - lora_params / full_params) * 100
print(f"Parameter reduction: {reduction:.2f}%")  # 99.61%
```
