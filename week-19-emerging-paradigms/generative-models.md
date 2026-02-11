# Week 19: Generative Recommendations

## Opening Problem: The Imagination Gap

*"Class, today we're going to ask a question that sounds almost philosophical..."*

**What if we could GENERATE the perfect item instead of selecting from existing inventory?**

Think about it. Every recommendation system we've built so far has the same fundamental limitation. We have a catalog of items, we have user preferences, and we find the best match. But what if the perfect item for you... doesn't exist yet?

*"I want a romantic comedy with the pacing of Tarantino, set in 1920s Tokyo, starring someone with Emma Stone's comedic timing."*

No traditional recommender can help here. But a generative system could:
1. Generate a description of this "ideal movie"
2. Match it to the closest existing films
3. Or even... create the screenplay outline itself

**This is the vision**: Recommendations that CREATE, not just SELECT.

> **Socratic Pause**: "Before we dive in, let me ask you something. When Netflix recommends 'Because you watched X,' what's actually being computed? Is it finding what you want, or finding what THEY have that's closest to what you might want? There's a difference. Keep that distinction in mind."

---

## The Fundamental Distinction: Generative vs Discriminative

*"Let's derive the mathematical difference because this isn't just terminology - it changes everything about how we approach the problem."*

### Discriminative Approach (What We've Done)

$$P(\text{item} | \text{user})$$

This says: "Given this user, what's the probability of each item being relevant?"

We model the **conditional distribution directly**. Matrix factorization, neural collaborative filtering, content-based filtering - all discriminative.

**What discriminative models learn**:
- Decision boundaries
- "User X likes items with these features"
- Direct mapping from input to output

### Generative Approach (What We're Learning Today)

$$P(\text{user}, \text{item}) = P(\text{user}) \cdot P(\text{item} | \text{user})$$

Or equivalently:

$$P(\text{user}, \text{item}) = P(\text{item}) \cdot P(\text{user} | \text{item})$$

**This changes everything**. Now we're modeling the **joint distribution** - how users and items relate in the full probability space.

**What generative models learn**:
- The underlying data distribution
- "What does a typical user-item interaction look like?"
- Can generate NEW samples from this distribution

### The Derivation That Matters

Let's formalize this for collaborative filtering:

$$\text{Discriminative: } \hat{y}_{ui} = f(u, i; \theta) \rightarrow \text{Score for user } u, \text{ item } i$$

$$\text{Generative: } P(x_u | z_u) \text{ where } z_u \sim P(z) \rightarrow \text{Generate user's preferences from latent code}$$

The generative view says: *there exists some compact representation $z$ of user preferences, and we can decode it into ANY possible preference pattern*.

> **Critical Insight**: "Discriminative models ask 'Which item?' Generative models ask 'What kind of item?' The first is selection. The second is creation."

---

## Variational Autoencoders for CF: The ELBO Derivation

*"Now let's derive VAE-CF properly. This is one of those derivations where every step matters."*

### The Setup

We observe user interaction vectors $x_u \in \{0,1\}^{|I|}$ (binary: interacted or not).

We assume there's a latent code $z_u \in \mathbb{R}^k$ that captures user preferences.

**Generative story**:
1. Sample latent code: $z_u \sim P(z) = \mathcal{N}(0, I)$
2. Decode to preferences: $x_u \sim P(x|z_u)$

### The Problem: Intractable Posterior

We want to learn $P(x|z)$ but we need $P(z|x)$ (the posterior) for inference.

**Bayes' theorem**:
$$P(z|x) = \frac{P(x|z) P(z)}{P(x)}$$

But $P(x) = \int P(x|z) P(z) dz$ is **intractable** - it's an integral over all possible latent codes!

### The Solution: Variational Inference

Instead of computing $P(z|x)$ exactly, we **approximate** it with $q_\phi(z|x)$.

**Step 1**: Write the log evidence

$$\log P(x) = \log \int P(x|z) P(z) dz$$

**Step 2**: Introduce the variational distribution

$$\log P(x) = \log \int P(x|z) P(z) \frac{q_\phi(z|x)}{q_\phi(z|x)} dz$$

**Step 3**: Apply Jensen's inequality (concavity of log)

$$\log P(x) \geq \int q_\phi(z|x) \log \frac{P(x|z) P(z)}{q_\phi(z|x)} dz$$

**Step 4**: Rearrange to get the ELBO

$$\log P(x) \geq \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log P_\theta(x|z)]}_{\text{Reconstruction}} - \underbrace{D_{KL}(q_\phi(z|x) \| P(z))}_{\text{Regularization}}$$

This is the **Evidence Lower BOund (ELBO)**!

### Interpreting Each Term

**Reconstruction term**: $\mathbb{E}_{q_\phi(z|x)}[\log P_\theta(x|z)]$
- "How well can we reconstruct interactions from the latent code?"
- For collaborative filtering: "Can we predict which items the user liked?"

**KL term**: $D_{KL}(q_\phi(z|x) \| P(z))$
- "How close is our learned posterior to the prior?"
- Regularization - prevents latent space from collapsing

### The Final VAE-CF Loss

$$\mathcal{L}(\theta, \phi; x) = -\mathbb{E}_{q_\phi(z|x)}[\log P_\theta(x|z)] + \beta \cdot D_{KL}(q_\phi(z|x) \| P(z))$$

Where $\beta$ controls the trade-off (often set to 0.2-0.5 for recommendations).

> **Professor's Note**: "The $\beta$ parameter is crucial. If $\beta$ is too high, latent codes become uninformative (posterior collapse). Too low, and you lose the regularization benefit. This is the 'posterior collapse' problem - something to always watch for."

---

## VAE-CF Architecture and Implementation

### Architecture

1. **Encoder** $q_\phi(z|x)$: User interaction history $\rightarrow$ latent distribution parameters $(\mu, \sigma^2)$
2. **Decoder** $P_\theta(x|z)$: Latent code $z$ $\rightarrow$ predicted preferences

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_CF(nn.Module):
    def __init__(self, n_items, latent_dim=64):
        super().__init__()

        # Encoder
        self.encoder_fc1 = nn.Linear(n_items, 600)
        self.encoder_fc2_mean = nn.Linear(600, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(600, latent_dim)

        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, 600)
        self.decoder_fc2 = nn.Linear(600, n_items)

    def encode(self, x):
        """
        Encode user interaction history to latent distribution.

        Args:
            x: [batch_size, n_items] binary interaction matrix

        Returns:
            mean, logvar: Parameters of latent distribution
        """
        h = torch.tanh(self.encoder_fc1(x))
        mean = self.encoder_fc2_mean(h)
        logvar = self.encoder_fc2_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """Reparameterization trick for backprop through sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        """
        Decode latent code to item preferences.

        Args:
            z: [batch_size, latent_dim]

        Returns:
            logits: [batch_size, n_items]
        """
        h = torch.tanh(self.decoder_fc1(z))
        logits = self.decoder_fc2(h)
        return logits

    def forward(self, x):
        """
        Forward pass through VAE.

        Args:
            x: [batch_size, n_items] interaction matrix

        Returns:
            recon_logits: Reconstructed interaction logits
            mean, logvar: Latent distribution parameters
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_logits = self.decode(z)
        return recon_logits, mean, logvar


def vae_loss(recon_logits, x, mean, logvar, beta=0.2):
    """
    VAE loss function.

    Args:
        recon_logits: Reconstructed logits
        x: Original interaction matrix
        mean, logvar: Latent distribution parameters
        beta: KL divergence weight

    Returns:
        loss: Total loss
    """
    # Reconstruction loss (binary cross-entropy)
    recon_loss = -torch.sum(
        F.log_softmax(recon_logits, dim=1) * x,
        dim=1
    ).mean()

    # KL divergence
    kl_div = -0.5 * torch.sum(
        1 + logvar - mean.pow(2) - logvar.exp(),
        dim=1
    ).mean()

    # Total loss
    loss = recon_loss + beta * kl_div

    return loss, recon_loss, kl_div


# Training
model = VAE_CF(n_items=10000, latent_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for user_interactions in train_loader:
        # user_interactions: [batch_size, n_items] binary matrix

        # Forward pass
        recon_logits, mean, logvar = model(user_interactions)

        # Compute loss
        loss, recon_loss, kl_div = vae_loss(recon_logits, user_interactions, mean, logvar)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Recon = {recon_loss:.4f}, KL = {kl_div:.4f}")


# Recommendation
def recommend(model, user_interaction, k=10):
    """
    Generate recommendations for user.

    Args:
        user_interaction: [n_items] binary vector
        k: Number of recommendations

    Returns:
        top_k_items: Recommended item indices
    """
    model.eval()

    with torch.no_grad():
        # Encode user
        mean, logvar = model.encode(user_interaction.unsqueeze(0))

        # Sample latent code
        z = model.reparameterize(mean, logvar)

        # Decode to item scores
        logits = model.decode(z).squeeze()

        # Mask already-interacted items
        logits[user_interaction == 1] = -float('inf')

        # Top-K
        top_k_items = torch.topk(logits, k).indices

    return top_k_items
```

---

## LLM-Based Recommendations: The New Frontier

*"Now we enter truly new territory. What if instead of learned embeddings, we use language models that can REASON about preferences?"*

### How GPT/Claude Can Reason About Preferences

LLMs bring something fundamentally different:
1. **World knowledge**: They know about items, genres, cultural context
2. **Reasoning**: They can explain WHY something might appeal
3. **Natural interaction**: Users can describe preferences in natural language
4. **Zero-shot generalization**: No training on your specific catalog needed

### Concrete Prompting Examples

**Example 1: Preference Understanding**

```
System: You are a movie recommendation expert. Analyze user preferences and
suggest movies they would enjoy.

User: I loved Inception for its mind-bending complexity, The Grand Budapest
Hotel for its visual style, and Parasite for its social commentary.
What should I watch next?

Assistant (analyzing preferences): Based on your choices, you appreciate:
- **Intellectual complexity**: Non-linear narratives, layered meanings
- **Strong visual identity**: Distinctive cinematography and art direction
- **Social depth**: Films that comment on class, society, culture

Recommendations:
1. "Snowpiercer" (2013) - Bong Joon-ho's class allegory with striking visuals
2. "The Lobster" (2015) - Yorgos Lanthimos's absurdist social critique
3. "Memento" (2000) - Nolan's earlier mind-bender
4. "The Handmaiden" (2016) - Park Chan-wook's gorgeous thriller with twists
```

**Example 2: Constraint-Based Generation**

```
System: Generate a product description for an item that matches these constraints.

User: I need running shoes for:
- Marathon training (high mileage)
- Flat feet (need stability)
- Budget under $150
- Prefer bright colors

Assistant (generating ideal item profile):
{
  "category": "running_shoes",
  "features": {
    "cushioning": "high",
    "stability": "motion_control",
    "drop": "8-10mm",
    "weight": "medium",
    "durability": "high_mileage",
    "price_range": [100, 150],
    "colors": ["neon_yellow", "electric_blue", "hot_pink"]
  },
  "use_case": "marathon_training",
  "arch_support": "flat_feet_friendly"
}

Matching catalog items:
1. Brooks Adrenaline GTS 23 - $139.95 (92% match)
2. ASICS GT-2000 11 - $129.95 (89% match)
3. New Balance 860v13 - $139.99 (87% match)
```

> **Socratic Question**: "Notice how the LLM generated a structured feature profile FIRST, then matched to catalog. What's the advantage of this two-step approach over direct recommendation?"

---

## Numerical Example: The Generative Process

*"Let's trace through exactly what happens when we generate item features and match to a catalog. This is where theory meets reality."*

### Scenario: Book Recommendation via Generation

**User profile** (from VAE encoder):
```
Latent code z = [0.8, -0.3, 0.5, 0.1, -0.7, 0.2, 0.4, -0.1]
```

**Step 1: Generate "ideal book" features via decoder**

The decoder transforms z into feature space:

```python
# Decoder weights (simplified)
W_decode = learned_weights  # [latent_dim x feature_dim]

# Generated features
ideal_features = sigmoid(W_decode @ z + bias)
```

**Generated ideal book profile**:
```
{
  "genre_fiction": 0.85,
  "genre_scifi": 0.72,
  "genre_fantasy": 0.31,
  "genre_romance": 0.08,
  "complexity_high": 0.78,
  "length_long": 0.65,
  "pace_fast": 0.82,
  "themes_philosophical": 0.71,
  "themes_adventure": 0.54,
  "writing_literary": 0.62
}
```

**Step 2: Match to catalog**

Catalog items with their features:

| Book | Fiction | SciFi | Complexity | Pace | Philosophical |
|------|---------|-------|------------|------|---------------|
| Dune | 0.90 | 0.88 | 0.82 | 0.70 | 0.85 |
| Foundation | 0.85 | 0.92 | 0.75 | 0.65 | 0.80 |
| Ender's Game | 0.88 | 0.85 | 0.55 | 0.90 | 0.45 |
| The Martian | 0.80 | 0.78 | 0.40 | 0.95 | 0.20 |

**Step 3: Compute similarity scores**

Using cosine similarity between generated profile and catalog items:

$$\text{similarity}(generated, item) = \frac{generated \cdot item}{||generated|| \cdot ||item||}$$

```
Dune:        similarity = 0.94
Foundation:  similarity = 0.91
Ender's Game: similarity = 0.76
The Martian: similarity = 0.68
```

**Final recommendation**: "Dune" - matches the generated "ideal book" profile best.

> **Key Insight**: "We didn't just find similar users or items. We GENERATED what the ideal item SHOULD look like, then found the closest real item. The generated profile is richer than any single item in the catalog."

---

## Conversational Recommendations: Multi-Turn Dialog

*"This is where LLM-based recommendations truly shine. Let's trace through a complete preference elicitation conversation."*

### Traced Conversation: Finding a Restaurant

```
TURN 1:
User: I'm looking for a place for dinner tonight.

System [Internal state]:
- Intent: restaurant_recommendation
- Constraints: {}
- Confidence: LOW (need more info)

System: I'd be happy to help! To find the perfect spot, I have a few questions.
What neighborhood are you thinking, and roughly what's your budget per person?
```

```
TURN 2:
User: Downtown, and maybe $30-50 per person? It's a date night.

System [Internal state]:
- Intent: restaurant_recommendation
- Constraints: {
    location: "downtown",
    budget: [30, 50],
    occasion: "date_night"
  }
- Inferred: romantic_atmosphere = HIGH, noise_level = LOW
- Confidence: MEDIUM

System: Great - date night downtown in the $30-50 range! That opens up nice options.
Any cuisine preferences, or things you definitely want to avoid?
```

```
TURN 3:
User: She's vegetarian, and I'd prefer something not too noisy.

System [Internal state]:
- Constraints updated: {
    location: "downtown",
    budget: [30, 50],
    occasion: "date_night",
    dietary: ["vegetarian_options"],
    atmosphere: "quiet",
    party_size: 2
  }
- Confidence: HIGH

System: Perfect. Looking for a quiet, romantic downtown spot with strong vegetarian
options in the $30-50 range. Here are my top picks:

1. **Verde Kitchen** - Mediterranean with creative vegetarian tasting menu.
   Intimate 30-seat space, candlelit. $42/person average.
   "Their mushroom shawarma is legendary" - great for impressing a date.

2. **Harvest Table** - Farm-to-table with 60% vegetarian menu.
   Exposed brick, quiet corners. $38/person average.
   Bonus: They have a great wine list.

3. **The Quiet Fork** - New American, fully vegetarian.
   Hushed atmosphere, beautiful plating. $45/person average.
   Perfect if you want to go all-in on vegetarian.

Would you like more details on any of these, or should I factor in anything else?
```

```
TURN 4:
User: Verde sounds good but she doesn't like mushrooms. Any other standout dishes?

System [Internal state]:
- Constraints updated: {
    ...previous...,
    avoid: ["mushrooms"]
  }
- Current focus: Verde Kitchen
- Mode: detail_exploration

System: Good catch! At Verde, besides the mushroom dish, their standouts are:

- **Cauliflower Steak** ($24) - harissa-spiced, with tahini and pomegranate
- **Stuffed Grape Leaves** ($14 appetizer) - the rice filling is incredible
- **Eggplant Moussaka** ($26) - their vegetarian take on the Greek classic

These are all vegetarian and mushroom-free. The cauliflower steak is actually
what most food critics highlight. Would you like me to check availability
for tonight?
```

### What Makes This Work

**Preference elicitation strategies**:
1. **Start broad, narrow progressively** - Don't ask 10 questions upfront
2. **Infer unstated preferences** - "Date night" implies quiet, romantic
3. **Remember context across turns** - Vegetarian + no mushrooms
4. **Provide explanations** - WHY each option fits
5. **Offer next actions** - Reservation help, more details

> **Socratic Reflection**: "Compare this to a traditional recommender. Could a matrix factorization model have this conversation? What's fundamentally different about maintaining dialog state?"

---

## What Can Go Wrong: Failure Modes of Generative Recommendations

*"Now let's talk about the dark side. These systems can fail in ways traditional recommenders cannot."*

### 1. Hallucination

**The Problem**: LLMs generate plausible-sounding but false information.

**Real Example**:
```
User: Recommend restaurants near Central Park.

LLM: I recommend "The Park View Bistro" at 845 5th Avenue. They have
an amazing brunch menu with views of the park...

[This restaurant doesn't exist]
```

**Why It Happens**:
- LLMs are trained to generate fluent text, not verify facts
- No grounding to actual inventory
- Confident tone masks uncertainty

**Mitigation**:
```python
def grounded_recommendation(llm_output, catalog):
    """
    Verify LLM recommendations against actual catalog.
    """
    recommendations = []
    for item in llm_output.items:
        # Fuzzy match against catalog
        matches = catalog.search(item.name, threshold=0.85)
        if matches:
            recommendations.append(matches[0])
        else:
            # Flag hallucination
            log_hallucination(item)
    return recommendations
```

### 2. Inventory Mismatch

**The Problem**: Generated "ideal items" may not exist or be available.

**Scenario**:
```
Generated ideal: "A 15-inch laptop with RTX 4090, 64GB RAM, under 3 lbs,
                 under $1500"

Reality: This combination doesn't exist at this price point.
```

**The Trade-off Triangle**:
```
      Performance
         /\
        /  \
       /    \
      /      \
   Weight --- Price

(Can't optimize all three simultaneously)
```

**Mitigation**: Constraint relaxation with explanation
```
System: The exact specs you want aren't available under $1500. Here are options:

Option A: Meet performance + weight, increase budget to $2200
Option B: Meet performance + price, accept 5.5 lbs weight
Option C: Meet weight + price, reduce to RTX 4070

Which trade-off works best for you?
```

### 3. Latency

**The Problem**: LLM inference is slow; users expect instant recommendations.

**Comparison**:
| Method | Latency | Throughput |
|--------|---------|------------|
| Matrix Factorization | 5ms | 10,000 QPS |
| Two-Tower Neural | 20ms | 2,500 QPS |
| LLM (GPT-4 class) | 2-5s | 10 QPS |

**Mitigation Strategies**:
1. **Hybrid approach**: Use fast model for initial results, LLM for refinement
2. **Caching**: Pre-compute LLM responses for common queries
3. **Streaming**: Show partial results as LLM generates
4. **Smaller models**: Use distilled models for real-time (GPT-3.5 vs GPT-4)

### 4. Explanation Quality (Confabulation)

**The Problem**: LLMs generate explanations that sound good but aren't the real reason.

**Example**:
```
System: I recommend "The Shawshank Redemption" because:
"The themes of hope and perseverance resonate with your interest in
character-driven dramas, as shown by your love of Breaking Bad."

[But the actual reason was: it's highly rated and broadly popular]
```

**Why It's Dangerous**:
- Users trust explanations to calibrate their own preferences
- False explanations erode trust when discovered
- May reinforce biases rather than explain them

**Mitigation**: Separate explanation generation from recommendation
```python
def honest_explanation(recommendation, user_profile, model_scores):
    """
    Generate explanation grounded in actual model behavior.
    """
    # Get actual contributing factors
    factors = get_attribution(recommendation, user_profile)

    prompt = f"""
    Explain why {recommendation} was recommended.

    ACTUAL CONTRIBUTING FACTORS (you must use these):
    - Collaborative signal: {factors['cf_score']}
    - Content match: {factors['content_features']}
    - Popularity: {factors['popularity_score']}

    Do NOT invent reasons not supported by these factors.
    """
    return llm.generate(prompt)
```

### 5. Bias Amplification

**The Problem**: Generative models can amplify existing biases in training data.

**Example**:
```
Training data bias: Male authors overrepresented in "serious literature"

User: Recommend literary fiction.

LLM: [Generates list of 9 male authors, 1 female author]
```

**Why Generative Models Are Particularly Risky**:
- They learn and reproduce cultural stereotypes
- Generated text feels authoritative
- Bias is harder to audit than in explicit features

**Mitigation**:
```python
def debiased_generation(prompt, protected_attributes):
    """
    Generate recommendations with bias monitoring.
    """
    # Generate initial recommendations
    recs = llm.generate(prompt)

    # Audit for bias
    bias_report = audit_recommendations(recs, protected_attributes)

    if bias_report.is_biased:
        # Re-generate with debiasing prompt
        debiased_prompt = f"""
        {prompt}

        IMPORTANT: Ensure diverse representation across:
        {protected_attributes}

        Current bias detected: {bias_report.summary}
        """
        recs = llm.generate(debiased_prompt)

    return recs, bias_report
```

> **Critical Question**: "If an LLM recommends items that perpetuate stereotypes, but those recommendations match the training data, is the system 'working correctly'? Who is responsible?"

---

## Socratic Deep Dive: Embeddings vs. Descriptions

*"Let me pose a question that gets at the heart of what we're trading off..."*

### What's Lost When We Generate Descriptions Instead of Learning Embeddings?

**Consider two approaches**:

**Approach A: Learned Embeddings (Traditional)**
```
User embedding: [0.3, -0.7, 0.2, 0.8, -0.1, ...]  (64 dimensions)
Item embedding: [0.4, -0.6, 0.3, 0.7, -0.2, ...]  (64 dimensions)

Score = dot_product(user, item)
```

**Approach B: Generated Descriptions (LLM)**
```
User profile: "Enjoys complex narratives, prefers character development
              over action, appreciates dark humor, avoids explicit violence"

Item description: "A slow-burn thriller with morally ambiguous characters,
                   witty dialogue, and psychological tension"

Score = LLM.score(user_profile, item_description)
```

### What Embeddings Capture That Descriptions Miss

1. **Implicit patterns**: "Users who buy diapers also buy beer" - no description captures this
2. **Nuanced taste gradients**: The difference between "likes sci-fi" and "likes HARD sci-fi"
3. **Combinatorial preferences**: How features interact (comedy + horror = different than comedy OR horror)
4. **Temporal dynamics**: How preferences evolve over time
5. **Privacy**: Embeddings don't reveal explicit preferences

### What Descriptions Capture That Embeddings Miss

1. **Explainability**: Can articulate WHY something is recommended
2. **Novel combinations**: Can describe items that don't exist yet
3. **Contextual reasoning**: "Good for a rainy Sunday" isn't a learnable feature
4. **Knowledge transfer**: Training on books helps recommend movies
5. **User agency**: Users can correct misunderstandings

### The Deeper Question

> "Embeddings are DISCOVERED from behavior. Descriptions are ASSERTED about meaning. Which do you trust more - what people DO or what they SAY?"

**Both have failure modes**:
- Embeddings fail on cold-start, can't explain, capture spurious correlations
- Descriptions fail on implicit preferences, can hallucinate, miss behavioral patterns

**The emerging consensus**: Hybrid systems that use embeddings for retrieval and LLMs for ranking/explanation may capture the best of both worlds.

---

## GANs for Data Augmentation

### Motivation

**Problem**: Limited user-item interactions (sparsity).

**Solution**: Generate synthetic interactions to augment training data.

**GAN approach**:
- **Generator**: Creates fake user interactions
- **Discriminator**: Distinguishes real vs. fake
- **Training**: Generator learns to fool discriminator

---

### Implementation

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=64, n_items=10000):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, n_items)

    def forward(self, z):
        """
        Generate fake user interactions.

        Args:
            z: [batch_size, latent_dim] random noise

        Returns:
            fake_interactions: [batch_size, n_items]
        """
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output in [0, 1]
        return x


class Discriminator(nn.Module):
    def __init__(self, n_items=10000):
        super().__init__()
        self.fc1 = nn.Linear(n_items, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Discriminate real vs. fake interactions.

        Args:
            x: [batch_size, n_items] interaction matrix

        Returns:
            logits: [batch_size] real/fake logits
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits.squeeze()


# Training
generator = Generator(latent_dim=64, n_items=10000)
discriminator = Discriminator(n_items=10000)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

criterion = nn.BCEWithLogitsLoss()

for epoch in range(100):
    for real_interactions in train_loader:
        batch_size = real_interactions.size(0)

        # Labels
        real_labels = torch.ones(batch_size)
        fake_labels = torch.zeros(batch_size)

        # ========== Train Discriminator ==========
        d_optimizer.zero_grad()

        # Real interactions
        d_real = discriminator(real_interactions)
        d_loss_real = criterion(d_real, real_labels)

        # Fake interactions
        z = torch.randn(batch_size, 64)
        fake_interactions = generator(z)
        d_fake = discriminator(fake_interactions.detach())
        d_loss_fake = criterion(d_fake, fake_labels)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # ========== Train Generator ==========
        g_optimizer.zero_grad()

        # Generate fake interactions
        z = torch.randn(batch_size, 64)
        fake_interactions = generator(z)

        # Fool discriminator
        d_fake = discriminator(fake_interactions)
        g_loss = criterion(d_fake, real_labels)  # Want discriminator to think they're real

        g_loss.backward()
        g_optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: D_loss = {d_loss:.4f}, G_loss = {g_loss:.4f}")


# Data augmentation
def augment_training_data(generator, n_synthetic=1000):
    """
    Generate synthetic user interactions.

    Args:
        n_synthetic: Number of synthetic users to generate

    Returns:
        synthetic_interactions: [n_synthetic, n_items]
    """
    generator.eval()

    with torch.no_grad():
        z = torch.randn(n_synthetic, 64)
        synthetic_interactions = generator(z)

        # Binarize (threshold at 0.5)
        synthetic_interactions = (synthetic_interactions > 0.5).float()

    return synthetic_interactions
```

---

## Diffusion Models for Recommendations

### Denoising Diffusion

**Idea**: Learn to denoise user preferences gradually.

**Process**:
1. **Forward**: Add noise to user interactions over T steps
2. **Reverse**: Train model to denoise (predict original from noisy)
3. **Generation**: Start from noise, iteratively denoise

**Application**: Generate personalized item sequences.

---

### Simplified Implementation

```python
class DiffusionRecommender(nn.Module):
    def __init__(self, n_items, hidden_dim=256, timesteps=100):
        super().__init__()
        self.n_items = n_items
        self.timesteps = timesteps

        # Noise schedule (variance at each timestep)
        self.betas = torch.linspace(0.0001, 0.02, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Denoising network
        self.denoiser = nn.Sequential(
            nn.Linear(n_items + 1, hidden_dim),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_items)
        )

    def add_noise(self, x, t):
        """
        Add noise to interactions at timestep t.

        Args:
            x: [batch_size, n_items] clean interactions
            t: [batch_size] timesteps

        Returns:
            x_t: Noisy interactions
            noise: Added noise
        """
        noise = torch.randn_like(x)

        # Get alpha for timestep
        alpha_t = self.alphas_cumprod[t].view(-1, 1)

        # Forward process: x_t = sqrt(alpha_t) * x + sqrt(1 - alpha_t) * noise
        x_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise

        return x_t, noise

    def denoise(self, x_t, t):
        """
        Predict noise at timestep t.

        Args:
            x_t: [batch_size, n_items] noisy interactions
            t: [batch_size] timesteps

        Returns:
            predicted_noise: [batch_size, n_items]
        """
        # Concatenate timestep as feature
        t_normalized = t.float() / self.timesteps
        input = torch.cat([x_t, t_normalized.unsqueeze(1)], dim=1)

        predicted_noise = self.denoiser(input)
        return predicted_noise

    def forward(self, x):
        """
        Training forward pass.

        Args:
            x: [batch_size, n_items] clean interactions

        Returns:
            loss: Denoising loss
        """
        batch_size = x.size(0)

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,))

        # Add noise
        x_t, noise = self.add_noise(x, t)

        # Predict noise
        predicted_noise = self.denoise(x_t, t)

        # Loss: MSE between predicted and actual noise
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def sample(self, batch_size=1):
        """
        Generate synthetic user interactions via reverse diffusion.

        Returns:
            generated: [batch_size, n_items]
        """
        # Start from pure noise
        x = torch.randn(batch_size, self.n_items)

        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            # Predict noise
            t_batch = torch.full((batch_size,), t, dtype=torch.long)
            predicted_noise = self.denoise(x, t_batch)

            # Remove predicted noise
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)

            x = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

            # Add noise for next step (except last step)
            if t > 0:
                noise = torch.randn_like(x)
                x = x * torch.sqrt(alpha_t_prev) + noise * torch.sqrt(1 - alpha_t_prev)

        # Binarize
        generated = (x > 0).float()

        return generated


# Training
model = DiffusionRecommender(n_items=10000, hidden_dim=256, timesteps=100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for user_interactions in train_loader:
        loss = model(user_interactions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Generate synthetic users
synthetic_users = model.sample(batch_size=100)
print(f"Generated {len(synthetic_users)} synthetic users")
```

---

## Counterfactual Generation

### Motivation

**Problem**: Can't observe all user-item interactions (missing data).

**Solution**: Generate counterfactual interactions ("What if user saw item X?").

**Use case**: Unbiased evaluation, debiasing models.

---

### Implementation

```python
def generate_counterfactuals(model, user, observed_items, candidate_items):
    """
    Generate counterfactual interactions.

    Args:
        model: Generative model (VAE or GAN)
        user: User ID
        observed_items: Items user actually interacted with
        candidate_items: Items to generate counterfactuals for

    Returns:
        counterfactual_prefs: Predicted preferences for candidate items
    """
    # Encode user from observed interactions
    user_vector = create_interaction_vector(observed_items, n_items=10000)

    # Encode to latent space
    with torch.no_grad():
        mean, logvar = model.encode(user_vector.unsqueeze(0))
        z = model.reparameterize(mean, logvar)

    # Decode to get full preference vector
    with torch.no_grad():
        full_prefs = torch.sigmoid(model.decode(z)).squeeze()

    # Extract counterfactual preferences for candidate items
    counterfactual_prefs = full_prefs[candidate_items]

    return counterfactual_prefs


# Example: Evaluate model with counterfactuals
observed_items = torch.tensor([10, 25, 47])
candidate_items = torch.tensor([100, 200, 300])  # Unobserved items

counterfactual_prefs = generate_counterfactuals(vae_model, user_id, observed_items, candidate_items)

print(f"Counterfactual preferences: {counterfactual_prefs}")
# Output: tensor([0.82, 0.15, 0.63])
```

---

## Summary

**Key Takeaways**:
1. **Generative vs Discriminative**: Modeling joint distribution vs conditional - fundamentally different approaches
2. **VAE-CF**: ELBO derivation, probabilistic latent representation for collaborative filtering
3. **LLM-based recommendations**: Reasoning, world knowledge, conversational interaction
4. **GANs**: Generate synthetic user interactions for data augmentation
5. **Diffusion**: Iterative denoising for high-quality generation
6. **Counterfactuals**: Fill in missing interactions for unbiased evaluation

**Critical Failure Modes** (Never forget these):
- Hallucination - generating non-existent items
- Inventory mismatch - ideal items may not exist
- Latency - LLMs are slow
- Confabulation - plausible but false explanations
- Bias amplification - stereotypes get reinforced

**The Emerging Paradigm**: Hybrid systems combining:
- Fast embedding-based retrieval
- LLM-based ranking and explanation
- Grounding to real inventory
- Human-in-the-loop verification

> **Final Socratic Question**: "If a generative system can create the 'perfect' recommendation that doesn't exist, have we succeeded or failed at the recommendation task?"

---

## Practice Problems

**Problem 1**: Train VAE-CF on MovieLens. Visualize latent space (t-SNE). Do similar users cluster together?

**Problem 2**: Use GAN to augment sparse users (< 5 interactions). Does it improve cold-start performance?

**Problem 3**: Implement diffusion model for sequential recommendation (generate item sequences instead of sets).

**Problem 4**: Generate counterfactual interactions for unbiased evaluation. Compare NDCG on observed vs. counterfactual data.

**Problem 5**: Build a conversational recommender using an LLM API. Implement hallucination detection that verifies recommendations against a real catalog.

**Problem 6**: Compare explanation quality between: (a) post-hoc LLM explanations, (b) attention-based explanations from neural CF. Which do users trust more? Which is more accurate?

---

## References

1. **Liang, D., et al. (2018)**. "Variational Autoencoders for Collaborative Filtering". *WWW*.

2. **Goodfellow, I., et al. (2014)**. "Generative Adversarial Networks". *NeurIPS*.

3. **Ho, J., et al. (2020)**. "Denoising Diffusion Probabilistic Models". *NeurIPS*.

4. **Wang, X., et al. (2021)**. "Counterfactual Data Augmentation for Neural Machine Translation". *NAACL*.

5. **Gao, Y., et al. (2023)**. "Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System". *arXiv*.

6. **Hou, Y., et al. (2024)**. "Large Language Models are Zero-Shot Rankers for Recommender Systems". *ECIR*.