# Week 17: Explainability Techniques

## The Opening Puzzle: When Recommendations Look Wrong

*Before we dive into the mathematics, let me show you something that will frame our entire discussion today.*

**Scenario**: A user who watched "The Shawshank Redemption," "Schindler's List," and "The Godfather" receives this recommendation:

> **Recommended for you: "Toy Story"**

*"Wait, what? The user watches serious dramas and you recommend an animated children's movie?"*

Your product manager is angry. The user is confused. The recommendation looks **completely wrong**.

**But then we show the explanation:**

> "Because you enjoyed films with themes of **hope and redemption** (Shawshank), **deep emotional journeys** (Schindler's List), and **complex character development** (Godfather) - Toy Story shares these narrative elements. Users with similar taste rated it 4.8/5."

*Suddenly, the recommendation makes sense.* Toy Story IS about hope, friendship, and character growth - just wrapped in animation.

**This is why explainability matters.** The same recommendation can seem absurd or brilliant depending on whether users understand the reasoning.

---

## Overview

**Post-hoc explainability**: Techniques to explain predictions of black-box models after training.

**Categories**:
1. **Local explanations**: Explain individual predictions (LIME, SHAP)
2. **Global explanations**: Explain overall model behavior (feature importance)
3. **Model-specific**: Attention visualization (transformers), influence functions
4. **Example-based**: Counterfactuals, prototypes

**The Central Question**: When a model makes a decision, can we answer "Why THIS recommendation for THIS user?"

---

## LIME (Local Interpretable Model-agnostic Explanations)

### Intuition

**Idea**: Approximate complex model locally with simple interpretable model.

*Think of it this way*: Your neural network is a complex mountain range. We cannot describe the entire landscape simply. But we CAN describe what the terrain looks like at one specific point - a flat plane tangent to the surface at that location.

**Process**:
1. Select prediction to explain
2. Generate perturbed samples around it
3. Get model predictions for perturbed samples
4. Train simple model (linear regression) on perturbed samples
5. Use simple model to explain prediction

---

### The LIME Derivation: Step by Step

*Let me walk you through exactly how LIME works, from intuition to implementation.*

**Step 1: The Formal Objective**

LIME finds an explanation by solving:

$$\xi(x) = \underset{g \in G}{\arg\min} \; \mathcal{L}(f, g, \pi_x) + \Omega(g)$$

where:
- $f$ = our black-box model (the neural network)
- $g$ = interpretable model (linear regression)
- $G$ = class of interpretable models
- $\pi_x$ = proximity measure to instance $x$
- $\Omega(g)$ = complexity penalty (keep explanation simple)

*In plain English: Find the simplest linear model that approximates our complex model near the point we care about.*

**Step 2: Proximity Weighting**

Not all perturbations are equally important. Closer perturbations should matter more:

$$\pi_x(z) = \exp\left(-\frac{D(x, z)^2}{\sigma^2}\right)$$

where $D(x, z)$ is distance (usually cosine distance for sparse features, Euclidean for dense).

**Step 3: The Weighted Linear Model**

We solve weighted least squares:

$$\min_{\beta} \sum_{i=1}^{N} \pi_x(z_i) \cdot \left(f(z_i) - g(z_i)\right)^2 + \lambda \|\beta\|_2^2$$

where $g(z) = \beta_0 + \sum_j \beta_j z_j$.

---

### Numerical Example: LIME with 5 Perturbations

*Let's trace through a concrete example to see exactly what happens.*

**Setup**: Recommending movie to User 123. Features:
- $x_1$ = action_preference (0.8)
- $x_2$ = comedy_preference (0.3)
- $x_3$ = item_rating (4.5)

**Original prediction**: $f(x) = 0.92$ (92% confidence)

**Step 1: Generate 5 perturbations** (turning features "on/off" or adding noise):

| Perturbation | $x_1$ | $x_2$ | $x_3$ | $f(z)$ |
|--------------|-------|-------|-------|--------|
| $z_1$ (original) | 0.8 | 0.3 | 4.5 | 0.92 |
| $z_2$ | 0.0 | 0.3 | 4.5 | 0.45 |
| $z_3$ | 0.8 | 0.0 | 4.5 | 0.88 |
| $z_4$ | 0.8 | 0.3 | 2.0 | 0.71 |
| $z_5$ | 0.0 | 0.0 | 4.5 | 0.42 |

**Step 2: Calculate distances and weights** ($\sigma = 0.5$):

| Perturbation | Distance $D(x, z)$ | Weight $w = e^{-D^2/\sigma^2}$ |
|--------------|-------------------|-------------------------------|
| $z_1$ | 0.00 | 1.000 |
| $z_2$ | 0.80 | 0.078 |
| $z_3$ | 0.30 | 0.699 |
| $z_4$ | 2.50 | 0.000 (nearly) |
| $z_5$ | 0.85 | 0.056 |

*Notice: $z_1$ (original) and $z_3$ (nearby) get high weights. $z_4$ (far away) is essentially ignored.*

**Step 3: Fit weighted linear regression**:

Solving weighted least squares gives us:

$$g(z) = 0.15 + 0.59 \cdot x_1 + 0.05 \cdot x_2 + 0.08 \cdot x_3$$

**Step 4: Interpret coefficients**:

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| action_preference | **+0.59** | Most important! |
| item_rating | +0.08 | Moderately important |
| comedy_preference | +0.05 | Minor influence |

**Explanation**: "This movie was recommended primarily because you like action films ($\beta_1 = 0.59$). The high rating also contributed, while your comedy preference had minimal impact."

---

### Algorithm

**Steps**:

1. **Original prediction**: $f(x) = 0.9$ (90% confidence recommendation)

2. **Generate perturbed samples**:
   - Randomly modify features
   - Example: For movie recommendation, set some genres to 0

3. **Get predictions**: $f(x')$ for each perturbed $x'$

4. **Weight samples** by proximity to $x$:
   $$w_i = \exp\left(-\frac{d(x, x_i)^2}{2\sigma^2}\right)$$

5. **Train linear model**:
   $$g(x') = \beta_0 + \beta_1 x'_1 + \beta_2 x'_2 + \ldots$$

6. **Interpret**: $\beta_i$ = importance of feature $i$

---

### Implementation

```python
import numpy as np
from sklearn.linear_model import Ridge

class LIME:
    def __init__(self, model, n_samples=1000, kernel_width=0.25):
        self.model = model
        self.n_samples = n_samples
        self.kernel_width = kernel_width

    def explain(self, instance, feature_names):
        """
        Explain a single prediction.

        Args:
            instance: Feature vector to explain
            feature_names: Names of features

        Returns:
            explanation: Dict mapping features to importance
        """
        # Generate perturbed samples
        perturbed_samples = self._generate_samples(instance)

        # Get model predictions
        predictions = self.model.predict(perturbed_samples)

        # Compute sample weights
        distances = np.linalg.norm(perturbed_samples - instance, axis=1)
        weights = np.exp(-(distances ** 2) / (2 * self.kernel_width ** 2))

        # Train linear model
        linear_model = Ridge(alpha=1.0)
        linear_model.fit(perturbed_samples, predictions, sample_weight=weights)

        # Extract feature importance
        coefficients = linear_model.coef_

        # Create explanation
        explanation = dict(zip(feature_names, coefficients))

        # Sort by absolute importance
        explanation = dict(sorted(explanation.items(),
                                 key=lambda x: abs(x[1]),
                                 reverse=True))

        return explanation

    def _generate_samples(self, instance):
        """Generate perturbed samples around instance."""
        n_features = len(instance)

        # Random perturbations
        perturbed = instance + np.random.normal(0, 0.1, (self.n_samples, n_features))

        return perturbed


# Example usage for recommendation
def explain_recommendation(user, item, model):
    """
    Explain why item was recommended to user.
    """
    # Extract features
    user_features = extract_user_features(user)
    item_features = extract_item_features(item)
    features = np.concatenate([user_features, item_features])

    feature_names = (
        ['user_age', 'user_genre_pref_action', 'user_genre_pref_comedy'] +
        ['item_year', 'item_rating', 'item_popularity']
    )

    # LIME explanation
    explainer = LIME(model, n_samples=1000)
    explanation = explainer.explain(features, feature_names)

    # Display top features
    print(f"Why we recommended {item.title}:\n")
    for feature, importance in list(explanation.items())[:5]:
        if importance > 0:
            print(f"  + {feature}: +{importance:.3f}")
        else:
            print(f"  - {feature}: {importance:.3f}")

    return explanation


# Example
user = get_user(user_id=123)
item = get_item(item_id=456)
model = load_model()

explanation = explain_recommendation(user, item, model)
# Output:
# Why we recommended Inception:
#   + user_genre_pref_action: +0.452
#   + item_rating: +0.318
#   + item_year: +0.245
#   - item_popularity: -0.102
```

---

## SHAP (SHapley Additive exPlanations)

### Socratic Moment: The Fair Attribution Problem

*Before I show you the formula, let me pose a challenge.*

**Scenario**: Three features together produce a prediction of $f(x) = 0.9$:
- Feature A (user likes action)
- Feature B (movie is highly rated)
- Feature C (movie is new release)

**Question**: How do we **fairly** assign credit to each feature?

*Think about it. Take 30 seconds before reading on.*

**Attempt 1: Just remove each feature**
- Remove A: prediction drops from 0.9 to 0.5 (A contributes 0.4?)
- Remove B: prediction drops from 0.9 to 0.7 (B contributes 0.2?)
- Remove C: prediction drops from 0.9 to 0.8 (C contributes 0.1?)

**Problem**: 0.4 + 0.2 + 0.1 = 0.7, but we need to explain a change of 0.9 from baseline!

**Attempt 2: Start with nothing, add each feature**
- Start with nothing: 0.3 (baseline)
- Add A: 0.3 to 0.6 (A contributes 0.3?)
- Add A+B: 0.6 to 0.85 (B contributes 0.25?)
- Add A+B+C: 0.85 to 0.9 (C contributes 0.05?)

**Problem**: The order matters! If we add B first, it might get a different credit.

*How do we make this fair?*

---

### Shapley Values: The Game-Theoretic Solution

**Origin**: Game theory (Lloyd Shapley, 1953 - Nobel Prize 2012).

**Original Problem**: How should a coalition fairly split the payout of a cooperative game?

**Key Insight**: A "fair" split must satisfy these axioms:
1. **Efficiency**: Credits sum to total value
2. **Symmetry**: Equal contributors get equal credit
3. **Null Player**: Zero contributors get zero credit
4. **Additivity**: Values combine linearly across games

**Remarkable Theorem**: There is exactly ONE way to split credit that satisfies all axioms - the Shapley value!

---

### The Shapley Formula Derivation

**Idea**: Consider ALL possible orderings of features. For each ordering, give a feature credit for its marginal contribution when it joins.

**The Formula**:
$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|}! [f(S \cup \{i\}) - f(S)]$$

*Let me break down each piece:*

- $F$ = all features (the "grand coalition")
- $S$ = a subset of features that arrived before feature $i$
- $|S|!$ = number of orderings where exactly $S$ comes before $i$
- $(|F| - |S| - 1)!$ = number of orderings for features after $i$
- $|F|!$ = total possible orderings
- $f(S \cup \{i\}) - f(S)$ = marginal contribution of adding $i$ to $S$

**Intuition**: This computes the **average marginal contribution** of feature $i$ across all possible "arrival orderings."

---

### Numerical SHAP Example: Three Features

*Let's compute Shapley values by hand for intuition.*

**Setup**: 3 features (A, B, C), need to explain prediction change from baseline 0.3 to final 0.9.

**Model outputs for all subsets**:
| Subset | $f(S)$ |
|--------|--------|
| {} | 0.30 |
| {A} | 0.55 |
| {B} | 0.45 |
| {C} | 0.40 |
| {A,B} | 0.75 |
| {A,C} | 0.70 |
| {B,C} | 0.55 |
| {A,B,C} | 0.90 |

**Computing $\phi_A$ (Shapley value for feature A)**:

We consider all orderings and A's marginal contribution:

| Ordering | S (before A) | Marginal $f(S \cup A) - f(S)$ |
|----------|--------------|-------------------------------|
| A,B,C | {} | $f(\{A\}) - f(\{\}) = 0.55 - 0.30 = 0.25$ |
| A,C,B | {} | $0.55 - 0.30 = 0.25$ |
| B,A,C | {B} | $f(\{A,B\}) - f(\{B\}) = 0.75 - 0.45 = 0.30$ |
| B,C,A | {B,C} | $f(\{A,B,C\}) - f(\{B,C\}) = 0.90 - 0.55 = 0.35$ |
| C,A,B | {C} | $f(\{A,C\}) - f(\{C\}) = 0.70 - 0.40 = 0.30$ |
| C,B,A | {B,C} | $0.90 - 0.55 = 0.35$ |

$$\phi_A = \frac{0.25 + 0.25 + 0.30 + 0.35 + 0.30 + 0.35}{6} = \frac{1.80}{6} = 0.30$$

*Similarly computing for B and C:*
- $\phi_B = 0.20$
- $\phi_C = 0.10$

**Verification (Efficiency Axiom)**:
$$\phi_A + \phi_B + \phi_C = 0.30 + 0.20 + 0.10 = 0.60 = f(\{A,B,C\}) - f(\{\}) = 0.90 - 0.30$$

*The credits sum to exactly the prediction change!*

---

### Properties of Shapley Values

**Properties**:
- **Efficiency**: $\sum_i \phi_i = f(x) - f(\emptyset)$
- **Symmetry**: Features with same contribution get same value
- **Additivity**: Contributions are additive

*These aren't just nice properties - they're the ONLY attribution method that satisfies all of them simultaneously.*

---

### The Computational Problem

*There's a catch. Did you notice it?*

**For $n$ features, we need to evaluate $2^n$ subsets!**

- 10 features: 1,024 evaluations
- 20 features: 1,048,576 evaluations
- 50 features: $> 10^{15}$ evaluations

*This is exponentially expensive - completely intractable for real models.*

**Solutions**:
1. **Sampling**: Approximate with random permutations (KernelSHAP)
2. **Model-specific algorithms**: TreeSHAP for trees (polynomial time!)

---

### Implementation

```python
import shap
import torch

class ShapExplainer:
    def __init__(self, model, background_data):
        """
        Args:
            model: PyTorch model
            background_data: Reference dataset for SHAP baseline
        """
        self.model = model
        self.explainer = shap.DeepExplainer(model, background_data)

    def explain(self, instance, feature_names):
        """
        Compute SHAP values for instance.
        """
        # Compute SHAP values
        shap_values = self.explainer.shap_values(instance)

        # Create explanation
        explanation = dict(zip(feature_names, shap_values[0]))

        return explanation

    def plot_waterfall(self, instance, feature_names):
        """
        Waterfall plot showing how features contribute to prediction.
        """
        shap_values = self.explainer.shap_values(instance)

        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=instance[0],
                feature_names=feature_names
            )
        )

    def plot_force(self, instance, feature_names):
        """
        Force plot showing positive and negative contributions.
        """
        shap_values = self.explainer.shap_values(instance)

        shap.force_plot(
            self.explainer.expected_value,
            shap_values[0],
            instance[0],
            feature_names=feature_names
        )


# Example usage
import torch.nn as nn

class RecommenderModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


# Load model and data
model = load_trained_model()
background_data = torch.randn(100, 10)  # Sample background data

# Create explainer
explainer = ShapExplainer(model, background_data)

# Explain prediction
user_item_features = torch.randn(1, 10)
feature_names = ['feature_' + str(i) for i in range(10)]

explanation = explainer.explain(user_item_features, feature_names)
print(explanation)

# Visualize
explainer.plot_waterfall(user_item_features, feature_names)
```

---

## TreeSHAP: Polynomial-Time SHAP for Trees

### The Efficiency Breakthrough

*Remember the exponential problem? TreeSHAP solves it elegantly for tree-based models.*

**Key Insight**: Tree structure allows computing exact Shapley values in $O(TLD^2)$ time:
- $T$ = number of trees
- $L$ = number of leaves
- $D$ = maximum depth

**Why It Works**: In a decision tree, a feature either:
1. Appears on the path from root to leaf (contributes)
2. Doesn't appear (zero contribution for that path)

The tree structure lets us recursively compute contributions without enumerating all $2^n$ subsets!

### Algorithm Intuition

**For each leaf node**, TreeSHAP tracks:
1. Which features were used to reach this leaf
2. The proportion of training data that would reach this leaf if certain features were "unknown"

**The recursion**:
```
TREESHAP(node, remaining_features):
    if node is leaf:
        return weighted prediction

    if node.feature in remaining_features:
        # Feature known: follow the appropriate branch
        return TREESHAP(appropriate_child, remaining_features - {node.feature})
    else:
        # Feature unknown: average over both branches
        return weight_left * TREESHAP(left_child, remaining_features) +
               weight_right * TREESHAP(right_child, remaining_features)
```

### Complexity Comparison

| Method | Complexity | 20 features | 100 features |
|--------|------------|-------------|--------------|
| Exact Shapley | $O(2^n)$ | 1 million | $10^{30}$ |
| KernelSHAP | $O(k \cdot n^2)$ | 4,000 | 100,000 |
| **TreeSHAP** | $O(TLD^2)$ | **400** | **2,000** |

*TreeSHAP is often 1000x faster than sampling approaches, with exact results!*

### TreeSHAP Implementation

```python
import xgboost as xgb
import shap

# Train XGBoost model
X_train, y_train = load_data()

model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# SHAP explainer (automatically uses TreeSHAP for tree models)
explainer = shap.TreeExplainer(model)

# Explain test instance
X_test = load_test_instance()
shap_values = explainer.shap_values(X_test)

# Summary plot: global feature importance
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Individual explanation
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0],
    matplotlib=True
)
```

---

## Attention as Explanation

### Transformer Attention Weights

**Transformers**: Use attention mechanism to weight importance of input tokens.

**Idea**: Visualize attention weights to see which items model focused on.

**Application**: Sequential recommendation (BERT4Rec, SASRec).

---

### When Attention IS a Valid Explanation

**Attention CAN be explanatory when**:

1. **Single-layer attention**: Direct connection from input to output
2. **The model genuinely uses attention for decisions**: Not just a computational shortcut
3. **Attention is necessary**: Zeroing attention actually changes the prediction proportionally

**Example where attention works**: Simple attention-pooling for sequential recommendation

```
User history: [Movie A, Movie B, Movie C] --> Attention weights --> Next item
```

If attention weight on Movie B is 0.7, and removing Movie B changes the prediction significantly, attention IS explanatory.

---

### When Attention is NOT a Valid Explanation

**The Faithfulness Problem**: Attention weights can be **plausible but unfaithful**.

**Jain & Wallace (2019) showed**:
1. Models with very different attention weights can make identical predictions
2. Attention weights can be manipulated without changing outputs
3. Gradient-based importance often disagrees with attention

**Example of unfaithful attention**:

```python
# Two different attention distributions, same prediction!
attention_1 = [0.8, 0.1, 0.1]  # Focus on item 1
attention_2 = [0.1, 0.8, 0.1]  # Focus on item 2
# Both produce prediction = 0.85 (somehow!)
```

*If we show users attention_1 as an explanation, we might be misleading them.*

---

### The Faithful vs. Plausible Distinction

| Property | Definition | Attention |
|----------|------------|-----------|
| **Plausible** | Explanation seems reasonable to humans | Often yes |
| **Faithful** | Explanation reflects actual model reasoning | Often no! |

**Key Test for Faithfulness**:
- Perturb the features with high attention
- Does the prediction change proportionally?
- If not, attention is plausible but unfaithful

---

### Implementation

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

class TransformerRecommender(nn.Module):
    def __init__(self, n_items, d_model=128, n_heads=4):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.fc = nn.Linear(d_model, n_items)

    def forward(self, item_sequence):
        """
        Args:
            item_sequence: [batch_size, seq_len]

        Returns:
            logits: [batch_size, n_items]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        # Embed items
        embeddings = self.item_embedding(item_sequence)  # [batch, seq_len, d_model]

        # Transpose for attention
        embeddings = embeddings.transpose(0, 1)  # [seq_len, batch, d_model]

        # Self-attention
        attn_output, attn_weights = self.attention(
            embeddings, embeddings, embeddings,
            need_weights=True
        )

        # Take last token for prediction
        last_output = attn_output[-1]  # [batch, d_model]

        # Predict next item
        logits = self.fc(last_output)

        return logits, attn_weights


def visualize_attention(model, item_sequence, item_names):
    """
    Visualize attention weights for recommendation.
    """
    model.eval()

    with torch.no_grad():
        logits, attn_weights = model(item_sequence)

    # Average attention across heads
    attn_weights = attn_weights.squeeze(0).mean(dim=0)  # [seq_len, seq_len]

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn_weights.numpy(),
        xticklabels=item_names,
        yticklabels=item_names,
        cmap='viridis',
        annot=True,
        fmt='.2f'
    )
    plt.title("Attention Weights: Which past items influenced recommendation?")
    plt.xlabel("Past Items (Key)")
    plt.ylabel("Items (Query)")
    plt.show()


def test_attention_faithfulness(model, item_sequence, item_names, top_k=2):
    """
    Test if attention weights are faithful explanations.
    """
    model.eval()

    with torch.no_grad():
        original_logits, attn_weights = model(item_sequence)
        original_pred = torch.softmax(original_logits, dim=-1).max().item()

    # Find items with highest attention (last row = attention for final prediction)
    last_attention = attn_weights.squeeze(0).mean(dim=0)[-1]  # [seq_len]
    top_attended = torch.topk(last_attention, top_k).indices

    print(f"Original prediction confidence: {original_pred:.3f}")
    print(f"Top attended items: {[item_names[i] for i in top_attended]}")

    # Zero out top attended items and see if prediction changes
    modified_sequence = item_sequence.clone()
    for idx in top_attended:
        modified_sequence[0, idx] = 0  # Set to padding/unknown

    with torch.no_grad():
        modified_logits, _ = model(modified_sequence)
        modified_pred = torch.softmax(modified_logits, dim=-1).max().item()

    print(f"After removing top attended: {modified_pred:.3f}")
    print(f"Change: {abs(original_pred - modified_pred):.3f}")

    if abs(original_pred - modified_pred) > 0.1:
        print("Attention appears FAITHFUL (removing attended items changes prediction)")
    else:
        print("WARNING: Attention may be UNFAITHFUL (prediction barely changed)")


# Example
model = TransformerRecommender(n_items=1000, d_model=128, n_heads=4)
item_sequence = torch.tensor([[10, 25, 47, 103, 256]])  # User's history

item_names = ['Item A', 'Item B', 'Item C', 'Item D', 'Item E']

visualize_attention(model, item_sequence, item_names)
test_attention_faithfulness(model, item_sequence, item_names)
```

**Interpretation**:
- High attention weight MAY indicate item influenced recommendation
- But always verify with faithfulness tests!
- Example: If recommending action movie, high attention on past action movies

---

## Numerical Walkthrough: Same Recommendation, Three Methods

*Let's explain the SAME recommendation using LIME, SHAP, and attention, then compare results.*

### Setup

**User**: Watched action movies (Die Hard, Matrix), rated highly (4.5 avg)
**Item**: "John Wick" (action, high rating, 2014)
**Prediction**: 0.92 (92% confident recommendation)

**Features**:
- $x_1$: user_action_preference = 0.85
- $x_2$: user_avg_rating = 4.5
- $x_3$: item_is_action = 1.0
- $x_4$: item_avg_rating = 4.2
- $x_5$: item_year = 2014

### LIME Explanation

```python
# LIME with 1000 perturbations, kernel_width=0.5
lime_explanation = {
    'user_action_preference': +0.42,
    'item_is_action': +0.38,
    'user_avg_rating': +0.15,
    'item_avg_rating': +0.12,
    'item_year': -0.02
}
```

**LIME says**: "Action preferences (user and item) dominate."

### SHAP Explanation

```python
# TreeSHAP values
shap_values = {
    'user_action_preference': +0.28,
    'item_is_action': +0.25,
    'user_avg_rating': +0.08,
    'item_avg_rating': +0.09,
    'item_year': -0.01
}
# Sum: 0.69 = 0.92 - 0.23 (base rate)
```

**SHAP says**: "Action features most important, but more balanced attribution."

### Attention Explanation

```python
# For sequential model: attention on user's watch history
# [Die Hard, Matrix, Avengers, Comedy X, Romance Y]
attention_weights = [0.35, 0.40, 0.15, 0.06, 0.04]
```

**Attention says**: "Focus on Matrix (0.40) and Die Hard (0.35)."

### Comparison Summary

| Feature/Item | LIME | SHAP | Attention |
|--------------|------|------|-----------|
| Action preference | **0.42** | **0.28** | - |
| Matrix (action) | - | - | **0.40** |
| Die Hard (action) | - | - | **0.35** |
| Item is action | 0.38 | 0.25 | - |

**Key Observations**:

1. **All three agree**: Action is the key factor
2. **LIME** gives highest feature weights (less normalized)
3. **SHAP** provides guaranteed-sum-to-prediction attribution
4. **Attention** shows WHICH past items matter (not engineered features)

**When to use which**:
- **LIME**: Quick, interpretable, any model
- **SHAP**: When you need mathematically rigorous attribution
- **Attention**: Sequential models, showing "similar items" explanations

---

## Influence Functions

### Intuition

**Question**: Which training examples most influenced this prediction?

**Idea**: Trace back through optimization to find influential training points.

**Formula** (simplified):
$$\mathcal{I}(z, z_{test}) = -\nabla_\theta L(z_{test}, \theta) \cdot H_\theta^{-1} \nabla_\theta L(z, \theta)$$

where:
- $z$ = training example
- $z_{test}$ = test example
- $H_\theta$ = Hessian of loss

**Interpretation**:
- Positive influence: Training example increased prediction
- Negative influence: Training example decreased prediction

---

### Implementation

```python
import torch

class InfluenceFunctions:
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader

    def compute_influence(self, test_instance, test_label):
        """
        Find training examples that most influenced test prediction.
        """
        # Compute test gradient
        test_grad = self._compute_gradient(test_instance, test_label)

        # Approximate inverse Hessian-vector product
        ihvp = self._inverse_hvp(test_grad)

        # Compute influence for each training example
        influences = []

        for train_batch in self.train_loader:
            train_inputs, train_labels = train_batch

            for i in range(len(train_inputs)):
                train_grad = self._compute_gradient(train_inputs[i], train_labels[i])

                # Influence = -dot(test_grad, H^-1 * train_grad)
                influence = -torch.dot(test_grad.flatten(), train_grad.flatten())

                influences.append((train_inputs[i], influence.item()))

        # Sort by influence
        influences.sort(key=lambda x: abs(x[1]), reverse=True)

        return influences

    def _compute_gradient(self, instance, label):
        """Compute gradient of loss w.r.t. parameters."""
        self.model.zero_grad()

        output = self.model(instance.unsqueeze(0))
        loss = nn.BCELoss()(output, label.unsqueeze(0).float())

        loss.backward()

        # Concatenate all parameter gradients
        grad = torch.cat([p.grad.flatten() for p in self.model.parameters()])

        return grad

    def _inverse_hvp(self, v, damping=0.01):
        """
        Approximate inverse Hessian-vector product using conjugate gradient.
        """
        # Simplified: return v (identity approximation)
        # Full implementation would use CG or LBFGS
        return v


# Example usage
model = load_trained_model()
train_loader = load_train_data()

influence_fn = InfluenceFunctions(model, train_loader)

# Find training examples that influenced this recommendation
test_user_item = torch.randn(10)
test_label = torch.tensor(1)  # Positive recommendation

top_influences = influence_fn.compute_influence(test_user_item, test_label)

print("Top 5 influential training examples:")
for i, (train_example, influence) in enumerate(top_influences[:5]):
    print(f"{i+1}. Influence: {influence:.4f}")
```

**Use case**: Debug model by identifying problematic training examples.

---

## Counterfactual Explanations

### Intuition

**Question**: What minimal change to input would flip the prediction?

**Example**:
```
Original: "Movie not recommended"
Counterfactual: "If user had watched 2 more action movies, it would be recommended"
```

---

### Algorithm

**Goal**: Find minimal change $\delta$ such that prediction changes.

**Optimization**:
$$\min_\delta \|\delta\|_1 \quad \text{s.t.} \quad f(x + \delta) \neq f(x)$$

where $\|\delta\|_1$ = sparsity (few features changed).

---

### Implementation

```python
import torch
import torch.optim as optim

def find_counterfactual(model, instance, target_prediction, max_iter=1000):
    """
    Find minimal perturbation to change prediction.

    Args:
        model: Recommendation model
        instance: Current feature vector
        target_prediction: Desired prediction (0 or 1)
        max_iter: Maximum optimization iterations

    Returns:
        counterfactual: Modified feature vector
        changes: Dict of feature changes
    """
    # Initialize counterfactual as copy of instance
    counterfactual = instance.clone().detach().requires_grad_(True)

    optimizer = optim.Adam([counterfactual], lr=0.01)

    for iteration in range(max_iter):
        optimizer.zero_grad()

        # Prediction for counterfactual
        prediction = model(counterfactual.unsqueeze(0))

        # Loss: prediction error + L1 regularization (sparsity)
        prediction_loss = (prediction - target_prediction) ** 2
        sparsity_loss = torch.norm(counterfactual - instance, p=1)

        loss = prediction_loss + 0.1 * sparsity_loss

        loss.backward()
        optimizer.step()

        # Check if target reached
        if (prediction > 0.5 and target_prediction == 1) or \
           (prediction < 0.5 and target_prediction == 0):
            break

    # Identify changes
    changes = {}
    for i, (orig, cf) in enumerate(zip(instance, counterfactual)):
        if abs(orig - cf) > 0.01:
            changes[f'feature_{i}'] = {
                'original': orig.item(),
                'counterfactual': cf.item(),
                'change': (cf - orig).item()
            }

    return counterfactual, changes


# Example
model = load_trained_model()
user_item_features = torch.tensor([0.5, 0.3, 0.8, 0.2, 0.6])

# Current prediction: not recommended (0)
# Find counterfactual for recommendation (1)

counterfactual, changes = find_counterfactual(
    model,
    user_item_features,
    target_prediction=1.0
)

print("To get this item recommended, you should:")
for feature, change in changes.items():
    print(f"  - Change {feature} from {change['original']:.2f} to {change['counterfactual']:.2f}")

# Output:
# To get this item recommended, you should:
#   - Change feature_1 from 0.30 to 0.55 (watch more comedy movies)
#   - Change feature_4 from 0.60 to 0.80 (increase rating threshold)
```

---

## Prototype-Based Explanations

### Intuition

**Idea**: Explain recommendation by showing similar examples.

**Example**:
```
"We recommended Inception because it's similar to these movies you liked:
  - The Matrix (similarity: 0.89)
  - Interstellar (similarity: 0.85)
  - Shutter Island (similarity: 0.78)"
```

---

### Implementation

```python
def explain_with_prototypes(user, recommendation, user_history, k=3):
    """
    Explain recommendation with similar items user liked.
    """
    # Get embeddings
    rec_embedding = get_item_embedding(recommendation)

    # Find similar items in user history
    similarities = []

    for past_item in user_history:
        if user.liked(past_item):
            past_embedding = get_item_embedding(past_item)

            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                rec_embedding, past_embedding, dim=0
            )

            similarities.append((past_item, similarity.item()))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Generate explanation
    explanation = f"We recommended {recommendation.title} because it's similar to:\n"

    for i, (item, sim) in enumerate(similarities[:k]):
        explanation += f"  {i+1}. {item.title} (similarity: {sim:.2f})\n"

    return explanation, similarities[:k]


# Example
user = get_user(user_id=123)
recommendation = get_item(item_id=456)  # Inception
user_history = user.get_watch_history()

explanation, prototypes = explain_with_prototypes(user, recommendation, user_history)
print(explanation)

# Output:
# We recommended Inception because it's similar to:
#   1. The Matrix (similarity: 0.89)
#   2. Interstellar (similarity: 0.85)
#   3. Shutter Island (similarity: 0.78)
```

---

## What Can Go Wrong: Failure Modes of Explainability

*Now for the cautionary tales. Explainability is powerful, but it can fail in subtle and dangerous ways.*

### 1. Unfaithful Explanations

**Problem**: Explanation doesn't reflect actual model reasoning.

**Example**:
```python
# Model actually uses: user_age > 30 AND item_popularity > 1000
# LIME explanation says: "Because you liked action movies"

# The explanation is PLAUSIBLE but WRONG
```

**Why it happens**:
- LIME uses linear approximation (model may be highly nonlinear locally)
- Sampling may miss important regions
- Explanation method assumptions don't match model behavior

**Detection**: Compare explanations across methods. If LIME, SHAP, and gradients disagree significantly, be suspicious.

**Mitigation**:
```python
def validate_explanation(model, instance, explanation, n_tests=100):
    """Test if explanation is faithful by perturbation."""
    top_feature = max(explanation, key=lambda k: abs(explanation[k]))

    original_pred = model.predict(instance)

    # Perturb top feature
    perturbed = instance.copy()
    perturbed[top_feature] = 0
    perturbed_pred = model.predict(perturbed)

    expected_change = abs(explanation[top_feature])
    actual_change = abs(original_pred - perturbed_pred)

    if actual_change < expected_change * 0.5:
        print(f"WARNING: Explanation may be unfaithful!")
        print(f"Expected change: {expected_change:.3f}, Actual: {actual_change:.3f}")
```

---

### 2. Gaming Explanations

**Problem**: Bad actors can manipulate systems to show misleading explanations.

**Scenario**: A fraudulent seller wants to appear legitimate.

```python
# Model legitimately flags item as suspicious
# But explanation shows: "Similar to popular items"
# Seller manipulates item features to change explanation
# while keeping fraudulent behavior hidden in other features
```

**Real-world example**:
- Loan denial explanation: "Income too low"
- Applicant increases stated income
- New explanation: "Employment history too short"
- Model was actually using zip code (illegal discrimination)!

**Mitigation**:
- Don't show full explanations externally
- Audit explanations for gaming patterns
- Use multiple explanation methods

---

### 3. Explanation Complexity

**Problem**: Technically correct explanations that users can't understand.

**Bad explanation**:
> "Recommended because: user_embedding[47] * item_embedding[23] + attention_weight_layer3_head2 * context_vector[15] = 0.847"

*This is faithful but useless to users.*

**Good explanation**:
> "Because you enjoyed similar sci-fi thrillers like Inception"

**The trade-off**: Simpler explanations are more understandable but less faithful.

```python
def simplify_explanation(detailed_explanation, max_features=3):
    """Convert technical explanation to user-friendly format."""

    # Map technical features to user concepts
    concept_map = {
        'genre_vec_0': 'action movies',
        'genre_vec_1': 'comedies',
        'user_age_bucket': 'your age group',
        'item_popularity_log': 'trending items'
    }

    # Take top features
    top_features = sorted(detailed_explanation.items(),
                         key=lambda x: abs(x[1]),
                         reverse=True)[:max_features]

    # Generate natural language
    reasons = []
    for feature, importance in top_features:
        if importance > 0:
            concept = concept_map.get(feature, feature)
            reasons.append(f"you enjoy {concept}")

    return "Recommended because " + " and ".join(reasons)
```

---

### 4. User Misunderstanding

**Problem**: Users interpret explanations incorrectly.

**Example**:
- Explanation: "Recommended because of your watch history"
- User thinks: "The system is spying on me!"
- Reality: Completely benign collaborative filtering

**Research finding** (Eslami et al.): Users often develop *folk theories* about algorithms that are completely wrong.

**Common misunderstandings**:
| What explanation says | What user might think |
|-----------------------|----------------------|
| "Based on your activity" | "They're tracking everything I do" |
| "Users like you also bought" | "They're comparing me to strangers" |
| "This item is trending" | "They're pushing popular stuff for profit" |

**Mitigation**:
- User test your explanations
- Provide educational context
- Allow users to ask follow-up questions

---

### 5. Explanation Inconsistency

**Problem**: Same item, different explanations at different times.

**Example**:
```
Monday: "Recommended because you like action movies"
Tuesday: "Recommended because it's highly rated"
```

**Why it happens**:
- Model updated between queries
- Random sampling in LIME
- Context features changed

**User impact**: Erodes trust ("The system is making things up!")

**Mitigation**:
```python
def consistent_explanation(model, instance, cache, cache_key, ttl_hours=24):
    """Return cached explanation for consistency."""

    if cache_key in cache and cache[cache_key]['timestamp'] > time.time() - ttl_hours * 3600:
        return cache[cache_key]['explanation']

    # Generate new explanation
    explanation = generate_explanation(model, instance)

    # Cache it
    cache[cache_key] = {
        'explanation': explanation,
        'timestamp': time.time()
    }

    return explanation
```

---

### Summary: Explanation Failure Checklist

Before deploying explanations, verify:

- [ ] **Faithfulness**: Do explanations reflect actual model behavior?
- [ ] **Consistency**: Same inputs produce same explanations?
- [ ] **Robustness**: Small input changes don't wildly change explanations?
- [ ] **Comprehensibility**: Can target users understand them?
- [ ] **Actionability**: Can users do something with this information?
- [ ] **Gaming resistance**: Can bad actors manipulate explanations?

---

## Summary

**Key Takeaways**:
1. **LIME**: Local approximation with linear model - fast, model-agnostic, but may be unfaithful
2. **SHAP**: Game-theoretic fair attribution (Shapley values) - mathematically rigorous, guaranteed properties
3. **TreeSHAP**: Polynomial-time exact SHAP for tree models - use this when possible
4. **Attention**: Visualize which inputs model focused on - but verify faithfulness!
5. **Influence functions**: Find influential training examples
6. **Counterfactuals**: Minimal changes to flip prediction
7. **Prototypes**: Explain with similar examples

**Choosing technique**:
- **LIME/SHAP**: Model-agnostic, widely applicable
- **Attention**: For transformer-based models (with caution)
- **Influence functions**: Debug training data issues
- **Counterfactuals**: Actionable feedback for users
- **Prototypes**: Intuitive, user-friendly

**Critical Warnings**:
- Plausible does not mean faithful
- Always validate explanations
- User-test before deployment
- Simpler explanations may sacrifice accuracy

---

## Practice Problems

**Problem 1**: Implement LIME for a matrix factorization recommender. What challenges do you face?

**Problem 2**: Compare LIME and SHAP explanations for the same prediction. Are they consistent?

**Problem 3**: Design a counterfactual explanation system for a news recommendation app. What constraints would you add (e.g., realistic changes only)?

**Problem 4**: Implement attention visualization for a sequential recommendation model. Which items in the sequence are most influential?

**Problem 5 (New)**: Given the TreeSHAP algorithm, compute exact Shapley values by hand for a simple decision tree with 3 features and 4 leaves. Verify efficiency property.

**Problem 6 (New)**: Design an experiment to test whether attention weights in your recommender are faithful or merely plausible. What would convince you of faithfulness?

**Problem 7 (New)**: A user complains: "Your explanation said I'd like this movie because of the director, but I've never watched any of that director's films!" Debug this unfaithful explanation - what might have gone wrong?

---

## References

1. **Ribeiro, M. T., et al. (2016)**. "Why Should I Trust You?: Explaining the Predictions of Any Classifier". *KDD*.

2. **Lundberg, S. M., & Lee, S. I. (2017)**. "A Unified Approach to Interpreting Model Predictions". *NeurIPS*.

3. **Lundberg, S. M., et al. (2020)**. "From local explanations to global understanding with explainable AI for trees". *Nature Machine Intelligence*.

4. **Koh, P. W., & Liang, P. (2017)**. "Understanding Black-box Predictions via Influence Functions". *ICML*.

5. **Wachter, S., et al. (2017)**. "Counterfactual Explanations Without Opening the Black Box". *Harvard Journal of Law & Technology*.

6. **Vaswani, A., et al. (2017)**. "Attention Is All You Need". *NeurIPS*.

7. **Jain, S., & Wallace, B. C. (2019)**. "Attention is not Explanation". *NAACL*.

8. **Wiegreffe, S., & Pinter, Y. (2019)**. "Attention is not not Explanation". *EMNLP*.

9. **Shapley, L. S. (1953)**. "A Value for n-Person Games". *Contributions to the Theory of Games*.
