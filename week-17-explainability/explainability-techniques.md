# Week 17: Explainability Techniques

## Overview

**Post-hoc explainability**: Techniques to explain predictions of black-box models after training.

**Categories**:
1. **Local explanations**: Explain individual predictions (LIME, SHAP)
2. **Global explanations**: Explain overall model behavior (feature importance)
3. **Model-specific**: Attention visualization (transformers), influence functions
4. **Example-based**: Counterfactuals, prototypes

---

## LIME (Local Interpretable Model-agnostic Explanations)

### Intuition

**Idea**: Approximate complex model locally with simple interpretable model.

**Process**:
1. Select prediction to explain
2. Generate perturbed samples around it
3. Get model predictions for perturbed samples
4. Train simple model (linear regression) on perturbed samples
5. Use simple model to explain prediction

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
            print(f"  ✓ {feature}: +{importance:.3f}")
        else:
            print(f"  ✗ {feature}: {importance:.3f}")

    return explanation


# Example
user = get_user(user_id=123)
item = get_item(item_id=456)
model = load_model()

explanation = explain_recommendation(user, item, model)
# Output:
# Why we recommended Inception:
#   ✓ user_genre_pref_action: +0.452
#   ✓ item_rating: +0.318
#   ✓ item_year: +0.245
#   ✗ item_popularity: -0.102
```

---

## SHAP (SHapley Additive exPlanations)

### Shapley Values

**Origin**: Game theory (fair credit assignment).

**Idea**: Feature's contribution = average marginal contribution across all feature subsets.

**Formula**:
$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$$

where:
- $F$ = all features
- $S$ = subset of features
- $\phi_i$ = Shapley value for feature $i$

**Properties**:
- **Efficiency**: $\sum_i \phi_i = f(x) - f(\emptyset)$
- **Symmetry**: Features with same contribution get same value
- **Additivity**: Contributions are additive

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

### TreeSHAP for Gradient Boosting

**TreeSHAP**: Efficient SHAP for tree-based models (XGBoost, LightGBM).

```python
import xgboost as xgb
import shap

# Train XGBoost model
X_train, y_train = load_data()

model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# SHAP explainer
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

## Attention Visualization

### Transformer Attention Weights

**Transformers**: Use attention mechanism to weight importance of input tokens.

**Idea**: Visualize attention weights to see which items model focused on.

**Application**: Sequential recommendation (BERT4Rec, SASRec).

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


# Example
model = TransformerRecommender(n_items=1000, d_model=128, n_heads=4)
item_sequence = torch.tensor([[10, 25, 47, 103, 256]])  # User's history

item_names = ['Item A', 'Item B', 'Item C', 'Item D', 'Item E']

visualize_attention(model, item_sequence, item_names)
```

**Interpretation**:
- High attention weight → Item influenced recommendation
- Example: If recommending action movie, high attention on past action movies

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
- Positive influence → Training example increased prediction
- Negative influence → Training example decreased prediction

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

## Summary

**Key Takeaways**:
1. **LIME**: Local approximation with linear model
2. **SHAP**: Game-theoretic fair attribution (Shapley values)
3. **Attention**: Visualize which inputs model focused on
4. **Influence functions**: Find influential training examples
5. **Counterfactuals**: Minimal changes to flip prediction
6. **Prototypes**: Explain with similar examples

**Choosing technique**:
- **LIME/SHAP**: Model-agnostic, widely applicable
- **Attention**: For transformer-based models
- **Influence functions**: Debug training data issues
- **Counterfactuals**: Actionable feedback for users
- **Prototypes**: Intuitive, user-friendly

---

## Practice Problems

**Problem 1**: Implement LIME for a matrix factorization recommender. What challenges do you face?

**Problem 2**: Compare LIME and SHAP explanations for the same prediction. Are they consistent?

**Problem 3**: Design a counterfactual explanation system for a news recommendation app. What constraints would you add (e.g., realistic changes only)?

**Problem 4**: Implement attention visualization for a sequential recommendation model. Which items in the sequence are most influential?

---

## References

1. **Ribeiro, M. T., et al. (2016)**. "Why Should I Trust You?: Explaining the Predictions of Any Classifier". *KDD*.

2. **Lundberg, S. M., & Lee, S. I. (2017)**. "A Unified Approach to Interpreting Model Predictions". *NeurIPS*.

3. **Koh, P. W., & Liang, P. (2017)**. "Understanding Black-box Predictions via Influence Functions". *ICML*.

4. **Wachter, S., et al. (2017)**. "Counterfactual Explanations Without Opening the Black Box". *Harvard Journal of Law & Technology*.

5. **Vaswani, A., et al. (2017)**. "Attention Is All You Need". *NeurIPS*.
