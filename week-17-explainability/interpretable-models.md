# Week 17: Inherently Interpretable Models

## Overview

**Inherently interpretable**: Models that are transparent by design (no post-hoc explanation needed).

**Examples**:
1. **Linear models**: Weights directly show feature importance
2. **Matrix factorization**: Latent factors are interpretable
3. **Decision trees**: Rules are human-readable
4. **Rule-based systems**: Explicit IF-THEN logic

**Trade-off**: Interpretability vs. accuracy
- Simple models → more interpretable, less accurate
- Complex models (deep learning) → less interpretable, more accurate

---

## Matrix Factorization Interpretability

### Latent Factors

**Matrix factorization**: $R \approx U^T V$

where:
- $U \in \mathbb{R}^{d \times |users|}$ = user latent factors
- $V \in \mathbb{R}^{d \times |items|}$ = item latent factors
- $d$ = number of latent dimensions (e.g., 50-100)

**Interpretability**: What do latent dimensions represent?

---

### Interpreting Latent Dimensions

**Approach**: Examine items with highest/lowest values in each dimension.

**Example** (movie recommendations):
```
Dimension 1:
  High: Star Wars, Star Trek, Blade Runner → Sci-Fi
  Low: Titanic, The Notebook, Pride & Prejudice → Romance

Dimension 2:
  High: The Godfather, Goodfellas, Scarface → Crime/Drama
  Low: Toy Story, Finding Nemo, Shrek → Animation

Dimension 3:
  High: The Dark Knight, Mad Max, John Wick → Action
  Low: Moonlight, The Shape of Water → Art House
```

**Interpretation**: Dimensions capture genres and themes.

---

### Implementation

```python
import torch
import torch.nn as nn
import pandas as pd

class InterpretableMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=10):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)

        # Initialize with small random values
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_factors(user_ids)
        item_emb = self.item_factors(item_ids)

        prediction = (user_emb * item_emb).sum(dim=1)
        return prediction

    def interpret_factors(self, item_metadata, top_k=10):
        """
        Interpret latent factors by examining top items per dimension.
        """
        item_factors = self.item_factors.weight.detach().numpy()

        interpretations = {}

        for dim in range(item_factors.shape[1]):
            # Get item scores for this dimension
            scores = item_factors[:, dim]

            # Top-K highest
            top_indices = scores.argsort()[-top_k:][::-1]
            top_items = [item_metadata[i] for i in top_indices]

            # Top-K lowest
            bottom_indices = scores.argsort()[:top_k]
            bottom_items = [item_metadata[i] for i in bottom_indices]

            interpretations[f'Dimension {dim}'] = {
                'high': top_items,
                'low': bottom_items,
                'interpretation': infer_theme(top_items, bottom_items)
            }

        return interpretations


def infer_theme(high_items, low_items):
    """
    Infer theme from top/bottom items (simplified).
    """
    # Extract genres
    high_genres = [item['genre'] for item in high_items]
    low_genres = [item['genre'] for item in low_items]

    # Most common genres
    from collections import Counter
    high_common = Counter(high_genres).most_common(1)[0][0]
    low_common = Counter(low_genres).most_common(1)[0][0]

    return f"{high_common} vs. {low_common}"


# Example usage
model = InterpretableMF(n_users=1000, n_items=500, n_factors=10)

# Train model...

# Interpret factors
item_metadata = load_item_metadata()  # List of dicts with 'title', 'genre', etc.

interpretations = model.interpret_factors(item_metadata, top_k=10)

for dim, info in interpretations.items():
    print(f"\n{dim}: {info['interpretation']}")
    print(f"  High: {[item['title'] for item in info['high'][:3]]}")
    print(f"  Low: {[item['title'] for item in info['low'][:3]]}")

# Output:
# Dimension 0: Action vs. Romance
#   High: ['Mad Max', 'John Wick', 'Die Hard']
#   Low: ['The Notebook', 'Titanic', 'La La Land']
#
# Dimension 1: Sci-Fi vs. Comedy
#   High: ['Blade Runner', 'Interstellar', 'The Matrix']
#   Low: ['Superbad', 'Bridesmaids', 'The Hangover']
```

---

### User Profiles

**Interpretability**: What are user preferences in each dimension?

```python
def explain_user_profile(user_id, model, factor_interpretations):
    """
    Explain user's preferences using latent factors.
    """
    user_factors = model.user_factors.weight[user_id].detach().numpy()

    print(f"User {user_id} preferences:\n")

    for dim, value in enumerate(user_factors):
        interpretation = factor_interpretations[f'Dimension {dim}']['interpretation']

        if value > 0.5:
            print(f"  ✓ Strong preference for: {interpretation} (score: {value:.2f})")
        elif value < -0.5:
            print(f"  ✗ Avoids: {interpretation} (score: {value:.2f})")
        else:
            print(f"  ~ Neutral on: {interpretation} (score: {value:.2f})")


# Example
explain_user_profile(user_id=42, model=model, factor_interpretations=interpretations)

# Output:
# User 42 preferences:
#   ✓ Strong preference for: Action vs. Romance (score: 0.82)
#   ✓ Strong preference for: Sci-Fi vs. Comedy (score: 0.65)
#   ~ Neutral on: Drama vs. Horror (score: 0.12)
```

---

## Sparse Linear Models

### Linear Regression for Recommendations

**Model**: $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n$

**Interpretability**: Coefficients $\beta_i$ directly show feature importance.

**Example**:
```
Recommendation score = 0.5 + 0.8 * (user likes action) + 0.6 * (item is new) - 0.3 * (item is expensive)
```

**Challenge**: With many features, some are irrelevant.

**Solution**: Sparsity (Lasso regression) → set irrelevant coefficients to 0.

---

### Lasso Regression

**Objective**:
$$\min_\beta \sum_{i=1}^n (y_i - \beta^T x_i)^2 + \lambda \sum_{j=1}^p |\beta_j|$$

where $\lambda$ controls sparsity.

**Effect**: Many $\beta_j = 0$ → only important features remain.

---

### Implementation

```python
from sklearn.linear_model import Lasso
import numpy as np

class SparseLinearRecommender:
    def __init__(self, alpha=0.1):
        """
        Args:
            alpha: Regularization strength (higher = more sparse)
        """
        self.model = Lasso(alpha=alpha)

    def fit(self, X_train, y_train):
        """
        Train sparse linear model.

        Args:
            X_train: [n_samples, n_features]
            y_train: [n_samples] (ratings or click labels)
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Predict recommendation scores."""
        return self.model.predict(X_test)

    def explain_prediction(self, instance, feature_names):
        """
        Explain prediction by showing feature contributions.
        """
        coefficients = self.model.coef_
        intercept = self.model.intercept_

        contributions = {}

        for feature, coef, value in zip(feature_names, coefficients, instance):
            contribution = coef * value
            if abs(contribution) > 0.01:  # Filter small contributions
                contributions[feature] = contribution

        # Sort by absolute contribution
        contributions = dict(sorted(contributions.items(),
                                   key=lambda x: abs(x[1]),
                                   reverse=True))

        # Generate explanation
        prediction = self.model.predict([instance])[0]

        print(f"Prediction: {prediction:.2f}")
        print(f"Base score: {intercept:.2f}\n")
        print("Feature contributions:")

        for feature, contrib in contributions.items():
            sign = '+' if contrib > 0 else ''
            print(f"  {feature}: {sign}{contrib:.3f}")

        return contributions


# Example usage
X_train, y_train = load_training_data()

# Features: [user_age, user_genre_pref_action, item_year, item_rating, item_popularity, ...]
feature_names = ['user_age', 'user_genre_pref_action', 'item_year',
                 'item_rating', 'item_popularity']

# Train model
model = SparseLinearRecommender(alpha=0.1)
model.fit(X_train, y_train)

# Explain prediction for specific user-item pair
test_instance = np.array([28, 0.9, 2020, 8.5, 0.7])  # Feature values

explanation = model.explain_prediction(test_instance, feature_names)

# Output:
# Prediction: 4.35
# Base score: 2.10
#
# Feature contributions:
#   user_genre_pref_action: +1.260
#   item_rating: +0.850
#   item_popularity: +0.210
#   user_age: -0.084
```

---

### Advantages and Limitations

**Advantages**:
- Direct interpretation from coefficients
- Fast training and prediction
- No hyperparameter tuning (besides $\lambda$)

**Limitations**:
- Assumes linear relationships
- Doesn't capture complex interactions
- Lower accuracy than deep models

---

## Decision Trees and Rule-Based Systems

### Decision Trees for Recommendations

**Interpretability**: Trees produce human-readable rules.

**Example**:
```
IF user.age < 25 AND user.genre_pref = 'Action'
    THEN recommend: Action movies

ELIF user.age >= 25 AND user.has_kids
    THEN recommend: Family movies

ELSE
    THEN recommend: Drama
```

---

### Implementation

```python
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt
from sklearn import tree

class InterpretableTreeRecommender:
    def __init__(self, max_depth=5):
        """
        Args:
            max_depth: Maximum tree depth (controls interpretability)
        """
        self.model = DecisionTreeClassifier(max_depth=max_depth)

    def fit(self, X_train, y_train, feature_names):
        """
        Train decision tree.

        Args:
            y_train: Binary labels (recommend or not)
        """
        self.model.fit(X_train, y_train)
        self.feature_names = feature_names

    def predict(self, X_test):
        """Predict recommendations."""
        return self.model.predict(X_test)

    def visualize_tree(self):
        """Visualize decision tree."""
        plt.figure(figsize=(20, 10))
        tree.plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=['Not Recommend', 'Recommend'],
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.show()

    def extract_rules(self):
        """Extract human-readable rules."""
        rules = export_text(self.model, feature_names=self.feature_names)
        return rules

    def explain_prediction(self, instance):
        """
        Explain prediction by showing decision path.
        """
        # Get decision path
        decision_path = self.model.decision_path([instance])

        # Extract path
        node_index = decision_path.indices

        print("Decision path:")

        for node in node_index:
            # Check if leaf node
            if self.model.tree_.feature[node] == -2:
                # Leaf node
                prediction = self.model.tree_.value[node].argmax()
                print(f"  → Prediction: {'Recommend' if prediction == 1 else 'Not Recommend'}")
            else:
                # Internal node
                feature = self.feature_names[self.model.tree_.feature[node]]
                threshold = self.model.tree_.threshold[node]

                feature_value = instance[self.model.tree_.feature[node]]

                if feature_value <= threshold:
                    print(f"  {feature} <= {threshold:.2f} (value: {feature_value:.2f})")
                else:
                    print(f"  {feature} > {threshold:.2f} (value: {feature_value:.2f})")


# Example usage
X_train, y_train = load_training_data()
feature_names = ['user_age', 'user_genre_pref_action', 'item_year', 'item_rating']

model = InterpretableTreeRecommender(max_depth=5)
model.fit(X_train, y_train, feature_names)

# Visualize tree
model.visualize_tree()

# Extract rules
rules = model.extract_rules()
print(rules)

# Output:
# |--- user_genre_pref_action <= 0.50
# |   |--- item_rating <= 7.50
# |   |   |--- class: Not Recommend
# |   |--- item_rating > 7.50
# |   |   |--- class: Recommend
# |--- user_genre_pref_action > 0.50
# |   |--- item_year <= 2015
# |   |   |--- class: Not Recommend
# |   |--- item_year > 2015
# |   |   |--- class: Recommend

# Explain specific prediction
test_instance = np.array([28, 0.9, 2020, 8.5])
model.explain_prediction(test_instance)

# Output:
# Decision path:
#   user_genre_pref_action > 0.50 (value: 0.90)
#   item_year > 2015 (value: 2020.00)
#   → Prediction: Recommend
```

---

### Rule Mining

**Alternative**: Mine association rules from user behavior.

**Example**:
```
Rule: {watched The Matrix} → {recommend Inception}
Support: 5%
Confidence: 80%
```

**Implementation**:
```python
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

def mine_recommendation_rules(user_item_matrix, min_support=0.01, min_confidence=0.5):
    """
    Mine association rules from user-item interactions.

    Args:
        user_item_matrix: Binary matrix (1 if user interacted with item)
        min_support: Minimum frequency of rule
        min_confidence: Minimum confidence for rule
    """
    # Find frequent itemsets
    frequent_itemsets = apriori(
        user_item_matrix,
        min_support=min_support,
        use_colnames=True
    )

    # Generate rules
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    # Sort by lift (interest)
    rules = rules.sort_values('lift', ascending=False)

    return rules


# Example
user_item_matrix = pd.DataFrame({
    'The Matrix': [1, 0, 1, 1, 0],
    'Inception': [1, 0, 1, 1, 0],
    'Interstellar': [0, 1, 1, 1, 1],
    'The Notebook': [0, 1, 0, 0, 1],
})

rules = mine_recommendation_rules(user_item_matrix, min_support=0.2, min_confidence=0.7)

print("Top recommendation rules:")
for idx, rule in rules.head(5).iterrows():
    antecedents = ', '.join(list(rule['antecedents']))
    consequents = ', '.join(list(rule['consequents']))

    print(f"IF user liked {antecedents}")
    print(f"  THEN recommend {consequents}")
    print(f"  (confidence: {rule['confidence']:.2f}, lift: {rule['lift']:.2f})\n")

# Output:
# IF user liked The Matrix
#   THEN recommend Inception
#   (confidence: 1.00, lift: 1.67)
```

---

## Accuracy vs. Interpretability Trade-off

### Comparison of Models

| Model | Interpretability | Accuracy | Training Time |
|-------|-----------------|----------|---------------|
| Linear Regression | ⭐⭐⭐⭐⭐ | ⭐⭐ | Fast |
| Decision Tree | ⭐⭐⭐⭐ | ⭐⭐⭐ | Fast |
| Random Forest | ⭐⭐ | ⭐⭐⭐⭐ | Medium |
| Matrix Factorization | ⭐⭐⭐ | ⭐⭐⭐⭐ | Medium |
| Deep Neural Network | ⭐ | ⭐⭐⭐⭐⭐ | Slow |

---

### When to Choose Interpretability

**Choose interpretable models when**:
1. **Regulatory compliance**: GDPR, fair lending laws
2. **High-stakes decisions**: Medical, legal, financial
3. **User trust critical**: Users need to understand recommendations
4. **Debugging required**: Frequently need to inspect model behavior
5. **Domain knowledge integration**: Experts need to validate model

**Choose accuracy when**:
1. **Competition**: Need best performance (e.g., Kaggle)
2. **Low-stakes**: Movie recommendations, music playlists
3. **Post-hoc explanations sufficient**: Can use SHAP/LIME
4. **Large datasets**: Deep learning excels with more data

---

### Hybrid Approaches

**Idea**: Combine interpretable and complex models.

**Strategies**:
1. **Cascade**: Use simple model first, complex model for edge cases
2. **Ensemble**: Combine predictions from interpretable and complex models
3. **Distillation**: Train complex model, distill into interpretable model

---

### Model Distillation

**Goal**: Transfer knowledge from complex model (teacher) to simple model (student).

**Process**:
1. Train complex model (e.g., deep neural network)
2. Generate predictions on large dataset
3. Train simple model (e.g., decision tree) to mimic predictions

**Benefits**: Retain accuracy of complex model, interpretability of simple model.

```python
def distill_model(teacher_model, X_train, n_samples=10000, max_depth=10):
    """
    Distill complex model into interpretable decision tree.

    Args:
        teacher_model: Complex trained model
        X_train: Training features
        n_samples: Number of samples for distillation
        max_depth: Max depth of student tree
    """
    # Generate predictions from teacher
    teacher_predictions = teacher_model.predict(X_train[:n_samples])

    # Train student tree to match teacher predictions
    student_model = DecisionTreeClassifier(max_depth=max_depth)
    student_model.fit(X_train[:n_samples], teacher_predictions)

    # Evaluate agreement
    agreement = (teacher_predictions == student_model.predict(X_train[:n_samples])).mean()

    print(f"Teacher-student agreement: {agreement:.2%}")

    return student_model


# Example
teacher_model = load_deep_learning_model()  # Complex model
X_train = load_training_data()

student_model = distill_model(teacher_model, X_train, n_samples=5000, max_depth=8)

# Now use student_model for interpretable predictions
# Extract rules
rules = export_text(student_model, feature_names=feature_names)
print(rules)
```

---

## Summary

**Key Takeaways**:
1. **Matrix factorization**: Interpret latent dimensions by examining top items
2. **Sparse linear models**: Lasso regression for feature selection
3. **Decision trees**: Human-readable rules, easy to visualize
4. **Rule mining**: Association rules from user behavior
5. **Trade-off**: Accuracy vs. interpretability
6. **Hybrid**: Combine models or use distillation

**Best practices**:
- Choose interpretability level based on application requirements
- Use sparse models when feature selection is important
- Visualize decision trees for stakeholder communication
- Consider distillation to get best of both worlds

---

## Practice Problems

**Problem 1**: Train a matrix factorization model on MovieLens dataset. Interpret the top-3 latent dimensions. What genres do they represent?

**Problem 2**: Compare accuracy and interpretability of:
- Lasso regression
- Decision tree (depth=5)
- Random forest (100 trees)
- Neural network (3 layers)

Which would you choose for a financial recommendation system? Why?

**Problem 3**: Implement model distillation. What is the accuracy loss when distilling a neural network into a decision tree?

**Problem 4**: Mine association rules from Amazon co-purchase data. What are the top-5 rules with highest lift?

---

## References

1. **Koren, Y., et al. (2009)**. "Matrix Factorization Techniques for Recommender Systems". *IEEE Computer*.

2. **Tibshirani, R. (1996)**. "Regression Shrinkage and Selection via the Lasso". *Journal of the Royal Statistical Society*.

3. **Quinlan, J. R. (1986)**. "Induction of Decision Trees". *Machine Learning*.

4. **Hinton, G., et al. (2015)**. "Distilling the Knowledge in a Neural Network". *NeurIPS Workshop*.

5. **Rudin, C. (2019)**. "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions". *Nature Machine Intelligence*.
