# Week 17: Explainability and Interpretability - Practice Problems

## Overview
Master explainability techniques (LIME, SHAP), attention visualization, and building trust through interpretable recommendations.

---

## Problem 1: Types of Explanations
**Difficulty:** Easy

**Scenario:** Recommend "Inception" to user

**Explanation types:**
1. **Content-based:** "Because you liked 'The Matrix'" (similar content)
2. **Collaborative:** "Users like you enjoyed this"
3. **Feature-based:** "Sci-fi, Action, Christopher Nolan"
4. **Social:** "Your friend Alice recommended this"

**Questions:**
1. Which is most convincing to users?
2. Which is easiest to generate?
3. Which requires what model type?
4. Design multi-faceted explanation combining types

**Learning Outcomes:** Understand explanation types, choose appropriate explanations, design user-facing explanations

---

## Problem 2: SHAP for Feature Importance
**Difficulty:** Hard

**Shapley values** quantify each feature's contribution to a prediction

**Given prediction:** Score = 4.5
**Features:** [genre=sci-fi, director=Nolan, year=2010, runtime=148min]

**SHAP values:**
- genre: +0.8
- director: +0.5
- year: -0.1
- runtime: +0.2

**Questions:**
1. Interpret: Which feature mattered most?
2. How do you compute SHAP values?
3. Computational cost? (Exponential in features)
4. Generate explanation text from SHAP values

**Learning Outcomes:** Compute SHAP values, interpret contributions, generate explanations

---

## Problem 3: Attention Visualization
**Difficulty:** Medium

**Transformer-based recommender:** Attention weights show which past items influenced prediction

**User history:** [Inception, The Matrix, Interstellar]
**Attention weights:** [0.5, 0.3, 0.2]

**Recommendation:** "Tenet"

**Explanation:** "We recommended Tenet because you loved Inception (50%), The Matrix (30%), and Interstellar (20%)"

**Questions:**
1. How do you extract attention weights?
2. How to visualize for users?
3. Are attention weights causal explanations?
4. Can attention be misleading?

**Learning Outcomes:** Visualize attention, interpret weights, recognize limitations

---

## Problem 4: LIME for Black-Box Models
**Difficulty:** Hard

**LIME:** Explain any model by approximating locally with interpretable model

**Algorithm:**
1. Perturb input (e.g., remove features)
2. Get predictions on perturbations
3. Fit linear model on perturbed data
4. Linear coefficients = feature importances

**Questions:**
1. Why perturb locally instead of globally?
2. How do you perturb recommendation features?
3. How many perturbations needed?
4. Compare LIME vs. SHAP

**Learning Outcomes:** Implement LIME, understand local approximations, compare methods

---

## Problem 5: Counterfactual Explanations
**Difficulty:** Hard

**Counterfactual:** "If you had rated action movies higher, we would have recommended X"

**Given:**
- Current: User rated action=2, sci-fi=5, drama=3 → Recommended "Arrival" (sci-fi drama)
- Counterfactual: If action=5, sci-fi=5, drama=3 → Would recommend "Blade Runner" (sci-fi action)

**Questions:**
1. How do you generate counterfactuals?
2. What makes a good counterfactual? (minimal changes, actionable)
3. Design algorithm to find nearest counterfactual
4. How to present to users?

**Learning Outcomes:** Generate counterfactuals, design actionable explanations, optimize for user understanding

---

## Programming Exercises

### Exercise 1: Implement SHAP for Recommendations

```python
import shap

# Train model
model = train_recommendation_model(data)

# Explain prediction
explainer = shap.TreeExplainer(model)  # For tree models
shap_values = explainer.shap_values(user_features)

# Visualize
shap.force_plot(explainer.expected_value, shap_values, user_features)
shap.summary_plot(shap_values, user_features)
```

---

### Exercise 2: Attention-Based Explanations

```python
class ExplainableRecommender(nn.Module):
    def __init__(self):
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=1)
        # ... other layers ...

    def forward(self, user_history):
        # ... encoding ...

        # Compute attention (which past items matter)
        output, attention_weights = self.attention(query, key, value)

        return predictions, attention_weights

# Generate explanation
predictions, attn = model(user_history)
top_influencers = attn.topk(3)  # Top 3 influential items
explanation = f"Recommended because you liked {items[top_influencers[0]]}"
```

---

### Exercise 3: LIME Implementation

```python
def explain_with_lime(model, user_features, n_samples=1000):
    # Generate perturbations
    perturbations = []
    for _ in range(n_samples):
        perturbed = user_features.copy()
        # Randomly zero out features
        mask = np.random.binomial(1, 0.5, size=len(perturbed))
        perturbed = perturbed * mask
        perturbations.append(perturbed)

    # Get predictions on perturbations
    predictions = [model.predict(p) for p in perturbations]

    # Fit linear model
    from sklearn.linear_model import Ridge
    explainer = Ridge(alpha=1.0)
    explainer.fit(perturbations, predictions)

    # Coefficients = feature importances
    importances = explainer.coef_
    return importances
```

---

### Exercise 4: Generate Counterfactual Explanations

```python
def find_counterfactual(model, user_features, target_item, max_changes=3):
    # Optimize to find minimal feature changes that flip recommendation
    best_cf = None
    min_distance = float('inf')

    for feature_subset in itertools.combinations(range(len(user_features)), max_changes):
        # Try changing these features
        cf_features = user_features.copy()

        # Grid search over feature values
        for values in grid_search_values:
            for i, feat_idx in enumerate(feature_subset):
                cf_features[feat_idx] = values[i]

            # Check if this leads to target recommendation
            if model.predict(cf_features) == target_item:
                distance = np.linalg.norm(cf_features - user_features)
                if distance < min_distance:
                    best_cf = cf_features
                    min_distance = distance

    return best_cf, min_distance
```

---

## Discussion Questions

1. **Trust:** Do explanations increase user trust? Or just give illusion of understanding?
2. **Faithfulness:** Attention weights may not reflect true reasoning. How to validate?
3. **Simplicity vs. Accuracy:** Simple explanations are preferred, but may not be accurate
4. **Actionability:** Should explanations help users change recommendations?
5. **Regulation:** GDPR "right to explanation". How to comply?
6. **Gaming:** Can users game the system if they understand it?

---

## References
1. Ribeiro, M. T., et al. (2016). "Why should I trust you?: Explaining the predictions of any classifier". KDD. (LIME)
2. Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions". NIPS. (SHAP)
3. Tintarev, N., & Masthoff, J. (2012). "Evaluating the effectiveness of explanations for recommender systems". User Modeling and User-Adapted Interaction.

---

*Return to [Week 17 Main Page](README.md)*
