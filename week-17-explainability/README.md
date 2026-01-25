# Week 17: Explainability and Interpretability

## Overview

Explainable recommendations build user trust, enable debugging, and support regulatory compliance. This week covers techniques for interpreting and explaining recommendations.

## Topics

### [1. Why Explainability Matters](importance.md)
- User trust and transparency
- Debugging model failures
- Regulatory compliance (GDPR, AI Act)
- Actionable feedback for users

### [2. Explainability Techniques](techniques.md)
- **Attention visualization**
- **LIME**: Local Interpretable Model-agnostic Explanations
- **SHAP**: SHapley Additive exPlanations
- **Influence functions**
- **Counterfactual explanations**
- Post-hoc rationalization

### [3. Inherently Interpretable Models](interpretable-models.md)
- Matrix factorization interpretability
- Sparse linear models
- Decision trees and rules
- Trade-offs: accuracy vs. interpretability

## Explanation Types

**Content-Based**:
- "Because you liked *The Matrix*" (similar items)

**Collaborative**:
- "Users similar to you enjoyed this" (user similarity)

**Attribute-Based**:
- "Matches your preferences: Sci-fi, Action" (feature matching)

**Social**:
- "Your friend Alice recommended this"

## SHAP for Recommendations

**Shapley Value**: Contribution of each feature to prediction

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$$

**Application**: Explain why item recommended to user

## Trade-offs

| Approach | Accuracy | Interpretability | Complexity |
|----------|----------|------------------|------------|
| Deep Neural Networks | Very High | Low | High |
| Matrix Factorization | High | Medium | Medium |
| Decision Trees | Medium | Very High | Low |
| Linear Models | Low-Medium | Very High | Low |

*Return to [Main Course Page](../README.md)*
