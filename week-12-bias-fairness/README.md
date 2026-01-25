# Week 12: Bias, Fairness, and Ethics

## Overview

Recommendation systems can perpetuate and amplify biases, leading to unfair outcomes for users and providers. This week examines types of bias, debiasing techniques, fairness frameworks, and ethical considerations.

## Topics

### [1. Types of Bias](types-of-bias.md)
- **Popularity bias**: Matthew effect
- **Selection bias**: Self-selection
- **Position bias**: Click patterns
- **Conformity bias**: Social influence

### [2. Debiasing Techniques](debiasing.md)
- **Inverse Propensity Scoring (IPS)**
- Causal inference approaches
- Doubly Robust estimation
- Unbiased learning from biased feedback

### [3. Fairness in Recommendations](fairness.md)
**Consumer Fairness**:
- Demographic parity
- Equal opportunity
- Individual vs. group fairness

**Provider Fairness**:
- Exposure fairness for creators
- Calibration
- Multi-stakeholder optimization

### [4. Ethical Considerations](ethics.md)
- Filter bubbles and echo chambers
- Radicalization risks
- Privacy (GDPR implications)
- Transparency and explainability
- Addictive design patterns

## Debiasing Formula: IPS

**Problem**: Observed data biased by position, popularity

**Solution**: Weight by inverse propensity

$$\mathcal{L}_{IPS} = \frac{1}{N} \sum_{(u,i) \in O} \frac{1}{p(o=1|u,i)} \cdot \text{loss}(r_{ui}, \hat{r}_{ui})$$

where $p(o=1|u,i)$ = probability of observing interaction

## Fairness Metrics

**Demographic Parity**:
$$P(\hat{y}=1 | A=0) = P(\hat{y}=1 | A=1)$$

**Equal Opportunity**:
$$P(\hat{y}=1 | A=0, Y=1) = P(\hat{y}=1 | A=1, Y=1)$$

## Case Studies

- YouTube: Reducing borderline content and misinformation
- Spotify: Artist fairness and exposure
- Amazon: Detecting and preventing review manipulation

*Return to [Main Course Page](../README.md)*
