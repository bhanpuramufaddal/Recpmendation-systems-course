# Week 12: Ethical Considerations

## Overview

**Ethics in RecSys**: Beyond technical performance, consider societal impact.

**Key issues**:
1. **Filter bubbles**: Echo chambers limiting diverse viewpoints
2. **Manipulation**: Addictive design, dark patterns
3. **Privacy**: Data collection, user tracking
4. **Transparency**: Black-box algorithms
5. **Harmful content**: Amplification of misinformation, radicalization

---

## Filter Bubbles

### Problem

**Filter bubble**: Users only see content aligned with existing beliefs.

**Mechanism**:
1. User clicks political content (Party A)
2. System recommends more Party A content
3. User never sees Party B perspectives
4. Beliefs become more extreme

**Impact**: Political polarization, reduced civic discourse.

---

### Mitigation

**1. Diversity injection**: Force 10-20% diverse viewpoints.

**2. Burst filter bubble**: Occasionally show opposite perspective.

**3. Transparency**: "Why this recommendation?" with diverse options.

```python
def diversify_recommendations(ranked_items, user_interests, diversity_ratio=0.2, k=10):
    """Inject diverse items to burst filter bubble"""
    n_diverse = int(k * diversity_ratio)
    n_similar = k - n_diverse

    # Take top similar items
    similar = ranked_items[:n_similar]

    # Find diverse items (low overlap with user interests)
    diverse = [item for item in ranked_items
               if item not in user_interests][:n_diverse]

    # Interleave
    result = []
    for i in range(k):
        if i % (1 / diversity_ratio) == 0 and diverse:
            result.append(diverse.pop(0))
        elif similar:
            result.append(similar.pop(0))

    return result
```

---

## Addictive Design

### Problem

**Optimization for engagement** → addictive patterns.

**Examples**:
- Autoplay next video (YouTube, Netflix)
- Infinite scroll (TikTok, Instagram)
- Notification manipulation

**Impact**: Screen addiction, mental health issues.

---

### Ethical Design

**Time well spent** (Center for Humane Technology):

**Principles**:
1. **Respect user time**: Don't maximize time-on-site
2. **Support goals**: Help users achieve their goals
3. **No dark patterns**: Don't manipulate

**Example**: YouTube "Take a break" reminders.

---

## Privacy

### Data Collection

**RecSys requires**: User interactions, demographics, browsing history.

**Privacy risks**:
- De-anonymization (Netflix Prize incident)
- Data breaches
- Third-party selling

---

### Privacy-Preserving Techniques

**1. Differential Privacy**:
- Add noise to data before training
- Guarantee: Individual data doesn't significantly affect output

$$P(M(D) \in S) \leq e^\epsilon \cdot P(M(D') \in S)$$

where $D, D'$ differ by one record.

**2. Federated Learning**:
- Train model on user devices
- Only send model updates (not raw data)

**3. Homomorphic Encryption**:
- Compute on encrypted data
- Decrypt only final result

---

### GDPR Compliance

**EU General Data Protection Regulation**:

**Requirements**:
1. **Consent**: Explicit opt-in for data collection
2. **Right to access**: Users can request their data
3. **Right to deletion**: "Right to be forgotten"
4. **Right to explanation**: Explain algorithmic decisions

**Impact on RecSys**: Must explain recommendations, allow data deletion.

---

## Transparency & Explainability

### Black Box Problem

**Issue**: Users don't understand why they see recommendations.

**Consequences**:
- Distrust
- Can't contest biased recommendations
- Regulatory issues (GDPR Article 22)

---

### Explanations

**Types**:
1. **Content-based**: "Because you liked similar items"
2. **Collaborative**: "Users like you also liked..."
3. **Hybrid**: "Based on your interest in X and popularity in Y"

**Example** (Spotify Discover Weekly):
> "We picked these songs because you listen to [Artist A] and [Artist B], and fans of those artists also love [Artist C]."

---

## Harmful Content

### Radicalization

**Problem**: Recommendation algorithms amplify extremist content.

**Mechanism** (YouTube):
1. User watches political video
2. Algorithm recommends more extreme version (higher engagement)
3. User progressively exposed to radical content

**Evidence**: Studies link YouTube recommendations to radicalization.

---

### Misinformation

**Problem**: False information spreads faster than truth.

**RecSys role**: Viral misinformation gets amplified (high engagement).

**Example** (COVID-19): Conspiracy theories recommended alongside news.

---

### Mitigation

**1. Content moderation**: Human reviewers + automated detection.

**2. Demonetization**: Reduce incentive for clickbait/misinformation.

**3. Context labels**: "Checked by fact-checkers", "Missing context".

**4. Algorithmic changes**: De-rank borderline content.

```python
def content_safety_score(item, safety_classifier):
    """Adjust ranking based on content safety"""
    base_score = item['relevance_score']
    safety_score = safety_classifier.predict(item['content'])  # 0-1

    # Penalize unsafe content
    if safety_score < 0.5:
        return base_score * safety_score
    return base_score
```

---

## Algorithmic Accountability

### Auditing

**Internal audits**: Regular fairness/bias checks.

**External audits**: Third-party review (e.g., AlgorithmWatch).

**Metrics**:
- Demographic parity across groups
- Exposure distribution (Gini)
- Harmful content amplification rate

---

### Regulation

**Emerging laws**:
- EU AI Act: High-risk AI systems (including RecSys) must be auditable
- US Algorithmic Accountability Act: Require impact assessments

---

## Case Study: YouTube 2019 Changes

### Problem

**Criticism**: YouTube radicalization pipeline (moderate → extreme content).

### Response

**Policy changes**:
1. **Borderline content**: Reduce recommendations of conspiracy theories, misinformation
2. **Information panels**: Add context from authoritative sources
3. **Raise authoritative voices**: Boost news from credible sources

**Results**:
- 70% reduction in borderline content watch time
- Increased news from authoritative sources
- User satisfaction stable (no engagement loss)

**Lesson**: Ethical changes don't always harm engagement.

---

## Summary

**Key Takeaways**:
1. **Filter bubbles**: Inject diversity, burst echo chambers
2. **Addictive design**: Respect user time, no dark patterns
3. **Privacy**: Differential privacy, federated learning, GDPR
4. **Transparency**: Explainable recommendations
5. **Harmful content**: Moderation, de-ranking, context labels
6. **Accountability**: Audits, regulation

**Ethical AI Principles** (ACM Code of Ethics):
- Avoid harm
- Be fair and not discriminate
- Respect privacy
- Be transparent

**Next**: Production systems and MLOps.

---

## References

1. **Tufekci, Z. (2018)**. "YouTube, the Great Radicalizer". *New York Times*.
2. **Zuboff, S. (2019)**. *The Age of Surveillance Capitalism*. Public Affairs.
3. **O'Neil, C. (2016)**. *Weapons of Math Destruction*. Crown.
4. **Solove, D. J. (2021)**. "The Myth of the Privacy Paradox". *George Washington Law Review*.
