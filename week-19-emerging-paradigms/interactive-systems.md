# Week 19: Interactive and Conversational Recommendations

## Overview

**Interactive recommendation**: Multi-turn dialogue between user and system to refine preferences.

**Conversational recommendation**: Natural language interaction for preference elicitation.

**Key difference from static systems**:
- Static: One-shot recommendations
- Interactive: Iterative refinement through feedback
- Conversational: Natural language dialogue

**Applications**: Voice assistants (Alexa, Siri), chatbots, virtual shopping assistants.

---

## Critique-Based Recommendations

### User Critiques

**Idea**: User provides feedback on recommendations to refine results.

**Example**:
```
System: Recommends "The Matrix"
User: "Too old, show me newer movies"
System: Recommends "Dune" (2021)
User: "Similar, but more action"
System: Recommends "Mad Max: Fury Road"
```

**Critique types**:
- **Attribute-based**: "More action", "Less romance"
- **Item-based**: "Similar to X", "Different from Y"
- **Directional**: "Cheaper", "Higher rating"

---

## Summary

**Key Takeaways**:
1. **Critique-based**: Iterative refinement through user feedback
2. **Conversational**: Natural language preference elicitation
3. **Multi-armed bandits**: Balance exploration and exploitation
4. **RL-based**: Optimize long-term user satisfaction
5. **Dialogue management**: Track conversation state, generate responses

**Benefits**:
- Better cold-start handling
- More transparent recommendations
- Higher user engagement
- Personalized interaction

**Challenges**:
- Natural language understanding
- Conversation flow management
- Balancing efficiency (few turns) vs. accuracy

**Best practices**:
- Start with simple critiques (binary feedback)
- Use RL for multi-turn optimization
- Provide explanations for transparency
- Limit conversation length (3-5 turns optimal)

---

## Practice Problems

**Problem 1**: Implement critique-based system for movie recommendations. Allow users to critique by genre, year, and rating. Measure how many turns needed to find satisfactory recommendation.

**Problem 2**: Build conversational recommender using GPT-4. Compare with traditional RecSys on cold-start users.

**Problem 3**: Implement Thompson Sampling for interactive recommendation. Compare with UCB and ε-greedy.

**Problem 4**: Design RL-based dialogue system with state, actions, rewards. Train with DQN. Evaluate on average reward per episode.

---

## References

1. **Chen, L., & Pu, P. (2012)**. "Critiquing-based Recommenders: Survey and Emerging Trends". *UMUAI*.

2. **Christakopoulou, K., et al. (2016)**. "Towards Conversational Recommender Systems". *KDD*.

3. **Sun, Y., & Zhang, Y. (2018)**. "Conversational Recommender System". *SIGIR*.

4. **Lei, W., et al. (2020)**. "Estimation-Action-Reflection: Towards Deep Interaction Between Conversational and Recommender Systems". *WSDM*.
