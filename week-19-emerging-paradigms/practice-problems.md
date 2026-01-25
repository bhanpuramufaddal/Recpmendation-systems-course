# Week 19: Emerging Paradigms and Future Directions - Practice Problems

## Overview
Explore cutting-edge topics: generative recommendations, foundation models, conversational systems, and grand challenges for the future.

---

## Problem 1: Diffusion Models for Recommendations
**Difficulty:** Very Hard

**Idea:** Use diffusion models (like DALL-E) to generate recommendation lists

**Approach:**
1. Train diffusion model on user interaction sequences
2. At inference: Denoise random noise → personalized item list
3. Advantages: Captures complex distributions, generates diverse recommendations

**Questions:**
1. How do you represent items for diffusion? (embeddings, discrete IDs)
2. How to condition on user preferences?
3. Computational cost vs. standard retrieval?
4. When would diffusion help? (Creative domains, diverse recommendations)

**Learning Outcomes:** Apply generative models to recommendations, understand diffusion, evaluate trade-offs

---

## Problem 2: Foundation Models for RecSys
**Difficulty:** Hard

**Vision:** One model for all recommendation tasks (like GPT for NLP)

**Challenges:**
1. **Data heterogeneity:** Movies, music, products have different features
2. **Task heterogeneity:** Rating prediction, ranking, generation
3. **Personalization:** How to incorporate billions of users?

**Questions:**
1. Design unified input representation for all domains
2. How to pre-train? (Masked prediction, contrastive learning)
3. How to personalize at scale? (Adapters, prompts)
4. Compare: Foundation model vs. domain-specific models

**Learning Outcomes:** Design unified models, handle heterogeneity, think about future architectures

---

## Problem 3: Conversational Recommendations
**Difficulty:** Hard

**Scenario:** User: "Find me a sci-fi movie like Inception but shorter"

**System components:**
1. **Intent understanding:** Parse natural language query
2. **Constraint extraction:** Genre=sci-fi, similar-to=Inception, runtime<120min
3. **Retrieval:** Find matching items
4. **Dialogue management:** Ask clarifying questions
5. **Explanation:** "I recommend Arrival because..."

**Questions:**
1. How to handle ambiguous queries?
2. When to ask clarifying questions vs. return results?
3. How to incorporate feedback? ("Not that one, something newer")
4. Design dialogue policy

**Learning Outcomes:** Build conversational systems, handle natural language, design dialogue flow

---

## Problem 4: Causal Recommendations
**Difficulty:** Very Hard

**Problem:** Correlation ≠ causation. User watched movie BECAUSE recommended, or would have watched anyway?

**Causal inference:**
- **Counterfactual:** What would user have done WITHOUT recommendation?
- **Instrumental variables:** Use randomization to estimate causal effects
- **Debiasing:** Correct for confounding factors

**Questions:**
1. Why does causality matter for recommendations?
2. How to estimate causal effects from observational data?
3. Design randomized experiment to measure causal impact
4. Compare: Causal vs. predictive models

**Learning Outcomes:** Think causally, design experiments, understand limitations of correlation

---

## Problem 5: Long-Term User Well-Being
**Difficulty:** Very Hard

**Challenge:** Optimize for long-term satisfaction, not just immediate engagement

**Approaches:**
1. **Reinforcement learning:** Maximize cumulative reward over time
2. **Counterfactual reasoning:** Estimate long-term impact of recommendations
3. **Constraints:** Ensure diversity, limit addictive content
4. **User surveys:** Direct feedback on satisfaction

**Questions:**
1. How to define and measure well-being?
2. How to balance short-term engagement and long-term satisfaction?
3. Design reward function for RL
4. Ethical considerations

**Learning Outcomes:** Think long-term, design for well-being, balance objectives

---

## Programming Exercises

### Exercise 1: LLM-Based Conversational Recommender

```python
from transformers import pipeline

class ConversationalRecommender:
    def __init__(self, llm, item_database):
        self.llm = llm
        self.database = item_database
        self.conversation_history = []

    def recommend(self, user_query):
        # Build prompt
        prompt = self.build_prompt(user_query, self.conversation_history)

        # LLM generates response
        response = self.llm(prompt)

        # Extract constraints
        constraints = self.parse_constraints(response)

        # Retrieve items
        candidates = self.database.search(constraints)

        # Rank and return
        return candidates[:10]

    def build_prompt(self, query, history):
        context = "\n".join(history)
        return f"{context}\nUser: {query}\nAssistant: I recommend"
```

---

### Exercise 2: Simulate Long-Term RL

```python
class LongTermRecommender:
    def __init__(self, policy):
        self.policy = policy

    def simulate_trajectory(self, user, T=100):
        state = user.initial_state()
        total_reward = 0
        trajectory = []

        for t in range(T):
            action = self.policy.select_action(state)
            reward_immediate = user.interact(action)

            # Measure long-term effects
            reward_longterm = user.satisfaction_change()  # Future engagement

            reward = reward_immediate + 0.5 * reward_longterm
            total_reward += (0.99 ** t) * reward  # Discounted

            state = user.update_state(action, reward)
            trajectory.append((state, action, reward))

        return total_reward, trajectory
```

---

### Exercise 3: Causal Effect Estimation

```python
def estimate_causal_effect(recommendations, outcomes, propensities):
    # Inverse propensity scoring
    treated = recommendations == 1
    control = recommendations == 0

    # Estimate treatment effect
    treated_outcome = np.mean(outcomes[treated] / propensities[treated])
    control_outcome = np.mean(outcomes[control] / propensities[control])

    causal_effect = treated_outcome - control_outcome
    return causal_effect

# Doubly robust estimator (combines IPS + prediction)
def doubly_robust(recs, outcomes, propensities, predicted_outcomes):
    treated = recs == 1
    ate = np.mean(predicted_outcomes[treated]) - np.mean(predicted_outcomes[~treated])

    # Correction term
    correction = np.mean((outcomes[treated] - predicted_outcomes[treated]) / propensities[treated]) - \
                 np.mean((outcomes[~treated] - predicted_outcomes[~treated]) / propensities[~treated])

    return ate + correction
```

---

## Discussion Questions

1. **Foundation Models:** Will one model replace all specialized recommenders? Or will domain-specific models remain?
2. **Privacy in Foundation Models:** How to personalize without compromising privacy?
3. **Generative Recommendations:** What are risks of AI-generated recommendation lists?
4. **Human-AI Collaboration:** Should AI recommend or collaborate with human curators?
5. **Ethical AI:** How to ensure recommendation systems promote societal good?
6. **Open Challenges:** What are the hardest unsolved problems in RecSys?

---

## Grand Challenges (Research Directions)

### 1. Causal Recommendations
**Goal:** Move from "users who clicked this also clicked that" to "showing this CAUSED users to click that"

**Why it matters:** Better understanding of causal mechanisms → better interventions

### 2. Long-Term Optimization
**Goal:** Optimize for user lifetime value, not just next click

**Challenges:** Delayed rewards, attribution, measuring well-being

### 3. Multi-Stakeholder Fairness
**Goal:** Fair recommendations for users, providers, and society

**Challenges:** Conflicting objectives, defining fairness, measuring impact

### 4. Sustainable Recommendations
**Goal:** Reduce carbon footprint of training and serving

**Approaches:** Efficient models, green computing, model compression

### 5. Bridging Online and Offline
**Goal:** Seamless recommendations across digital and physical experiences

**Examples:** AR shopping, location-based recommendations

---

## Final Reflection

**You've learned:**
- Classical methods (CF, MF)
- Deep learning (NCF, GNN, Transformers)
- Production systems (two-tower, ANN, MLOps)
- Industry applications (Netflix, TikTok, Airbnb, Cursor AI)
- Ethics and fairness

**Next steps:**
1. **Build:** Implement end-to-end recommendation system
2. **Research:** Explore open problems (causality, long-term optimization)
3. **Contribute:** Open-source projects (RecBole, Surprise, Cornac)
4. **Learn more:** RecSys conference, research papers, industry blogs

**Keep learning. Keep building. Keep questioning.**

---

## References
1. Zhang, S., et al. (2023). "Diffusion Recommender Model". Emerging work on generative recommendations.
2. Pearl, J. (2009). "Causality: Models, Reasoning, and Inference". Cambridge University Press.
3. Jannach, D., & Adomavicius, G. (2016). "Recommendations with a Purpose". RecSys.

---

## Congratulations!

You've completed all 19 weeks of CS 329R: Recommendation Systems.

You now have the knowledge and skills to:
- Design and implement recommendation algorithms
- Build production-scale systems
- Evaluate and improve models
- Consider ethical implications
- Contribute to the field

**The future of recommendations is yours to build.**

---

*Return to [Week 19 Main Page](README.md) | [Main Course Page](../README.md)*
