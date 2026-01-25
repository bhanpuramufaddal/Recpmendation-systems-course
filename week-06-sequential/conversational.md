# Week 6: Sequential Recommendations - Conversational Recommendation

## Overview

**Conversational recommendation systems** interact with users through natural language dialogue to understand preferences and provide personalized recommendations.

**Key difference from traditional RecSys**:
- **Traditional**: Passive (user views/clicks, system recommends)
- **Conversational**: Active (system asks questions, user responds)

**Examples**:
- **Alexa**: "Alexa, recommend a good Italian restaurant nearby"
- **Movie chatbot**: "What kind of movies do you like?" → "I like action movies" → "How about Mission Impossible?"

**Business impact**: Conversational systems improve user engagement by 30-50% in cold-start scenarios (source: industry reports).

---

## Learning Objectives

By the end of this section, you will:
- Understand conversational recommendation fundamentals
- Master multi-turn dialogue strategies
- Implement preference elicitation through questions
- Apply critiquing-based recommendation
- Build conversational agents for recommendations

---

## Why Conversational Recommendation?

### Advantages Over Traditional Systems

**1. Cold Start**:
- **Problem**: New users have no history
- **Solution**: Ask questions to bootstrap preferences

**Example**:
```
System: "What type of cuisine do you prefer?"
User: "Italian"
System: "Any dietary restrictions?"
User: "Vegetarian"
System: "How about this vegetarian Italian restaurant?"
```

---

**2. Preference Clarification**:
- **Problem**: User query is ambiguous
- **Solution**: Ask clarifying questions

**Example**:
```
User: "Recommend a laptop"
System: "What will you use it for? Gaming, work, or general use?"
User: "Gaming"
System: "What's your budget?"
User: "Around dollar 1500"
System: "Here are high-performance gaming laptops in your budget..."
```

---

**3. Transparency**:
- **Problem**: Users don't understand why item was recommended
- **Solution**: Explain through dialogue

**Example**:
```
System: "I recommend this restaurant because it's Italian, vegetarian-friendly, and within 2 miles of your location."
User: "Sounds good!"
```

---

**4. Exploration**:
- **Problem**: Filter bubble (only similar items recommended)
- **Solution**: Suggest diverse options, let user critique

**Example**:
```
System: "You usually like action movies. Would you like to try a thriller?"
User: "Sure!"
System: "How about 'Gone Girl'?"
```

---

## Conversational Recommendation Paradigms

### 1. Slot-Filling (Form-Based)

**Approach**: Ask structured questions to fill preference "slots".

**Example** (Restaurant recommendation):
```
Slots: {cuisine, price_range, location, dietary_restrictions}

System: "What cuisine?"
User: "Italian"
[Fills: cuisine = Italian]

System: "What's your budget? ($, $$, $$$)"
User: "$$"
[Fills: price_range = $$]

System: "Any dietary restrictions?"
User: "None"
[Fills: dietary_restrictions = None]

System: "Recommend: Restaurant X (Italian, $$, nearby, no restrictions)"
```

**Pros**: Simple, structured
**Cons**: Rigid, feels like a form

---

### 2. Critiquing

**Approach**: Show recommendation, let user critique (e.g., "cheaper", "closer").

**Example**:
```
System: "How about Restaurant A? (Italian, $$$, 5 miles)"
User: "Too expensive"
System: "How about Restaurant B? (Italian, $$, 5 miles)"
User: "Too far"
System: "How about Restaurant C? (Italian, $$, 2 miles)"
User: "Perfect!"
```

**Pros**: Intuitive, learns preferences incrementally
**Cons**: May require multiple turns

---

### 3. Free-Form Dialogue

**Approach**: Natural language conversation (like chatbot).

**Example**:
```
User: "I'm looking for a good place to eat tonight"
System: "Sure! What kind of food are you in the mood for?"
User: "Something light and healthy"
System: "How about sushi or salad?"
User: "Sushi sounds great"
System: "I recommend Sushi Bar X. They have fresh fish and a cozy atmosphere."
User: "Perfect, thanks!"
```

**Pros**: Natural, flexible
**Cons**: Requires NLU (natural language understanding), more complex

---

## Multi-Turn Recommendation Strategy

### Dialogue State Tracking

**Goal**: Maintain conversation state across turns.

**State** (at each turn):
```python
dialogue_state = {
    "user_preferences": {"cuisine": "Italian", "price": "$$"},
    "recommended_items": [restaurant_1, restaurant_2],
    "critiques": ["too expensive", "too far"],
    "turn_number": 3
}
```

---

### Strategy: Question Selection

**Problem**: Which question to ask next?

**Approaches**:

**1. Fixed Order**:
- Ask predefined questions in sequence
- Simple, but may ask unnecessary questions

**2. Entropy-Based**:
- Ask question that maximally reduces uncertainty
- **Information gain**: $IG(Q) = H(\text{before}) - H(\text{after})$

**Example**:
```
Items: 100 restaurants
  50 Italian, 30 Chinese, 20 Mexican

Question 1: "What cuisine?" (high information gain - splits evenly)
Question 2: "Indoor or outdoor?" (low gain if most are indoor)

→ Ask Question 1 first
```

---

**3. User-Driven**:
- Let user guide conversation
- More natural, but harder to optimize

---

## Preference Elicitation

### Asking Effective Questions

**Goal**: Learn user preferences with few questions.

**Types of questions**:

**1. Categorical**:
```
"What genre do you prefer? (Action, Comedy, Drama, Sci-Fi)"
```

**2. Numerical**:
```
"What's your budget? ($0-$50, $50-$100, $100+)"
```

**3. Binary**:
```
"Do you have any dietary restrictions? (Yes/No)"
```

**4. Open-ended**:
```
"What are you looking for in a laptop?"
```

---

### Active Learning for Recommendations

**Idea**: Ask questions about items to maximize learning.

**Process**:
1. Select candidate item
2. Ask: "Would you like this item?"
3. Use answer to refine user profile

**Example**:
```
System: "Would you enjoy 'The Matrix'? (Sci-Fi, Action)"
User: "Yes!"
[Update profile: likes Sci-Fi, Action]

System: "How about 'Inception'? (Sci-Fi, Thriller)"
User: "Yes!"
[Update profile: likes Sci-Fi, Action, Thriller]

System: "Would you like 'The Notebook'? (Romance, Drama)"
User: "No"
[Update profile: dislikes Romance, Drama]

→ Refined profile: Sci-Fi + Action + Thriller, avoid Romance + Drama
```

---

## Implementation: Simple Conversational RecSys

### Slot-Filling Chatbot

```python
class SlotFillingRecommender:
    def __init__(self, items, slots):
        """
        items: list of dicts with item attributes
        slots: list of slot names to fill, e.g., ["cuisine", "price", "location"]
        """
        self.items = items
        self.slots = slots
        self.filled_slots = {}
        self.current_slot_index = 0

    def next_question(self):
        """Ask next question to fill slot."""
        if self.current_slot_index >= len(self.slots):
            return None  # All slots filled

        slot = self.slots[self.current_slot_index]
        return f"What {slot} do you prefer?"

    def fill_slot(self, slot_value):
        """Fill current slot with user's answer."""
        slot = self.slots[self.current_slot_index]
        self.filled_slots[slot] = slot_value
        self.current_slot_index += 1

    def recommend(self):
        """Recommend items matching filled slots."""
        candidates = self.items

        # Filter by filled slots
        for slot, value in self.filled_slots.items():
            candidates = [item for item in candidates if item.get(slot) == value]

        # Return top candidate
        return candidates[0] if candidates else None


# Example usage
items = [
    {"name": "Restaurant A", "cuisine": "Italian", "price": "$$$", "location": "Downtown"},
    {"name": "Restaurant B", "cuisine": "Italian", "price": "$$", "location": "Uptown"},
    {"name": "Restaurant C", "cuisine": "Chinese", "price": "$$", "location": "Downtown"},
]

chatbot = SlotFillingRecommender(items, slots=["cuisine", "price", "location"])

# Conversation
print(chatbot.next_question())  # "What cuisine do you prefer?"
chatbot.fill_slot("Italian")

print(chatbot.next_question())  # "What price do you prefer?"
chatbot.fill_slot("$$")

print(chatbot.next_question())  # "What location do you prefer?"
chatbot.fill_slot("Uptown")

recommendation = chatbot.recommend()
print(f"I recommend: {recommendation['name']}")  # "Restaurant B"
```

---

### Critiquing-Based System

```python
class CritiquingRecommender:
    def __init__(self, items):
        self.items = items
        self.current_recommendation = None
        self.constraints = {}

    def initial_recommendation(self):
        """Show first recommendation."""
        self.current_recommendation = self.items[0]
        return self.current_recommendation

    def apply_critique(self, attribute, direction):
        """
        Apply user's critique.

        attribute: e.g., "price"
        direction: e.g., "cheaper" or "more expensive"
        """
        # Store constraint
        self.constraints[attribute] = direction

        # Find next recommendation satisfying constraints
        for item in self.items:
            if self._satisfies_constraints(item):
                self.current_recommendation = item
                return item

        return None  # No item satisfies all constraints

    def _satisfies_constraints(self, item):
        """Check if item satisfies all constraints."""
        for attribute, direction in self.constraints.items():
            current_value = self.current_recommendation.get(attribute)
            item_value = item.get(attribute)

            if direction == "cheaper" and item_value >= current_value:
                return False
            if direction == "more expensive" and item_value <= current_value:
                return False
            if direction == "closer" and item_value >= current_value:
                return False
            # Add more constraint types as needed

        return True


# Example
items = [
    {"name": "Restaurant A", "price": 50, "distance": 5},
    {"name": "Restaurant B", "price": 30, "distance": 5},
    {"name": "Restaurant C", "price": 30, "distance": 2},
]

critic_bot = CritiquingRecommender(items)

# Conversation
rec = critic_bot.initial_recommendation()
print(f"How about {rec['name']}? (price: ${rec['price']}, distance: {rec['distance']} miles)")
# User: "Too expensive"

rec = critic_bot.apply_critique("price", "cheaper")
print(f"How about {rec['name']}? (price: ${rec['price']}, distance: {rec['distance']} miles)")
# User: "Too far"

rec = critic_bot.apply_critique("distance", "closer")
print(f"How about {rec['name']}? (price: ${rec['price']}, distance: {rec['distance']} miles)")
# User: "Perfect!"
```

---

## Natural Language Understanding (NLU)

### Intent Recognition

**Goal**: Understand user's intent from natural language.

**Example intents**:
- `request_recommendation`: "I need a restaurant recommendation"
- `provide_preference`: "I like Italian food"
- `critique`: "Too expensive"
- `accept`: "Sounds good!"
- `reject`: "No, thanks"

---

### Slot Extraction

**Goal**: Extract entities (slots) from user's utterance.

**Example**:
```
User: "I'm looking for a cheap Italian restaurant downtown"

Extracted slots:
  - cuisine: Italian
  - price: cheap
  - location: downtown
```

---

### Implementation with Pre-Trained NLU

```python
# Simplified example using rule-based approach
import re

class SimpleNLU:
    def __init__(self):
        self.cuisine_keywords = ["Italian", "Chinese", "Mexican", "Japanese"]
        self.price_keywords = {"cheap": "$", "moderate": "$$", "expensive": "$$$"}
        self.location_keywords = ["downtown", "uptown", "nearby"]

    def extract_slots(self, utterance):
        """Extract slots from user utterance."""
        slots = {}

        # Extract cuisine
        for cuisine in self.cuisine_keywords:
            if cuisine.lower() in utterance.lower():
                slots["cuisine"] = cuisine

        # Extract price
        for price_word, price_symbol in self.price_keywords.items():
            if price_word in utterance.lower():
                slots["price"] = price_symbol

        # Extract location
        for location in self.location_keywords:
            if location in utterance.lower():
                slots["location"] = location

        return slots


# Example
nlu = SimpleNLU()
utterance = "I want a cheap Italian restaurant downtown"
slots = nlu.extract_slots(utterance)
print(f"Extracted slots: {slots}")
# {"cuisine": "Italian", "price": "$", "location": "downtown"}
```

**Production systems**: Use pre-trained models (BERT, Rasa NLU, Dialogflow).

---

## Evaluation Metrics

### Dialogue Quality

**1. Task Success Rate**:
$$\text{Success Rate} = \frac{\text{successful recommendations}}{\text{total conversations}}$$

**Successful**: User accepts recommendation.

---

**2. Average Turns to Success**:
$$\text{Avg Turns} = \frac{\sum \text{turns in successful convs}}{\text{successful conversations}}$$

**Fewer turns** = more efficient.

---

**3. User Satisfaction**:
- Survey after conversation (1-5 scale)
- Measures: Ease of use, recommendation quality, naturalness

---

### Recommendation Quality

**Same as traditional RecSys**:
- Precision, Recall, NDCG

**Difference**: Measure after dialogue ends.

---

## Real-World Applications

### 1. Amazon Alexa

**Use case**: Voice-based product recommendations.

**Example**:
```
User: "Alexa, recommend a laptop"
Alexa: "Sure! What will you use it for?"
User: "Gaming"
Alexa: "I recommend the ASUS ROG Strix. It has an RTX 3070 GPU and 16GB RAM."
User: "Add to cart"
```

---

### 2. Google Assistant

**Use case**: Local business recommendations.

**Example**:
```
User: "Find me a good sushi place"
Assistant: "I found Sushi Bar X with 4.5 stars. It's 1 mile away. Want directions?"
User: "Yes"
```

---

### 3. Movie Recommendation Chatbots

**Use case**: Help users find movies to watch.

**Example**:
```
Bot: "What genre do you feel like watching?"
User: "Something funny"
Bot: "Comedy! Do you prefer romantic comedies or silly comedies?"
User: "Silly"
Bot: "How about 'The Hangover'? It's a hilarious comedy about a wild bachelor party."
User: "I've already seen it"
Bot: "How about '21 Jump Street'? Similar humor."
User: "Perfect!"
```

---

### 4. Fashion Stylist Bots

**Use case**: Personalized outfit recommendations.

**Example**:
```
Bot: "What's the occasion?"
User: "Job interview"
Bot: "Formal or business casual?"
User: "Business casual"
Bot: "I recommend this navy blazer with khaki pants and brown shoes."
User: "Show me more options"
```

---

## Challenges and Future Directions

### Challenges

**1. Natural Language Understanding**:
- Ambiguity, typos, slang
- Requires robust NLU models

**2. Dialogue Management**:
- Handling unexpected user responses
- Maintaining context over long conversations

**3. Scalability**:
- Large item catalogs (millions of items)
- Real-time response requirements

**4. Privacy**:
- Voice data collection
- User preference storage

---

### Future Directions

**1. Large Language Models (LLMs)**:
- Use GPT-4, ChatGPT for natural dialogue
- Better understanding, generation

**2. Multimodal Conversational RecSys**:
- Combine text, voice, images
- "Show me a picture" → display image

**3. Proactive Recommendations**:
- System initiates conversation: "Hi! Based on your history, would you like to try X?"

**4. Emotionally Aware Systems**:
- Detect user mood from text/voice
- Adapt recommendations: "You seem stressed. How about a relaxing movie?"

---

## Summary

**Key Takeaways**:
1. **Conversational RecSys**: Active dialogue to understand preferences
2. **Paradigms**: Slot-filling, critiquing, free-form dialogue
3. **Benefits**: Cold start, clarification, transparency, exploration
4. **Implementation**: NLU + dialogue management + recommendation
5. **Evaluation**: Success rate, turns to success, user satisfaction

**Best Practices**:
- Start simple (slot-filling)
- Add critiquing for flexibility
- Use pre-trained NLU (Rasa, Dialogflow)
- Keep conversations short (<5 turns ideal)
- Provide explanations

**When to use**:
- **Cold start**: New users, no history
- **Complex domains**: Many attributes (real estate, jobs)
- **Voice interfaces**: Alexa, Google Assistant
- **High engagement**: Users willing to chat

**Next**: Week 7: Graph Neural Networks for recommendations.

---

## References

1. **Christakopoulou, K., Radlinski, F., & Hofmann, K. (2016)**. "Towards Conversational Recommender Systems". *KDD*.
   - Foundations of conversational RecSys

2. **Sun, Y., & Zhang, Y. (2018)**. "Conversational Recommender System". *SIGIR*.
   - Survey of conversational recommendation

3. **Li, R., et al. (2018)**. "Towards Deep Conversational Recommendations". *NeurIPS*.
   - Deep learning for conversational RecSys

4. **Chen, Q., et al. (2019)**. "Towards Knowledge-Based Recommender Dialog System". *EMNLP*.
   - Knowledge graphs + dialogue

5. **Zou, L., et al. (2020)**. "A Survey on Application of Knowledge Graph". *arXiv*.
   - Using knowledge graphs in conversational systems

---

## Practice Problems

### Problem 1: Slot-Filling

**Design slot-filling dialogue** for hotel recommendations.

**Slots**: location, check-in date, check-out date, number of guests, budget

**Solution**:
```
Q1: "Where would you like to stay?"
Q2: "What are your check-in and check-out dates?"
Q3: "How many guests?"
Q4: "What's your budget per night?"

Filter hotels matching all constraints.
```

---

### Problem 2: Critique Application

**Given**:
```
Current recommendation: Restaurant A (price: $50, distance: 5 miles)
User critique: "Too expensive"
```

**Find next recommendation** from:
```
Restaurant B (price: $30, distance: 5 miles)
Restaurant C (price: $60, distance: 2 miles)
Restaurant D (price: $20, distance: 10 miles)
```

**Solution**:
```
Restaurant B (price < $50, satisfies "cheaper" critique)
```

---

### Problem 3: Information Gain

**100 items**:
- 50 Action, 30 Comedy, 20 Drama

**Question**: "What genre?"

**Compute**: Information gain (entropy reduction).

**Solution**:
```
Before: H = -(0.5log0.5 + 0.3log0.3 + 0.2log0.2) = 1.49 bits

After:
  If Action: 50 items (no more uncertainty for this subset)
  If Comedy: 30 items
  If Drama: 20 items

Expected H after = 0 (no more uncertainty within each genre)

Information Gain = 1.49 - 0 = 1.49 bits
```
