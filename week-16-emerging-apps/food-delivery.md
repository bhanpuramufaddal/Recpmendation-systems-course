# Week 16: Food Delivery Recommendations (Uber Eats, DoorDash)

## Overview

**Food delivery platforms**: Uber Eats, DoorDash, Grubhub.

**Unique challenges**:
1. **Time-sensitive**: Meal times (lunch 12-2pm, dinner 6-9pm)
2. **Contextual**: Location, weather, dietary restrictions
3. **Supply-side**: Restaurant availability, delivery time
4. **Variety-seeking**: Users want different food each day

---

## Restaurant Ranking

### Multi-Objective Optimization

**Balance**:
1. **Relevance**: User preferences (cuisine, price)
2. **Delivery time**: Faster delivery ranked higher
3. **Restaurant quality**: Ratings, reviews
4. **Diversity**: Not all same cuisine

**Combined score**:
$$\text{Score} = w_1 \cdot P(\text{order}) + w_2 \cdot \text{Quality} - w_3 \cdot \text{DeliveryTime} + w_4 \cdot \text{Diversity}$$

---

### Features

**User**:
- Past orders (cuisine preferences)
- Price sensitivity (average order value)
- Dietary restrictions (vegetarian, gluten-free)
- Location

**Restaurant**:
- Cuisine type
- Average rating
- Price range ($, $$, $$$)
- Estimated prep time
- Distance from user

**Context**:
- Time of day (breakfast, lunch, dinner)
- Day of week (weekday vs. weekend)
- Weather (rainy → comfort food)

---

## Contextual Personalization

### Time of Day

**Breakfast (7-11am)**: Coffee shops, bagels, breakfast burritos.

**Lunch (12-2pm)**: Quick meals (sandwiches, salads, fast food).

**Dinner (6-9pm)**: Full meals (pizza, Chinese, Italian).

**Late night (10pm+)**: Fast food, 24-hour diners.

**Implementation**:
```python
def time_based_boost(restaurant, hour):
    """Boost restaurants appropriate for time of day"""
    if 7 <= hour < 11 and restaurant['type'] == 'breakfast':
        return 1.5
    elif 12 <= hour < 14 and restaurant['avg_prep_time'] < 20:  # Fast lunch
        return 1.3
    elif 18 <= hour < 22 and restaurant['type'] in ['dinner', 'full_service']:
        return 1.2
    return 1.0

score *= time_based_boost(restaurant, current_hour)
```

---

### Weather

**Rainy/cold**: Comfort food (soup, pizza, pasta) ranked higher.

**Hot/sunny**: Light meals (salads, smoothies).

**Implementation**:
```python
def weather_boost(restaurant, weather):
    if weather == 'rainy' and restaurant['comfort_food']:
        return 1.2
    elif weather == 'hot' and restaurant['category'] in ['salad', 'smoothie']:
        return 1.15
    return 1.0
```

---

## Reordering and Shortcuts

### Personalized Shortcuts

**Top of app**: "Order again" with recent favorites.

**Benefits**:
- **Convenience**: One-tap reorder
- **Conversion**: Higher order rate

**Ranking favorites**:
1. Frequency (how often ordered)
2. Recency (when last ordered)
3. Rating (user's rating of restaurant)

```python
def rank_favorites(user_order_history):
    """Rank user's favorite restaurants"""
    restaurant_scores = {}

    for order in user_order_history:
        restaurant = order['restaurant']
        days_ago = (today - order['date']).days

        # Recency decay
        recency = 1 / (1 + days_ago / 30)  # Decay over 30 days

        # Frequency
        frequency = user_order_history.count(restaurant)

        # Rating
        rating = order.get('rating', 3) / 5.0

        score = 0.4 * recency + 0.3 * frequency + 0.3 * rating
        restaurant_scores[restaurant] = restaurant_scores.get(restaurant, 0) + score

    return sorted(restaurant_scores.items(), key=lambda x: x[1], reverse=True)[:5]
```

---

## Delivery Time Optimization

### ETA Prediction

**Model**: Predict delivery time based on:
- Restaurant prep time (historical average)
- Driver availability (current supply)
- Distance (travel time)
- Traffic (real-time data)

**Formula**:
$$\text{ETA} = \text{Prep time} + \text{Travel time} + \text{Buffer}$$

**Impact on ranking**: Fast ETA → higher rank.

---

### Supply-Demand Balancing

**Problem**: Busy restaurants have long wait times.

**Solution**:
1. **De-rank** overloaded restaurants (spread demand)
2. **Promote** underutilized restaurants (increase supply usage)

---

## Discovery vs. Exploitation

### Exploration

**Challenge**: Users order from same restaurants → miss new options.

**Solution**: Inject 10-20% exploratory recommendations.

**Methods**:
1. **New restaurants**: Recently added to platform
2. **Trending**: Popular among similar users
3. **Diverse cuisines**: Different from recent orders

---

### Implementation

```python
def generate_recommendations(user, k=10):
    recs = []

    # 80% exploitative (personalized)
    personalized = rank_restaurants(user)
    recs.extend(personalized[:int(k * 0.8)])

    # 20% exploratory
    exploratory = []

    # New restaurants
    new_restaurants = get_new_restaurants(user['location'], days=30)
    exploratory.extend(new_restaurants[:k//4])

    # Diverse cuisines
    recent_cuisines = [o['cuisine'] for o in user['recent_orders']]
    diverse = get_restaurants_excluding_cuisines(recent_cuisines)
    exploratory.extend(diverse[:k//4])

    recs.extend(exploratory)

    return recs[:k]
```

---

## Dietary Restrictions

### Filtering

**Hard constraints**:
- Vegetarian → exclude non-veg restaurants
- Gluten-free → only show gluten-free options
- Allergies (nuts, dairy) → filter out

**Implementation**:
```python
def apply_dietary_filters(restaurants, user_restrictions):
    """Filter restaurants by dietary restrictions"""
    filtered = []

    for restaurant in restaurants:
        if 'vegetarian' in user_restrictions and not restaurant['vegetarian_options']:
            continue
        if 'gluten_free' in user_restrictions and not restaurant['gluten_free_options']:
            continue
        filtered.append(restaurant)

    return filtered
```

---

## Promotional Strategy

### Coupons and Discounts

**Incentivize**:
1. **First-time users**: $10 off first order
2. **Lapsed users**: $5 off if haven't ordered in 30 days
3. **High-value users**: Free delivery

**Personalized offers**:
```python
def personalize_promotion(user):
    if user['order_count'] == 0:
        return "50% off first order"
    elif days_since_last_order(user) > 30:
        return "$5 off your next order"
    elif user['lifetime_value'] > 500:
        return "Free delivery this week"
    return None
```

---

## Summary

**Key Takeaways**:
1. **Contextual**: Time of day, weather, location
2. **Delivery time**: Optimize for fast ETA
3. **Reordering**: Personalized shortcuts for convenience
4. **Exploration**: 20% diverse recommendations
5. **Dietary**: Hard filters for restrictions

**Metrics**: Order conversion rate, repeat rate, user satisfaction.

---

## References

1. **Uber Eats Engineering Blog**: "Powering Uber Eats Recommendations" (2019).
2. **DoorDash Engineering Blog**: "Improving Restaurant Recommendations" (2020).
