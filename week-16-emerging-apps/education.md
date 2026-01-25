# Week 16: Education Platforms (Coursera, Khan Academy)

## Overview

**Online education platforms**: Coursera (100M+ learners), Udacity, Khan Academy, edX.

**Recommendation surfaces**:
1. **Course recommendations**: Suggest courses based on goals
2. **Learning paths**: Sequential course progression
3. **Next lesson/video**: Within-course recommendations
4. **Peer learners**: Connect with similar students
5. **Career guidance**: Courses for target jobs

**Unique challenges**:
1. **Knowledge prerequisites**: Must learn X before Y
2. **Skill levels**: Beginner vs. advanced content
3. **Goal diversity**: Career change vs. hobby vs. degree
4. **Completion rates**: Low engagement (5-10% complete MOOCs)
5. **Time constraints**: Balancing commitment with availability

---

## Course Recommendations

### Goal-Based Filtering

**Problem**: Different learners have different goals.

**Goal types**:
- **Career change**: Software engineering, data science
- **Skill upgrade**: Learn new framework (React, PyTorch)
- **Academic**: Supplement university courses
- **Hobby**: Photography, music theory

**Approach**: Explicit goal collection + personalization.

```python
def recommend_courses(user, k=10):
    """
    Recommend courses based on user goals.
    """
    goal = user.goal  # 'career_change', 'skill_upgrade', etc.
    target_role = user.target_role  # 'Data Scientist', 'Web Developer', etc.

    # Get courses relevant to goal
    if goal == 'career_change':
        relevant_courses = get_career_path_courses(target_role)
    elif goal == 'skill_upgrade':
        relevant_courses = get_advanced_courses(user.current_skills)
    else:
        relevant_courses = get_all_courses()

    # Score courses
    scored_courses = []
    for course in relevant_courses:
        score = 0

        # Skill match
        if user.has_prerequisites(course):
            score += 1.0
        else:
            score -= 0.5  # Penalize if missing prerequisites

        # Difficulty match
        if course.difficulty == user.skill_level:
            score += 0.5

        # Time commitment match
        if course.hours_per_week <= user.available_hours:
            score += 0.3

        # Popularity (social proof)
        score += 0.1 * (course.enrollments / 10000)

        scored_courses.append((course, score))

    # Sort by score
    scored_courses.sort(key=lambda x: x[1], reverse=True)
    return [course for course, _ in scored_courses[:k]]


def get_career_path_courses(target_role):
    """
    Get courses for career path.
    """
    career_paths = {
        'Data Scientist': [
            'Python for Data Science',
            'Statistics Fundamentals',
            'Machine Learning',
            'Deep Learning',
            'SQL and Databases',
            'Data Visualization'
        ],
        'Web Developer': [
            'HTML & CSS',
            'JavaScript Fundamentals',
            'React.js',
            'Node.js',
            'Databases (SQL & NoSQL)',
            'Web Security'
        ]
        # ... more roles
    }

    course_titles = career_paths.get(target_role, [])
    return find_courses_by_titles(course_titles)
```

---

### Collaborative Filtering

**Approach**: Users with similar course history have similar interests.

**Matrix factorization** on user-course enrollment matrix:

```python
import torch
import torch.nn as nn

class CourseMF(nn.Module):
    def __init__(self, n_users, n_courses, embedding_dim=64):
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.course_embeddings = nn.Embedding(n_courses, embedding_dim)

        # Bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.course_bias = nn.Embedding(n_courses, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_ids, course_ids):
        user_emb = self.user_embeddings(user_ids)
        course_emb = self.course_embeddings(course_ids)

        # Dot product + biases
        interaction = (user_emb * course_emb).sum(dim=1)
        user_b = self.user_bias(user_ids).squeeze()
        course_b = self.course_bias(course_ids).squeeze()

        prediction = interaction + user_b + course_b + self.global_bias
        return prediction


# Training with completion rate as target
model = CourseMF(n_users=1000000, n_courses=10000, embedding_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(10):
    for user_ids, course_ids, completion_rates in train_loader:
        # completion_rates: 0-1 (fraction of course completed)
        predictions = model(user_ids, course_ids)

        loss = criterion(predictions, completion_rates)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Prerequisite Knowledge Modeling

### Knowledge Graph

**Goal**: Model prerequisite relationships between concepts.

**Example**:
```
Calculus → Linear Algebra → Machine Learning
Python Basics → NumPy → Pandas → Data Analysis
```

**Implementation**:
```python
import networkx as nx

def build_knowledge_graph():
    """
    Build directed graph of prerequisite relationships.
    """
    G = nx.DiGraph()

    # Add edges: prerequisite → advanced
    edges = [
        ('Python Basics', 'NumPy'),
        ('NumPy', 'Pandas'),
        ('Pandas', 'Data Analysis'),
        ('Calculus', 'Linear Algebra'),
        ('Linear Algebra', 'Machine Learning'),
        ('Python Basics', 'Web Development'),
        # ... many more
    ]

    G.add_edges_from(edges)
    return G


def check_prerequisites(user_knowledge, course, kg):
    """
    Check if user has prerequisites for course.
    """
    course_prerequisites = get_course_prerequisites(course, kg)

    # Check if user has all prerequisites
    missing = course_prerequisites - user_knowledge
    return len(missing) == 0, missing


def get_course_prerequisites(course, kg):
    """
    Get all prerequisites for course (transitive).
    """
    prerequisites = set()

    # Get all ancestors in knowledge graph
    course_concepts = course.concepts

    for concept in course_concepts:
        if concept in kg:
            ancestors = nx.ancestors(kg, concept)
            prerequisites.update(ancestors)

    return prerequisites


# Example
kg = build_knowledge_graph()
user_knowledge = {'Python Basics', 'NumPy'}
course = Course(name='Data Analysis', concepts=['Pandas', 'Data Visualization'])

has_prereqs, missing = check_prerequisites(user_knowledge, course, kg)
if not has_prereqs:
    print(f"Missing prerequisites: {missing}")
    # Output: Missing prerequisites: {'Pandas'}
```

---

### Adaptive Prerequisite Testing

**Goal**: Assess user knowledge before recommending courses.

**Approach**: Short diagnostic quiz.

```python
def adaptive_assessment(user, concept, max_questions=10):
    """
    Adaptive quiz to assess user knowledge of concept.
    """
    questions = get_questions_for_concept(concept)

    # Sort by difficulty
    questions.sort(key=lambda q: q.difficulty)

    score = 0
    for i, question in enumerate(questions[:max_questions]):
        answer = user.answer(question)

        if answer.is_correct():
            score += 1

            # If user gets easy questions right, skip to harder ones
            if i < 3 and score == i + 1:
                questions = [q for q in questions if q.difficulty >= 'medium']

        else:
            # If user struggles, stay at current difficulty
            break

    # Estimate proficiency
    proficiency = score / max_questions
    return proficiency


def recommend_with_assessment(user, course):
    """
    Assess prerequisites before recommending course.
    """
    prerequisites = course.prerequisites

    for prereq in prerequisites:
        proficiency = adaptive_assessment(user, prereq)

        if proficiency < 0.6:  # Below threshold
            # Recommend prerequisite course
            prereq_course = find_course_for_concept(prereq)
            return prereq_course

    # User has prerequisites, recommend target course
    return course
```

---

## Learning Path Personalization

### Sequential Course Recommendations

**Goal**: Recommend next course in learning path.

**Approach**: Markov chains on course transitions.

```python
def build_transition_model(user_histories):
    """
    Build Markov chain of course transitions.
    """
    transitions = {}

    for history in user_histories:
        for i in range(len(history) - 1):
            current_course = history[i]
            next_course = history[i + 1]

            if current_course not in transitions:
                transitions[current_course] = {}

            if next_course not in transitions[current_course]:
                transitions[current_course][next_course] = 0

            transitions[current_course][next_course] += 1

    # Normalize to probabilities
    for current_course, next_courses in transitions.items():
        total = sum(next_courses.values())
        for next_course in next_courses:
            transitions[current_course][next_course] /= total

    return transitions


def recommend_next_course(user_history, transitions, k=5):
    """
    Recommend next course based on past courses.
    """
    last_course = user_history[-1]

    if last_course not in transitions:
        return []

    # Get next courses with probabilities
    next_courses = transitions[last_course]

    # Sort by probability
    sorted_courses = sorted(next_courses.items(),
                           key=lambda x: x[1],
                           reverse=True)

    return [course for course, _ in sorted_courses[:k]]


# Example
user_histories = [
    ['Python Basics', 'Data Analysis', 'Machine Learning'],
    ['Python Basics', 'Web Development', 'React.js'],
    ['Python Basics', 'Data Analysis', 'SQL'],
    # ... more histories
]

transitions = build_transition_model(user_histories)
user_history = ['Python Basics', 'Data Analysis']

next_courses = recommend_next_course(user_history, transitions, k=3)
print(next_courses)
# Output: ['Machine Learning', 'SQL', 'Deep Learning']
```

---

### Goal-Directed Path Planning

**Goal**: Find optimal path from current skills to target goal.

```python
def find_learning_path(current_skills, target_skill, kg, course_catalog):
    """
    Find shortest learning path from current to target skills.
    """
    # Find shortest path in knowledge graph
    paths = []

    for current_skill in current_skills:
        if current_skill in kg and target_skill in kg:
            try:
                path = nx.shortest_path(kg, current_skill, target_skill)
                paths.append(path)
            except nx.NetworkXNoPath:
                continue

    if not paths:
        return []

    # Get shortest path
    shortest_path = min(paths, key=len)

    # Map concepts to courses
    learning_path = []
    for concept in shortest_path[1:]:  # Skip first (current skill)
        course = find_course_for_concept(concept, course_catalog)
        if course:
            learning_path.append(course)

    return learning_path


# Example
kg = build_knowledge_graph()
course_catalog = load_courses()

current_skills = ['Python Basics']
target_skill = 'Machine Learning'

path = find_learning_path(current_skills, target_skill, kg, course_catalog)
print([c.name for c in path])
# Output: ['Linear Algebra', 'Calculus', 'Machine Learning Foundations']
```

---

## Difficulty Adaptation

### Skill Level Inference

**Goal**: Infer user's skill level from behavior.

**Signals**:
- Quiz scores
- Time to complete exercises
- Help requests
- Video replay frequency

```python
def infer_skill_level(user_behavior):
    """
    Infer skill level from user behavior.
    """
    score = 0

    # Quiz performance
    avg_quiz_score = user_behavior.avg_quiz_score
    score += avg_quiz_score

    # Completion speed (relative to average)
    speed_ratio = user_behavior.avg_completion_time / global_avg_completion_time

    if speed_ratio < 0.8:  # Faster than average
        score += 0.2
    elif speed_ratio > 1.2:  # Slower than average
        score -= 0.2

    # Help requests (fewer = higher skill)
    help_ratio = user_behavior.help_requests / user_behavior.lessons_completed

    if help_ratio < 0.1:
        score += 0.1
    elif help_ratio > 0.3:
        score -= 0.1

    # Map score to level
    if score >= 0.8:
        return 'advanced'
    elif score >= 0.5:
        return 'intermediate'
    else:
        return 'beginner'


def adapt_content_difficulty(user, lesson):
    """
    Adapt lesson difficulty based on user skill level.
    """
    skill_level = infer_skill_level(user.behavior)

    if skill_level == 'advanced':
        # Skip basic explanations, show advanced examples
        return lesson.advanced_version
    elif skill_level == 'beginner':
        # Include more explanations, simpler examples
        return lesson.beginner_version
    else:
        return lesson.standard_version
```

---

### Spaced Repetition

**Goal**: Optimize review timing for retention.

**Approach**: SM-2 algorithm (SuperMemo).

```python
def update_review_schedule(card, quality):
    """
    Update review schedule based on recall quality.

    Args:
        card: Flashcard with current interval and ease factor
        quality: 0-5 (0=wrong, 5=perfect recall)
    """
    if quality < 3:  # Failed to recall
        card.interval = 1  # Review tomorrow
    else:
        if card.repetitions == 0:
            card.interval = 1
        elif card.repetitions == 1:
            card.interval = 6
        else:
            card.interval = card.interval * card.ease_factor

    # Update ease factor
    card.ease_factor = card.ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))

    # Minimum ease factor
    if card.ease_factor < 1.3:
        card.ease_factor = 1.3

    card.repetitions += 1
    card.next_review = today + timedelta(days=card.interval)

    return card


# Example usage for course review
def schedule_lesson_review(user, lesson, recall_quality):
    """
    Schedule when user should review lesson.
    """
    if lesson.id not in user.review_cards:
        user.review_cards[lesson.id] = ReviewCard(
            interval=1,
            ease_factor=2.5,
            repetitions=0
        )

    card = user.review_cards[lesson.id]
    updated_card = update_review_schedule(card, recall_quality)

    user.review_cards[lesson.id] = updated_card
```

---

## Completion Likelihood Prediction

### Features

**Goal**: Predict if user will complete course.

**Features**:
- User engagement (login frequency, time spent)
- Course characteristics (length, difficulty)
- User's past completion rate
- Social factors (peer progress)

```python
import torch
import torch.nn as nn

class CompletionPredictor(nn.Module):
    def __init__(self, n_features=20):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)


def extract_features(user, course):
    """
    Extract features for completion prediction.
    """
    features = []

    # User features
    features.append(user.avg_completion_rate)
    features.append(user.login_frequency)
    features.append(user.avg_time_per_session)
    features.append(len(user.completed_courses))

    # Course features
    features.append(course.duration_weeks)
    features.append(course.difficulty_score)
    features.append(course.avg_rating)
    features.append(course.global_completion_rate)

    # User-course match
    features.append(int(user.has_prerequisites(course)))
    features.append(course.difficulty_score / user.skill_level)

    # Social features
    peers = get_similar_learners(user)
    peer_completion = sum(p.completed(course) for p in peers) / len(peers)
    features.append(peer_completion)

    # Temporal features
    features.append(user.days_since_last_course)
    features.append(course.days_since_launch)

    return torch.tensor(features, dtype=torch.float32)


# Training
model = CompletionPredictor(n_features=20)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(10):
    for users, courses, completions in train_loader:
        features = torch.stack([extract_features(u, c)
                               for u, c in zip(users, courses)])

        predictions = model(features).squeeze()
        loss = criterion(predictions, completions.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Prediction
def predict_completion(user, course, model):
    features = extract_features(user, course)
    with torch.no_grad():
        prob = model(features.unsqueeze(0)).item()
    return prob
```

---

## Peer Learner Recommendations

### Finding Similar Learners

**Goal**: Connect users with similar learning goals/progress.

**Approach**: Clustering on user features.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_learners(users, n_clusters=100):
    """
    Cluster learners by learning behavior.
    """
    # Extract features
    features = []
    for user in users:
        feature_vector = [
            user.skill_level,
            user.learning_pace,  # Lessons per week
            user.goal_orientation,  # Career vs. hobby
            user.preferred_difficulty,
            len(user.completed_courses),
            user.avg_quiz_score,
            # ... more features
        ]
        features.append(feature_vector)

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)

    # Assign clusters
    for user, cluster in zip(users, clusters):
        user.cluster = cluster

    return clusters


def recommend_peers(user, all_users, k=10):
    """
    Recommend peers in same cluster.
    """
    # Find users in same cluster
    cluster_users = [u for u in all_users
                     if u.cluster == user.cluster and u.id != user.id]

    # Score by similarity
    scored_users = []
    for other_user in cluster_users:
        # Course overlap
        common_courses = set(user.courses) & set(other_user.courses)
        score = len(common_courses)

        # Similar progress
        progress_diff = abs(user.progress - other_user.progress)
        score += (1 - progress_diff)

        scored_users.append((other_user, score))

    # Sort by score
    scored_users.sort(key=lambda x: x[1], reverse=True)
    return [user for user, _ in scored_users[:k]]
```

---

## Instructor Quality Signals

### Rating Prediction

**Goal**: Predict instructor rating for course recommendations.

**Features**:
- Past course ratings
- Student feedback sentiment
- Engagement metrics (response time, forum activity)
- Credentials and experience

```python
def instructor_quality_score(instructor):
    """
    Compute instructor quality score.
    """
    score = 0

    # Average rating across courses
    avg_rating = instructor.avg_rating
    score += avg_rating / 5.0  # Normalize to 0-1

    # Number of courses taught (experience)
    experience = len(instructor.courses)
    score += min(experience / 10, 1.0)  # Cap at 10 courses

    # Student feedback sentiment
    sentiment = analyze_feedback_sentiment(instructor.reviews)
    score += sentiment  # 0-1

    # Engagement (forum response rate)
    response_rate = instructor.forum_response_rate
    score += response_rate

    # Credentials (PhD, industry experience)
    if instructor.has_phd:
        score += 0.5

    if instructor.years_industry_experience > 5:
        score += 0.5

    return score / 5.0  # Normalize to 0-1


def analyze_feedback_sentiment(reviews):
    """
    Sentiment analysis on instructor reviews.
    """
    from transformers import pipeline

    sentiment_analyzer = pipeline("sentiment-analysis")

    positive_count = 0
    for review in reviews:
        result = sentiment_analyzer(review.text)[0]
        if result['label'] == 'POSITIVE':
            positive_count += 1

    return positive_count / len(reviews) if reviews else 0.5
```

---

## Certification Value and ROI

### Career Outcome Prediction

**Goal**: Predict career impact of certification.

**Data**:
- Job placement rates
- Salary increase post-certification
- Time to employment

```python
def estimate_roi(certification, user):
    """
    Estimate return on investment for certification.
    """
    # Cost
    cost = certification.price + (user.hourly_rate * certification.hours_required)

    # Benefit: Expected salary increase
    alumni = certification.alumni

    # Filter alumni similar to user
    similar_alumni = [a for a in alumni
                     if a.background_similar_to(user)]

    if not similar_alumni:
        return 0

    # Average salary increase
    salary_increases = [a.salary_after - a.salary_before
                       for a in similar_alumni]

    avg_salary_increase = sum(salary_increases) / len(salary_increases)

    # Estimate benefit over 2 years
    benefit = avg_salary_increase * 2

    # ROI
    roi = (benefit - cost) / cost

    return roi


def recommend_certifications(user, k=5):
    """
    Recommend certifications with highest ROI.
    """
    all_certs = get_certifications()

    scored_certs = []
    for cert in all_certs:
        # Check prerequisites
        if not user.has_prerequisites(cert):
            continue

        # Estimate ROI
        roi = estimate_roi(cert, user)

        # Time feasibility
        if cert.hours_required <= user.available_hours_per_week * 52:
            scored_certs.append((cert, roi))

    # Sort by ROI
    scored_certs.sort(key=lambda x: x[1], reverse=True)
    return [cert for cert, _ in scored_certs[:k]]
```

---

## Summary

**Key Takeaways**:
1. **Course recommendations**: Goal-based filtering + CF
2. **Prerequisite modeling**: Knowledge graph for dependencies
3. **Learning paths**: Sequential recommendations via Markov chains
4. **Difficulty adaptation**: Infer skill level, adapt content
5. **Completion prediction**: ML model for engagement likelihood
6. **Peer matching**: Clustering learners by behavior
7. **Instructor quality**: Multi-signal scoring
8. **ROI estimation**: Career outcome prediction

**Metrics**: Completion rate, engagement time, skill assessment scores, job placement rate.

---

## References

1. **Coursera Engineering Blog**: "Personalized Course Recommendations" (2018).
2. **Piech, C., et al. (2015)**. "Deep Knowledge Tracing". *NeurIPS*.
3. **Pardos, Z., et al. (2014)**. "Affective States and State Tests". *LAK*.
