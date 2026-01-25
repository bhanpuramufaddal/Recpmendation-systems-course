# Week 16: GitHub and Developer Tools

## Overview

**GitHub**: 100M+ developers, 330M+ repositories.

**Recommendation surfaces**:
1. **Repository recommendations**: Discover new repos based on interests
2. **Code completion**: GitHub Copilot (LLM-based)
3. **Trending**: Real-time detection of popular repos
4. **Developer network**: "People to follow" suggestions
5. **Package/dependency**: Recommend libraries and tools
6. **Learning paths**: Technology roadmaps

**Unique challenges**:
1. **Code understanding**: Semantic comprehension of source code
2. **Developer expertise**: Skill level inference from activity
3. **Technology diversity**: 500+ programming languages
4. **Temporal relevance**: Technologies become outdated quickly
5. **Quality signals**: Stars ≠ actual quality

---

## Repository Recommendations

### Activity-Based Filtering

**Goal**: Recommend repos based on developer activity.

**Signals**:
- **Stars**: Bookmarked repos (explicit interest)
- **Forks**: Intention to contribute
- **Commits**: Active contributions
- **Pull requests**: Collaboration patterns
- **Issues**: Engagement with projects

**Weighted scoring**:
```python
def activity_score(user, repo):
    """
    Score repo relevance based on user activity.
    """
    score = 0

    if user.starred(repo):
        score += 1.0  # Explicit interest

    if user.forked(repo):
        score += 2.0  # Strong signal

    if user.contributed_to(repo):
        score += 3.0  # Highest signal

    # Recent activity weighted higher
    days_ago = (today - user.last_activity(repo)).days
    recency = 1 / (1 + days_ago / 30)

    return score * recency
```

---

### Collaborative Filtering on Stars

**Approach**: Users who starred similar repos have similar interests.

**Matrix factorization** on user-repo star matrix:
$$R \approx U^T V$$

where:
- $U \in \mathbb{R}^{d \times |users|}$ = user embeddings
- $V \in \mathbb{R}^{d \times |repos|}$ = repo embeddings

**Prediction**: $\hat{r}_{ui} = u_i^T v_u$

**Implementation**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RepoMF(nn.Module):
    def __init__(self, n_users, n_repos, embedding_dim=128):
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.repo_embeddings = nn.Embedding(n_repos, embedding_dim)

    def forward(self, user_ids, repo_ids):
        user_emb = self.user_embeddings(user_ids)
        repo_emb = self.repo_embeddings(repo_ids)

        # Dot product
        scores = (user_emb * repo_emb).sum(dim=1)
        return scores

    def recommend(self, user_id, k=10, exclude_starred=True):
        """
        Recommend top-K repos for user.
        """
        user_emb = self.user_embeddings(torch.tensor([user_id]))

        # Score all repos
        all_repo_embs = self.repo_embeddings.weight
        scores = torch.matmul(user_emb, all_repo_embs.T).squeeze()

        # Exclude already starred
        if exclude_starred:
            starred_repos = get_starred_repos(user_id)
            scores[starred_repos] = -float('inf')

        # Top-K
        top_k_indices = torch.topk(scores, k).indices
        return top_k_indices.tolist()


# Training
model = RepoMF(n_users=1000000, n_repos=10000000, embedding_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for user_ids, repo_ids, labels in train_loader:
        # labels: 1 if starred, 0 if not
        scores = model(user_ids, repo_ids)
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

### Content-Based: Code Embeddings

**Goal**: Recommend repos with similar code/topics.

**Features**:
- **README**: Natural language description
- **Topics/Tags**: User-assigned labels
- **Languages**: Primary programming language
- **Code embeddings**: Learned from source code

**Code2Vec**: Learn embeddings from abstract syntax trees (ASTs).

**Implementation**:
```python
from transformers import AutoTokenizer, AutoModel

# Use CodeBERT for code embeddings
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

def embed_code(code_snippet):
    """
    Embed code using CodeBERT.
    """
    inputs = tokenizer(code_snippet, return_tensors="pt",
                       truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    # Use [CLS] token embedding
    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return embedding


def embed_repository(repo_files):
    """
    Embed entire repository by averaging file embeddings.
    """
    file_embeddings = []

    for file_path, code in repo_files.items():
        # Filter for code files only
        if file_path.endswith(('.py', '.js', '.java', '.cpp')):
            emb = embed_code(code)
            file_embeddings.append(emb)

    # Average embeddings
    if len(file_embeddings) > 0:
        repo_embedding = torch.stack(file_embeddings).mean(dim=0)
    else:
        repo_embedding = torch.zeros(768)  # CodeBERT dim

    return repo_embedding


def find_similar_repos(query_repo, all_repos, k=10):
    """
    Find repos with similar code.
    """
    query_emb = embed_repository(query_repo)

    similarities = []
    for repo in all_repos:
        repo_emb = embed_repository(repo)
        sim = F.cosine_similarity(query_emb, repo_emb, dim=0)
        similarities.append((repo, sim.item()))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [repo for repo, _ in similarities[:k]]
```

---

## Trending Repositories Detection

### Real-Time Signals

**Problem**: Identify repos gaining rapid popularity.

**Metrics**:
- **Star velocity**: Stars per hour/day
- **Fork velocity**: Forks per day
- **Traffic surge**: Page views spike
- **Social mentions**: Twitter, Reddit, Hacker News

**Algorithm**:
```python
def detect_trending(repo, window_hours=24):
    """
    Detect if repo is trending.
    """
    current_stars = repo.stars_count
    past_stars = repo.stars_count_ago(hours=window_hours)

    # Star velocity
    star_velocity = (current_stars - past_stars) / window_hours

    # Baseline: average velocity over past month
    baseline_velocity = repo.avg_star_velocity(days=30)

    # Trending if velocity >> baseline
    if star_velocity > 5 * baseline_velocity and star_velocity > 10:
        return True

    return False


class TrendingDetector:
    def __init__(self, alpha=0.1):
        self.alpha = alpha  # Exponential moving average weight
        self.baseline_velocities = {}

    def update(self, repo_id, current_stars):
        """
        Update baseline velocity with exponential moving average.
        """
        if repo_id not in self.baseline_velocities:
            self.baseline_velocities[repo_id] = 0

        # EMA update
        velocity = current_stars - self.get_previous_stars(repo_id)
        self.baseline_velocities[repo_id] = (
            self.alpha * velocity +
            (1 - self.alpha) * self.baseline_velocities[repo_id]
        )

    def is_trending(self, repo_id, current_velocity, threshold=3.0):
        """
        Check if current velocity >> baseline.
        """
        baseline = self.baseline_velocities.get(repo_id, 0)
        return current_velocity > threshold * baseline
```

---

### Topic-Specific Trending

**Goal**: Trending within specific domains (ML, web dev, etc.).

**Implementation**:
```python
def trending_by_topic(repos, topic='machine-learning', k=10):
    """
    Find trending repos within topic.
    """
    # Filter repos by topic
    topic_repos = [r for r in repos if topic in r.topics]

    # Compute trending score
    scores = []
    for repo in topic_repos:
        velocity = compute_star_velocity(repo, window_hours=24)
        baseline = compute_baseline_velocity(repo, window_days=30)

        # Trending score
        score = velocity / (baseline + 1)  # Avoid division by zero
        scores.append((repo, score))

    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    return [repo for repo, _ in scores[:k]]
```

---

## Developer Network Recommendations

### People You May Know

**Goal**: Suggest developers to follow.

**Signals**:
1. **Common repo contributions**: Work on same projects
2. **Common stars**: Similar interests
3. **Follower overlap**: Mutual followers
4. **Organization membership**: Same company/team

**Graph-based approach**:
```python
import networkx as nx

def build_developer_graph(developers, repos):
    """
    Build graph where edges = shared repos.
    """
    G = nx.Graph()

    for dev in developers:
        G.add_node(dev.id)

    # Add edges for shared repos
    for repo in repos:
        contributors = repo.contributors

        # Connect all pairs of contributors
        for i, dev1 in enumerate(contributors):
            for dev2 in contributors[i+1:]:
                if G.has_edge(dev1.id, dev2.id):
                    G[dev1.id][dev2.id]['weight'] += 1
                else:
                    G.add_edge(dev1.id, dev2.id, weight=1)

    return G


def recommend_developers(user_id, graph, k=10):
    """
    Recommend developers to follow using graph algorithms.
    """
    # Get current connections (following)
    current_following = get_following(user_id)

    # Personalized PageRank
    scores = nx.pagerank(graph, personalization={user_id: 1.0})

    # Remove already following
    candidates = [(dev, score) for dev, score in scores.items()
                  if dev != user_id and dev not in current_following]

    # Sort by score
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [dev for dev, _ in candidates[:k]]
```

---

### Expertise-Based Matching

**Goal**: Find experts in specific technologies.

**Expertise signals**:
- Commits to repos in domain
- Stars on advanced repos
- Maintainer of popular projects
- StackOverflow reputation (if linked)

```python
def compute_expertise(developer, technology):
    """
    Score developer's expertise in technology.
    """
    score = 0

    # Repos contributed to
    relevant_repos = [r for r in developer.repos
                      if technology in r.languages or technology in r.topics]

    for repo in relevant_repos:
        # Weighted by repo popularity
        popularity = repo.stars_count

        # Weighted by contribution size
        contribution = developer.commits_to(repo) / repo.total_commits

        score += popularity * contribution

    return score


def find_experts(technology, k=10):
    """
    Find top experts in technology.
    """
    all_developers = get_developers()

    expertise_scores = []
    for dev in all_developers:
        score = compute_expertise(dev, technology)
        if score > 0:
            expertise_scores.append((dev, score))

    # Sort by expertise
    expertise_scores.sort(key=lambda x: x[1], reverse=True)
    return [dev for dev, _ in expertise_scores[:k]]
```

---

## Package and Dependency Recommendations

### Dependency Co-Occurrence

**Goal**: Suggest packages often used together.

**Approach**: Mine `package.json`, `requirements.txt`, etc.

**Association rules**:
```python
from itertools import combinations

def mine_dependencies(projects):
    """
    Find packages frequently used together.
    """
    dependency_sets = []

    for project in projects:
        deps = set(project.dependencies)
        dependency_sets.append(deps)

    # Count co-occurrences
    co_occurrence = {}

    for deps in dependency_sets:
        for pkg1, pkg2 in combinations(deps, 2):
            pair = tuple(sorted([pkg1, pkg2]))
            co_occurrence[pair] = co_occurrence.get(pair, 0) + 1

    return co_occurrence


def recommend_packages(current_dependencies, co_occurrence, k=5):
    """
    Recommend packages given current dependencies.
    """
    candidates = {}

    for pkg in current_dependencies:
        # Find packages that co-occur with pkg
        for (pkg1, pkg2), count in co_occurrence.items():
            if pkg1 == pkg:
                candidate = pkg2
            elif pkg2 == pkg:
                candidate = pkg1
            else:
                continue

            # Skip if already using
            if candidate in current_dependencies:
                continue

            # Aggregate scores
            candidates[candidate] = candidates.get(candidate, 0) + count

    # Sort by score
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [pkg for pkg, _ in sorted_candidates[:k]]


# Example
projects = load_projects()
co_occurrence = mine_dependencies(projects)

# User has React, recommend complementary packages
current = {'react', 'react-dom'}
recommendations = recommend_packages(current, co_occurrence, k=5)
print(recommendations)
# Output: ['react-router', 'redux', 'axios', 'styled-components', 'prop-types']
```

---

### Functional Similarity

**Goal**: Recommend packages with similar functionality.

**README embeddings**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_package(package):
    """
    Embed package description.
    """
    text = f"{package.name} {package.description}"
    embedding = model.encode(text)
    return embedding


def find_similar_packages(query_package, all_packages, k=10):
    """
    Find packages with similar functionality.
    """
    query_emb = embed_package(query_package)

    similarities = []
    for pkg in all_packages:
        pkg_emb = embed_package(pkg)
        sim = cosine_similarity(query_emb, pkg_emb)
        similarities.append((pkg, sim))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [pkg for pkg, _ in similarities[:k]]
```

---

## Learning Path Recommendations

### Prerequisite Knowledge Graph

**Goal**: Recommend technologies to learn based on prerequisites.

**Knowledge graph**:
```
Python → NumPy → Pandas → Scikit-learn → PyTorch
JavaScript → React → Next.js
```

**Implementation**:
```python
import networkx as nx

def build_tech_graph():
    """
    Build technology prerequisite graph.
    """
    G = nx.DiGraph()

    # Add edges: prerequisite → advanced
    edges = [
        ('Python', 'NumPy'),
        ('NumPy', 'Pandas'),
        ('Pandas', 'Scikit-learn'),
        ('Scikit-learn', 'PyTorch'),
        ('JavaScript', 'React'),
        ('React', 'Next.js'),
        # ... many more
    ]

    G.add_edges_from(edges)
    return G


def recommend_learning_path(user_skills, target_skill, graph):
    """
    Find shortest learning path from current skills to target.
    """
    # Find closest current skill to target
    shortest_paths = []

    for skill in user_skills:
        if skill in graph and target_skill in graph:
            try:
                path = nx.shortest_path(graph, skill, target_skill)
                shortest_paths.append(path)
            except nx.NetworkXNoPath:
                continue

    # Return shortest path
    if shortest_paths:
        shortest = min(shortest_paths, key=len)
        return shortest[1:]  # Exclude starting skill
    else:
        return []


# Example
graph = build_tech_graph()
user_skills = ['Python', 'NumPy']
target = 'PyTorch'

path = recommend_learning_path(user_skills, target, graph)
print(f"Learning path: {' → '.join(path)}")
# Output: Pandas → Scikit-learn → PyTorch
```

---

### Skill Gap Analysis

**Goal**: Identify missing skills for career goals.

```python
def analyze_skill_gap(user_skills, target_role):
    """
    Identify skills to learn for target role.
    """
    # Load typical skills for role
    role_skills = get_required_skills(target_role)

    # Find missing skills
    missing = set(role_skills) - set(user_skills)

    # Prioritize by frequency in job postings
    skill_importance = get_skill_frequency(target_role)

    prioritized = sorted(missing,
                        key=lambda s: skill_importance.get(s, 0),
                        reverse=True)

    return prioritized


def get_required_skills(role):
    """
    Get typical skills for role (from job postings).
    """
    role_skills = {
        'Machine Learning Engineer': [
            'Python', 'PyTorch', 'TensorFlow', 'Scikit-learn',
            'Docker', 'Kubernetes', 'SQL', 'Git'
        ],
        'Frontend Developer': [
            'JavaScript', 'React', 'TypeScript', 'CSS',
            'HTML', 'Webpack', 'Git'
        ],
        # ... more roles
    }
    return role_skills.get(role, [])
```

---

## Issue and Pull Request Recommendations

### Good First Issues

**Goal**: Recommend beginner-friendly issues to new contributors.

**Signals**:
- Labeled "good first issue"
- Small code change required
- Clear description
- Active maintainer responses

```python
def recommend_issues(user, k=10):
    """
    Recommend issues for user to work on.
    """
    # Get user's interests (starred repos, languages)
    languages = user.primary_languages
    topics = user.interested_topics

    # Find repos in user's domain
    relevant_repos = find_repos_by_topics(topics)

    # Get open issues
    issues = []
    for repo in relevant_repos:
        repo_issues = repo.get_issues(state='open', labels=['good first issue'])
        issues.extend(repo_issues)

    # Score issues
    scored_issues = []
    for issue in issues:
        score = 0

        # Language match
        if issue.repo.primary_language in languages:
            score += 1.0

        # Topic match
        topic_overlap = len(set(issue.repo.topics) & set(topics))
        score += topic_overlap * 0.5

        # Issue quality (clear description, recent activity)
        if len(issue.description) > 100:
            score += 0.5

        if issue.days_since_updated < 7:
            score += 0.5

        scored_issues.append((issue, score))

    # Sort by score
    scored_issues.sort(key=lambda x: x[1], reverse=True)
    return [issue for issue, _ in scored_issues[:k]]
```

---

### Matching Contributors to PRs

**Goal**: Suggest reviewers for pull requests.

**Signals**:
- Code ownership (CODEOWNERS file)
- Past contributions to same files
- Expertise in languages/frameworks

```python
def recommend_reviewers(pull_request, k=3):
    """
    Recommend reviewers for PR.
    """
    repo = pull_request.repo
    changed_files = pull_request.changed_files

    # Get potential reviewers
    contributors = repo.contributors

    # Score each contributor
    scores = {}

    for contributor in contributors:
        score = 0

        # Past edits to same files
        for file in changed_files:
            edits = contributor.edit_count(file)
            score += edits

        # Expertise in language
        pr_languages = set(get_languages(changed_files))
        contributor_languages = set(contributor.primary_languages)

        if pr_languages & contributor_languages:
            score += 10

        # Recent activity (prefer active contributors)
        days_since_last_commit = (today - contributor.last_commit_date).days
        recency = 1 / (1 + days_since_last_commit / 30)
        score *= recency

        scores[contributor] = score

    # Sort by score
    sorted_reviewers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [contributor for contributor, _ in sorted_reviewers[:k]]
```

---

## Code Completion and Copilot

### Context-Aware Suggestions

**Goal**: Suggest code completions based on context.

**LLM-based approach** (GitHub Copilot):
- **Model**: Fine-tuned GPT (Codex)
- **Context**: Current file + open files
- **Prompt**: Code up to cursor position
- **Output**: Next lines of code

**Simplified example**:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

def suggest_completion(code_context, max_length=50):
    """
    Suggest code completion given context.
    """
    inputs = tokenizer(code_context, return_tensors="pt")

    outputs = model.generate(
        inputs.input_ids,
        max_length=inputs.input_ids.shape[1] + max_length,
        num_return_sequences=1,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove context, return only new code
    new_code = completion[len(code_context):]
    return new_code


# Example
context = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
"""

completion = suggest_completion(context)
print(completion)
# Output: "    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)"
```

---

### Ranking Multiple Suggestions

**Problem**: Model generates multiple completions, which to show?

**Ranking features**:
- **Model confidence**: Log probability
- **Code validity**: Parses without syntax errors
- **Style consistency**: Matches repo coding style
- **Contextual relevance**: Uses variables in scope

```python
def rank_completions(completions, context):
    """
    Rank code completions by quality.
    """
    scored_completions = []

    for completion in completions:
        score = 0

        # Model confidence (log probability)
        log_prob = completion.log_probability
        score += log_prob

        # Syntactic validity
        if is_valid_python(context + completion.text):
            score += 5.0

        # Style consistency
        if matches_style(completion.text, context):
            score += 2.0

        # Uses in-scope variables
        in_scope_vars = extract_variables(context)
        completion_vars = extract_variables(completion.text)

        if completion_vars.issubset(in_scope_vars):
            score += 1.0

        scored_completions.append((completion, score))

    # Sort by score
    scored_completions.sort(key=lambda x: x[1], reverse=True)
    return [comp for comp, _ in scored_completions]


def is_valid_python(code):
    """Check if code is syntactically valid Python."""
    import ast
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def matches_style(code, context):
    """Check if code matches existing style (indentation, naming)."""
    # Simplified: check indentation consistency
    context_indent = detect_indentation(context)
    code_indent = detect_indentation(code)
    return context_indent == code_indent
```

---

## Summary

**Key Takeaways**:
1. **Repository recommendations**: CF on stars + code embeddings
2. **Trending detection**: Real-time velocity vs. baseline
3. **Developer network**: Graph-based PYMK using shared repos
4. **Package recommendations**: Dependency co-occurrence + functional similarity
5. **Learning paths**: Prerequisite knowledge graph + skill gap analysis
6. **Issue recommendations**: Match user interests with "good first issue" labels
7. **Code completion**: LLM-based (Copilot) with context-aware ranking

**Metrics**: Star conversion rate, PR acceptance rate, completion acceptance rate.

---

## References

1. **GitHub Engineering Blog**: "How GitHub Uses Machine Learning" (2020).
2. **Chen, M., et al. (2021)**. "Evaluating Large Language Models Trained on Code". *arXiv*.
3. **Iyer, S., et al. (2018)**. "Mapping Language to Code in Programmatic Context". *EMNLP*.
