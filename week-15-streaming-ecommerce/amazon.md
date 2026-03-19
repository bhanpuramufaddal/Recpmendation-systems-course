# System Design: Amazon Product Recommendations

## Problem Statement & Requirements

### Interview Prompt

> "Design a product recommendation system for an e-commerce platform like Amazon with 350M+ products and 300M+ active customers."

### Functional Requirements

1. **"Customers who bought this also bought"**: Item-to-item recommendations
2. **Personalized homepage**: Custom product feed for each user
3. **Product detail page**: Related products, frequently bought together
4. **Search ranking**: Personalized search results
5. **Cart recommendations**: "Add to cart" suggestions
6. **Email recommendations**: Personalized product emails
7. **Replenishment**: "Buy again" for consumables

### Non-Functional Requirements

1. **Latency**: Homepage load < 100ms for recommendations
2. **Scale**: 350M products, 300M customers, 500K QPS peak (Prime Day)
3. **Availability**: 99.99% uptime
4. **Freshness**: Inventory and price updates in real-time
5. **Revenue impact**: 35%+ of revenue from recommendations

### Scope

**In scope**:
- Product recommendation algorithm
- Multiple recommendation surfaces
- Real-time personalization
- Candidate retrieval and ranking

**Out of scope**:
- Inventory management
- Pricing system
- Order fulfillment
- Ad auction

---

## Scale Estimation (Back-of-Envelope)

### Users & Traffic

```
Users:
- Active customers: 300M
- Daily active users (DAU): 50M
- Prime members: 200M

Traffic:
- Page views per user per day: 10
- Recommendation slots per page: 5
- Total recommendation requests: 50M × 10 × 5 = 2.5B/day

QPS Calculation:
- Average QPS: 2.5B / 86,400 = 29,000 QPS
- Peak QPS (Prime Day): 500,000 QPS (17x average)
- Black Friday peak: 300,000 QPS
```

### Product Scale

```
Products:
- Total products: 350M+
- Active products (in stock): 200M
- Products with engagement data: 100M
- New products per day: 500K

Categories:
- Top-level categories: 30
- Sub-categories: 30,000+
- Leaf categories: 500,000+
```

### Storage

```
Product Embeddings:
- Products: 350M
- Embedding dimension: 256
- Storage: 350M × 256 × 4 bytes = 350GB

User Embeddings:
- Users: 300M
- Embedding dimension: 256
- Storage: 300M × 256 × 4 bytes = 300GB

Item-Item Similarity Matrix:
- Top 200M products × 100 neighbors each
- Storage: 200M × 100 × 12 bytes = 240GB (sparse)

Feature Store:
- User features: 300M × 2KB = 600GB
- Product features: 350M × 1KB = 350GB
```

### Latency Budget (Homepage)

```
Total budget: 100ms

Component breakdown:
- Feature store lookup: 10ms
- Candidate retrieval (parallel sources): 20ms
- Feature hydration: 15ms
- Ranking model inference: 30ms
- Re-ranking (diversity, business rules): 15ms
- Network overhead: 10ms
```

---

## High-Level Architecture

```
User Request
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Candidate Generation (Parallel)                 │
│                                                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ │
│  │ Item-Item  │ │ User-Item  │ │ Purchase   │ │ Session  │ │
│  │ CF         │ │ CF         │ │ History    │ │ Based    │ │
│  └────────────┘ └────────────┘ └────────────┘ └──────────┘ │
│                                                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ │
│  │ Category   │ │ Trending   │ │ Search     │ │ Sponsored│ │
│  │ Popular    │ │            │ │ History    │ │          │ │
│  └────────────┘ └────────────┘ └────────────┘ └──────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │ ~1000 candidates
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Ranking Model (DNN)                       │
│         P(purchase), P(add_to_cart), P(click)               │
└──────────────────────────┬──────────────────────────────────┘
                           │ ~100 ranked
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Re-Ranking Layer                           │
│  - Category diversity                                        │
│  - Price range diversity                                     │
│  - Inventory filtering                                       │
│  - Sponsored product insertion                               │
│  - Business rules                                            │
└──────────────────────────┬──────────────────────────────────┘
                           │ ~50 final
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       Response                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Model

### User Features

```python
class UserProfile:
    user_id: str

    # Account
    account_type: str           # "prime", "regular"
    account_age_days: int
    lifetime_value: float       # Total spend

    # Demographics (inferred)
    predicted_age_range: str
    predicted_gender: str
    location_zip: str

    # Shopping behavior
    purchase_history: List[str]     # Last 1000 purchases
    browse_history: List[str]       # Last 500 viewed products
    search_history: List[str]       # Last 100 searches
    cart_history: List[str]         # Items added to cart

    # Preferences (learned)
    category_affinity: Dict[str, float]   # {"Electronics": 0.4, ...}
    brand_affinity: Dict[str, float]      # {"Apple": 0.7, "Sony": 0.5}
    price_sensitivity: float              # 0-1 (0=price sensitive)
    avg_order_value: float

    # Embeddings
    taste_embedding: List[float]    # 256-dim
    session_embedding: List[float]  # 256-dim (current session)

    # Engagement patterns
    preferred_shopping_time: str    # "morning", "evening"
    device_preference: str          # "mobile", "desktop"
```

### Product Features

```python
class Product:
    product_id: str
    asin: str                       # Amazon Standard ID

    # Basic info
    title: str
    description: str
    category_tree: List[str]        # ["Electronics", "Computers", "Laptops"]
    brand: str

    # Pricing
    price: float
    list_price: float               # MSRP
    discount_pct: float
    price_history: List[float]      # Last 30 days

    # Inventory
    in_stock: bool
    stock_quantity: int
    fulfillment_type: str           # "FBA", "FBM", "Amazon"
    delivery_days: int

    # Quality signals
    avg_rating: float               # 1-5
    num_ratings: int
    review_sentiment: float         # -1 to 1
    return_rate: float

    # Engagement
    total_views: int
    purchase_rate: float            # Purchases / Views
    add_to_cart_rate: float
    click_through_rate: float

    # Embeddings
    content_embedding: List[float]  # 256-dim (from title + description)
    image_embedding: List[float]    # 256-dim (from product images)
    cf_embedding: List[float]       # 256-dim (from collaborative filtering)

    # Business
    profit_margin: float
    seller_id: str
    is_amazon_choice: bool
    is_best_seller: bool
```

---

## Candidate Generation

### Source 1: Item-to-Item Collaborative Filtering

**Course Connection**: Week 2-3 (Memory-Based CF, Matrix Factorization)

This is Amazon's most famous algorithm, published in 2003 by Linden et al.

```python
class ItemToItemCF:
    """
    Classic Amazon item-to-item collaborative filtering.

    "Customers who bought this also bought..."
    """
    def __init__(self):
        self.item_similarity = {}  # Pre-computed

    def precompute_similarities(self, purchase_matrix: csr_matrix):
        """
        Compute cosine similarity between all item pairs.

        Run weekly as batch job.
        """
        n_items = purchase_matrix.shape[1]

        for i in range(n_items):
            item_vector = purchase_matrix[:, i].toarray().flatten()

            similarities = []
            for j in range(n_items):
                if i != j:
                    other_vector = purchase_matrix[:, j].toarray().flatten()
                    sim = cosine_similarity(item_vector, other_vector)
                    if sim > 0.1:  # Threshold
                        similarities.append((j, sim))

            # Keep top 100 neighbors
            similarities.sort(key=lambda x: -x[1])
            self.item_similarity[i] = similarities[:100]

    def get_candidates(self, seed_items: List[str], k: int = 200) -> List[str]:
        """
        Get items similar to seed items.
        """
        candidates = defaultdict(float)

        for item in seed_items:
            if item in self.item_similarity:
                for similar_item, score in self.item_similarity[item]:
                    candidates[similar_item] += score

        # Sort by aggregated score
        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
        return [item for item, _ in sorted_candidates[:k]]
```

**Why item-item (not user-user)?**
- Items change slower than users → more stable similarities
- Can precompute offline
- Scales better (fewer items than users)
- More interpretable ("bought X, so recommend Y")

### Source 2: User-Item Collaborative Filtering

**Course Connection**: Week 3 (Matrix Factorization)

```python
class UserItemCF:
    """
    Matrix factorization for personalized recommendations.
    """
    def __init__(self, embedding_dim=256):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=embedding_dim,
            regularization=0.1,
            iterations=30
        )

    def train(self, purchase_matrix: csr_matrix):
        """
        Train on implicit feedback (purchases, views, cart adds).

        Weighting: purchases > cart adds > views
        """
        # Weight matrix: confidence = 1 + alpha * f(interaction)
        # where f(interaction) = 10 for purchase, 3 for cart, 1 for view
        self.model.fit(purchase_matrix)

    def get_candidates(self, user_id: str, k: int = 200) -> List[str]:
        """
        Get personalized recommendations.
        """
        user_idx = self.user_to_idx[user_id]
        item_ids, scores = self.model.recommend(
            user_idx,
            self.purchase_matrix[user_idx],
            N=k,
            filter_already_liked_items=True
        )
        return [self.idx_to_item[i] for i in item_ids]
```

### Source 3: Session-Based Recommendations

**Course Connection**: Week 6 (Sequential Models)

```python
class SessionBasedCandidates:
    """
    RNN-based model for current session behavior.

    Captures short-term intent (what user is looking for right now).
    """
    def __init__(self):
        self.model = GRU4Rec(hidden_size=256, n_items=350_000_000)

    def get_candidates(self, session_items: List[str], k: int = 200) -> List[str]:
        """
        Predict next items based on current session.
        """
        # Encode session
        item_indices = [self.item_to_idx[i] for i in session_items]
        session_tensor = torch.tensor(item_indices).unsqueeze(0)

        # Get predictions
        with torch.no_grad():
            logits = self.model(session_tensor)

        # Top-k predictions
        top_k = torch.topk(logits[0, -1], k).indices
        return [self.idx_to_item[i] for i in top_k.tolist()]
```

### Source 4: Purchase History Based

```python
class PurchaseHistorySource:
    """
    Recommendations based on purchase patterns.
    """
    def get_replenishment_candidates(self, user: UserProfile) -> List[str]:
        """
        Items user might need to reorder (consumables).
        """
        candidates = []

        for purchase in user.purchase_history:
            product = get_product(purchase.product_id)

            # Check if consumable
            if product.category in ['Grocery', 'Health', 'Baby', 'Pet']:
                days_since = (now() - purchase.date).days
                avg_reorder_time = self.get_avg_reorder_time(purchase.product_id)

                if days_since > avg_reorder_time * 0.8:
                    candidates.append(purchase.product_id)

        return candidates

    def get_cross_sell_candidates(self, user: UserProfile, k: int = 100) -> List[str]:
        """
        Products commonly bought with user's recent purchases.
        """
        recent_purchases = user.purchase_history[:10]

        # Frequent itemset mining (pre-computed)
        cross_sell = []
        for purchase in recent_purchases:
            associated = self.frequently_bought_together[purchase.product_id]
            cross_sell.extend(associated)

        return list(set(cross_sell))[:k]
```

### Candidate Merging

```python
def generate_candidates(user: UserProfile, context: Context) -> List[Product]:
    """
    Merge candidates from all sources.
    """
    candidates = {}

    # Item-to-item CF (from recent views/purchases)
    seed_items = user.browse_history[:20] + user.purchase_history[:10]
    item_cf_candidates = item_cf.get_candidates(seed_items, k=300)
    for item in item_cf_candidates:
        candidates[item] = {'item_cf': 1.0}

    # User-item CF (personalized)
    user_cf_candidates = user_cf.get_candidates(user.user_id, k=300)
    for item in user_cf_candidates:
        if item in candidates:
            candidates[item]['user_cf'] = 1.0
        else:
            candidates[item] = {'user_cf': 1.0}

    # Session-based (current intent)
    session_candidates = session_model.get_candidates(context.session_items, k=200)
    for item in session_candidates:
        if item in candidates:
            candidates[item]['session'] = 1.2  # Higher weight for current intent
        else:
            candidates[item] = {'session': 1.2}

    # Category popular (fallback)
    if len(candidates) < 500:
        popular = get_category_popular(user.preferred_categories, k=200)
        for item in popular:
            if item not in candidates:
                candidates[item] = {'popular': 0.5}

    return list(candidates.keys())[:1000]
```

---

## Ranking Model

### Architecture

**Course Connection**: Week 5 (Neural CF), Week 8 (Multi-Task Learning)

```python
class AmazonRankingModel(nn.Module):
    """
    Multi-task deep learning model for product ranking.
    """
    def __init__(self, user_dim=256, product_dim=256, context_dim=64):
        super().__init__()

        input_dim = user_dim + product_dim + context_dim + 100  # +100 for cross features

        # Shared tower
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Task-specific heads
        self.purchase_head = nn.Linear(128, 1)    # P(purchase)
        self.cart_head = nn.Linear(128, 1)        # P(add to cart)
        self.click_head = nn.Linear(128, 1)       # P(click)

    def forward(self, user_emb, product_emb, context, cross_features):
        x = torch.cat([user_emb, product_emb, context, cross_features], dim=1)
        shared = self.shared(x)

        return {
            'purchase': torch.sigmoid(self.purchase_head(shared)),
            'cart': torch.sigmoid(self.cart_head(shared)),
            'click': torch.sigmoid(self.click_head(shared))
        }
```

### Revenue-Optimized Ranking

```python
def compute_ranking_score(predictions: Dict[str, float],
                          product: Product,
                          user: UserProfile) -> float:
    """
    Revenue-optimized ranking score.

    Key insight: Weight purchase probability by item value.
    """
    # Expected revenue
    expected_revenue = (
        predictions['purchase'] * product.price * (1 - product.return_rate)
    )

    # Engagement score (for non-revenue optimization)
    engagement_score = (
        0.2 * predictions['click'] +
        0.3 * predictions['cart'] +
        0.5 * predictions['purchase']
    )

    # Combine (balance revenue and engagement)
    # High margin items get boost
    margin_factor = 1 + 0.5 * product.profit_margin

    # Prime delivery boost (better user experience)
    prime_boost = 1.2 if product.fulfillment_type == 'FBA' else 1.0

    # In-stock penalty (don't show out of stock)
    stock_penalty = 1.0 if product.in_stock else 0.1

    score = (
        0.6 * expected_revenue +
        0.4 * engagement_score
    ) * margin_factor * prime_boost * stock_penalty

    return score
```

### Features Used (~1000 Features)

| Category | Example Features | Count |
|----------|------------------|-------|
| **User** | Account age, Prime status, lifetime value, purchase frequency | ~50 |
| **Product** | Price, rating, reviews, category, brand, inventory | ~100 |
| **User-Product** | Category affinity match, brand affinity, price fit | ~100 |
| **Session** | Items viewed, items carted, search queries | ~50 |
| **Historical** | User-product past interactions, purchase history | ~100 |
| **Cross** | User category × Product category, price sensitivity × price | ~500 |
| **Embeddings** | User embedding, product embedding, dot product | ~100 |

---

## Re-Ranking Layer

### Diversity Constraints

```python
class DiversityReranker:
    """
    Apply diversity and business rules to ranked products.
    """
    def rerank(self, ranked_products: List[Product], k: int = 50) -> List[Product]:
        final = []
        category_counts = defaultdict(int)
        brand_counts = defaultdict(int)

        for product in ranked_products:
            # Category diversity: max 3 per category in top 20
            if len(final) < 20:
                if category_counts[product.category] >= 3:
                    continue

            # Brand diversity: max 2 per brand in top 10
            if len(final) < 10:
                if brand_counts[product.brand] >= 2:
                    continue

            # Price diversity: ensure mix of price points
            if len(final) >= 10:
                prices = [p.price for p in final]
                if product.price > np.percentile(prices, 90):
                    # Too many expensive items
                    if sum(1 for p in final if p.price > np.median(prices)) > len(final) * 0.5:
                        continue

            final.append(product)
            category_counts[product.category] += 1
            brand_counts[product.brand] += 1

            if len(final) >= k:
                break

        return final
```

### Business Rules

```python
class BusinessRulesFilter:
    """
    Apply business rules and constraints.
    """
    def apply_rules(self, products: List[Product]) -> List[Product]:
        filtered = []

        for product in products:
            # Must be in stock (or back-orderable)
            if not product.in_stock and not product.backorder_available:
                continue

            # Minimum rating threshold
            if product.avg_rating < 3.0 and product.num_ratings > 10:
                continue

            # Exclude flagged products
            if product.is_flagged:
                continue

            # Seller quality threshold
            if product.seller_rating < 3.5:
                continue

            filtered.append(product)

        return filtered
```

### Sponsored Product Integration

```python
def integrate_sponsored(organic: List[Product],
                        sponsored: List[Product],
                        positions: List[int] = [3, 7, 15]) -> List[Product]:
    """
    Insert sponsored products at specific positions.

    Sponsored products come from separate ad auction.
    """
    result = organic.copy()

    for i, pos in enumerate(positions):
        if i < len(sponsored) and pos <= len(result):
            result.insert(pos, sponsored[i])

    return result
```

---

## Training Pipeline

### Data Collection

**Course Connection**: Week 3 (Implicit Feedback), Week 11 (Evaluation)

```python
class AmazonTrainingData:
    """
    Collect multi-signal training data.
    """
    def create_training_example(self, impression: Impression) -> Example:
        """
        Labels hierarchy:
        - Purchase > Add to cart > Click > View
        """
        user_features = get_user_features(impression.user_id)
        product_features = get_product_features(impression.product_id)
        context = get_context(impression)

        # Compute cross features
        cross_features = compute_cross_features(user_features, product_features)

        labels = {
            'purchase': impression.purchased,
            'cart': impression.added_to_cart,
            'click': impression.clicked
        }

        # Position bias correction
        position_weight = 1.0 / self.position_ctr[impression.position]

        return Example(
            features=(user_features, product_features, context, cross_features),
            labels=labels,
            weight=position_weight
        )
```

### Training Schedule

```
Model training cadence:
- Item-item similarity: Weekly (full recompute)
- User-item CF (ALS): Daily (incremental)
- Ranking model: Daily (fine-tune), Weekly (full retrain)
- Session model: Weekly

Feature updates:
- User embeddings: Daily
- Product embeddings: Daily
- Real-time features: Streaming (Kinesis)
```

---

## Serving Infrastructure

### Architecture

**Course Connection**: Week 13 (Production Systems)

```
┌─────────────────────────────────────────────────────────────┐
│                    BATCH LAYER                               │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Item-Item    │  │ User         │  │ Product          │   │
│  │ Similarity   │  │ Embedding    │  │ Embedding        │   │
│  │ (Weekly)     │  │ (Daily)      │  │ (Daily)          │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ Model        │  │ Feature      │                         │
│  │ Training     │  │ Engineering  │                         │
│  └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    REAL-TIME LAYER                           │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Candidate    │  │ Ranking      │  │ Feature          │   │
│  │ Generation   │  │ Service      │  │ Store            │   │
│  │ Service      │  │              │  │ (DynamoDB)       │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ ANN Index    │  │ Inventory    │  │ Price            │   │
│  │ (FAISS)      │  │ Service      │  │ Service          │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   STREAMING LAYER                            │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Session      │  │ Inventory    │  │ User Activity    │   │
│  │ Tracking     │  │ Updates      │  │ Events           │   │
│  │ (Kinesis)    │  │ (Kinesis)    │  │ (Kinesis)        │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Feature Store

```python
class AmazonFeatureStore:
    """
    Real-time and batch feature serving.
    """
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.redis = redis.Redis()

    async def get_user_features(self, user_id: str) -> Dict:
        """
        Get user features with caching.
        """
        # L1: Redis cache (10 min TTL)
        cached = self.redis.get(f"user:{user_id}")
        if cached:
            return json.loads(cached)

        # L2: DynamoDB
        features = self.dynamodb.get_item(
            TableName='user_features',
            Key={'user_id': user_id}
        )

        # Cache for next request
        self.redis.setex(f"user:{user_id}", 600, json.dumps(features))

        return features

    async def get_product_features_batch(self, product_ids: List[str]) -> List[Dict]:
        """
        Batch get product features.
        """
        # Batch DynamoDB query
        return self.dynamodb.batch_get_item(
            RequestItems={
                'product_features': {
                    'Keys': [{'product_id': pid} for pid in product_ids]
                }
            }
        )
```

### Latency Optimization

```python
class LatencyOptimizations:
    """
    Key optimizations for sub-100ms latency.
    """
    # 1. Parallel candidate retrieval
    async def get_candidates_parallel(self, user, context):
        results = await asyncio.gather(
            self.item_cf.get_candidates(user),
            self.user_cf.get_candidates(user),
            self.session_model.get_candidates(context),
            self.popular.get_candidates(user)
        )
        return merge(results)

    # 2. Pre-computed similarities
    # Item-item similarity computed offline, stored in memory

    # 3. Model quantization
    # INT8 quantization for 2x inference speedup

    # 4. Tiered caching
    cache_config = {
        'user_features': ('redis', 300),      # 5 min
        'product_features': ('memcached', 60), # 1 min (price changes)
        'item_similarity': ('local_memory', 3600),  # 1 hour
        'recommendations': ('cdn', 30)         # 30 sec (personalized)
    }
```

---

## Metrics & Evaluation

### Online Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **CVR** | Conversion rate (purchases/views) | > 3% |
| **RPV** | Revenue per visitor | > $10 |
| **Add-to-cart rate** | Cart adds / impressions | > 8% |
| **Click-through rate** | Clicks / impressions | > 5% |
| **Recommendation revenue %** | Revenue from recs / total | > 35% |

### A/B Testing

```
Amazon experimentation:
- Platform: Weblab (internal)
- Typical experiment: 1-5% of traffic
- Duration: 1-2 weeks
- Key metrics: RPV, CVR, cart add rate, session value

Prime Day special handling:
- Pre-tested algorithms deployed
- Real-time monitoring
- Instant rollback capability
```

### Guardrail Metrics

```python
class GuardrailMetrics:
    def category_coverage(self, recommendations):
        """Ensure diverse categories are represented."""
        categories = set(r.category for r in recommendations)
        return len(categories) >= 5  # At least 5 categories in top 20

    def price_distribution(self, recommendations):
        """Ensure price diversity."""
        prices = [r.price for r in recommendations]
        # Should have items in multiple price brackets
        return np.std(prices) / np.mean(prices) > 0.5

    def seller_fairness(self, recommendations):
        """Don't over-concentrate on few sellers."""
        sellers = [r.seller_id for r in recommendations]
        gini = gini_coefficient(Counter(sellers).values())
        return gini < 0.6  # Lower is more fair
```

---

## Trade-offs & Deep Dives

### Key Design Decisions

#### 1. Item-Item vs User-Item CF

**Decision**: Use both, item-item for "also bought", user-item for personalization

**Reasoning**:
- Item-item: Stable, explainable, good for product pages
- User-item: Better personalization, good for homepage
- Combine signals in ranking

#### 2. Revenue vs Engagement Optimization

**Decision**: 60% revenue weight, 40% engagement weight

**Reasoning**:
- Pure revenue optimization leads to expensive items only
- Pure engagement leads to low-margin items
- Balance ensures user satisfaction + business goals

#### 3. Real-time vs Batch Features

**Decision**: Most features batch, session features real-time

**Reasoning**:
- User preferences stable (batch is fine)
- Current session intent changes rapidly (needs real-time)
- Inventory/price must be real-time

### Common Interview Follow-ups

**Q: How do you handle cold start for new products?**
> Content-based similarity using product title, description, and images. We extract embeddings from product catalog data and find similar existing products. After ~100 views, collaborative signals take over.

**Q: How do you balance relevance vs revenue?**
> Multi-objective optimization with tunable weights. We A/B test different weight combinations. Also use "exploration budget" - 10% of recommendations from high-margin items that might not rank highest organically.

**Q: How do you handle Prime Day scale (500K QPS)?**
> Pre-warming caches, pre-computed recommendations for top users, model serving auto-scaling, circuit breakers for graceful degradation. We run load tests at 2x expected peak.

**Q: How do you personalize for new users?**
> Start with browse-based recommendations immediately. After first session, we have enough signal for basic personalization. Explicit category selection on signup helps. Full personalization after 5+ sessions.

---

## Course Concepts Applied

| Concept | Week | Application in Amazon |
|---------|------|----------------------|
| **User-Based CF** | 2 | "Customers like you" signals |
| **Item-Based CF** | 2 | "Also bought" recommendations |
| **Matrix Factorization** | 3 | User-item embeddings (ALS) |
| **Content-Based** | 4 | Product embeddings from descriptions |
| **Neural CF** | 5 | Deep ranking model |
| **Sequential Models** | 6 | Session-based recommendations |
| **Graph-Based** | 7 | Product relationship graphs |
| **Two-Tower** | 8 | User/product embedding retrieval |
| **Multi-Task Learning** | 8 | Click/cart/purchase joint prediction |
| **Embeddings** | 9 | Product and user representations |
| **Bandits** | 10 | Exploration for long-tail products |
| **Evaluation** | 11 | RPV, CVR, A/B testing |
| **Fairness** | 12 | Seller exposure fairness |
| **Production** | 13 | Feature store, latency optimization |

---

## Summary

**Amazon's recommendation system** is the gold standard for e-commerce:

1. **Scale**: 350M products, 300M users, 500K QPS peak
2. **Item-to-Item CF**: The original 2003 algorithm still core today
3. **Multi-signal ranking**: Purchase > Cart > Click
4. **Revenue optimization**: Expected value ranking with margin consideration
5. **Real-time inventory**: Never show out-of-stock items
6. **Session awareness**: Capture current shopping intent

**Key innovations**:
- Item-item CF (Linden et al., 2003)
- Multi-stage pipeline for latency
- Revenue-weighted ranking
- Cross-sell recommendations ("Frequently bought together")

**Business impact**: 35%+ of Amazon revenue from recommendations.

---

## References

1. **Linden, G., Smith, B., & York, J. (2003)**. "Amazon.com Recommendations: Item-to-Item Collaborative Filtering". *IEEE Internet Computing*.
   - **Seminal paper** on item-to-item CF

2. **Smith, B., & Linden, G. (2017)**. "Two Decades of Recommender Systems at Amazon.com". *IEEE Internet Computing*.
   - Evolution and lessons learned

3. **Hidasi, B., et al. (2016)**. "Session-based Recommendations with Recurrent Neural Networks". *ICLR*.
   - GRU4Rec for session modeling

4. **Amazon Science Blog**. Various posts on recommendation systems.
   - Recent techniques and scale

5. **He, R., & McAuley, J. (2016)**. "Ups and Downs: Modeling the Visual Evolution of Fashion Trends with One-Class Collaborative Filtering". *WWW*.
   - Image-based recommendations
