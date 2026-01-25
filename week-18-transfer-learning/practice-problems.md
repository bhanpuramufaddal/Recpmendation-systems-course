# Week 18: Cross-Domain and Transfer Learning - Practice Problems

## Overview
Master cross-domain recommendations, meta-learning (MAML), federated learning, and knowledge transfer for cold-start problems.

---

## Problem 1: Cross-Domain Transfer
**Difficulty:** Medium

**Scenario:** Transfer from movies (data-rich) to books (data-poor)

**Approaches:**
1. **Shared embeddings:** Learn user embeddings on movies, apply to books
2. **Domain adaptation:** Fine-tune movie model on book data
3. **Multi-domain:** Joint model for both domains
4. **Content bridge:** Use content features common to both

**Questions:**
1. When does transfer help? (similar domains, shared users)
2. What if domains are very different? (movies vs. restaurants)
3. How much book data needed for successful transfer?
4. Design evaluation: Compare transfer vs. train-from-scratch

**Learning Outcomes:** Transfer knowledge across domains, choose transfer strategies, evaluate transfer effectiveness

---

## Problem 2: Meta-Learning for Few-Shot Recommendations
**Difficulty:** Hard

**MAML (Model-Agnostic Meta-Learning):** Learn initialization that quickly adapts to new users

**Algorithm:**
1. Sample batch of users (tasks)
2. For each user:
   - Adapt model with few examples (inner loop)
   - Compute loss on held-out examples
3. Meta-update: Update initialization to minimize adaptation loss (outer loop)

**Questions:**
1. Why is MAML better than standard fine-tuning?
2. How many examples per user for adaptation?
3. Computational cost? (Two gradient steps per task)
4. Apply to cold-start user problem

**Learning Outcomes:** Implement meta-learning, understand few-shot learning, solve cold-start

---

## Problem 3: Federated Learning for Privacy
**Difficulty:** Hard

**Challenge:** Train recommendation model without centralizing user data

**Federated averaging:**
1. Server sends model to devices
2. Devices train locally on private data
3. Devices send gradients to server
4. Server aggregates and updates model

**Questions:**
1. What privacy guarantees does this provide?
2. Challenge: Non-IID data (different users have different distributions)
3. How to handle stragglers (slow devices)?
4. Compare: Federated vs. centralized (privacy vs. accuracy)

**Learning Outcomes:** Implement federated learning, preserve privacy, handle non-IID data

---

## Problem 4: Pre-training for Recommendations
**Difficulty:** Hard

**Approach:** Pre-train on large corpus, fine-tune on target task

**Pre-training tasks:**
1. **Masked item prediction:** Mask items in sequence, predict
2. **Next-item prediction:** Predict next item given history
3. **Contrastive learning:** Pull similar user sequences together

**Questions:**
1. What dataset for pre-training? (Public datasets, synthetic)
2. How much does pre-training help cold-start?
3. How to fine-tune? (Freeze encoder, train only head?)
4. Compare: Pre-trained vs. random initialization

**Learning Outcomes:** Design pre-training tasks, fine-tune models, measure benefits

---

## Problem 5: Multi-Task Transfer
**Difficulty:** Hard

**Scenario:** Joint model for multiple tasks (CTR, rating, watch time)

**Knowledge sharing:** Shared representations help all tasks

**Questions:**
1. How to share parameters? (Shared encoder, task-specific heads)
2. What if tasks conflict? (Optimizing one hurts another)
3. How to balance losses from different tasks?
4. When does multi-task help vs. hurt?

**Learning Outcomes:** Design multi-task architectures, handle task conflicts, optimize jointly

---

## Programming Exercises

### Exercise 1: Cross-Domain Embedding Transfer

```python
# Train on source domain (movies)
movie_model = train_model(movie_data)
user_embeddings = movie_model.get_user_embeddings()

# Transfer to target domain (books)
book_model = BookRecommender()
book_model.set_user_embeddings(user_embeddings)  # Initialize with transferred
book_model.train(book_data, freeze_users=False)  # Fine-tune

# Evaluate
baseline = train_from_scratch(book_data)
print(f"Baseline NDCG: {evaluate(baseline, book_test)}")
print(f"Transfer NDCG: {evaluate(book_model, book_test)}")
```

---

### Exercise 2: Implement MAML

```python
class MAML:
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    def adapt(self, support_set):
        # Inner loop: Adapt to task
        adapted_model = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        for x, y in support_set:
            loss = F.mse_loss(adapted_model(x), y)
            loss.backward()
            optimizer.step()

        return adapted_model

    def meta_update(self, tasks):
        # Outer loop: Meta-learning
        meta_loss = 0

        for support_set, query_set in tasks:
            adapted_model = self.adapt(support_set)

            # Evaluate on query set
            for x, y in query_set:
                loss = F.mse_loss(adapted_model(x), y)
                meta_loss += loss

        meta_loss.backward()
        self.meta_optimizer.step()
```

---

### Exercise 3: Federated Learning

```python
class FederatedRecommender:
    def __init__(self, global_model):
        self.global_model = global_model

    def federated_round(self, clients):
        client_models = []

        # Each client trains locally
        for client in clients:
            local_model = copy.deepcopy(self.global_model)
            local_model.train(client.data, epochs=1)
            client_models.append(local_model)

        # Aggregate (FedAvg)
        self.global_model = self.aggregate(client_models)

    def aggregate(self, models):
        # Average parameters
        avg_params = {}
        for name, param in self.global_model.named_parameters():
            avg_params[name] = torch.mean(torch.stack([m.state_dict()[name] for m in models]), dim=0)

        self.global_model.load_state_dict(avg_params)
        return self.global_model
```

---

### Exercise 4: Pre-training with Masked Prediction

```python
def pretrain_masked_prediction(model, sequences, mask_prob=0.15):
    for seq in sequences:
        # Mask random items
        masked_seq = seq.copy()
        masked_indices = random.sample(range(len(seq)), int(len(seq) * mask_prob))

        for idx in masked_indices:
            masked_seq[idx] = MASK_TOKEN

        # Predict masked items
        predictions = model(masked_seq)
        targets = [seq[idx] for idx in masked_indices]

        loss = F.cross_entropy(predictions[masked_indices], targets)
        loss.backward()
        optimizer.step()

# Fine-tune on target task
model.finetune(target_data)
```

---

## Discussion Questions

1. **Negative Transfer:** When does transfer hurt? (Very different domains)
2. **Data Efficiency:** How much target data needed for successful transfer?
3. **Privacy-Utility Trade-off:** Federated learning preserves privacy but may reduce accuracy
4. **Pre-training Data:** What makes good pre-training data for recommendations?
5. **Cold Start:** How does transfer help cold-start users/items?
6. **Multi-Domain Systems:** Single model for all domains vs. domain-specific models?

---

## References
1. Finn, C., et al. (2017). "Model-agnostic meta-learning for fast adaptation of deep networks". ICML. (MAML)
2. McMahan, B., et al. (2017). "Communication-efficient learning of deep networks from decentralized data". AISTATS. (Federated Learning)
3. Hu, G., et al. (2018). "Conet: Collaborative cross networks for cross-domain recommendation". CIKM.

---

*Return to [Week 18 Main Page](README.md)*
