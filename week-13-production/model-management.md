# Week 13: Model Management and MLOps

## Overview

**MLOps**: DevOps for machine learning - deploy, monitor, update models.

**Key processes**:
1. **Versioning**: Track models, data, code
2. **Deployment**: Canary, blue-green, shadow
3. **Monitoring**: Performance degradation, drift
4. **A/B testing**: Validate improvements
5. **Rollback**: Quick recovery from failures

---

## Model Versioning

### Why Version?

**Problem**: Which model version is in production?

**Need to track**:
- Model architecture
- Hyperparameters
- Training data
- Code version

---

### Tools

**MLflow**:
```python
import mlflow

mlflow.start_run()

# Log params
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param("embedding_dim", 128)

# Train model
model = train_model(...)

# Log metrics
mlflow.log_metric("ndcg", 0.75)

# Save model
mlflow.pytorch.log_model(model, "model")

mlflow.end_run()
```

**DVC** (Data Version Control):
```bash
# Track data
dvc add data/interactions.csv
dvc add models/recommender.pt

# Commit to git
git add data/.gitignore models/.gitignore
git commit -m "Track data and model"

# Push data to cloud storage
dvc push
```

---

## Deployment Strategies

### Canary Deployment

**Idea**: Deploy to small % of traffic, monitor, gradually increase.

**Process**:
1. **5% traffic** → new model
2. Monitor metrics (24 hours)
3. If good → **20% traffic**
4. Repeat → **100%**

```python
def get_model(user_id):
    if hash(user_id) % 100 < 5:  # 5% traffic
        return new_model
    else:
        return old_model
```

---

### Blue-Green Deployment

**Two environments**:
- **Blue**: Current production
- **Green**: New version

**Process**:
1. Deploy to green
2. Test green
3. Switch traffic: blue → green
4. Keep blue as backup

---

### Shadow Mode

**Run new model** alongside production, **don't serve** results.

**Purpose**: Measure performance without risk.

**Example**:
```python
def recommend(user_id):
    # Production model
    prod_recs = prod_model.predict(user_id)

    # Shadow model (log predictions, don't serve)
    shadow_recs = shadow_model.predict(user_id)
    log_shadow_predictions(user_id, shadow_recs)

    return prod_recs  # Serve only production
```

---

## Monitoring

### Metrics to Track

**1. Model metrics**:
- Accuracy: NDCG, Precision@K
- Diversity, coverage
- Latency (p50, p95, p99)

**2. Business metrics**:
- CTR, conversion rate
- Revenue per user
- User engagement (session length, return rate)

**3. System metrics**:
- QPS (queries per second)
- Error rate
- CPU/memory usage

---

### Drift Detection

**Concept drift**: Data distribution changes → model degrades.

**Example**: COVID-19 → user preferences shift (travel ↓, home entertainment ↑).

---

### Statistical Tests

**KS Test** (Kolmogorov-Smirnov): Compare feature distributions.

```python
from scipy.stats import ks_2samp

# Training data distribution
train_features = [0.2, 0.5, 0.3, ...]

# Production data distribution
prod_features = [0.1, 0.6, 0.4, ...]

# Test
statistic, p_value = ks_2samp(train_features, prod_features)

if p_value < 0.05:
    print("Distribution shift detected! Retrain model.")
```

---

### Alerting

**Set thresholds**:
- NDCG drops >5% → **alert**
- Latency p99 >200ms → **alert**
- Error rate >1% → **alert**

**Tools**: Prometheus, Grafana, PagerDuty

---

## A/B Testing

### Design

**Control**: Existing model
**Treatment**: New model

**Randomization**: Assign users randomly (50/50 split).

**Duration**: 1-2 weeks (statistical significance).

---

### Statistical Significance

**Hypothesis test**:
- $H_0$: Control = Treatment
- $H_1$: Treatment > Control

**t-test**:
```python
from scipy.stats import ttest_ind

control_metrics = [0.7, 0.72, 0.68, ...]  # NDCG per user
treatment_metrics = [0.73, 0.75, 0.71, ...]

t_stat, p_value = ttest_ind(control_metrics, treatment_metrics)

if p_value < 0.05:
    print(f"Statistically significant! Deploy treatment.")
else:
    print("No significant difference.")
```

---

### Sample Size

**Question**: How many users needed for test?

**Power analysis**:
$$n = \frac{2(Z_{\alpha/2} + Z_\beta)^2 \sigma^2}{\delta^2}$$

where:
- $\delta$ = minimum detectable effect (e.g., 1% CTR improvement)
- $\sigma$ = standard deviation
- $Z_{\alpha/2}$ = 1.96 (95% confidence)
- $Z_\beta$ = 0.84 (80% power)

**Typical**: 10K-100K users per variant.

---

## Continuous Training

### Workflow

**Automate**:
1. **Data collection**: Stream interactions → data lake
2. **Feature engineering**: Daily batch job
3. **Model training**: Trigger when new data available
4. **Evaluation**: Offline metrics on validation set
5. **Deployment**: If metrics improve → deploy

**Tools**: Airflow, Kubeflow, MLflow

---

### Example Pipeline

```python
# Airflow DAG
from airflow import DAG
from airflow.operators import PythonOperator

dag = DAG('recsys_training', schedule_interval='@daily')

def extract_data():
    # Pull interactions from DB
    pass

def train_model():
    # Train on new data
    pass

def evaluate_model():
    # Compute NDCG
    pass

def deploy_model():
    # If NDCG > threshold, deploy
    pass

extract = PythonOperator(task_id='extract', python_callable=extract_data, dag=dag)
train = PythonOperator(task_id='train', python_callable=train_model, dag=dag)
evaluate = PythonOperator(task_id='evaluate', python_callable=evaluate_model, dag=dag)
deploy = PythonOperator(task_id='deploy', python_callable=deploy_model, dag=dag)

extract >> train >> evaluate >> deploy
```

---

## Rollback Strategy

### When to Rollback

**Triggers**:
- Metrics drop >10%
- Error rate spikes
- User complaints increase

**Process**:
1. **Detect issue** (monitoring alerts)
2. **Rollback** to previous version (< 5 min)
3. **Root cause analysis**
4. **Fix and redeploy**

---

### Feature Flags

**Control deployment** without code changes.

```python
if feature_flag('new_ranking_model'):
    recs = new_model.predict(user_id)
else:
    recs = old_model.predict(user_id)
```

**Benefits**:
- **Quick rollback**: Flip flag (no redeploy)
- **Gradual rollout**: Increase % over time
- **A/B testing**: Easy to configure

---

## Summary

**Key Takeaways**:
1. **Versioning**: MLflow, DVC for reproducibility
2. **Deployment**: Canary (gradual), shadow (safe)
3. **Monitoring**: Metrics, drift detection, alerts
4. **A/B testing**: Statistical significance, power analysis
5. **Continuous training**: Automated pipelines (Airflow)
6. **Rollback**: Feature flags, quick recovery

**MLOps Best Practices**:
- Version everything (data, model, code)
- Monitor continuously (metrics + drift)
- Test before deploying (shadow mode, canary)
- Automate retraining (stay fresh)
- Plan for failure (rollback strategy)

**Next**: Industry case studies.

---

## References

1. **Sculley, D., et al. (2015)**. "Hidden Technical Debt in Machine Learning Systems". *NeurIPS*.
2. **Breck, E., et al. (2017)**. "The ML Test Score: A Rubric for ML Production Readiness". *NIPS MLSys Workshop*.
3. **Polyzotis, N., et al. (2018)**. "Data Lifecycle Challenges in Production Machine Learning". *SIGMOD*.
