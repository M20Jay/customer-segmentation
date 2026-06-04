# Week 3 — Customer Segmentation Pipeline
**Martin James Ng'ang'a · MLOps Engineer · Nairobi, Kenya 🇰🇪**
`github.com/M20Jay` · Week 3 of 15

---

## Overview

Unsupervised ML pipeline that segments 7,032 telecom customers into 4 actionable behavioural groups. KMeans clustering with PCA visualisation, MLflow experiment tracking, Evidently drift monitoring, Streamlit dashboard, and FastAPI inference — all containerised with Docker on AWS EC2 Frankfurt.

**The core insight:** Most businesses treat all customers the same. This pipeline finds the 4 groups they never formally defined — and gives each a different action.

---

## The 4 Segments

| Segment | Customers | Avg ARPU | Churn Rate | Action |
|---------|-----------|----------|------------|--------|
| 🏆 High Value Loyal | 1,696 | $89.77 | 24.29% | Reward and retain |
| ⚠️ High Value At Risk | 2,043 | $75.28 | 28.29% | Intervene immediately |
| 📈 Low Value Mid-Tenure | 2,219 | $27.12 | 26.86% | Upsell opportunity |
| 👴 Senior High Value | 1,074 | $83.27 | 25.98% | Senior-specific offers |

---

## Final Results

| Metric | Result |
|--------|--------|
| Dataset | IBM Telco — 7,032 customers · 21 features |
| Optimal K | 4 clusters |
| Silhouette Score | 0.4293 at K=4 |
| Live API | http://18.184.3.203:8004/docs |
| Endpoints | GET /health · POST /segment |
| Dashboard | Streamlit — live on AWS EC2 |

---

## 7-Day Build Plan

| Day | Task | Status |
|-----|------|--------|
| Day 1 | EDA + Feature Engineering | ✅ |
| Day 2 | StandardScaler + PCA + Elbow + Silhouette | ✅ |
| Day 3 | KMeans K=4 + Segment Profiling | ✅ |
| Day 4 | MLflow Experiment Tracking | ✅ |
| Day 5 | FastAPI + PostgreSQL + Docker | ✅ |
| Day 6 | Streamlit dashboard + Evidently drift | ✅ |
| Day 7 | README + Deploy to AWS EC2 | ✅ |

---

## Project Structure

```
customer-segmentation/
├── data/                       IBM Telco dataset
├── models/                     Saved KMeans + scaler + PCA
├── notebooks/                  EDA — exploratory analysis
├── screenshots/                Dashboard and chart screenshots
├── src/
│   ├── data/
│   │   └── preprocessing.py    Load, clean, encode features
│   ├── features/
│   │   └── feature_engineering.py  ARPU, tenure groups
│   └── models/
│       ├── train.py            KMeans training + MLflow tracking
│       └── evaluate.py         Silhouette, elbow, PCA visualisation
├── api/
│   ├── main.py                 FastAPI app — loads model at startup
│   └── routes/
│       ├── health.py           GET /health
│       └── segment.py          POST /segment
├── streamlit_app.py            Interactive business dashboard
├── evidently_report.py         Data drift detection
├── Dockerfile
├── docker-compose.yml          API + PostgreSQL + Streamlit
└── requirements.txt
```

---

## Pipeline Architecture

```
IBM Telco CSV (7,032 customers)
    ↓
src/data/preprocessing.py       → cleaned df
    ↓
src/features/feature_engineering.py  → ARPU, tenure groups
    ↓
StandardScaler                  → scaled features
    ↓
PCA                             → dimensionality reduction (visualisation)
    ↓
KMeans (K=4)                    → cluster labels
    ↓ tracked by MLflow
src/models/train.py             → saves kmeans.pkl + scaler.pkl + pca.pkl
    ↓
api/main.py                     → loads artifacts at startup
    ↓
POST /segment request
    ↓
scale → predict cluster → return segment name + action
    ↓
PostgreSQL                      → save prediction
    ↓
Streamlit dashboard             → reads PostgreSQL → live visualisation
    ↓
Evidently                       → drift detection on new batches
```

---

## Key Concepts

### Why K=4 Not K=6

```python
# K=6 scored marginally higher on silhouette (0.4555 vs 0.4293)
# But K=4 was chosen — why?

# Principle: interpretability over marginal metric gain
# K=4 maps cleanly to actionable business language:
#   High Value Loyal → reward programme
#   High Value At Risk → retention call today
#   Low Value Mid-Tenure → upsell campaign
#   Senior High Value → senior-specific offers

# K=6 gives two extra segments nobody knows what to do with
# A model that confuses the business is worse than a simpler one
```

### Finding Optimal K — Two Methods

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Method 1 — Elbow Method
inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
# Plot inertias — find elbow (point where curve flattens)

# Method 2 — Silhouette Score
silhouette_scores = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
# Higher silhouette = better defined clusters
# Range: -1 (wrong cluster) to +1 (perfect cluster)
# 0.4293 is good for real-world business data
```

### Why StandardScaler Before KMeans

```python
from sklearn.preprocessing import StandardScaler

# KMeans uses Euclidean distance
# Without scaling: monthly_charges (range 18-119) dominates tenure (range 1-72)
# The feature with largest range controls the clusters — wrong

# With scaling: all features have mean=0, std=1
# Distance is meaningful across all features equally

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['tenure', 'monthly_charges', 'total_charges', 'arpu']])

# CRITICAL: fit scaler on training data only
# transform both train and new inference data with same scaler
```

### PCA for Visualisation

```python
from sklearn.decomposition import PCA

# PCA reduces dimensions for 2D/3D visualisation
# Does NOT change the clustering — clustering runs on full feature space
# PCA is only for seeing the clusters visually

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters in 2D PCA space
import matplotlib.pyplot as plt
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
```

### MLflow Experiment Tracking for K Selection

```python
import mlflow

mlflow.set_experiment("customer-segmentation")

for k in range(2, 11):
    with mlflow.start_run(run_name=f"KMeans_K{k}"):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, labels)

        mlflow.log_param("k", k)
        mlflow.log_param("n_init", 10)
        mlflow.log_metric("silhouette_score", sil_score)
        mlflow.log_metric("inertia", km.inertia_)

# MLflow UI shows all K values with their scores side by side
# Makes the K=4 decision documented and reproducible
```

### Evidently Drift Detection

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(
    reference_data=df_reference,  # training data
    current_data=df_current       # new batch of customers
)
report.save_html("reports/drift_report.html")

# If feature distributions shift → segment boundaries may need updating
# E.g. monthly charges distribution shifts → High Value threshold changes
```

### FastAPI Segment Endpoint

```python
from fastapi import APIRouter
from api.schemas.request import CustomerInput
from api.schemas.response import SegmentResponse
from api.main import kmeans, scaler

router = APIRouter()

@router.post("/segment", response_model=SegmentResponse)
async def segment(customer: CustomerInput):
    try:
        features = [[
            customer.tenure,
            customer.monthly_charges,
            customer.total_charges,
            customer.arpu
        ]]
        features_scaled = scaler.transform(features)
        cluster = int(kmeans.predict(features_scaled)[0])

        segment_names = {
            0: "High Value Loyal",
            1: "High Value At Risk",
            2: "Low Value Mid-Tenure",
            3: "Senior High Value"
        }
        actions = {
            0: "Reward and retain",
            1: "Intervene immediately",
            2: "Upsell opportunity",
            3: "Senior-specific offers"
        }

        return SegmentResponse(
            segment=cluster,
            segment_name=segment_names[cluster],
            recommended_action=actions[cluster]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## CLI Reference

### Training Pipeline

```bash
# Run from project root
PYTHONPATH=. python src/data/preprocessing.py
PYTHONPATH=. python src/features/feature_engineering.py
PYTHONPATH=. python src/models/train.py
PYTHONPATH=. python src/models/evaluate.py

# Start MLflow UI to compare K values
mlflow ui --port 5000
# Open http://127.0.0.1:5000

# Generate Evidently drift report
PYTHONPATH=. python evidently_report.py
open reports/drift_report.html
```

### API Testing

```bash
# Health check
curl -s http://localhost:8004/health | python3 -m json.tool

# Segment a customer
curl -s -X POST http://localhost:8004/segment \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 34,
    "monthly_charges": 56.95,
    "total_charges": 1889.50,
    "arpu": 55.57,
    "senior_citizen": 0
  }' | python3 -m json.tool

# Expected response:
# {
#   "segment": 1,
#   "segment_name": "High Value At Risk",
#   "recommended_action": "Intervene immediately"
# }
```

### Streamlit Dashboard

```bash
# Run dashboard locally
streamlit run streamlit_app.py

# Dashboard shows:
# - Segment distribution pie chart
# - ARPU by segment bar chart
# - Churn rate by segment
# - Individual customer lookup
```

### Docker Commands

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs api --tail=20

# Restart API
docker compose restart api

# Stop everything
docker compose down

# Check port on server
sudo ss -tlnp | grep 8004
```

### Database Inspection

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U segmentation -d segmentation

# Inside psql:
\dt
SELECT COUNT(*) FROM predictions;
SELECT segment_name, COUNT(*), AVG(monthly_charges)
  FROM predictions GROUP BY segment_name ORDER BY COUNT(*) DESC;
\q
```

### Git Workflow

```bash
git status
git add src/models/train.py streamlit_app.py
git commit -m "feat: add KMeans K=4 with MLflow tracking"
git push origin main
git log --oneline -5
```

---

## Debugging Reference

### Common Errors and Fixes

| Error | Fix |
|-------|-----|
| `KMeans poor cluster separation` | Check StandardScaler applied before fitting |
| `Silhouette score negative` | Too many clusters — reduce K |
| `MLflow experiment not found` | Create experiment first: `mlflow.create_experiment("customer-segmentation")` |
| `Evidently column mismatch` | Reference and current data must have same column names |
| `Streamlit connection refused` | Check Streamlit running: `ps aux | grep streamlit` |
| `Scaler feature mismatch` | Fit scaler on same features used during training |

### Debugging Order

```
1. Check all containers: docker compose ps
2. Check API logs: docker compose logs api --tail=50
3. Verify model loaded: curl http://localhost:8004/health
4. Check PostgreSQL: docker compose exec postgres psql -U segmentation -d segmentation
5. Verify scaler features match API input features
```

---

## AWS EC2 Deployment

```bash
# SSH to server
ssh -i ~/Documents/GitHub/mlops-key.pem ubuntu@18.184.3.203

# Start segmentation API
cd ~/customer-segmentation
docker compose up -d seg_fastapi

# Verify running
docker ps | grep seg
curl -s http://localhost:8004/health

# Check logs
docker compose logs seg_fastapi --tail=20
```

---

## Key Learnings from Week 3

- **Interpretability over marginal metric gain** — K=4 beats K=6 even though K=6 has slightly better silhouette score — four segments the business can act on beats six segments nobody understands
- **Always scale before KMeans** — StandardScaler is mandatory — unscaled features make the cluster with largest range dominate
- **PCA is for visualisation only** — clustering runs on full feature space — PCA just helps you see the clusters in 2D
- **MLflow makes K selection auditable** — every K value tested is logged with its silhouette score — the K=4 decision is documented and reproducible
- **Evidently catches distribution shift** — when new customer data arrives, Evidently checks if the feature distributions have changed — if so, segments may need retraining

---

*Week 3 of 15 · Customer Segmentation Pipeline · Built in Nairobi, Kenya 🇰🇪*
*Live API: http://18.184.3.203:8004/docs · Repository: https://github.com/M20Jay/customer-segmentation*
