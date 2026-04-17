# Customer Segmentation Pipeline

**Author:** Martin James Ng'ang'a | [github.com/M20Jay](https://github.com/M20Jay)  
**Status:** 🔨 In Progress — Week 3 of 15  
**Stack:** KMeans · PCA · StandardScaler · MLflow · Evidently · Streamlit · PostgreSQL · AWS

---

## Business Problem

Most businesses treat all customers the same — sending identical campaigns to high value loyal customers and dormant low value ones alike. The result is wasted budget, wrong conversations, and missed revenue.

This pipeline uses unsupervised machine learning to identify distinct customer groups from behavioural data — enabling targeted, data-driven intervention for each segment.

---

## The 4 Segments

| Segment | Description | Action |
|---------|-------------|--------|
| 🏆 High Value Loyal | Long tenure · High ARPU · Low churn risk | Reward and retain |
| ⚠️ High Value At Risk | High ARPU · High churn probability | Intervene immediately |
| 📈 Low Value Engaged | Low ARPU · Active usage | Upsell opportunity |
| 💤 Low Value Dormant | Low ARPU · Low engagement | Win-back campaign |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| KMeans | Customer clustering |
| PCA | Dimensionality reduction for visualisation |
| StandardScaler | Feature scaling before clustering |
| MLflow | Experiment tracking — comparing K values |
| Evidently | Data drift monitoring |
| Streamlit | Interactive dashboard |
| PostgreSQL | Storing segment assignments |
| AWS EC2 | Production deployment |

---

## Progress

| Day | Task | Status |
|-----|------|--------|
| Day 1 | EDA + Feature Engineering | ✅ Complete |
| Day 2 | StandardScaler + PCA + Elbow + Silhouette | ✅ Complete |
| Day 3 | KMeans training + Segment profiling | ⏳ In Progress |
| Day 4 | FastAPI + PostgreSQL + Docker | ⏳ Pending |
| Day 5 | Streamlit dashboard | ⏳ Pending |
| Day 6 | Evidently drift report + Next Best Offer | ⏳ Pending |
| Day 7 | README + GitHub + LinkedIn post | ⏳ Pending |

---

## Dataset

IBM Telco Customer Churn Dataset  
7,032 customers · 21 features · Behavioural and demographic data

---

*Part of a 15-week MLOps programme building production ML systems from scratch.*  
*Week 3 of 15 — Building in public. No shortcuts. 🇰🇪*