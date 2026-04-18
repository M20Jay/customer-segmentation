# ── Customer Segmentation FastAPI ────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from sqlalchemy import create_engine,text
from prometheus_fastapi_instrumentator import Instrumentator

# ── App Setup ─────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Segmentation API",
    description="Telecom customer segmentation using KMeans clustering — 4 behavioural segments. Built by Martin James — MLOps Engineer",
    version="1.0.0"
)

Instrumentator().instrument(app).expose(app)

# ── Database Connection ───────────────────────────────────────────
# Connects to PostgreSQL to store predictions
# DATABASE_URL reads from environment variable
# Falls back to local Docker connection if not set

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://martin:martin123@localhost:5434/segmentation_db"

)
try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS segment_predictions(
                id                  SERIAL PRIMARY KEY,
                tenure              FLOAT,
                monthly_charges     FLOAT,  
                total_charges       FLOAT,
                arpu                FLOAT,
                senior_citizen      INTEGER,
                segment             INTEGER,
                segment_name        VARCHAR(50),
                recommended_action  VARCHAR(100),
                predicted_at        TIMESTAMP DEFAULT NOW()
            )

        """))
        conn.commit()
    print("✅ Database connected and table ready")
except Exception as e:
    print(f" ⚠️ Database not available: {e}")
    engine = None

# ── Input Model ───────────────────────────────────────────────────
# Defines what data the API accepts
# Pydantic validates automatically
class CustomerFeatures(BaseModel):
    tenure: float
    monthly_charges: float
    total_charges: float
    arpu: float
    senior_citizen: int

# ── Segment Mapping ───────────────────────────────────────────────
segment_names = {
    0: "Low Value Mid-Tenure",
    1: "High Value At Risk",
    2: "High Value Loyal",
    3: "Senior High Value"
}

recommended_actions = {
    0: "Upsell to higher plan",
    1: "Immediate retention call",
    2: "Loyalty rewards programme",
    3: "Senior-specific offers"
}
# ── Load Model ────────────────────────────────────────────────────
# kmeans_model.pkl → saved from notebook Section 10
# scaler.pkl → saved from notebook Section 6
# Both saved to models/ folder
MODEL_PATH = "models/kmeans_model.pkl"
SCALER_PATH = "models/scaler.pkl"

try:
    kmeans_model = joblib.load(MODEL_PATH)
    scaler_model = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully")

except Exception as e:
    print(f"⚠️ Model not found: {e}")
    kmeans_model = None
    scaler_model = None
# ── Health Endpoint ───────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status" : "ok",
        "model loaded" : kmeans_model is not None,
        "database connected" : engine is not None
    }
# ── Segment Endpoint ──────────────────────────────────────────────
@app.post("/segment")
def predict_segment(customer: CustomerFeatures):

    if kmeans_model is None:
        return {"error": "Model not loaded"}

    # Prepare input — from CustomerFeatures input model
    input_data = pd.DataFrame([{
        'tenure': customer.tenure,
        'MonthlyCharges': customer.monthly_charges,
        'TotalCharges': customer.total_charges,
        'ARPU': customer.arpu,
        'SeniorCitizen': customer.senior_citizen
    }])

    # Scale using saved scaler
    input_scaled = scaler_model.transform(input_data)

    # Predict segment
    segment = int(kmeans_model.predict(input_scaled)[0])
    name = segment_names[segment]
    action = recommended_actions[segment]

    # Save to PostgreSQL
    if engine is not None:
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO segment_predictions
                    (tenure, monthly_charges, total_charges,
                     arpu, senior_citizen, segment,
                     segment_name, recommended_action)
                    VALUES (:tenure, :monthly_charges,
                            :total_charges, :arpu,
                            :senior_citizen, :segment,
                            :segment_name, :recommended_action)
                """), {
                    "tenure": customer.tenure,
                    "monthly_charges": customer.monthly_charges,
                    "total_charges": customer.total_charges,
                    "arpu": customer.arpu,
                    "senior_citizen": customer.senior_citizen,
                    "segment": segment,
                    "segment_name": name,
                    "recommended_action": action
                })
                conn.commit()
        except Exception as e:
            print(f"⚠️ Could not save prediction: {e}")

    return {
        "segment": segment,
        "segment_name": name,
        "recommended_action": action,
        "input": {
            "tenure": customer.tenure,
            "monthly_charges": customer.monthly_charges,
            "arpu": customer.arpu
        }
    }