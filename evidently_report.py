# ── Evidently Drift Report ────────────────────────────────────────
# Author: Martin James Ng'ang'a | github.com/M20Jay
import pandas as pd
from sqlalchemy import create_engine
from evidently.future.report import Report
from evidently.future.presets import DataDriftPreset

# ── Load Training Data (Reference) ────────────────────────────────
reference_data = pd.read_csv(
    '/Users/martinjames/Documents/GitHub/churn-prediction-pipeline/data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
)

reference_data = reference_data[[
    'tenure', 'MonthlyCharges', 'TotalCharges'
]].copy()

reference_data['TotalCharges'] = pd.to_numeric(
    reference_data['TotalCharges'], errors='coerce'
)
reference_data = reference_data.dropna()
reference_data = reference_data.head(100)

# ── Load Current Data (Predictions) ───────────────────────────────
try:
    engine = create_engine(
        "postgresql://martin:martin123@localhost:5434/segmentation_db"
    )
    current_data = pd.read_sql("""
        SELECT tenure, monthly_charges, total_charges
        FROM segment_predictions
        ORDER BY predicted_at DESC
        LIMIT 100
    """, engine)
    print(f"✅ Loaded {len(current_data)} predictions from database")
except Exception as e:
    print(f"⚠️ Database error: {e}")
    exit()

current_data.columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

# ── Generate Drift Report ──────────────────────────────────────────
report = Report([DataDriftPreset()])

# run() returns a Snapshot object — save_html is on the snapshot
snapshot = report.run(
    reference_data=reference_data,
    current_data=current_data
)

snapshot.save_html('screenshots/evidently_drift_report.html')
print("✅ Drift report saved → screenshots/evidently_drift_report.html")