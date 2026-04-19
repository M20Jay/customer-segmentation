# ── Customer Segmentation Streamlit Dashboard ─────────────────────
# Reads from PostgreSQL and calls FastAPI /segment endpoint
import streamlit as st
import pandas as pd
import requests
import psycopg2
import plotly.express as px
import os
# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────────────────
st.title("📊 Customer Segmentation Dashboard")
st.markdown("**Built by Martin James — MLOps Engineer | github.com/M20Jay**")
st.divider()
# ── Database Connection ───────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://martin:martin123@localhost:5434/segmentation_db"
)
@st.cache_data(ttl=60)
def load_predictions():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        df = pd.read_sql("""
            SELECT segment_name, arpu, monthly_charges, tenure, recommended_action, predicted_at
            FROM segment_predictions
            ORDER BY predicted_at DESC
        """, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Database not connected: {e}")
        return pd.DataFrame()
# ── Load Data ─────────────────────────────────────────────────────
df = load_predictions()
# ── Metrics Row ───────────────────────────────────────────────────
st.subheader("📈 Live Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customer Scored", len(df))

with col2:
    if not df.empty:
        avg_arpu = df['arpu'].mean()
        st.metric("Average ARPU", f"${avg_arpu:.2f}")
    else:
        st.metric("Average ARPU", "N/A")
with col3:
    if not df.empty:
        high_value = len(df[df['segment_name'] == 'High Value Loyal'])
        st.metric("High Value Loyal", high_value)
    else:
        st.metric("High Value Loyal", "N/A")
with col4:
    if not df.empty:
        at_risk = len(df[df['segment_name'] == 'High Value At Risk'])
        st.metric("High Value At Risk", at_risk)
    else:
        st.metric("High Value At Risk", "N/A")

st.divider()
# ── Charts ────────────────────────────────────────────────────────
st.subheader("📊 Segment Distribution")

if not df.empty:
    col1, col2 = st.columns(2)

    with col1:
        # Segment count chart
        segment_counts = df['segment_name'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        fig1 = px.bar(
            segment_counts,
            x='Segment',
            y='Count',
            title='Customers per Segment',
            color='Segment',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Average ARPU per segment
        arpu_by_segment = df.groupby('segment_name')['arpu'].mean().reset_index()
        arpu_by_segment.columns = ['Segment', 'Average ARPU']
        fig2 = px.bar(
            arpu_by_segment,
            x='Segment',
            y='Average ARPU',
            title='Average ARPU per Segment',
            color='Segment',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("No predictions yet. Use the form below to score customers.")

st.divider()

# ── Prediction Form ───────────────────────────────────────────────
st.subheader("🎯 Score a New Customer")
with st.form("segment_form"):
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=50.0
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=600.0
        )

    with col2:
        arpu = st.number_input(
            "ARPU ($)",
            min_value=0.0,
            max_value=200.0,
            value=50.0
        )
        senior_citizen = st.selectbox(
            "Senior Citizen",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    submitted = st.form_submit_button("🎯 Predict Segment")
    
if submitted:
    response =requests.post(
        "http://localhost:8001/segment",
        json={
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "arpu": arpu,
            "senior_citizen": senior_citizen
        }
    )
    if response.status_code == 200:
        result = response.json()
        st.success(f"✅ Segment: **{result['segment_name']}**")
        st.info(f"💡 Recommended Action: {result['recommended_action']}")
    else:
        st.error("❌ API not available. Start the FastAPI server.")
st.divider()
# ── Recent Predictions Table ──────────────────────────────────────
st.subheader("🕐 Recent Predictions")
if not df.empty:
    st.dataframe(
        df[['segment_name', 'arpu', 'tenure',
            'monthly_charges', 'recommended_action',
            'predicted_at']].head(20),
        use_container_width=True
    )
else:
    st.info("No predictions yet.")
# ── Footer ────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align: center; color: grey;'>
    Built by <b>Martin James </b> — MLOps Engineer | 
    <a href='https://github.com/M20Jay'>github.com/M20Jay</a> | 
    Week 3 of 15
    </div>
    """,
    unsafe_allow_html=True
)