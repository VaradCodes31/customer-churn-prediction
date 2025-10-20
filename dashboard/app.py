import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add models to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_model import ChurnPredictor
from intervention_engine import InterventionEngine

st.set_page_config(
    page_title="Customer Churn Prevention System",
    page_icon="🚀",
    layout="wide"
)

# Initialize systems
@st.cache_resource
def load_systems():
    predictor = ChurnPredictor()
    
    try:
        # Try to load existing model
        predictor.load_model()
    except FileNotFoundError:
        # If model doesn't exist, train it
        st.info("🔄 Training model for the first time... This may take a minute.")
        df = pd.read_csv('data/telco_processed.csv')
        predictor.train(df)
        predictor.save_model()
        st.success("✅ Model trained successfully!")
    
    engine = InterventionEngine()
    return predictor, engine

predictor, engine = load_systems()

# Load sample data
@st.cache_data
def load_data():
    return pd.read_csv('data/customer_data.csv')

df = load_data()

# Sidebar
st.sidebar.title("🎯 Churn Prevention System")
st.sidebar.markdown("Identify at-risk customers and take proactive actions")

# Main dashboard
st.title("🚀 Context-Aware Customer Churn Prediction")
st.markdown("Predict who will churn, understand why, and take action")

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

total_customers = len(df)
churned_customers = df['churned'].sum()
avg_churn_risk = df['churn_risk'].mean()

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Churn Rate", f"{(churned_customers/total_customers)*100:.1f}%")
col4.metric("Avg Churn Risk", f"{avg_churn_risk:.1%}")

# Customer analysis section
st.header("🔍 Individual Customer Analysis")

# Customer selector
customer_ids = df['customer_id'].tolist()
selected_customer_id = st.selectbox("Select Customer", customer_ids[:100])

if selected_customer_id:
    customer_data = df[df['customer_id'] == selected_customer_id].iloc[0].to_dict()
    
    # Get prediction and explanation
    risk_score, prediction = predictor.predict_churn_risk(customer_data)
    
    # Diagnose reasons
    reasons = engine.diagnose_churn_reason(customer_data)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Risk Score"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Churn Diagnosis")
        
        if reasons:
            primary_reason = reasons[0]
            intervention = engine.recommend_intervention(risk_score, primary_reason, customer_data)
            
            st.error(f"**Primary Risk Factor:** {primary_reason['reason']}")
            st.write(f"*{primary_reason['description']}*")
            
            st.success(f"**Recommended Action:** {intervention['intervention']}")
            st.info(f"**Urgency:** {intervention['urgency'].replace('_', ' ').title()}")
            
            # Show all reasons
            with st.expander("View All Risk Factors"):
                for i, reason in enumerate(reasons[:3]):
                    st.write(f"{i+1}. **{reason['reason']}** - {reason['description']}")
        else:
            st.success("✅ No significant churn risk factors identified")

# Customer details
st.header("📋 Customer Details")
if selected_customer_id:
    customer_details = df[df['customer_id'] == selected_customer_id].iloc[0]
    st.dataframe(customer_details, use_container_width=True)

# Feature importance
st.header("📊 Model Insights")
col1, col2 = st.columns(2)

with col1:
    # Customer distribution by risk
    fig = px.histogram(
        df, x='churn_risk',
        title="Customer Distribution by Churn Risk",
        nbins=20
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Churn by contract type
    fig = px.box(
        df, x='contract_type', y='churn_risk',
        title="Churn Risk by Contract Type"
    )
    st.plotly_chart(fig, use_container_width=True)

# Batch analysis
st.header("📈 High-Risk Customer Portfolio")

high_risk_threshold = st.slider("High Risk Threshold", 0.1, 1.0, 0.7)
high_risk_customers = df[df['churn_risk'] > high_risk_threshold]

st.metric("High-Risk Customers", len(high_risk_customers))

if not high_risk_customers.empty:
    # Show high-risk customers table
    st.dataframe(
        high_risk_customers[['customer_id', 'churn_risk', 'tenure_days', 'monthly_charges', 'support_calls_3mo']].head(10),
        use_container_width=True
    )
    
    # Export option
    if st.button("📥 Export High-Risk List"):
        high_risk_customers.to_csv('high_risk_customers.csv', index=False)
        st.success("High-risk customer list exported!")