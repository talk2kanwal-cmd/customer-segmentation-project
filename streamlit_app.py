import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from pipeline import ChurnPredictionPipeline
from data_generator import generate_customer_data
from config import SAVED_MODELS_DIR

# Page config
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Metric Card Styling - Dark box as requested */
    .stMetric {
        background-color: #1e2130 !important;
        padding: 15px !important;
        border-radius: 10px !important;
        border: 1px solid #3e4150 !important;
    }
    /* Force light text color ONLY inside the dark metric cards */
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    [data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }
    
    /* We REMOVED global backgrounds and global tab/button overrides 
       to ensure visibility in Light Mode (white text on white background fix) */
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("🔮 Customer Churn Prediction Dashboard")
st.markdown("Predict customer churn risk using advanced machine learning models.")

# Sidebar
st.sidebar.header("Settings")

# Model selection
available_models = [f.stem for f in SAVED_MODELS_DIR.glob("*.pkl")]
if not available_models:
    st.error("No trained models found. Please run 'python main.py --mode train' first.")
    st.stop()

selected_model = st.sidebar.selectbox(
    "Select Prediction Model",
    options=available_models,
    index=available_models.index("xgboost") if "xgboost" in available_models else 0
)

# Load pipeline
@st.cache_resource
def get_pipeline(model_name):
    pipeline = ChurnPredictionPipeline(model_name=model_name)
    pipeline.load_model_and_scaler()
    return pipeline

try:
    pipeline = get_pipeline(selected_model)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input Options
st.header("📥 Customer Data Input")
tab1, tab2 = st.tabs(["📁 Upload CSV", "🎲 Generate Sample"])

input_df = None

with tab1:
    uploaded_file = st.file_uploader("Upload customer data CSV", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {len(input_df)} records.")

with tab2:
    n_samples = st.slider("Number of samples to generate", 5, 100, 20)
    if st.button("Generate & Load Sample Data"):
        input_df = generate_customer_data(n_samples=n_samples)
        if 'churned' in input_df.columns:
            input_df = input_df.drop(columns=['churned'])
        st.success(f"Generated {n_samples} customer records.")

# Prediction and Analysis
if input_df is not None:
    st.divider()
    st.header("🚀 Prediction Results")
    
    with st.spinner(f"Predicting with {selected_model}..."):
        try:
            results = pipeline.predict(input_df)
            
            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            avg_prob = results['churn_probability'].mean()
            high_risk_count = len(results[results['risk_category'] == 'High Risk'])
            
            col1.metric("Total Customers", len(results))
            col2.metric("Avg. Churn Prob", f"{avg_prob:.1%}")
            col3.metric("High Risk Users", high_risk_count)
            col4.metric("Model Used", selected_model)
            
            # Displays results table
            st.subheader("📋 Prediction Detail")
            # Replace NaNs for display
            display_df = results.copy()
            st.dataframe(display_df, use_container_width=True)
            
            # Visualizations
            st.divider()
            st.header("📊 Risk Analysis")
            
            vcol1, vcol2 = st.columns(2)
            
            with vcol1:
                st.subheader("Risk Category Distribution")
                risk_counts = results['risk_category'].value_counts()
                fig_pie = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    color=risk_counts.index,
                    color_discrete_map={
                        'Low Risk': '#10b981',
                        'Medium Risk': '#f59e0b',
                        'High Risk': '#ef4444'
                    },
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with vcol2:
                st.subheader("Probability Distribution")
                fig_his = px.histogram(
                    results, 
                    x="churn_probability",
                    nbins=20,
                    color_discrete_sequence=['#636EFA']
                )
                fig_his.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Churn Threshold")
                st.plotly_chart(fig_his, use_container_width=True)
                
            # Download button
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f'churn_predictions_{selected_model}.csv',
                mime='text/csv',
            )
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)
else:
    st.info("Please upload a CSV file or generate sample data to see predictions.")

# Footer
st.sidebar.divider()
st.sidebar.info("Customer Churn Prediction System v1.0")
st.sidebar.write("Developed by sumara kanwal as practice projects")
