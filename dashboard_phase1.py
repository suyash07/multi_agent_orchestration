import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from agents.ingestion import IngestionAgent
from agents.eda import EDAAgent
from agents.modeling import ModelingAgent
from agents.insight import InsightAgent

st.set_page_config(
    page_title="Bank Churn Intelligence",
    page_icon="🏦",
    layout="wide"
)

@st.cache_data
def run_pipeline():
    ingestion = IngestionAgent()
    X, X_raw, y, schema = ingestion.run()
    stats = EDAAgent().run(X_raw, y)
    model, metrics, probas = ModelingAgent().run(X, y)
    insights = InsightAgent().run(stats, metrics)
    return ingestion, X, X_raw, y, schema, stats, model, metrics, probas, insights

st.title("🏦 Bank Customer Churn Intelligence")
st.caption("Multi-agent analysis pipeline • Random Forest • GPT-4o-mini insights")

with st.spinner("Running pipeline across all agents..."):
    ingestion, X, X_raw, y, schema, stats, model, metrics, probas, insights = run_pipeline()

# KPI Cards
st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{schema['n_rows']:,}")
col2.metric("Churn Rate", f"{schema['churn_rate']}%")
col3.metric("Model ROC-AUC", f"{metrics['roc_auc']}")
col4.metric("High Risk Customers",
            f"{(probas > 0.7).sum():,}",
            help="Customers with >70% churn probability")

st.divider()

# Charts row 1
st.subheader("Churn by Segment")
col1, col2 = st.columns(2)

with col1:
    geo_data = pd.DataFrame({
        "Segment": list(stats["churn_by_geography"].keys()),
        "Churn Rate (%)": list(stats["churn_by_geography"].values())
    })
    fig = px.bar(geo_data, x="Segment", y="Churn Rate (%)",
                 title="Churn by Geography",
                 color="Churn Rate (%)",
                 color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    prod_data = pd.DataFrame({
        "Num Products": [str(k) for k in stats["churn_by_products"].keys()],
        "Churn Rate (%)": list(stats["churn_by_products"].values())
    })
    fig = px.bar(prod_data, x="Num Products", y="Churn Rate (%)",
                 title="Churn by Number of Products",
                 color="Churn Rate (%)",
                 color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

# Charts row 2
col1, col2 = st.columns(2)

with col1:
    active_data = pd.DataFrame({
        "Status": list(stats["churn_by_active"].keys()),
        "Churn Rate (%)": list(stats["churn_by_active"].values())
    })
    fig = px.bar(active_data, x="Status", y="Churn Rate (%)",
                 title="Churn by Membership Activity",
                 color="Churn Rate (%)",
                 color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    top_features = dict(list(metrics["feature_importance"].items())[:8])
    feat_data = pd.DataFrame({
        "Feature": list(top_features.keys()),
        "Importance": list(top_features.values())
    }).sort_values("Importance")
    fig = px.bar(feat_data, x="Importance", y="Feature",
                 orientation="h",
                 title="Top Feature Importances (Random Forest)",
                 color="Importance",
                 color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# AI Insight Report
st.subheader("🤖 AI Business Report")
st.info(insights["report"])

st.divider()

# High risk customer table
st.subheader("🚨 Highest Risk Customers")
df_display = X_raw.copy()
df_display["Churn Probability"] = probas.values
df_display["Risk Level"] = pd.cut(
    probas.values,
    bins=[0, 0.3, 0.7, 1.0],
    labels=["Low", "Medium", "High"]
)
high_risk = (df_display[df_display["Churn Probability"] > 0.7]
             .sort_values("Churn Probability", ascending=False)
             .head(20))
st.dataframe(
    high_risk.style.background_gradient(
        subset=["Churn Probability"], cmap="Reds"
    ),
    use_container_width=True
)

st.divider()

# Single customer predictor
st.subheader("🔍 Single Customer Churn Predictor")
st.caption("Enter a customer's details to get their churn probability")

col1, col2, col3 = st.columns(3)
with col1:
    credit_score = st.slider("Credit Score", 300, 850, 650)
    age = st.slider("Age", 18, 92, 40)
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    balance = st.number_input("Balance ($)", 0, 250000, 50000)
with col2:
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Has Credit Card", [1, 0],
                               format_func=lambda x: "Yes" if x == 1 else "No")
    is_active = st.selectbox("Is Active Member", [1, 0],
                             format_func=lambda x: "Yes" if x == 1 else "No")
    salary = st.number_input("Estimated Salary ($)", 0, 200000, 100000)
with col3:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict Churn Probability", type="primary"):
    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary,
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0
    }])

    numeric_cols = ["CreditScore", "Age", "Tenure", "Balance",
                    "NumOfProducts", "EstimatedSalary"]
    input_data[numeric_cols] = ingestion.scaler.transform(
        input_data[numeric_cols]
    )

    proba = model.predict_proba(input_data)[0][1]

    if proba > 0.7:
        st.error(f"🔴 High Risk: {proba:.1%} churn probability")
    elif proba > 0.3:
        st.warning(f"🟡 Medium Risk: {proba:.1%} churn probability")
    else:
        st.success(f"🟢 Low Risk: {proba:.1%} churn probability")