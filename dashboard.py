import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage

from agents.ingestion import IngestionAgent
from agents.eda import EDAAgent
from agents.modeling import ModelingAgent
from agents.insight import InsightAgent
from tools import tools

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Intelligence",
    page_icon="🏦",
    layout="wide"
)

# ─────────────────────────────────────────
# LIGHT THEME STYLE
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .block-container { padding-top: 2rem; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 2px solid #e0e0e0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
        color: #555555;
        padding: 8px 4px;
    }
    .stTabs [aria-selected="true"] {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #1f77b4;
    }
    .chat-message-user {
        background: #e8f4fd;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 3px solid #1f77b4;
    }
    .chat-message-assistant {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 3px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# PIPELINE (cached, runs once)
# ─────────────────────────────────────────
@st.cache_resource
def run_pipeline():
    ingestion = IngestionAgent()
    X, X_raw, y, schema = ingestion.run()
    stats = EDAAgent().run(X_raw, y)
    model, metrics, probas = ModelingAgent().run(X, y)
    insights = InsightAgent().run(stats, metrics)
    return ingestion, X, X_raw, y, schema, stats, model, metrics, probas, insights

@st.cache_resource
def get_agent_executor():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
You are a senior bank analyst assistant with access to a
customer churn prediction system.

You help bank analysts, customer service reps, and managers
understand and act on customer churn risk.

You have access to four tools:
- get_portfolio_summary: overall churn picture
- get_churn_drivers: why customers are churning
- get_high_risk_customers: who is most at risk
- predict_churn: churn probability for one customer

RULES:
1. Always use a tool to answer. Never make up numbers.
2. Be concise and actionable. This is a business context.
3. When listing customers, prioritize actionability.
4. If asked something outside your tools, say so honestly.
5. Always end with a clear recommended action.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True
    )

# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────
st.title("🏦 Bank Churn Intelligence")
st.caption("Multi-agent pipeline · Random Forest · GPT-4o-mini")

with st.spinner("Running pipeline..."):
    ingestion, X, X_raw, y, schema, stats, model, metrics, probas, insights = run_pipeline()
    executor = get_agent_executor()

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Ask the Analyst"])

# ═════════════════════════════════════════
# TAB 1: DASHBOARD
# ═════════════════════════════════════════
with tab1:

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
                     color_continuous_scale="Blues",
                     template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        prod_data = pd.DataFrame({
            "Num Products": [str(k) for k in
                             stats["churn_by_products"].keys()],
            "Churn Rate (%)": list(stats["churn_by_products"].values())
        })
        fig = px.bar(prod_data, x="Num Products", y="Churn Rate (%)",
                     title="Churn by Number of Products",
                     color="Churn Rate (%)",
                     color_continuous_scale="Blues",
                     template="plotly_white")
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
                     color_continuous_scale="Blues",
                     template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_features = dict(list(
            metrics["feature_importance"].items())[:8])
        feat_data = pd.DataFrame({
            "Feature": list(top_features.keys()),
            "Importance": list(top_features.values())
        }).sort_values("Importance")
        fig = px.bar(feat_data, x="Importance", y="Feature",
                     orientation="h",
                     title="Feature Importances (Random Forest)",
                     color="Importance",
                     color_continuous_scale="Blues",
                     template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # AI Report
    st.subheader("🤖 AI Business Report")
    st.info(insights["report"])

    st.divider()

    # High risk table
    st.subheader("🚨 Highest Risk Customers")
    df_display = X_raw.copy()
    df_display["Churn Probability"] = probas.values
    df_display["Risk Level"] = pd.cut(
        probas.values,
        bins=[0, 0.3, 0.7, 1.0],
        labels=["Low", "Medium", "High"]
    )
    high_risk = (
        df_display[df_display["Churn Probability"] > 0.7]
        .sort_values("Churn Probability", ascending=False)
        .head(20)
    )
    st.dataframe(
        high_risk.style.background_gradient(
            subset=["Churn Probability"], cmap="Blues"
        ),
        use_container_width=True
    )

    st.divider()

    # Single customer predictor
    st.subheader("🔍 Single Customer Predictor")
    st.caption("Enter customer details to get churn probability")

    col1, col2, col3 = st.columns(3)
    with col1:
        credit_score = st.slider("Credit Score", 300, 850, 650)
        age          = st.slider("Age", 18, 92, 40)
        tenure       = st.slider("Tenure (years)", 0, 10, 5)
        balance      = st.number_input("Balance ($)", 0, 250000, 50000)
    with col2:
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card  = st.selectbox("Has Credit Card", [1, 0],
                                    format_func=lambda x:
                                    "Yes" if x == 1 else "No")
        is_active    = st.selectbox("Is Active Member", [1, 0],
                                    format_func=lambda x:
                                    "Yes" if x == 1 else "No")
        salary       = st.number_input("Estimated Salary ($)",
                                       0, 200000, 100000)
    with col3:
        geography = st.selectbox("Geography",
                                 ["France", "Germany", "Spain"])
        gender    = st.selectbox("Gender", ["Male", "Female"])

    if st.button("Predict Churn Probability", type="primary"):
        input_data = pd.DataFrame([{
            "CreditScore":      credit_score,
            "Age":              age,
            "Tenure":           tenure,
            "Balance":          balance,
            "NumOfProducts":    num_products,
            "HasCrCard":        has_cr_card,
            "IsActiveMember":   is_active,
            "EstimatedSalary":  salary,
            "Geography_Germany": 1 if geography == "Germany" else 0,
            "Geography_Spain":   1 if geography == "Spain" else 0,
            "Gender_Male":       1 if gender == "Male" else 0
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

# ═════════════════════════════════════════
# TAB 2: CHAT INTERFACE
# ═════════════════════════════════════════
with tab2:
    st.subheader("💬 Ask the Analyst")
    st.caption("Ask any question about customer churn in plain English.")

    # suggested questions
    st.markdown("**Try asking:**")
    cols = st.columns(3)
    suggestions = [
        "Give me an overview of churn",
        "Which customers should I call this week?",
        "Why are German customers churning?",
        "Is cross-selling a good idea?",
        "Who are the top 5 customers at risk?",
        "Predict churn for a 52 year old German female with $180k balance, 3 products, inactive"
    ]
    for i, suggestion in enumerate(suggestions):
        if cols[i % 3].button(suggestion, key=f"suggestion_{i}"):
            st.session_state.pending_question = suggestion

    st.divider()

    # initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "lc_history" not in st.session_state:
        st.session_state.lc_history = []

    # display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-message-user">🧑 <b>You:</b> {msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-message-assistant">🤖 <b>Analyst:</b> {msg["content"]}</div>',
                unsafe_allow_html=True
            )

    # chat input
    user_input = st.chat_input("Ask about churn...")

    # handle suggestion button clicks
    if "pending_question" in st.session_state:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question

    if user_input:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.spinner("Analyst is thinking..."):
            try:
                response = executor.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.lc_history
                })
                answer = response["output"]

                st.session_state.lc_history.append(
                    HumanMessage(content=user_input)
                )
                st.session_state.lc_history.append(
                    AIMessage(content=answer)
                )
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Sorry, I ran into an issue: {str(e)}"
                })

        st.rerun()