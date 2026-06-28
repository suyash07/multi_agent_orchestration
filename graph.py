from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from agents.ingestion import IngestionAgent
from agents.eda import EDAAgent
from agents.modeling import ModelingAgent
from agents.insight import InsightAgent
import operator

# ─────────────────────────────────────────
# 1. STATE
# This is the shared memory between all nodes
# Every agent reads from and writes to this
# ─────────────────────────────────────────
class PipelineState(TypedDict):
    # Ingestion outputs
    X: object
    X_raw: object
    y: object
    schema: dict
    ingestion_agent: object   # keep scaler for dashboard predictor

    # EDA outputs
    stats: dict

    # Modeling outputs
    model: object
    metrics: dict
    probas: object

    # Insight outputs
    insights: dict

    # Orchestrator decisions
    model_approved: bool
    retry_count: int
    messages: list

# ─────────────────────────────────────────
# 2. NODES
# Each node is one agent's job
# Takes state in, returns updated state out
# ─────────────────────────────────────────
def ingestion_node(state: PipelineState) -> PipelineState:
    print("\n[Graph] Running IngestionNode...")
    agent = IngestionAgent()
    X, X_raw, y, schema = agent.run()
    return {
        **state,
        "X": X,
        "X_raw": X_raw,
        "y": y,
        "schema": schema,
        "ingestion_agent": agent
    }

def eda_node(state: PipelineState) -> PipelineState:
    print("[Graph] Running EDANode...")
    stats = EDAAgent().run(state["X_raw"], state["y"])
    return {**state, "stats": stats}

def modeling_node(state: PipelineState) -> PipelineState:
    print("[Graph] Running ModelingNode...")
    model, metrics, probas = ModelingAgent().run(
        state["X"], state["y"]
    )
    return {
        **state,
        "model": model,
        "metrics": metrics,
        "probas": probas,
        "retry_count": state.get("retry_count", 0)
    }

def insight_node(state: PipelineState) -> PipelineState:
    print("[Graph] Running InsightNode...")
    insights = InsightAgent().run(state["stats"], state["metrics"])
    return {**state, "insights": insights}

# ─────────────────────────────────────────
# 3. CONDITIONAL ROUTING
# This is what makes it an orchestrator
# not just a script
# ─────────────────────────────────────────
def should_retry_model(state: PipelineState) -> str:
    auc = state["metrics"]["roc_auc"]
    retry_count = state.get("retry_count", 0)

    if auc < 0.80 and retry_count < 1:
        print(f"[Graph] AUC {auc} below threshold. Retrying with XGBoost...")
        return "retry"
    else:
        print(f"[Graph] AUC {auc} approved. Moving to insights...")
        return "approved"

def retry_with_xgboost(state: PipelineState) -> PipelineState:
    print("[Graph] Running XGBoost retry...")
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    X, y = state["X"], state["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    xgb = XGBClassifier(
        scale_pos_weight=4,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    print(f"[Graph] XGBoost ROC-AUC: {auc:.4f}")

    import pandas as pd
    probas = pd.Series(
        xgb.predict_proba(X)[:, 1],
        index=X.index
    ).round(4)

    updated_metrics = {
        **state["metrics"],
        "best_model": "XGBoost",
        "roc_auc": round(auc, 4),
        "feature_importance": dict(zip(
            list(X.columns),
            xgb.feature_importances_.round(4)
        ))
    }

    return {
        **state,
        "model": xgb,
        "metrics": updated_metrics,
        "probas": probas,
        "retry_count": state.get("retry_count", 0) + 1
    }

# ─────────────────────────────────────────
# 4. BUILD THE GRAPH
# ─────────────────────────────────────────
def build_graph():
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("ingestion", ingestion_node)
    graph.add_node("eda", eda_node)
    graph.add_node("modeling", modeling_node)
    graph.add_node("retry_xgboost", retry_with_xgboost)
    graph.add_node("insight", insight_node)

    # Add edges (fixed sequence)
    graph.add_edge("ingestion", "eda")
    graph.add_edge("eda", "modeling")

    # Conditional edge after modeling
    graph.add_conditional_edges(
        "modeling",
        should_retry_model,
        {
            "retry": "retry_xgboost",
            "approved": "insight"
        }
    )

    # After retry, go to insight
    graph.add_edge("retry_xgboost", "insight")
    graph.add_edge("insight", END)

    # Entry point
    graph.set_entry_point("ingestion")

    return graph.compile()

# ─────────────────────────────────────────
# 5. RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("\n" + "="*60)
    print("LANGGRAPH ORCHESTRATOR STARTING")
    print("="*60)

    pipeline = build_graph()

    final_state = pipeline.invoke({
        "X": None,
        "X_raw": None,
        "y": None,
        "schema": {},
        "ingestion_agent": None,
        "stats": {},
        "model": None,
        "metrics": {},
        "probas": None,
        "insights": {},
        "model_approved": False,
        "retry_count": 0,
        "messages": []
    })

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print(f"  Churn rate:    {final_state['schema']['churn_rate']}%")
    print(f"  Model:         {final_state['metrics']['best_model']}")
    print(f"  ROC-AUC:       {final_state['metrics']['roc_auc']}")
    print(f"  High risk:     {(final_state['probas'] > 0.7).sum()} customers")
    print(f"  Top driver:    {list(final_state['metrics']['feature_importance'].keys())[0]}")
    print("="*60 + "\n")