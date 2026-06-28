from agents.ingestion import IngestionAgent
from agents.eda import EDAAgent
from agents.modeling import ModelingAgent
from agents.insight import InsightAgent

def run():
    print("\n" + "="*60)
    print("MULTI-AGENT CHURN PIPELINE STARTING")
    print("="*60 + "\n")

    # Agent 1
    ingestion = IngestionAgent()
    X, X_raw, y, schema = ingestion.run()

    # Agent 2
    stats = EDAAgent().run(X_raw, y)

    # Agent 3
    model, metrics, probas = ModelingAgent().run(X, y)

    # Agent 4
    insights = InsightAgent().run(stats, metrics)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print(f"  Churn rate:    {schema['churn_rate']}%")
    print(f"  Model AUC:     {metrics['roc_auc']}")
    print(f"  High risk:     {(probas > 0.7).sum()} customers")
    print(f"  Top driver:    {list(metrics['feature_importance'].keys())[0]}")
    print("="*60 + "\n")

    return {
        "schema": schema,
        "stats": stats,
        "metrics": metrics,
        "probas": probas,
        "insights": insights
    }

if __name__ == "__main__":
    run()

import subprocess
import sys

if __name__ == "__main__":
    results = run()
    print("\nLaunching dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", 
                   "run", "dashboard.py"])