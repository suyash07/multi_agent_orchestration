import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from agents.ingestion import IngestionAgent
from agents.eda import EDAAgent
from agents.modeling import ModelingAgent
from config import RISK_THRESHOLDS, MODEL_CONFIG, DATA_CONFIG, TOOL_CONFIG

# ─────────────────────────────────────────
# BOOTSTRAP PIPELINE ONCE
# ─────────────────────────────────────────
print("Bootstrapping pipeline for tools...")
_ingestion = IngestionAgent()
_X, _X_raw, _y, _schema = _ingestion.run()
_stats = EDAAgent().run(_X_raw, _y)
_model, _metrics, _probas = ModelingAgent().run(_X, _y)

# ─────────────────────────────────────────
# SCHEMA DRIVEN COLUMN DETECTION
# No hardcoded column names
# ─────────────────────────────────────────
GEOGRAPHY_COLS = [c for c in _X_raw.columns if c.startswith("Geography_")]
GENDER_COLS    = [c for c in _X_raw.columns if c.startswith("Gender_")]

def decode_category(row, cols, base_category):
    """
    Decodes a one-hot encoded category back to its label.
    Falls back to base_category if no column is 1.
    Works for any one-hot encoded column group.
    """
    for col in cols:
        if row[col] == 1:
            # strips prefix e.g. Geography_Germany → Germany
            return col.split("_", 1)[1]
    return base_category

def decode_row(row):
    """
    Decodes a full customer row into human readable dict.
    Driven entirely by config, no hardcoded values.
    """
    return {
        "geography": decode_category(
            row, GEOGRAPHY_COLS,
            DATA_CONFIG["base_categories"]["Geography"]
        ),
        "gender": decode_category(
            row, GENDER_COLS,
            DATA_CONFIG["base_categories"]["Gender"]
        ),
        "age":        row.get("Age", "N/A"),
        "balance":    row.get("Balance", 0),
        "products":   row.get("NumOfProducts", "N/A"),
        "active":     "Yes" if row.get("IsActiveMember", 0) == 1 else "No",
        "credit":     row.get("CreditScore", "N/A"),
        "tenure":     row.get("Tenure", "N/A"),
        "salary":     row.get("EstimatedSalary", 0)
    }

def get_risk_label(proba: float) -> tuple:
    """
    Returns risk label and recommended action.
    Driven by config thresholds not hardcoded values.
    """
    if proba >= RISK_THRESHOLDS["high"]:
        return "HIGH RISK", "Immediate retention outreach recommended"
    elif proba >= RISK_THRESHOLDS["medium"]:
        return "MEDIUM RISK", "Monitor closely, consider proactive engagement"
    else:
        return "LOW RISK", "No immediate action needed"

print(f"Pipeline ready.")
print(f"Risk thresholds: high={RISK_THRESHOLDS['high']}, "
      f"medium={RISK_THRESHOLDS['medium']}")
print(f"Geography columns detected: {GEOGRAPHY_COLS}")
print(f"Gender columns detected:    {GENDER_COLS}\n")

# ─────────────────────────────────────────
# TOOL 1: Get high risk customers
# ─────────────────────────────────────────
@tool
def get_high_risk_customers(
    threshold: float = None,
    top_n: int = None
) -> str:
    """
    Returns the top N customers with highest churn probability.
    Use this when asked about which customers to prioritize,
    who to call, or who is most at risk.
    threshold: minimum churn probability (default from config)
    top_n: number of customers to return (default from config)
    """
    # use config defaults if not specified by the agent
    threshold = threshold or RISK_THRESHOLDS["high"]
    top_n     = min(top_n or TOOL_CONFIG["default_top_n"],
                    TOOL_CONFIG["max_top_n"])

    df = _X_raw.copy()
    df["churn_probability"] = _probas.values
    df["customer_id"] = range(1, len(df) + 1)

    high_risk = (
        df[df["churn_probability"] >= threshold]
        .sort_values("churn_probability", ascending=False)
        .head(top_n)
    )

    result = (f"Top {len(high_risk)} customers with churn probability"
              f" >= {threshold:.0%}:\n\n")

    for _, row in high_risk.iterrows():
        decoded = decode_row(row)
        risk_label, action = get_risk_label(row["churn_probability"])
        result += (
            f"Customer {int(row['customer_id'])}: "
            f"{row['churn_probability']:.0%} | "
            f"{risk_label} | "
            f"Age: {decoded['age']:.0f} | "
            f"Balance: ${decoded['balance']:,.0f} | "
            f"Products: {decoded['products']:.0f} | "
            f"Active: {decoded['active']} | "
            f"Geography: {decoded['geography']} | "
            f"Gender: {decoded['gender']}\n"
        )

    total = (df["churn_probability"] >= threshold).sum()
    result += f"\nTotal customers above {threshold:.0%} threshold: {total}"
    return result

# ─────────────────────────────────────────
# TOOL 2: Get churn drivers
# ─────────────────────────────────────────
@tool
def get_churn_drivers(segment: str = "overall") -> str:
    """
    Returns the key drivers of churn and segment statistics.
    Use this when asked why customers are churning, what factors
    drive churn, or about patterns in a specific segment.
    segment options: 'overall', 'germany', 'inactive', 'products'
    """
    result = f"Churn drivers analysis ({segment}):\n\n"

    result += "TOP FEATURE IMPORTANCES:\n"
    for feat, imp in list(_metrics["feature_importance"].items())[:5]:
        bar = "█" * int(imp * 50)
        result += f"  {feat:<22} {bar} {imp:.4f}\n"

    result += "\nSEGMENT CHURN RATES:\n"

    if segment in ("overall", "germany"):
        result += "  Geography:\n"
        for geo, rate in _stats["churn_by_geography"].items():
            result += f"    {geo}: {rate}%\n"

    if segment in ("overall", "inactive"):
        result += "  Activity status:\n"
        for status, rate in _stats["churn_by_active"].items():
            result += f"    {status}: {rate}%\n"

    if segment in ("overall", "products"):
        result += "  Number of products:\n"
        for n, rate in _stats["churn_by_products"].items():
            result += f"    {n} products: {rate}%\n"

    result += f"\nMODEL: {_metrics['best_model']} | "
    result += f"ROC-AUC: {_metrics['roc_auc']} | "
    result += f"Overall churn: {_stats['overall_churn_rate']}%\n"

    return result

# ─────────────────────────────────────────
# TOOL 3: Predict churn for one customer
# ─────────────────────────────────────────
@tool
def predict_churn(
    age: int,
    balance: float,
    credit_score: int,
    num_products: int,
    is_active: int,
    geography: str,
    gender: str,
    tenure: int = 5,
    has_cr_card: int = 1,
    estimated_salary: float = 100000
) -> str:
    """
    Predicts churn probability for a single customer.
    Use this when asked about a specific customer profile.
    geography: France, Germany, or Spain
    gender: Male or Female
    is_active: 1 for active, 0 for inactive
    """
    # Build input dynamically using schema
    # no hardcoded Geography_Germany or Gender_Male
    input_dict = {
        "CreditScore":     credit_score,
        "Age":             age,
        "Tenure":          tenure,
        "Balance":         balance,
        "NumOfProducts":   num_products,
        "HasCrCard":       has_cr_card,
        "IsActiveMember":  is_active,
        "EstimatedSalary": estimated_salary
    }

    # dynamically add one-hot columns from schema
    for col in GEOGRAPHY_COLS:
        country = col.split("_", 1)[1]
        input_dict[col] = 1 if geography == country else 0

    for col in GENDER_COLS:
        g = col.split("_", 1)[1]
        input_dict[col] = 1 if gender == g else 0

    input_data = pd.DataFrame([input_dict])

    # scale using same fitted scaler
    input_data[DATA_CONFIG["numeric_cols"]] = (
        _ingestion.scaler.transform(
            input_data[DATA_CONFIG["numeric_cols"]]
        )
    )

    proba = _model.predict_proba(input_data)[0][1]
    risk_label, action = get_risk_label(proba)

    return f"""
Customer Churn Prediction:
  Churn Probability: {proba:.1%}
  Risk Level:        {risk_label}
  Action:            {action}

Profile:
  Age: {age} | Geography: {geography} | Gender: {gender}
  Balance: ${balance:,.0f} | Products: {num_products}
  Active: {'Yes' if is_active == 1 else 'No'}
  Credit Score: {credit_score} | Tenure: {tenure} years

Thresholds used:
  High risk  >= {RISK_THRESHOLDS['high']:.0%}
  Medium risk >= {RISK_THRESHOLDS['medium']:.0%}
"""

# ─────────────────────────────────────────
# TOOL 4: Portfolio summary
# ─────────────────────────────────────────
@tool
def get_portfolio_summary() -> str:
    """
    Returns a high level summary of the entire portfolio.
    Use this when asked for an overview or general state of churn.
    """
    high   = (_probas >= RISK_THRESHOLDS["high"]).sum()
    medium = ((_probas >= RISK_THRESHOLDS["medium"]) &
              (_probas < RISK_THRESHOLDS["high"])).sum()
    low    = (_probas < RISK_THRESHOLDS["medium"]).sum()
    total  = _schema["n_rows"]

    top_drivers = list(_metrics["feature_importance"].keys())[:3]

    return f"""
Portfolio Churn Summary:
  Total customers:       {total:,}
  Overall churn rate:    {_schema['churn_rate']}%

Risk Distribution:
  High   (>= {RISK_THRESHOLDS['high']:.0%}):  {high:,} ({high/total*100:.1f}%)
  Medium ({RISK_THRESHOLDS['medium']:.0%}-{RISK_THRESHOLDS['high']:.0%}): {medium:,} ({medium/total*100:.1f}%)
  Low    (<  {RISK_THRESHOLDS['medium']:.0%}):  {low:,} ({low/total*100:.1f}%)

Model:
  Algorithm:  {_metrics['best_model']}
  ROC-AUC:    {_metrics['roc_auc']}

Top Drivers:  {', '.join(top_drivers)}

High Risk Segments:
  {chr(10).join(_stats['high_risk_segments'])}
"""

# ─────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────
tools = [
    get_high_risk_customers,
    get_churn_drivers,
    predict_churn,
    get_portfolio_summary
]

if __name__ == "__main__":
    print("Testing tools...\n")
    print(get_portfolio_summary.invoke({}))
    print(get_churn_drivers.invoke({"segment": "overall"}))