import os

# ─────────────────────────────────────────
# RISK THRESHOLDS
# Change these via environment variables
# without touching any code
# ─────────────────────────────────────────
RISK_THRESHOLDS = {
    "high":   float(os.getenv("HIGH_RISK_THRESHOLD", 0.7)),
    "medium": float(os.getenv("MEDIUM_RISK_THRESHOLD", 0.3))
}

# ─────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────
MODEL_CONFIG = {
    "auc_threshold":    float(os.getenv("AUC_THRESHOLD", 0.80)),
    "test_size":        float(os.getenv("TEST_SIZE", 0.2)),
    "random_state":     int(os.getenv("RANDOM_STATE", 42)),
    "n_estimators":     int(os.getenv("N_ESTIMATORS", 100)),
    "xgb_scale_pos_weight": float(os.getenv("XGB_SCALE_POS_WEIGHT", 4.0))
}

# ─────────────────────────────────────────
# DATA CONFIG
# ─────────────────────────────────────────
DATA_CONFIG = {
    "path":             os.getenv("DATA_PATH", "data/churn.csv"),
    "target_col":       os.getenv("TARGET_COL", "Exited"),
    "drop_cols":        ["RowNumber", "CustomerId", "Surname"],
    "categorical_cols": ["Geography", "Gender"],
    "numeric_cols":     [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "EstimatedSalary"
    ],
    "base_categories": {
        "Geography": "France",
        "Gender":    "Female"
    }
}

# ─────────────────────────────────────────
# TOOL CONFIG
# ─────────────────────────────────────────
TOOL_CONFIG = {
    "default_top_n":        int(os.getenv("DEFAULT_TOP_N", 10)),
    "max_top_n":            int(os.getenv("MAX_TOP_N", 100)),
}