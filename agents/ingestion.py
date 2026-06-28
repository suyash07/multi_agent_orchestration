import pandas as pd
from sklearn.preprocessing import StandardScaler

class IngestionAgent:
    def __init__(self):
        self.scaler = StandardScaler()

    def run(self):
        print("[IngestionAgent] Loading data...")
        df = pd.read_csv("data/churn.csv")
        df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

        y = df["Exited"]
        X = df.drop(columns=["Exited"])

        X = pd.get_dummies(X, columns=["Geography", "Gender"], drop_first=True)

        numeric_cols = ["CreditScore", "Age", "Tenure", "Balance",
                        "NumOfProducts", "EstimatedSalary"]

        # Keep raw copy before scaling for EDA
        X_raw = X.copy()

        X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])

        schema = {
            "n_rows": len(X),
            "n_features": X.shape[1],
            "feature_names": list(X.columns),
            "churn_rate": round(y.mean() * 100, 2)
        }

        print(f"[IngestionAgent] Done. {schema['n_rows']} rows, "
              f"{schema['n_features']} features, "
              f"{schema['churn_rate']}% churn rate.")

        return X, X_raw, y, schema