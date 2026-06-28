import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix)

class ModelingAgent:
    def __init__(self):
        self.model = None
        self.feature_names = None

    def run(self, X, y):
        print("[ModelingAgent] Training model...")

        self.feature_names = list(X.columns)

        # Stratified split preserves churn ratio in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Logistic regression baseline with class weights for imbalance
        lr = LogisticRegression(class_weight="balanced",
                                max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
        print(f"[ModelingAgent] Logistic Regression ROC-AUC: {lr_auc:.4f}")

        # Random forest
        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
        print(f"[ModelingAgent] Random Forest ROC-AUC:      {rf_auc:.4f}")

        # Pick the better model
        if rf_auc >= lr_auc:
            self.model = rf
            best_name = "Random Forest"
            best_auc = rf_auc
        else:
            self.model = lr
            best_name = "Logistic Regression"
            best_auc = lr_auc

        print(f"[ModelingAgent] Selected: {best_name} (AUC: {best_auc:.4f})")

        # Predictions on test set
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Feature importance
        if best_name == "Random Forest":
            importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_.round(4)
            ))
        else:
            importance = dict(zip(
                self.feature_names,
                abs(self.model.coef_[0]).round(4)
            ))

        importance = dict(sorted(importance.items(),
                                 key=lambda x: x[1], reverse=True))

        # Per customer churn probability on full dataset
        all_probas = self.model.predict_proba(X)[:, 1]
        churn_probabilities = pd.Series(all_probas,
                                        index=X.index,
                                        name="churn_probability").round(4)

        metrics = {
            "best_model": best_name,
            "roc_auc": round(best_auc, 4),
            "lr_auc": round(lr_auc, 4),
            "rf_auc": round(rf_auc, 4),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "feature_importance": importance,
            "n_train": len(X_train),
            "n_test": len(X_test)
        }

        print(f"[ModelingAgent] Done. AUC: {best_auc:.4f}")
        print(f"[ModelingAgent] Top 3 features: "
              f"{list(importance.keys())[:3]}")

        return self.model, metrics, churn_probabilities