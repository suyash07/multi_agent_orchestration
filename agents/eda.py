import pandas as pd

class EDAAgent:
    def run(self, X_raw, y):
        print("[EDAAgent] Running exploratory analysis...")

        df = X_raw.copy()
        df["Exited"] = y.values

        stats = {}

        stats["overall_churn_rate"] = round(y.mean() * 100, 2)

        stats["churn_by_geography"] = (
            df.groupby("Geography_Germany")["Exited"]
            .mean()
            .rename({0: "Non-Germany", 1: "Germany"})
            .mul(100).round(2)
            .to_dict()
        )

        stats["churn_by_active"] = (
            df.groupby("IsActiveMember")["Exited"]
            .mean()
            .rename({0: "Inactive", 1: "Active"})
            .mul(100).round(2)
            .to_dict()
        )

        stats["churn_by_products"] = (
            df.groupby("NumOfProducts")["Exited"]
            .mean()
            .mul(100).round(2)
            .to_dict()
        )

        stats["churn_by_gender"] = (
            df.groupby("Gender_Male")["Exited"]
            .mean()
            .rename({0: "Female", 1: "Male"})
            .mul(100).round(2)
            .to_dict()
        )

        high_risk = []
        for segment, rates in stats["churn_by_geography"].items():
            if rates > 30:
                high_risk.append(f"Geography: {segment} ({rates}%)")
        for segment, rates in stats["churn_by_active"].items():
            if rates > 30:
                high_risk.append(f"Activity: {segment} ({rates}%)")
        for products, rates in stats["churn_by_products"].items():
            if rates > 30:
                high_risk.append(f"NumProducts: {products} ({rates}%)")

        stats["high_risk_segments"] = high_risk

        print(f"[EDAAgent] Overall churn: {stats['overall_churn_rate']}%")
        print(f"[EDAAgent] High risk segments found: {len(high_risk)}")
        for s in high_risk:
            print(f"           → {s}")

        return stats