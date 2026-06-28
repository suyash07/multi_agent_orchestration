from openai import OpenAI

class InsightAgent:
    def __init__(self):
        self.client = OpenAI()

    def run(self, stats, metrics):
        print("[InsightAgent] Generating insights with LLM...")

        context = f"""
You are a senior data scientist at a bank analyzing customer churn.
Here is what the analysis found:

CHURN STATISTICS:
- Overall churn rate: {stats['overall_churn_rate']}%
- Churn by geography: {stats['churn_by_geography']}
- Churn by active membership: {stats['churn_by_active']}
- Churn by number of products: {stats['churn_by_products']}
- Churn by gender: {stats['churn_by_gender']}
- High risk segments: {stats['high_risk_segments']}

MODEL PERFORMANCE:
- Best model: {metrics['best_model']}
- ROC-AUC score: {metrics['roc_auc']}
- Top 5 most important features: {dict(list(metrics['feature_importance'].items())[:5])}

Write a concise business report with three sections:
1. KEY FINDINGS: The 3 most important churn patterns discovered
2. HIGH RISK PROFILES: Describe the typical customer most likely to churn
3. RECOMMENDATIONS: 3 specific actionable steps the bank should take

Be specific, use the numbers, keep each section to 3-4 sentences.
Do not use jargon. Write for a business audience, not a technical one.
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": context}
            ],
            temperature=0.3
        )

        insight_text = response.choices[0].message.content

        insights = {
            "report": insight_text,
            "model_used": "gpt-4o-mini",
            "high_risk_segments": stats["high_risk_segments"],
            "top_features": list(metrics["feature_importance"].keys())[:5],
            "roc_auc": metrics["roc_auc"]
        }

        print("[InsightAgent] Done.")
        print()
        print("=" * 60)
        print(insight_text)
        print("=" * 60)

        return insights