import shap
import pandas as pd

def explain_model(model, X_train):

    print("\nExplaining model with SHAP")
    print("--------------------------")

    # SHAP works best with tree models
    explainer = shap.Explainer(model, X_train)

    shap_values = explainer(X_train)

    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": abs(shap_values.values).mean(axis=0)
    })

    importance = importance.sort_values(
        "importance", ascending=False
    )

    print("\nTop SHAP Features")

    for i, row in importance.head(10).iterrows():
        print(f"{row['feature']:30s} {row['importance']:.4f}")

    return importance