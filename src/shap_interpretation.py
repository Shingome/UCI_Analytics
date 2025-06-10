import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    model = joblib.load("../models/xgboost_model.pkl")
    df = pd.read_csv("../data/processed/train_balanced.csv")
    X = df.drop("income", axis=1)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("../outputs/shap_summary_plot.png")
    plt.close()