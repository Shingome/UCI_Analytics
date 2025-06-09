import pandas as pd
import joblib
from tensorflow.keras.models import load_model


def predict_logistic(X, model_path="../models/logistic_regression.pkl"):
    model = joblib.load(model_path)
    return model.predict(X)


def predict_xgboost(X, model_path="../models/xgboost_model.pkl"):
    model = joblib.load(model_path)
    return model.predict(X)


def predict_neural_network(X, model_path="../models/neural_network.h5"):
    model = load_model(model_path)
    y_proba = model.predict(X)
    return (y_proba > 0.5).astype("int32").flatten()


if __name__ == "__main__":
    input_path = "../data/processed/test.csv"

    df = pd.read_csv(input_path)
    X = df.drop("income", axis=1, errors='ignore')

    print("\n--- Logistic Regression ---")
    preds_logreg = predict_logistic(X)
    print(preds_logreg[:10])

    print("\n--- XGBoost ---")
    preds_xgb = predict_xgboost(X)
    print(preds_xgb[:10])

    print("\n--- Neural Network ---")
    preds_nn = predict_neural_network(X)
    print(preds_nn[:10])