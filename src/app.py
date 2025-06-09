from flask import Flask, request, jsonify
import pandas as pd
import os
from inference import predict_logistic, predict_xgboost, predict_neural_network

app = Flask(__name__)

MODEL_PATHS = {
    "logreg": "../models/logistic_regression.pkl",
    "xgboost": "../models/xgboost_model.pkl",
    "nn": "../models/neural_network.h5"
}


def preprocess_input(data):
    df = pd.DataFrame(data)
    if "income" in df.columns:
        df = df.drop("income", axis=1)
    return df


def predict(model_name, X):
    if model_name == "logreg":
        return predict_logistic(X, MODEL_PATHS["logreg"]).tolist()
    elif model_name == "xgboost":
        return predict_xgboost(X, MODEL_PATHS["xgboost"]).tolist()
    elif model_name == "nn":
        return predict_neural_network(X, MODEL_PATHS["nn"]).tolist()
    else:
        raise ValueError("Unknown model name")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    content = request.get_json()
    data = content.get("data")
    model_name = content.get("model", "xgboost") # По умолчанию

    if not data:
        return jsonify({"error": "Missing 'data' in request"}), 400

    try:
        X = preprocess_input(data)
        preds = predict(model_name, X)
        return jsonify({"predictions": preds})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)