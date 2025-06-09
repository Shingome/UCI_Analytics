import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging
import os
from pathlib import Path


BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "../data/processed"
MODEL_DIR = BASE_DIR / "../models"
OUTPUT_DIR = BASE_DIR / "../outputs"
TRAIN_DATA = "train_balanced.csv"
TEST_DATA = "test.csv"

logging.basicConfig(
    filename=OUTPUT_DIR / "training.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def log_paths():
    logging.info("===== Path Configuration =====")
    logging.info(f"Base directory: {BASE_DIR}")
    logging.info(f"Data directory: {DATA_DIR.resolve()}")
    logging.info(f"Model directory: {MODEL_DIR.resolve()}")
    logging.info(f"Output directory: {OUTPUT_DIR.resolve()}")
    logging.info(f"Train data: {DATA_DIR.resolve()}/{TRAIN_DATA}")
    logging.info(f"Test data: {DATA_DIR.resolve()}/{TEST_DATA}")


def load_data(train_path: Path, test_path: Path):
    logging.info(f"Loading train data from: {train_path}")
    logging.info(f"Loading test data from: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop("income", axis=1)
    y_train = train_df["income"]
    X_test = test_df.drop("income", axis=1)
    y_test = test_df["income"]

    return X_train, X_test, y_train, y_test


def evaluate_model(model_name: str, y_true, y_pred):
    logging.info(f"\nEvaluating {model_name}")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    report = \
        f"""===== {model_name} Evaluation =====
        Accuracy:  {acc:.4f}
        Precision: {prec:.4f}
        Recall:    {rec:.4f}
        F1-score:  {f1:.4f}"""

    print(report)
    logging.info(report)

    conf_matrix_path = OUTPUT_DIR / f"{model_name}_conf_matrix.png"
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["<=50K", ">50K"])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.savefig(conf_matrix_path)
    plt.close()
    logging.info(f"Confusion matrix saved to: {conf_matrix_path.resolve()}")


def train_logistic_regression(X_train, y_train, X_test, y_test):
    model_path = MODEL_DIR / "logistic_regression.pkl"
    logging.info(f"\nTraining Logistic Regression. Model will be saved to: {model_path.resolve()}")

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    evaluate_model("Logistic Regression", y_test, y_pred)
    joblib.dump(model, model_path)
    logging.info(f"Model saved to: {model_path.resolve()}")


def train_xgboost(X_train, y_train, X_test, y_test):
    model_path = MODEL_DIR / "xgboost_model.pkl"
    logging.info(f"\nTraining XGBoost. Model will be saved to: {model_path.resolve()}")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    evaluate_model("XGBoost", y_test, y_pred)
    joblib.dump(model, model_path)
    logging.info(f"Model saved to: {model_path.resolve()}")


def train_neural_network(X_train, y_train, X_test, y_test):
    model_path = MODEL_DIR / "neural_network.h5"
    logging.info(f"\nTraining Neural Network. Model will be saved to: {model_path.resolve()}")

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    evaluate_model("Neural Network", y_test, y_pred)
    model.save(model_path)
    logging.info(f"Model saved to: {model_path.resolve()}")


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    log_paths()

    train_path = DATA_DIR / TRAIN_DATA
    test_path = DATA_DIR / TEST_DATA
    X_train, X_test, y_train, y_test = load_data(train_path, test_path)

    train_logistic_regression(X_train, y_train, X_test, y_test)
    train_xgboost(X_train, y_train, X_test, y_test)
    train_neural_network(X_train, y_train, X_test, y_test)