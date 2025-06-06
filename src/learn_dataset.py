import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    data = pd.read_csv('../data/raw/adult.csv')

    print(data.head())
    print(data.columns)

    data.replace("?", pd.NA, inplace=True)

    missing_values = data.isna().sum()
    print("Missing vals:\n", missing_values[missing_values > 0])

    data_clean = data.dropna()

    X = data_clean.drop("income", axis=1)
    y = data_clean["income"].apply(lambda x: 1 if x == ">50K" else 0)

    numeric_features = X.select_dtypes(include=["int64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    X_encoded = pd.get_dummies(X, columns=categorical_features)

    scaler = StandardScaler()
    X_encoded[numeric_features] = scaler.fit_transform(X_encoded[numeric_features])

    print(X_encoded.shape, y.value_counts())

    # Визуализация
    plt.rcParams["figure.figsize"] = (12, 6)

    numeric_data = data_clean[numeric_features]
    numeric_data.hist(bins=30, figsize=(15, 10), layout=(2, 3))
    plt.suptitle("Распределения числовых признаков", fontsize=16)
    plt.tight_layout()
    plt.show()

    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Корреляционная матрица числовых признаков")
    plt.show()