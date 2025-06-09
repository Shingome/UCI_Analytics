import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # Смотрим набор данных
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

    # Балансируем классы, используя smote
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"\nAfter SMOTE:\n{y_train_smote.value_counts()}")

    train_balanced = pd.DataFrame(X_train_smote, columns=X_encoded.columns)
    train_balanced["income"] = y_train_smote.values

    test_df = pd.DataFrame(X_test, columns=X_encoded.columns)
    test_df["income"] = y_test.values

    train_path = "../data/processed/train_balanced.csv"
    test_path = "../data/processed/test.csv"

    train_balanced.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
