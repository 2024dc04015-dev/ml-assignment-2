
# ml_assignment.py
# ------------------------------------------------------------
# BITS ML Assignment-2: Train 6 classifiers, compute metrics,
# save comparison table and model pipelines for Streamlit.
# ------------------------------------------------------------

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle


def main():
    # -----------------------
    # 0) Paths & setup
    # -----------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "bank.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # -----------------------
    # 1) Load data
    # -----------------------
    # UCI Bank Marketing uses ';' as delimiter
    df = pd.read_csv(data_path, delimiter=";").drop_duplicates()

    # -----------------------
    # 2) Features / Target
    # -----------------------
    # Target is 'y' with values 'yes'/'no'
    y = df["y"].map({"no": 0, "yes": 1}).astype(int)
    X = df.drop(columns=["y"])

    # Identify column types (be explicit with "object" & "string")
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols     = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    # -----------------------
    # 3) Preprocessing
    # -----------------------
    # If your sklearn < 1.2, change sparse_output=False -> sparse=False
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # -----------------------
    # 4) Train/Test split
    # -----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------
    # 5) Build pipelines per model
    # -----------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":         GaussianNB(),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        ),
    }

    pipelines = {
        name: Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])
        for name, clf in models.items()
    }

    # -----------------------
    # 6) Train and save models
    # -----------------------
    for name, pipe in pipelines.items():
        print(f"\nTraining: {name}")
        pipe.fit(X_train, y_train)

        # Save each pipeline (preprocess + model) for Streamlit
        short = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = os.path.join(models_dir, f"{short}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(pipe, f)
        # Uncomment if you want to see save messages:
        # print(f"Saved: {out_path}")

    # -----------------------
    # 7) Evaluate once and save metrics once
    # -----------------------
    rows = []
    for name, pipe in pipelines.items():
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]  # all chosen models expose predict_proba

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred),
        })

    results_df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    print("\n=== Comparison Table (copy to README) ===")
    print(results_df.round(4))

    results_df.to_csv(os.path.join(base_dir, "model_results.csv"), index=False)
    print("\nSaved: model_results.csv")


if __name__ == "__main__":
    main()



