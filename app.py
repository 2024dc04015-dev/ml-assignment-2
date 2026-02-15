# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix
)

st.set_page_config(page_title="ML Assignment-2", layout="wide")
st.title("ML Assignment‑2: Bank Marketing Models")

MODEL_FILES = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree":       "models/decision_tree.pkl",
    "KNN":                 "models/knn.pkl",
    "Naive Bayes":         "models/naive_bayes.pkl",
    "Random Forest":       "models/random_forest.pkl",
    "XGBoost":             "models/xgboost.pkl",
}

model_choice = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
uploaded = st.file_uploader("Upload CSV (UCI uses ';' delimiter)", type=["csv"])
delimiter = st.text_input("CSV delimiter", value=";")

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

if uploaded:
    try:
        df = pd.read_csv(uploaded, delimiter=delimiter)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.subheader("Preview")
    st.dataframe(df.head())

    has_target = "y" in df.columns
    if has_target:
        y_true = df["y"].map({"no": 0, "yes": 1})
        mask = y_true.isna()
        if mask.any():
            st.warning(f"Dropping {mask.sum()} rows with unknown y values.")
            df = df.loc[~mask].copy()
            y_true = y_true.loc[~mask].astype(int)
        X_infer = df.drop(columns=["y"])
    else:
        y_true = None
        X_infer = df.copy()

    model = load_model(MODEL_FILES[model_choice])

    y_pred = model.predict(X_infer)
    try:
        y_prob = model.predict_proba(X_infer)[:, 1]
    except Exception:
        y_prob = None

    st.subheader("Predictions")
    pred_df = df.copy()
    pred_df["prediction"] = y_pred
    pred_df["prediction_label"] = pred_df["prediction"].map({0: "no", 1: "yes"})
    st.dataframe(pred_df.head())

    st.download_button(
        "Download Predictions CSV",
        data=pred_df.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv"
    )

    if has_target:
        st.subheader("Evaluation Metrics")
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan,
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred)
        }
        st.write({k: round(v, 4) for k, v in metrics.items()})

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        st.write(pd.DataFrame(cm,
                              index=["True:0(no)", "True:1(yes)"],
                              columns=["Pred:0(no)", "Pred:1(yes)"]))

        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred, digits=4))
else:
    st.info("Upload a CSV to start. Use ';' as delimiter for the UCI Bank Marketing dataset.")