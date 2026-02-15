# ML Assignment 2 — Bank Marketing (Classification)

## 1) Problem Statement
Predict whether a customer subscribes to a term deposit (target `y`: yes/no) using customer and campaign attributes.

## 2) Dataset
- Source: UCI Bank Marketing
- Instances: 45k+
- Features: 16
- Target: `y` (yes/no)
- Note: CSV uses `;` as delimiter.

## 3) Models & Metrics (Test Set)
> Values below are from `model_results.csv`.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.8917 | 0.9118 | 0.5429 | 0.3654 | 0.4368 | 0.3885 |
| Random Forest | 0.8873 | 0.8998 | 0.5217 | 0.2308 | 0.3200 | 0.2952 |
| Logistic Regression | 0.8928 | 0.8913 | 0.5636 | 0.2981 | 0.3899 | 0.3579 |
| Naive Bayes | 0.8210 | 0.7881 | 0.3092 | 0.4519 | 0.3672 | 0.2737 |
| KNN | 0.8840 | 0.7692 | 0.4872 | 0.1827 | 0.2657 | 0.2477 |
| Decision Tree | 0.8586 | 0.6942 | 0.4032 | 0.4088 | 0.4386 | 0.3602 |

## 4) Observations
- **XGBoost** achieved the highest AUC and strong overall ranking performance for the positive class.  
- **Random Forest** is competitive but shows lower recall than XGBoost.  
- **Logistic Regression** provides balanced metrics and interpretability.  
- **Naive Bayes** trades precision for recall due to independence assumptions.  
- **KNN** has reasonable accuracy but lower recall (class imbalance sensitivity).  
- **Decision Tree** is fast/interpretable but trails ensembles in AUC.

## 5) How to Run
```bash
pip install -r requirements.txt
python ml_assignment.py           # trains & saves models and model_results.csv
streamlit run app.py              # launches the web app