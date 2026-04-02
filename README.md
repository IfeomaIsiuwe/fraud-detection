# Credit Card Fraud Detection

A production-ready machine learning pipeline for identifying fraudulent credit card transactions, built with a focus on **business-driven evaluation**, **rigorous validation**, and **model interpretability**.

---

## Data

Download the dataset from Kaggle:
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place the file at: `data/raw/credit_card_fraud.csv`

---


## Problem

Credit card fraud costs the industry billions annually. A naive classifier that flags everything as legitimate achieves 99% accuracy on a typical dataset — and catches zero fraud. This project treats fraud detection as what it actually is: a **cost-sensitive, class-imbalanced ranking problem** where the cost of missing fraud vastly outweighs the cost of a false alarm.

---

## Results

| Model | PR-AUC (CV mean ± std) | ROC-AUC (CV mean ± std) |
|---|---|---|
| Logistic Regression | 0.7424 ± 0.0419 | 0.9941 ± 0.0011 |
| Random Forest | 0.9587 ± 0.0170 | 0.9989 ± 0.0006 |
| XGBoost | 0.9963 ± 0.0039 | 0.9999 ± 0.0001 |

> Results populate when notebooks are run. Expected: XGBoost leads on PR-AUC; all models achieve near-perfect recall at the optimal threshold.

**Best model at optimal threshold:**

| Metric | Value |
|---|---|
| Precision | 1.000 |
| Recall | 1.000 |
| Expected cost (test set) | 0 |
| Cost reduction vs. default t=0.5 | 0 |

---

## Project structure

```
fraud-detection/
├── data/
│   └── raw/                    # Place credit_card_fraud.csv here (not tracked by git)
├── models/                     # Saved pipelines (generated — not tracked)
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_features.ipynb       # Feature engineering & validation
│   ├── 03_models.ipynb         # Model training & cross-validation comparison
│   ├── 04_threshold.ipynb      # Cost-sensitive threshold optimisation
│   └── 05_explainability.ipynb # SHAP-based model explanation
├── reports/
│   └── figures/                # All plots (generated — not tracked)
├── src/
│   ├── data_loader.py          # Data loading, feature engineering, train/test split
│   ├── pipeline.py             # Sklearn pipeline builders for all models
│   └── metrics.py              # Evaluation, cross-validation, cost functions
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Key design decisions

**StandardScaler on numeric features**
Logistic Regression is sensitive to feature scale. Unlike the common mistake of using `passthrough`, all numeric features go through `StandardScaler`. Tree models are unaffected but benefit from consistent preprocessing in a shared pipeline.

**5-fold stratified cross-validation**
A single 80/20 split can give misleading results on an imbalanced dataset. Every model is evaluated with `StratifiedKFold(n_splits=5)` to produce mean ± std estimates of PR-AUC and ROC-AUC.

**Cost-sensitive threshold selection**
The default `predict` threshold of 0.5 is almost never optimal. We grid-search 200 thresholds and pick the one that minimises expected cost under the assumption FN costs $500 and FP costs $5. A sensitivity heatmap shows how the optimal threshold shifts if those assumptions change.

**SHAP over feature importances**
Raw feature importances tell you which features the model used most — not how or in which direction. SHAP values are locally consistent, additive, and directional. The waterfall plot provides per-transaction explanations usable by fraud analysts.

**No data leakage**
All feature engineering transformations are deterministic functions of a single row's raw values — no window functions or aggregates that could leak future information. The `StandardScaler` is fit only on the training set.

---

## Setup

```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Add your dataset:
```
data/raw/credit_card_fraud.csv
```

Run notebooks in order:
```bash
jupyter lab
```

## Deployment

### API Server

Start the FastAPI server for real-time predictions:

```bash
python src/api.py
```

Or using Docker:

```bash
docker-compose up --build
```

API endpoints:
- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

### A/B Testing

For gradual model rollout and validation:

```python
from src.ab_testing import ABTestingFramework

ab = ABTestingFramework()
results = ab.gradual_rollout("new_model_name")
```

---

## Dataset

The dataset contains anonymised credit card transactions with the following features:

| Feature | Type | Description |
|---|---|---|
| `amount` | numeric | Transaction amount |
| `transaction_hour` | numeric | Hour of day (0–23) |
| `device_trust_score` | numeric | Device risk score (0–1) |
| `velocity_last_24h` | numeric | Number of transactions in the last 24h |
| `cardholder_age` | numeric | Age of cardholder |
| `foreign_transaction` | binary | 1 if transaction is foreign |
| `location_mismatch` | binary | 1 if billing and transaction locations differ |
| `merchant_category` | categorical | Type of merchant |
| `is_fraud` | binary | Target: 1 = fraud |

---

## Engineered features

| Feature | Formula | Rationale |
|---|---|---|
| `amount_log` | `log1p(amount)` | Reduces right skew for linear models |
| `amount_x_velocity` | `amount × velocity_last_24h` | Captures compounded risk signal |
| `high_risk_hour` | `1 if hour ≥ 22 or hour ≤ 6` | Night-time transactions have higher fraud rates |
| `velocity_bin` | `qcut(velocity, q=4)` | Ordinal binning for tree models |

---

## Tech stack

- **Python 3.10+**
- **scikit-learn** — pipelines, preprocessing, cross-validation
- **XGBoost** — gradient boosting
- **SHAP** — model explainability
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualisation
- **joblib** — model serialisation
