"""
pipeline.py
-----------
Builds reusable sklearn pipelines for all models.
Centralises preprocessing so every notebook uses identical transformations.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor(use_engineered: bool = True) -> ColumnTransformer:
    """
    Returns a ColumnTransformer that:
      - StandardScales all numeric + engineered features  (fixes LR scale issue)
      - Passes binary features through unchanged
      - OneHotEncodes categorical features
    """
    numeric_cols = NUMERIC_FEATURES + (ENGINEERED_FEATURES if use_engineered else [])

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def build_lr_pipeline(use_engineered: bool = True) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(use_engineered)),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )),
    ])


def build_rf_pipeline(use_engineered: bool = True) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(use_engineered)),
        ("model", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_xgb_pipeline(use_engineered: bool = True) -> Pipeline:
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")
    return Pipeline([
        ("preprocess", build_preprocessor(use_engineered)),
        ("model", XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=99,   # mirrors class_weight="balanced" for XGB
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )),
    ])


def get_all_pipelines(use_engineered: bool = True) -> dict:
    """Return all available model pipelines as a dict."""
    pipes = {
        "Logistic Regression": build_lr_pipeline(use_engineered),
        "Random Forest":       build_rf_pipeline(use_engineered),
    }
    if XGBOOST_AVAILABLE:
        pipes["XGBoost"] = build_xgb_pipeline(use_engineered)
    return pipes
