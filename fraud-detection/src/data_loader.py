"""
data_loader.py
--------------
Centralised data loading and train/test splitting.
All notebooks import from here — no copy-pasted setup code.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "credit_card_fraud.csv")


# ---------------------------------------------------------------------------
# Feature groups  (single source of truth)
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "amount",
    "transaction_hour",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age",
]

BINARY_FEATURES = [
    "foreign_transaction",
    "location_mismatch",
]

CATEGORICAL_FEATURES = [
    "merchant_category",
]

ENGINEERED_FEATURES = [
    "amount_log",
    "amount_x_velocity",
    "high_risk_hour",
    "velocity_bin",
]

TARGET = "is_fraud"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_raw(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV and drop the ID column."""
    df = pd.read_csv(path)
    if "transaction_id" in df.columns:
        df = df.drop(columns=["transaction_id"])
    return df


def load_engineered(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw data and apply feature engineering."""
    df = load_raw(path)
    df = engineer_features(df)
    return df


def get_split(
    df: pd.DataFrame,
    target: str = TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Return (X_train, X_test, y_train, y_test) with stratification."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features on top of the raw columns.
    All transformations are deterministic — no leakage risk.
    """
    df = df.copy()

    # Log-transform skewed amount
    df["amount_log"] = df["amount"].apply(lambda x: pd.np.log1p(x) if hasattr(pd, 'np') else __import__('numpy').log1p(x))

    # Interaction: high amount + high velocity = elevated risk
    df["amount_x_velocity"] = df["amount"] * df["velocity_last_24h"]

    # Non-business hours (22:00–06:00) are higher risk
    df["high_risk_hour"] = df["transaction_hour"].apply(
        lambda h: 1 if (h >= 22 or h <= 6) else 0
    )

    # Ordinal velocity bin (0 = low … 3 = very high)
    df["velocity_bin"] = pd.qcut(
        df["velocity_last_24h"], q=4, labels=[0, 1, 2, 3]
    ).astype(int)

    return df
