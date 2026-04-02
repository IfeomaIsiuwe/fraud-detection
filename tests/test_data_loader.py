"""
Unit tests for data_loader.py
"""
import pytest
import numpy as np
import pandas as pd

from src.data_loader import (
    load_raw, load_engineered, get_split, engineer_features,
    NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, ENGINEERED_FEATURES, TARGET
)


def make_mock_df(n=200):
    """Create a minimal fake dataset that matches the real CSV structure."""
    np.random.seed(42)
    df = pd.DataFrame({
        "transaction_id": range(n),
        "amount": np.random.uniform(1, 5000, n),
        "transaction_hour": np.random.randint(0, 24, n),
        "device_trust_score": np.random.uniform(0, 1, n),
        "velocity_last_24h": np.random.uniform(0, 20, n),
        "cardholder_age": np.random.randint(18, 80, n),
        "foreign_transaction": np.random.randint(0, 2, n),
        "location_mismatch": np.random.randint(0, 2, n),
        "merchant_category": np.random.choice(["grocery", "travel", "online", "retail"], n),
        "is_fraud": np.random.choice([0, 1], n, p=[0.9, 0.1]),
    })
    return df


class TestDataLoader:
    """Test data loading and preprocessing functions."""

    def test_load_raw(self, tmp_path):
        """Test raw data loading."""
        # Save mock data to a temp CSV and load it
        mock_df = make_mock_df()
        csv_path = tmp_path / "credit_card_fraud.csv"
        mock_df.to_csv(csv_path, index=False)

        df = load_raw(path=str(csv_path))
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert TARGET in df.columns
        assert 'transaction_id' not in df.columns

    def test_load_engineered(self, tmp_path):
        """Test engineered data loading."""
        mock_df = make_mock_df()
        csv_path = tmp_path / "credit_card_fraud.csv"
        mock_df.to_csv(csv_path, index=False)

        df = load_engineered(path=str(csv_path))
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        for feat in ENGINEERED_FEATURES:
            assert feat in df.columns

    def test_engineer_features(self):
        """Test feature engineering."""
        mock_df = make_mock_df()
        # Remove transaction_id as load_raw would
        mock_df = mock_df.drop(columns=["transaction_id"])

        df_eng = engineer_features(mock_df.copy())

        assert 'amount_log' in df_eng.columns
        assert 'amount_x_velocity' in df_eng.columns
        assert 'high_risk_hour' in df_eng.columns
        assert 'velocity_bin' in df_eng.columns

        # Check amount_log is log-transformed
        expected = np.log1p(mock_df['amount'])
        pd.testing.assert_series_equal(df_eng['amount_log'], expected, check_names=False)

        # Check high_risk_hour logic
        assert df_eng['high_risk_hour'].isin([0, 1]).all()
        high_risk_mask = (df_eng['transaction_hour'] >= 22) | (df_eng['transaction_hour'] <= 6)
        assert (df_eng.loc[high_risk_mask, 'high_risk_hour'] == 1).all()
        assert (df_eng.loc[~high_risk_mask, 'high_risk_hour'] == 0).all()

    def test_get_split(self, tmp_path):
        """Test train/test splitting."""
        mock_df = make_mock_df()
        csv_path = tmp_path / "credit_card_fraud.csv"
        mock_df.to_csv(csv_path, index=False)

        df = load_engineered(path=str(csv_path))
        X_train, X_test, y_train, y_test = get_split(df)

        assert len(X_train) > len(X_test)
        assert len(y_train) > len(y_test)
        assert TARGET not in X_train.columns
        assert TARGET not in X_test.columns

        # Check stratification
        train_fraud_rate = y_train.mean()
        test_fraud_rate = y_test.mean()
        assert abs(train_fraud_rate - test_fraud_rate) < 0.05

    def test_feature_groups(self, tmp_path):
        """Test feature group definitions."""
        mock_df = make_mock_df()
        csv_path = tmp_path / "credit_card_fraud.csv"
        mock_df.to_csv(csv_path, index=False)

        df = load_engineered(path=str(csv_path))

        all_features = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES
        for feat in all_features:
            assert feat in df.columns

        # Check no overlap between groups
        numeric_set = set(NUMERIC_FEATURES)
        binary_set = set(BINARY_FEATURES)
        categorical_set = set(CATEGORICAL_FEATURES)
        engineered_set = set(ENGINEERED_FEATURES)

        assert len(numeric_set & binary_set) == 0
        assert len(numeric_set & categorical_set) == 0
        assert len(numeric_set & engineered_set) == 0
        assert len(binary_set & categorical_set) == 0
        assert len(binary_set & engineered_set) == 0
        assert len(categorical_set & engineered_set) == 0
