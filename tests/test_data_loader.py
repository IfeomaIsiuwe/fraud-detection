"""
Unit tests for data_loader.py
"""
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import (
    load_raw, load_engineered, get_split, engineer_features,
    NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, ENGINEERED_FEATURES, TARGET
)


class TestDataLoader:
    """Test data loading and preprocessing functions."""

    def test_load_raw(self):
        """Test raw data loading."""
        df = load_raw()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert TARGET in df.columns
        # Check no transaction_id after loading
        assert 'transaction_id' not in df.columns

    def test_load_engineered(self):
        """Test engineered data loading."""
        df = load_engineered()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        # Check engineered features are present
        for feat in ENGINEERED_FEATURES:
            assert feat in df.columns

    def test_engineer_features(self):
        """Test feature engineering."""
        df_raw = load_raw()
        df_eng = engineer_features(df_raw.copy())

        # Check new features
        assert 'amount_log' in df_eng.columns
        assert 'amount_x_velocity' in df_eng.columns
        assert 'high_risk_hour' in df_eng.columns
        assert 'velocity_bin' in df_eng.columns

        # Check amount_log is log-transformed
        assert (df_eng['amount_log'] == df_raw['amount'].apply(lambda x: pd.np.log1p(x) if hasattr(pd, 'np') else __import__('numpy').log1p(x))).all()

        # Check high_risk_hour logic
        assert df_eng['high_risk_hour'].isin([0, 1]).all()
        high_risk_mask = (df_eng['transaction_hour'] >= 22) | (df_eng['transaction_hour'] <= 6)
        assert (df_eng.loc[high_risk_mask, 'high_risk_hour'] == 1).all()
        assert (df_eng.loc[~high_risk_mask, 'high_risk_hour'] == 0).all()

    def test_get_split(self):
        """Test train/test splitting."""
        df = load_engineered()
        X_train, X_test, y_train, y_test = get_split(df)

        # Check splits
        assert len(X_train) > len(X_test)
        assert len(y_train) > len(y_test)
        assert TARGET not in X_train.columns
        assert TARGET not in X_test.columns

        # Check stratification
        train_fraud_rate = y_train.mean()
        test_fraud_rate = y_test.mean()
        # Should be very close due to stratification
        assert abs(train_fraud_rate - test_fraud_rate) < 0.01

    def test_feature_groups(self):
        """Test feature group definitions."""
        df = load_engineered()

        # Check all features exist
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