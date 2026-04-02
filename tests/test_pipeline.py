"""
Unit tests for pipeline.py
"""
import numpy as np
import pytest
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.pipeline import (
    build_preprocessor, build_lr_pipeline, build_rf_pipeline, build_xgb_pipeline, get_all_pipelines
)
from src.data_loader import load_engineered, get_split, NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, ENGINEERED_FEATURES


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


class TestPipeline:
    """Test pipeline building functions."""

    def test_build_preprocessor(self):
        """Test preprocessor construction."""
        # Test with engineered features
        preprocessor = build_preprocessor(use_engineered=True)
        assert isinstance(preprocessor, ColumnTransformer)

        # Check transformers are configured
        transformers = preprocessor.transformers
        assert len(transformers) == 3  # num, bin, cat

        # Check transformer names
        names = [name for name, _, _ in transformers]
        assert 'num' in names
        assert 'bin' in names
        assert 'cat' in names

        # Check numeric features include engineered
        numeric_features = NUMERIC_FEATURES + ENGINEERED_FEATURES
        num_transformer_entry = next((name, trans, cols) for name, trans, cols in transformers if name == 'num')
        assert len(num_transformer_entry[2]) == len(numeric_features)

        # Test without engineered features
        preprocessor_no_eng = build_preprocessor(use_engineered=False)
        num_transformer_no_eng_entry = next((name, trans, cols) for name, trans, cols in preprocessor_no_eng.transformers if name == 'num')
        assert len(num_transformer_no_eng_entry[2]) == len(NUMERIC_FEATURES)

    def test_build_lr_pipeline(self):
        """Test logistic regression pipeline."""
        pipeline = build_lr_pipeline()
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == 'preprocess'
        assert pipeline.steps[1][0] == 'model'

        # Check model parameters
        model = pipeline.named_steps['model']
        assert hasattr(model, 'class_weight')
        assert model.class_weight == 'balanced'

    def test_build_rf_pipeline(self):
        """Test random forest pipeline."""
        pipeline = build_rf_pipeline()
        assert isinstance(pipeline, Pipeline)

        model = pipeline.named_steps['model']
        assert hasattr(model, 'class_weight')
        assert model.class_weight == 'balanced_subsample'
        assert model.n_estimators == 400

    @pytest.mark.skipif(not hasattr(__import__('xgboost'), 'XGBClassifier'), reason="XGBoost not available")
    def test_build_xgb_pipeline(self):
        """Test XGBoost pipeline."""
        pipeline = build_xgb_pipeline()
        assert isinstance(pipeline, Pipeline)

        model = pipeline.named_steps['model']
        assert hasattr(model, 'scale_pos_weight')
        assert model.scale_pos_weight == 99

    def test_get_all_pipelines(self):
        """Test getting all available pipelines."""
        pipelines = get_all_pipelines()
        assert isinstance(pipelines, dict)
        assert 'Logistic Regression' in pipelines
        assert 'Random Forest' in pipelines

        try:
            import xgboost
            assert 'XGBoost' in pipelines
        except ImportError:
            assert 'XGBoost' not in pipelines

    def test_pipeline_fit_transform(self, tmp_path):
        """Test that pipelines can fit and transform data."""
        mock_df = make_mock_df(n=300)
        csv_path = tmp_path / "credit_card_fraud.csv"
        mock_df.to_csv(csv_path, index=False)

        df = load_engineered(path=str(csv_path))
        X_train, X_test, y_train, y_test = get_split(df)

        pipelines = get_all_pipelines()

        for name, pipeline in pipelines.items():
            pipeline.fit(X_train, y_train)
            X_transformed = pipeline[:-1].transform(X_test)
            assert hasattr(X_transformed, 'shape')
            assert X_transformed.shape[0] == len(X_test)
            assert X_transformed.shape[1] > 0
            proba = pipeline.predict_proba(X_test)
            assert proba.shape == (len(X_test), 2)
            assert np.all((proba >= 0) & (proba <= 1))
            assert np.allclose(proba.sum(axis=1), 1.0)
