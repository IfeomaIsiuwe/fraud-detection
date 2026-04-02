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
        num_transformer = next(trans for name, trans, cols in transformers if name == 'num')
        assert len(cols) == len(numeric_features)

        # Test without engineered features
        preprocessor_no_eng = build_preprocessor(use_engineered=False)
        num_transformer_no_eng = next(trans for name, trans, cols in preprocessor_no_eng.transformers if name == 'num')
        assert len(cols) == len(NUMERIC_FEATURES)

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
        assert model.scale_pos_weight == 99  # Approximately 1/0.01 for balanced

    def test_get_all_pipelines(self):
        """Test getting all available pipelines."""
        pipelines = get_all_pipelines()
        assert isinstance(pipelines, dict)
        assert 'Logistic Regression' in pipelines
        assert 'Random Forest' in pipelines

        # XGBoost may or may not be available
        try:
            import xgboost
            assert 'XGBoost' in pipelines
        except ImportError:
            assert 'XGBoost' not in pipelines

    def test_pipeline_fit_transform(self):
        """Test that pipelines can fit and transform data."""
        df = load_engineered()
        X_train, X_test, y_train, y_test = get_split(df)

        pipelines = get_all_pipelines()

        for name, pipeline in pipelines.items():
            # Fit pipeline
            pipeline.fit(X_train, y_train)

            # Transform test data
            X_transformed = pipeline[:-1].transform(X_test)  # Exclude model step

            # Check output is numeric array
            assert hasattr(X_transformed, 'shape')
            assert X_transformed.shape[0] == len(X_test)
            assert X_transformed.shape[1] > 0

            # Predict probabilities
            proba = pipeline.predict_proba(X_test)
            assert proba.shape == (len(X_test), 2)
            assert np.all((proba >= 0) & (proba <= 1))
            assert np.allclose(proba.sum(axis=1), 1.0)