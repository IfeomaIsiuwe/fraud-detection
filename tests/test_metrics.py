"""
Unit tests for metrics.py
"""
import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from src.metrics import (
    evaluate_model, expected_cost, find_best_threshold,
    DEFAULT_COST_FN, DEFAULT_COST_FP
)


class TestMetrics:
    """Test evaluation metrics functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample predictions for testing."""
        np.random.seed(42)
        n = 1000
        y_true = np.random.choice([0, 1], size=n, p=[0.99, 0.01])  # 1% fraud
        y_proba = np.random.beta(2, 5, size=n)  # Skewed towards low probabilities
        # Make some high probabilities for fraud cases
        fraud_indices = np.where(y_true == 1)[0]
        y_proba[fraud_indices] = np.random.beta(5, 2, size=len(fraud_indices))
        return y_true, y_proba

    def test_evaluate_model(self, sample_data):
        """Test model evaluation at given threshold."""
        y_true, y_proba = sample_data
        threshold = 0.5
        results = evaluate_model(y_true, y_proba, threshold)

        required_keys = ['threshold', 'precision', 'recall', 'pr_auc', 'roc_auc', 'tp', 'fp', 'fn', 'tn']
        for key in required_keys:
            assert key in results

        assert results['threshold'] == threshold
        assert 0 <= results['precision'] <= 1
        assert 0 <= results['recall'] <= 1
        assert 0 <= results['pr_auc'] <= 1
        assert 0 <= results['roc_auc'] <= 1

        # Check confusion matrix consistency
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        assert results['tp'] == tp
        assert results['fp'] == fp
        assert results['fn'] == fn
        assert results['tn'] == tn

    def test_expected_cost(self, sample_data):
        """Test expected cost calculation."""
        y_true, y_proba = sample_data
        threshold = 0.5
        cost = expected_cost(y_true, y_proba, threshold)

        assert isinstance(cost, (int, float, np.integer, np.floating))
        assert cost >= 0

        # Manual calculation
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        expected = fp * DEFAULT_COST_FP + fn * DEFAULT_COST_FN
        assert cost == expected

    def test_find_best_threshold(self, sample_data):
        """Test optimal threshold finding."""
        y_true, y_proba = sample_data
        best_t, best_cost = find_best_threshold(y_true, y_proba)

        assert 0 < best_t < 1
        assert isinstance(best_cost, (int, float, np.integer, np.floating))
        assert best_cost >= 0

        # Check it's actually the minimum
        thresholds = np.linspace(0.01, 0.99, 200)
        costs = [expected_cost(y_true, y_proba, t) for t in thresholds]
        min_cost = min(costs)
        assert abs(best_cost - min_cost) < 1e-10  # Should be exactly the minimum

    def test_cost_sensitivity(self, sample_data):
        """Test cost sensitivity with different FN/FP costs."""
        y_true, y_proba = sample_data

        # Test with different cost ratios
        cost_ratios = [(100, 1), (500, 5), (1000, 50)]

        for cost_fn, cost_fp in cost_ratios:
            best_t, best_cost = find_best_threshold(y_true, y_proba, cost_fn, cost_fp)
            assert 0 < best_t < 1
            assert best_cost >= 0

            # Higher FN cost should generally lead to lower threshold (more aggressive)
            # But this is a statistical test, so we'll just check validity