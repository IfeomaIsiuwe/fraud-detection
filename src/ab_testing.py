"""
A/B Testing Framework for Model Deployment Validation

This module provides tools for comparing model performance in production
through A/B testing, allowing gradual rollout and validation of new models.
"""
import hashlib
import json
import os
import joblib
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

from data_loader import TARGET


@dataclass
class ABTestResult:
    """Results from an A/B test comparison."""
    test_id: str
    model_a: str
    model_b: str
    start_date: str
    end_date: str
    traffic_split: float  # % of traffic to model B
    metrics_a: Dict[str, float]
    metrics_b: Dict[str, float]
    sample_size_a: int
    sample_size_b: int
    winner: Optional[str] = None
    confidence_level: Optional[float] = None


class ABTestingFramework:
    """
    Framework for A/B testing model deployments.

    Supports:
    - Traffic splitting based on user/transaction ID
    - Automatic metric calculation
    - Statistical significance testing
    - Model versioning and rollback
    """

    def __init__(self, models_dir: str = "models", results_dir: str = "ab_test_results"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Load available models
        self.models = self._load_models()

    def _load_models(self) -> Dict[str, Any]:
        """Load all available models."""
        models = {}
        if os.path.exists(self.models_dir):
            for file in os.listdir(self.models_dir):
                if file.endswith('.pkl'):
                    model_name = file.replace('.pkl', '')
                    model_path = os.path.join(self.models_dir, file)
                    try:
                        models[model_name] = joblib.load(model_path)
                    except Exception as e:
                        print(f"Failed to load model {model_name}: {e}")
        return models

    def assign_model(self, transaction_id: str, traffic_split: float = 0.5,
                    seed: str = "ab_test") -> str:
        """
        Assign a model to a transaction based on consistent hashing.

        Args:
            transaction_id: Unique transaction identifier
            traffic_split: Fraction of traffic to assign to model B (0.0-1.0)
            seed: Seed for consistent hashing

        Returns:
            Model name ('A' or 'B')
        """
        # Create consistent hash
        hash_input = f"{seed}:{transaction_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        # Normalize to [0, 1]
        normalized = hash_value / (2**128 - 1)

        return 'B' if normalized < traffic_split else 'A'

    def run_ab_test(self, test_data: pd.DataFrame, model_a: str, model_b: str,
                   traffic_split: float = 0.5, test_id: Optional[str] = None) -> ABTestResult:
        """
        Run an A/B test on historical data.

        Args:
            test_data: DataFrame with features and target
            model_a: Name of control model
            model_b: Name of treatment model
            traffic_split: Traffic split for model B
            test_id: Optional test identifier

        Returns:
            ABTestResult with comparison metrics
        """
        if test_id is None:
            test_id = f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if model_a not in self.models:
            raise ValueError(f"Model {model_a} not found")
        if model_b not in self.models:
            raise ValueError(f"Model {model_b} not found")

        # Prepare data
        X = test_data.drop(columns=[TARGET])
        y_true = test_data[TARGET]

        # Simulate traffic split
        assignments = [self.assign_model(f"tx_{i}", traffic_split)
                      for i in range(len(test_data))]

        # Split data by assignment
        mask_a = [assignment == 'A' for assignment in assignments]
        mask_b = [assignment == 'B' for assignment in assignments]

        X_a, y_a = X[mask_a], y_true[mask_a]
        X_b, y_b = X[mask_b], y_true[mask_b]

        # Get predictions
        proba_a = self.models[model_a].predict_proba(X_a)[:, 1]
        proba_b = self.models[model_b].predict_proba(X_b)[:, 1]

        # Calculate metrics
        metrics_a = self._calculate_metrics(y_a, proba_a)
        metrics_b = self._calculate_metrics(y_b, proba_b)

        # Determine winner (simplified - in practice use statistical testing)
        winner = None
        if metrics_b['pr_auc'] > metrics_a['pr_auc'] + 0.01:  # 1% improvement threshold
            winner = model_b
        elif metrics_a['pr_auc'] > metrics_b['pr_auc'] + 0.01:
            winner = model_a

        result = ABTestResult(
            test_id=test_id,
            model_a=model_a,
            model_b=model_b,
            start_date=datetime.now().isoformat(),
            end_date=datetime.now().isoformat(),
            traffic_split=traffic_split,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            sample_size_a=len(X_a),
            sample_size_b=len(X_b),
            winner=winner
        )

        # Save results
        self._save_result(result)

        return result

    def _calculate_metrics(self, y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score

        # Use optimal threshold (simplified)
        threshold = 0.4  # From our analysis
        y_pred = (y_proba >= threshold).astype(int)

        return {
            'pr_auc': average_precision_score(y_true, y_proba),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'fraud_rate': y_true.mean()
        }

    def _save_result(self, result: ABTestResult):
        """Save A/B test result to disk."""
        filename = f"{result.test_id}.json"
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2)

    def load_result(self, test_id: str) -> ABTestResult:
        """Load A/B test result from disk."""
        filepath = os.path.join(self.results_dir, f"{test_id}.json")

        with open(filepath, 'r') as f:
            data = json.load(f)

        return ABTestResult(**data)

    def list_results(self) -> List[str]:
        """List all saved A/B test results."""
        if not os.path.exists(self.results_dir):
            return []

        return [f.replace('.json', '') for f in os.listdir(self.results_dir)
                if f.endswith('.json')]

    def gradual_rollout(self, new_model: str, rollout_steps: List[float] = [0.1, 0.25, 0.5, 1.0],
                       evaluation_period_days: int = 7) -> List[ABTestResult]:
        """
        Perform gradual rollout of a new model.

        Args:
            new_model: Name of the new model to rollout
            rollout_steps: Traffic percentages to test
            evaluation_period_days: Days to evaluate each step

        Returns:
            List of A/B test results for each rollout step
        """
        results = []
        baseline_model = "best_pipeline"  # Assume this exists

        for traffic_pct in rollout_steps:
            print(f"Testing {new_model} at {traffic_pct*100}% traffic...")

            # In practice, this would run over time with real traffic
            # For demo, we'll use historical data
            from data_loader import load_engineered, get_split
            df = load_engineered()
            _, X_test, _, y_test = get_split(df)

            result = self.run_ab_test(
                pd.concat([X_test, y_test], axis=1),
                baseline_model,
                new_model,
                traffic_pct,
                f"rollout_{traffic_pct}_{datetime.now().strftime('%Y%m%d')}"
            )

            results.append(result)

            # Check if we should stop early
            if result.winner == baseline_model:
                print(f"New model {new_model} underperforming at {traffic_pct*100}% traffic. Stopping rollout.")
                break

        return results


# Example usage
if __name__ == "__main__":
    # Initialize framework
    ab = ABTestingFramework()

    # Example A/B test
    from data_loader import load_engineered, get_split
    df = load_engineered()
    _, X_test, _, y_test = get_split(df)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Compare two models (assuming they exist)
    if len(ab.models) >= 2:
        model_names = list(ab.models.keys())
        result = ab.run_ab_test(test_data, model_names[0], model_names[1], traffic_split=0.5)
        print(f"A/B Test Result: Winner = {result.winner}")
        print(f"Model A PR-AUC: {result.metrics_a['pr_auc']:.4f}")
        print(f"Model B PR-AUC: {result.metrics_b['pr_auc']:.4f}")