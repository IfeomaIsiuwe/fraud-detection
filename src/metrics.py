"""
metrics.py
----------
Evaluation helpers shared across all modelling notebooks.
Covers classification metrics, cost-sensitive analysis, and cross-validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


# ---------------------------------------------------------------------------
# Cost constants  (easily overridden by callers)
# ---------------------------------------------------------------------------

DEFAULT_COST_FN = 500   # cost of missing a fraud (false negative)
DEFAULT_COST_FP = 5     # cost of flagging a legit transaction (false positive)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(y_true, y_proba, threshold: float = 0.5) -> dict:
    """
    Full evaluation report for a given threshold.
    Returns a dict with all key metrics.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold":    threshold,
        "precision":    precision_score(y_true, y_pred, zero_division=0),
        "recall":       recall_score(y_true, y_pred, zero_division=0),
        "pr_auc":       average_precision_score(y_true, y_proba),
        "roc_auc":      roc_auc_score(y_true, y_proba),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def expected_cost(
    y_true,
    y_proba,
    threshold: float,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> float:
    """Total expected cost at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * cost_fp + fn * cost_fn


def find_best_threshold(
    y_true,
    y_proba,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    n_steps: int = 200,
) -> tuple[float, float]:
    """
    Grid-search for the threshold that minimises expected cost.
    Returns (best_threshold, best_cost).
    """
    grid = np.linspace(0.01, 0.99, n_steps)
    costs = [expected_cost(y_true, y_proba, t, cost_fn, cost_fp) for t in grid]
    idx = int(np.argmin(costs))
    return grid[idx], costs[idx]


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    pipeline,
    X,
    y,
    cv: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run stratified k-fold CV and return a summary DataFrame.
    Scores: PR-AUC, ROC-AUC (both computed from predict_proba).
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    results = cross_validate(
        pipeline, X, y,
        cv=skf,
        scoring={
            "pr_auc":  "average_precision",
            "roc_auc": "roc_auc",
        },
        return_train_score=False,
        n_jobs=-1,
    )

    summary = pd.DataFrame({
        "fold":    list(range(1, cv + 1)),
        "pr_auc":  results["test_pr_auc"],
        "roc_auc": results["test_roc_auc"],
    })
    summary.loc["mean"] = summary.mean()
    summary.loc["std"]  = summary.std()
    return summary


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_evaluation(name: str, y_true, y_proba, threshold: float = 0.5) -> None:
    metrics = evaluate_model(y_true, y_proba, threshold)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Threshold  : {metrics['threshold']:.3f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  PR-AUC     : {metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"    TP={metrics['tp']}  FP={metrics['fp']}")
    print(f"    FN={metrics['fn']}  TN={metrics['tn']}")


def model_comparison_table(results: dict) -> pd.DataFrame:
    """
    Build a clean comparison DataFrame from a dict of
    {model_name: metrics_dict}.
    """
    rows = []
    for name, m in results.items():
        rows.append({
            "Model":     name,
            "PR-AUC":    round(m["pr_auc"], 4),
            "ROC-AUC":   round(m["roc_auc"], 4),
            "Precision": round(m["precision"], 4),
            "Recall":    round(m["recall"], 4),
        })
    return pd.DataFrame(rows).set_index("Model")
