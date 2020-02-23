"""Tests for evaluate.py."""
import os
import numpy as np

from src.evaluate import evaluate, plot_pr_curve, plot_roc_curve


class _DummyModel:
    """Predict probability proportional to feature 0."""
    def predict_proba(self, X):
        x = np.asarray(X)[:, 0]
        x = (x - x.min()) / (x.max() - x.min() + 1e-9)
        return np.column_stack([1 - x, x])


def test_metrics_keys():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    y = (X[:, 0] > 0.5).astype(int)
    m = _DummyModel()
    out = evaluate(m, X, y)
    for k in ["precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix"]:
        assert k in out


def test_pr_curve_writes_file(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    y = (X[:, 0] > 0.0).astype(int)
    m = _DummyModel()
    out = plot_pr_curve(m, X, y, out_path=str(tmp_path / "pr.png"))
    assert os.path.exists(out)


def test_roc_curve_writes_file(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    y = (X[:, 0] > 0.0).astype(int)
    m = _DummyModel()
    out = plot_roc_curve(m, X, y, out_path=str(tmp_path / "roc.png"))
    assert os.path.exists(out)
