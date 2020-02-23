"""Tests for the threshold helpers."""
import numpy as np

from src.threshold import best_f1_threshold, threshold_for_recall


def _scores(seed=0):
    rng = np.random.RandomState(seed)
    n = 300
    y = (rng.rand(n) < 0.1).astype(int)
    # signal: positives have higher score
    proba = rng.rand(n)
    proba = np.where(y == 1, proba * 0.5 + 0.5, proba * 0.5)
    return y, proba


def test_best_f1_in_range():
    y, p = _scores()
    t, f1 = best_f1_threshold(y, p)
    assert 0.0 <= t <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_threshold_for_recall_meets_target():
    y, p = _scores()
    target = 0.8
    t, prec, rec = threshold_for_recall(y, p, target_recall=target)
    assert rec >= target - 1e-6
