"""Tests for resample.py."""
import numpy as np

from src.resample import smote_resample, random_over_resample, random_under_resample, resample


def _toy(n_majority=400, n_minority=20, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_majority + n_minority, 5)
    y = np.array([0] * n_majority + [1] * n_minority)
    return X, y


def test_smote_increases_minority():
    X, y = _toy()
    Xr, yr = smote_resample(X, y, sampling_strategy=0.5, random_state=42)
    assert sum(yr) > sum(y)
    # ratio of minority to majority should be ~0.5 (with rounding tolerance)
    ratio = sum(yr == 1) / sum(yr == 0)
    assert abs(ratio - 0.5) < 0.02


def test_random_over():
    X, y = _toy()
    Xr, yr = random_over_resample(X, y, sampling_strategy=0.3, random_state=42)
    assert sum(yr) > sum(y)


def test_random_under():
    X, y = _toy()
    Xr, yr = random_under_resample(X, y, sampling_strategy=0.5, random_state=42)
    # majority should shrink
    assert sum(yr == 0) < sum(y == 0)


def test_resample_dispatch_none():
    X, y = _toy()
    Xr, yr = resample("none", X, y)
    assert (Xr == X).all()
    assert (yr == y).all()
