"""Tests for preprocess.py.

We had flakiness in the train_test_split: not all seeds were respected
across pandas versions. Pinning the seed and using stratify=y fixes it.
"""
import numpy as np
import pandas as pd

from src.preprocess import make_split, scale_time_amount, split_xy


def _toy_df(n=200, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "Time": rng.rand(n) * 10000,
        "Amount": rng.rand(n) * 500,
        "Class": (rng.rand(n) < 0.05).astype(int),
    })
    for i in range(1, 29):
        df["V%d" % i] = rng.randn(n)
    return df


def test_split_shapes():
    df = _toy_df()
    Xtr, Xte, ytr, yte = make_split(df, test_size=0.25, random_state=42)
    assert len(Xtr) + len(Xte) == len(df)
    assert len(ytr) == len(Xtr)
    assert len(yte) == len(Xte)


def test_scale_only_time_amount():
    df = _toy_df()
    Xtr, Xte, _, _ = make_split(df, random_state=42)
    Xtr_s, Xte_s, _ = scale_time_amount(Xtr, Xte)
    # V1 should be unchanged
    assert np.allclose(Xtr_s["V1"].values, Xtr["V1"].values)
    # Time should be standardized on train (~0 mean, unit std)
    assert abs(Xtr_s["Time"].mean()) < 1e-6
