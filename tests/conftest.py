"""Shared pytest fixtures."""
import numpy as np
import pandas as pd
import pytest


def _toy_df(n=1000, fraud_rate=0.02, seed=0):
    rng = np.random.RandomState(seed)
    n_fraud = max(2, int(n * fraud_rate))
    df = pd.DataFrame({
        "Time": rng.rand(n) * 86400,
        "Amount": rng.exponential(50, size=n),
    })
    for i in range(1, 29):
        df["V%d" % i] = rng.randn(n)
    cls = np.zeros(n, dtype=int)
    cls[:n_fraud] = 1
    rng.shuffle(cls)
    df["Class"] = cls
    return df


@pytest.fixture
def toy_df():
    return _toy_df()
