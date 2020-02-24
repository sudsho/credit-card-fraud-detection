"""Tests for load_data."""
import os
import pandas as pd

from src.load_data import load


def test_load_csv(tmp_path):
    p = tmp_path / "creditcard.csv"
    df = pd.DataFrame({
        "Time": [0, 1, 2],
        "Amount": [10.0, 20.0, 30.0],
        "Class": [0, 0, 1],
        **{"V%d" % i: [0.0, 0.0, 0.0] for i in range(1, 29)},
    })
    df.to_csv(p, index=False)
    out = load(str(p))
    assert len(out) == 3
    assert "Class" in out.columns
