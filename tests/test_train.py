"""End-to-end smoke test for the training script using a toy CSV.

This is slow-ish but it shoots through the entire pipeline: load -> split ->
scale -> resample -> fit -> save. Catches things like config typos.
"""
import os
import yaml
import pytest


def _write_toy_csv(path, n=2000, fraud_rate=0.05, seed=0):
    import numpy as np
    import pandas as pd
    rng = np.random.RandomState(seed)
    n_fraud = max(20, int(n * fraud_rate))
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
    df.to_csv(path, index=False)


def _write_cfg(path, csv_path, model_type="logreg"):
    cfg = {
        "data": {"path": str(csv_path), "test_size": 0.25, "random_state": 42},
        "preprocess": {"scale_columns": ["Time", "Amount"]},
        "resample": {"strategy": "smote", "smote_strategy": 0.3, "random_state": 42},
        "model": {
            "model_type": model_type,
            "logreg": {"C": 1.0, "max_iter": 100},
            "randomforest": {"n_estimators": 20, "max_depth": 4},
            "xgboost": {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1, "scale_pos_weight": 1},
        },
        "eval": {"threshold": 0.5},
    }
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


@pytest.mark.parametrize("model_type", ["logreg", "xgboost"])
def test_train_smoke(tmp_path, monkeypatch, model_type):
    csv = tmp_path / "creditcard.csv"
    # use a slightly larger n so SMOTE k_neighbors has enough fraud rows
    _write_toy_csv(csv, n=4000, fraud_rate=0.05)
    cfg = tmp_path / "cfg.yaml"
    _write_cfg(cfg, csv, model_type=model_type)
    monkeypatch.chdir(tmp_path)
    from src.train import run
    model, metrics = run(str(cfg))
    # toy data is random; just check the metric is in [0, 1] and the file is on disk.
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert os.path.exists("models/model.pkl")
