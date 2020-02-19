"""Run all three model types under SMOTE and dump a comparison table.

usage:
    python -m src.benchmark
"""
import json
import os
import joblib
import yaml

from src.load_data import load
from src.preprocess import make_split, scale_time_amount
from src.resample import resample as do_resample
from src.evaluate import evaluate
from src.train import build_model


def bench(config_path="configs/default.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    df = load(cfg["data"]["path"])
    X_train, X_test, y_train, y_test = make_split(
        df, test_size=cfg["data"]["test_size"], random_state=cfg["data"]["random_state"])
    X_train, X_test, _ = scale_time_amount(X_train, X_test, cols=cfg["preprocess"]["scale_columns"])

    Xr, yr = do_resample(cfg["resample"]["strategy"], X_train, y_train,
                         sampling_strategy=cfg["resample"]["smote_strategy"],
                         random_state=cfg["resample"]["random_state"])

    rows = []
    for mt in ["logreg", "randomforest", "xgboost"]:
        cfg_local = dict(cfg)
        cfg_local["model"] = dict(cfg["model"])
        cfg_local["model"]["model_type"] = mt
        m = build_model(cfg_local)
        m.fit(Xr, yr)
        metrics = evaluate(m, X_test, y_test, threshold=cfg["eval"]["threshold"])
        rows.append({"model": mt, **metrics})
        print(mt, metrics)

    os.makedirs("reports", exist_ok=True)
    with open("reports/benchmark.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


if __name__ == "__main__":
    bench()
