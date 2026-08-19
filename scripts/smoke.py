"""Offline smoke test for the credit-card-fraud-detection pipeline.

The real Kaggle creditcard.csv is ~150MB and needs the kaggle CLI, so this
smoke synthesizes a schema-matched imbalanced dataset (Time, V1..V28, Amount,
Class with ~2% fraud) via sklearn's make_classification. It then drives the
repo's own functions end to end:

    build features  ->  split + scale  ->  resample  ->  train
    ->  evaluate (ROC-AUC, PR-AUC, confusion matrix)  ->  score one transaction

Everything runs offline with numpy + sklearn only. imbalanced-learn is
optional: when it is missing, src.resample transparently falls back to a
numpy random oversampler (reported below). The default model is logistic
regression, which ships with sklearn so nothing extra needs installing.

Run:
    python scripts/smoke.py
    python scripts/smoke.py --model xgboost   # if xgboost is installed
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# make the repo importable when run as `python scripts/smoke.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import FEATURES, TARGET
from src.preprocess import make_split, scale_time_amount
from src.resample import resample as do_resample, HAS_IMBLEARN
from src.train import build_model
from src.evaluate import evaluate
from src.predict import predict_one


def make_synthetic_frame(n=8000, fraud_rate=0.02, seed=42):
    """Build a schema-matched imbalanced fraud frame: Time, V1..V28, Amount, Class."""
    X, y = make_classification(
        n_samples=n,
        n_features=len(FEATURES),      # 30 = Time + V1..V28 + Amount
        n_informative=8,
        n_redundant=6,
        n_clusters_per_class=2,
        weights=[1.0 - fraud_rate, fraud_rate],
        flip_y=0.01,
        class_sep=1.2,
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=FEATURES)
    # give Time/Amount realistic raw scales so StandardScaler has real work to do
    rng = np.random.RandomState(seed)
    df["Time"] = rng.rand(n) * 172800.0            # two days in seconds
    df["Amount"] = np.abs(df["Amount"]) * 50.0 + rng.rand(n) * 5.0
    df[TARGET] = y.astype(int)
    return df


def build_cfg(model_type):
    return {
        "data": {"test_size": 0.2, "random_state": 42},
        "preprocess": {"scale_columns": ["Time", "Amount"]},
        "resample": {"strategy": "smote", "smote_strategy": 0.2, "random_state": 42},
        "model": {
            "model_type": model_type,
            "logreg": {"C": 1.0, "max_iter": 200},
            "randomforest": {"n_estimators": 80, "max_depth": 8},
            "xgboost": {"n_estimators": 120, "max_depth": 4,
                        "learning_rate": 0.1, "scale_pos_weight": 1},
        },
        "eval": {"threshold": 0.5},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logreg",
                    choices=["logreg", "randomforest", "xgboost"])
    ap.add_argument("--n", type=int, default=8000)
    args = ap.parse_args()

    print("=" * 64)
    print("credit-card-fraud-detection :: offline smoke")
    print("=" * 64)
    print("imbalanced-learn available :", HAS_IMBLEARN,
          "(fallback: numpy random oversampler)" if not HAS_IMBLEARN else "")
    cfg = build_cfg(args.model)

    # 1. build features (synthetic, schema-matched, offline)
    df = make_synthetic_frame(n=args.n, fraud_rate=0.02, seed=42)
    n_fraud = int(df[TARGET].sum())
    print("synthetic frame           : %d rows, %d fraud (%.2f%%), %d features"
          % (len(df), n_fraud, 100.0 * n_fraud / len(df), len(FEATURES)))

    # 2. split + scale
    X_train, X_test, y_train, y_test = make_split(
        df, test_size=cfg["data"]["test_size"], random_state=cfg["data"]["random_state"])
    X_train, X_test, scaler = scale_time_amount(
        X_train, X_test, cols=cfg["preprocess"]["scale_columns"])

    # 3. resample the training set (guarded imblearn -> numpy fallback)
    Xr, yr = do_resample(
        cfg["resample"]["strategy"], X_train, y_train,
        sampling_strategy=cfg["resample"]["smote_strategy"],
        random_state=cfg["resample"]["random_state"])
    print("after resample            : %d rows, fraud frac=%.3f"
          % (len(Xr), float(np.mean(yr))))

    # 4. train
    model = build_model(cfg)
    print("model                     : %s" % args.model)
    model.fit(Xr, yr)

    # 5. evaluate (prints ROC-AUC + PR-AUC + confusion matrix)
    metrics = evaluate(model, X_test, y_test, threshold=cfg["eval"]["threshold"])
    print("-" * 64)
    print("ROC-AUC                   : %.4f" % metrics["roc_auc"])
    print("PR-AUC (avg precision)    : %.4f" % metrics["pr_auc"])
    print("precision / recall / f1   : %.4f / %.4f / %.4f"
          % (metrics["precision"], metrics["recall"], metrics["f1"]))
    print("confusion matrix [[tn,fp],[fn,tp]] : %s" % metrics["confusion_matrix"])
    print("-" * 64)

    # 6. score one sample transaction through the serving path (predict_one).
    # predict_one expects RAW feature values and applies the scaler itself, so
    # pass the original (pre-scale) test row rather than the scaled one.
    raw_row = df.drop(columns=[TARGET]).loc[X_test.index[0]].to_dict()
    result = predict_one(model, scaler, raw_row, threshold=cfg["eval"]["threshold"])
    proba = result["fraud_probability"]
    print("scored one transaction    : fraud_probability=%.4f label=%d"
          % (proba, result["label"]))

    # assertions: this is what makes it a *verified* smoke
    assert 0.0 <= proba <= 1.0, "fraud probability out of [0,1]: %r" % proba
    assert set(["roc_auc", "pr_auc", "confusion_matrix"]).issubset(metrics), "missing metrics"
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0

    print("=" * 64)
    print("SMOKE OK: trained, evaluated, and scored a transaction offline.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
