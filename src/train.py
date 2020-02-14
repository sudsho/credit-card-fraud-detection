"""Train a fraud detection model.

usage:
    python -m src.train --config configs/default.yaml
"""
import argparse
import os
import joblib
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src.load_data import load
from src.preprocess import make_split, scale_time_amount
from src.resample import smote_resample
from src.evaluate import evaluate, plot_pr_curve, plot_roc_curve
from src.threshold import best_f1_threshold


def build_model(cfg):
    mt = cfg["model"]["model_type"]
    if mt == "logreg":
        p = cfg["model"]["logreg"]
        return LogisticRegression(C=p["C"], max_iter=p["max_iter"], solver="liblinear")
    if mt == "randomforest":
        p = cfg["model"]["randomforest"]
        return RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            random_state=cfg["data"]["random_state"], n_jobs=-1)
    if mt == "xgboost":
        p = cfg["model"]["xgboost"]
        return xgb.XGBClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            scale_pos_weight=p.get("scale_pos_weight", 1),
            objective="binary:logistic", n_jobs=-1)
    raise ValueError("unknown model_type: %s" % mt)


def run(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    df = load(cfg["data"]["path"])
    X_train, X_test, y_train, y_test = make_split(
        df, test_size=cfg["data"]["test_size"], random_state=cfg["data"]["random_state"])

    X_train, X_test, scaler = scale_time_amount(X_train, X_test, cols=cfg["preprocess"]["scale_columns"])

    if cfg["resample"]["strategy"] == "smote":
        X_train, y_train = smote_resample(
            X_train, y_train,
            sampling_strategy=cfg["resample"]["smote_strategy"],
            random_state=cfg["resample"]["random_state"])

    model = build_model(cfg)
    model.fit(X_train, y_train)

    # quick eval on the held-out test set
    metrics = evaluate(model, X_test, y_test, threshold=cfg["eval"]["threshold"])
    print("metrics:", metrics)

    proba = model.predict_proba(X_test)[:, 1]
    t_star, f1_star = best_f1_threshold(y_test, proba)
    print("best-F1 threshold = %.4f (F1=%.4f)" % (t_star, f1_star))

    plot_pr_curve(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("saved models/model.pkl")
    return model, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)
