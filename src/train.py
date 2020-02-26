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
from src.resample import resample as do_resample
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


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def prepare(cfg):
    """Load data, split, scale, resample. Returns (X_train, X_test, y_train, y_test, scaler)."""
    df = load(cfg["data"]["path"])
    X_train, X_test, y_train, y_test = make_split(
        df, test_size=cfg["data"]["test_size"], random_state=cfg["data"]["random_state"])
    X_train, X_test, scaler = scale_time_amount(
        X_train, X_test, cols=cfg["preprocess"]["scale_columns"])
    strat = cfg["resample"]["strategy"]
    if strat != "none":
        X_train, y_train = do_resample(
            strat, X_train, y_train,
            sampling_strategy=cfg["resample"]["smote_strategy"],
            random_state=cfg["resample"]["random_state"])
    return X_train, X_test, y_train, y_test, scaler


def fit_and_eval(model, X_train, X_test, y_train, y_test, threshold=0.5):
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test, threshold=threshold)
    return model, metrics


def save_artifacts(model, scaler, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))


def run(config_path):
    cfg = load_config(config_path)
    X_train, X_test, y_train, y_test, scaler = prepare(cfg)
    print("after resample:", len(X_train), "rows, fraud frac=",
          float(sum(y_train) / len(y_train)))

    model = build_model(cfg)
    model, metrics = fit_and_eval(
        model, X_train, X_test, y_train, y_test, threshold=cfg["eval"]["threshold"])
    print("metrics:", metrics)

    proba = model.predict_proba(X_test)[:, 1]
    t_star, f1_star = best_f1_threshold(y_test, proba)
    print("best-F1 threshold = %.4f (F1=%.4f)" % (t_star, f1_star))

    plot_pr_curve(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    save_artifacts(model, scaler)
    print("saved models/model.pkl")
    return model, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)
