"""Evaluation metrics for the fraud model.

Class imbalance is severe (~0.17% fraud), so accuracy is useless. We track:
    precision, recall, f1, ROC AUC, average precision (PR-AUC).
The PR curve is more informative than ROC for this kind of imbalance.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve,
)


def evaluate(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    out = {
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    return out


def report(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    print(classification_report(y_test, pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, proba))
    print("PR AUC :", average_precision_score(y_test, proba))


def plot_pr_curve(model, X_test, y_test, out_path="reports/pr_curve.png"):
    proba = model.predict_proba(X_test)[:, 1]
    p, r, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(r, p, label="AP=%.4f" % ap)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_curve(model, X_test, y_test, out_path="reports/roc_curve.png"):
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="AUC=%.4f" % auc)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC curve")
    ax.legend()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
