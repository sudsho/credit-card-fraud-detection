"""Evaluation metrics for the fraud model.

Class imbalance is severe (~0.17% fraud), so accuracy is useless. We track:
    precision, recall, f1, ROC AUC, average precision (PR-AUC).
"""
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
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
