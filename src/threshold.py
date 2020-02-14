"""Pick a threshold based on a target metric.

In fraud detection the default 0.5 threshold is rarely the right one. Often
we want to maximize F1, or to hit a target recall while keeping false-positive
volume manageable.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score


def best_f1_threshold(y_true, proba):
    p, r, t = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns one extra entry for p and r vs t
    f1 = 2 * p * r / (p + r + 1e-12)
    # ignore the last point which has no threshold
    idx = int(np.argmax(f1[:-1]))
    return float(t[idx]), float(f1[idx])


def threshold_for_recall(y_true, proba, target_recall=0.9):
    p, r, t = precision_recall_curve(y_true, proba)
    # find the highest threshold that still gives recall >= target
    ok = r[:-1] >= target_recall
    if not ok.any():
        return float(t[0]), float(p[0]), float(r[0])
    idx = int(np.where(ok)[0].max())
    return float(t[idx]), float(p[idx]), float(r[idx])
