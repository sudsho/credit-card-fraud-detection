"""Predict fraud probability for a single transaction or a batch."""
import joblib
import numpy as np


FEATURES = ["Time"] + ["V%d" % i for i in range(1, 29)] + ["Amount"]


def load_artifacts(model_path="models/model.pkl", scaler_path="models/scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_one(model, scaler, features, threshold=0.5):
    """features is a dict with keys Time, V1..V28, Amount."""
    row = np.array([[features[k] for k in FEATURES]], dtype=float)
    # scale Time and Amount in-place
    row[:, [0, -1]] = scaler.transform(row[:, [0, -1]])
    proba = float(model.predict_proba(row)[0, 1])
    label = int(proba >= threshold)
    return {"fraud_probability": proba, "label": label}
