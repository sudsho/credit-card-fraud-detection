"""Smoke tests for the Flask app.

We monkeypatch get_model so we don't need a real pickled model on disk.
"""
import numpy as np
import pytest

import app as flask_app
from src.predict import FEATURES


class _DummyModel:
    def predict_proba(self, X):
        x = np.asarray(X, dtype=float)
        # use Time feature as a faux signal
        s = (x[:, 0] - x[:, 0].min()) / max(1e-9, (x[:, 0].max() - x[:, 0].min()))
        return np.column_stack([1 - s, s])


class _DummyScaler:
    def transform(self, X):
        return X


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(flask_app, "get_model", lambda: (_DummyModel(), _DummyScaler()))
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


def _payload():
    return {k: 0.0 for k in FEATURES}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json=_payload())
    assert r.status_code == 200
    body = r.get_json()
    assert "fraud_probability" in body
    assert "label" in body


def test_predict_missing_features(client):
    r = client.post("/predict", json={"Time": 0.0})
    assert r.status_code == 400
    assert "missing" in r.get_json()
