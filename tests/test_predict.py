"""Tests for predict.py."""
import numpy as np
import pytest

from src.predict import predict_one, validate, FEATURES


class _DummyModel:
    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


class _DummyScaler:
    def transform(self, X):
        return X


def test_features_count():
    # 1 (Time) + 28 (V1..V28) + 1 (Amount) = 30
    assert len(FEATURES) == 30


def test_validate_pass():
    payload = {k: 0.0 for k in FEATURES}
    ok, missing = validate(payload)
    assert ok
    assert missing == []


def test_validate_fail():
    payload = {"Time": 1.0}
    ok, missing = validate(payload)
    assert not ok
    assert "Amount" in missing


def test_predict_one_threshold():
    payload = {k: 0.0 for k in FEATURES}
    out = predict_one(_DummyModel(), _DummyScaler(), payload, threshold=0.5)
    assert out["label"] == 1
    out2 = predict_one(_DummyModel(), _DummyScaler(), payload, threshold=0.9)
    assert out2["label"] == 0


def test_predict_one_raises_on_missing():
    with pytest.raises(ValueError):
        predict_one(_DummyModel(), _DummyScaler(), {"Time": 0.0})
