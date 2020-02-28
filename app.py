"""Flask app to serve the fraud model.

POST /predict  with JSON {"Time":..., "V1":..., ..., "V28":..., "Amount":...}
GET  /health
"""
import os
from flask import Flask, request, jsonify

from src.predict import load_artifacts, predict_one, FEATURES


app = Flask(__name__)
MODEL = None
SCALER = None
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))


def get_model():
    global MODEL, SCALER
    if MODEL is None:
        MODEL, SCALER = load_artifacts()
    return MODEL, SCALER


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True) or {}
    missing = [k for k in FEATURES if k not in payload]
    if missing:
        return jsonify({"error": "missing features", "missing": missing}), 400
    try:
        model, scaler = get_model()
        out = predict_one(model, scaler, payload, threshold=THRESHOLD)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # don't leak stack traces in prod
        app.logger.exception("predict failed")
        return jsonify({"error": "internal"}), 500
    return jsonify(out)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
