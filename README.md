# credit-card-fraud-detection

Fraud detection on the kaggle creditcard.csv dataset using SMOTE
oversampling and gradient boosted trees, served as a small Flask API.

## Quick start (runs offline)

No Kaggle download and no imbalanced-learn needed. The smoke synthesizes a
schema-matched imbalanced dataset (Time, V1..V28, Amount, Class at ~2% fraud)
with sklearn's make_classification, then drives the repo's real pipeline end to
end: split, scale, resample, train, evaluate, and score one transaction.

    pip install numpy pandas scikit-learn matplotlib PyYAML joblib
    python scripts/smoke.py        # or: make smoke

Real output on Python 3.11 (scikit-learn 1.8, numpy 2.4, imbalanced-learn absent):

    ================================================================
    credit-card-fraud-detection :: offline smoke
    ================================================================
    imbalanced-learn available : False (fallback: numpy random oversampler)
    synthetic frame           : 8000 rows, 200 fraud (2.50%), 30 features
    after resample            : 7488 rows, fraud frac=0.167
    model                     : logreg
    ----------------------------------------------------------------
    ROC-AUC                   : 0.7730
    PR-AUC (avg precision)    : 0.1738
    precision / recall / f1   : 0.2167 / 0.3250 / 0.2600
    confusion matrix [[tn,fp],[fn,tp]] : [[1513, 47], [27, 13]]
    ----------------------------------------------------------------
    scored one transaction    : fraud_probability=0.2096 label=0
    ================================================================
    SMOKE OK: trained, evaluated, and scored a transaction offline.
    ================================================================

Run the tests (22 tests, all offline):

    python -m pytest -q        # or: make test
    # 22 passed

`make smoke-xgb` runs the same smoke with gradient-boosted trees if xgboost is
installed (ROC-AUC ~0.83, PR-AUC ~0.66 on the synthetic frame).

### Notes on the numbers and the environment

- These metrics are on **synthetic** data and only prove the pipeline runs; they
  are not the real-dataset numbers. On the actual Kaggle creditcard.csv, XGBoost
  with SMOTE typically reaches ROC-AUC ~0.97 and PR-AUC ~0.85 (see Metrics
  below). Fetch the real data with `bash scripts/download_data.sh` (needs the
  kaggle CLI) and run `make train`.
- **imbalanced-learn is optional.** When it is not installed, `src/resample.py`
  transparently falls back to a numpy random oversampler that hits the same
  target minority ratio (it duplicates minority rows instead of synthesizing
  SMOTE interpolations). Install `imbalanced-learn` for true SMOTE.
- The original hard pins (scikit-learn 0.22, pandas 1.0, Flask 1.1, Python 3.8)
  do not install on Python 3.11, so `requirements.txt` was relaxed to modern
  lower bounds. The code itself needed no API changes for sklearn 1.8 / numpy 2.

## Problem

European cardholder transactions over two days, ~284,807 rows, of which only
~492 are labelled as fraud. The fraud rate is ~0.17%, which is the kind of
class imbalance that breaks plain accuracy as a metric: a model that always
predicts "not fraud" gets 99.83% accuracy and is useless.

## Data

Kaggle: `mlg-ulb/creditcardfraud`. The CSV columns are:
- `Time` (seconds since the first transaction in the dataset)
- `V1`..`V28` (PCA-anonymized features)
- `Amount` (transaction amount)
- `Class` (1 = fraud, 0 = not fraud)

The V's are PCA-transformed for privacy, so we know nothing about what they
actually mean. Time and Amount are the only raw, unscaled columns; we run
StandardScaler on those two before training.

Download with `bash scripts/download_data.sh` (needs the kaggle CLI).

## Approach

1. Train/test split, stratified on `Class`.
2. StandardScaler on `Time` and `Amount` only (V1..V28 are already
   roughly standardized from PCA).
3. Resample the training set with SMOTE to a configurable minority ratio
   (default 0.1 = 10% fraud after oversampling). Test set is left alone.
4. Train one of: logistic regression, random forest, or XGBoost.
5. Evaluate with precision, recall, F1, ROC AUC, and PR AUC. Plot the
   precision-recall curve (more informative than ROC under heavy imbalance).
6. Tune the decision threshold to maximize F1, or to hit a target recall.

### Why SMOTE

Plain class weighting works for logreg and tree models but doesn't always
play well with downstream calibration. SMOTE explicitly synthesizes new
minority points in feature space using k-nearest-neighbours interpolation,
which gives the model more data to fit a decision boundary on. We oversample
to a moderate ratio (0.1) rather than full balance to keep the geometry of
the original problem.

### Why XGBoost vs LogReg vs RandomForest

XGBoost handles the non-linear interactions between the V's well and is
robust to feature scale, so it tends to win on this dataset. Logistic
regression is the cheap baseline and is easier to ship. Random forest sits
in the middle. The benchmark script (`src.benchmark`) trains all three
under the same SMOTE setup and dumps a JSON comparison.

## Metrics

With XGBoost, SMOTE strategy 0.1, threshold 0.5 we typically see:

- ROC AUC: ~0.97
- PR AUC: ~0.85
- Precision @ default threshold: ~0.87
- Recall @ default threshold: ~0.83

Numbers vary across seeds. Use `src.threshold.best_f1_threshold` to pick
a threshold that trades off precision for recall depending on the
operational cost of false positives.

## Usage

Install:

    pip install -r requirements.txt

Train:

    python -m src.train --config configs/default.yaml

Benchmark all three models:

    python -m src.benchmark

Serve locally:

    python app.py

POST a transaction to `/predict`:

    curl -X POST http://localhost:5000/predict \
        -H 'Content-Type: application/json' \
        -d '{"Time":0,"V1":0,"V2":0,...,"V28":0,"Amount":50.0}'

Health check:

    curl http://localhost:5000/health

## Deploy

The repo includes a Dockerfile, docker-compose.yml, a Heroku Procfile, and a
Travis CI config. To run in Docker:

    docker build -t fraud .
    docker run -p 5000:5000 fraud

Or with compose:

    docker-compose up

## Repo layout

```
.
|-- app.py                  Flask app (/predict, /health)
|-- configs/default.yaml    config for training and serving
|-- src/
|   |-- load_data.py        load and validate creditcard.csv
|   |-- preprocess.py       split + StandardScaler on Time/Amount
|   |-- resample.py         SMOTE / random over / random under
|   |-- threshold.py        threshold tuning helpers
|   |-- evaluate.py         metrics + PR/ROC curves
|   |-- train.py            training script
|   |-- benchmark.py        run logreg/rf/xgb side by side
|   `-- predict.py          predict_one used by the Flask app
|-- tests/                  pytest suite
|-- notebooks/eda.ipynb     class imbalance + V-distribution + heatmap
|-- scripts/
|   `-- download_data.sh    kaggle CLI download
|-- Dockerfile
|-- docker-compose.yml
|-- Procfile                heroku web dyno
|-- runtime.txt             python-3.8.0
|-- .travis.yml             CI: install + pytest
|-- requirements.txt
`-- LICENSE                 MIT
```

## License

MIT.
