.PHONY: smoke test install train benchmark serve clean

# Offline smoke: synthesizes an imbalanced dataset, trains, prints
# ROC-AUC / PR-AUC / confusion matrix, and scores one transaction.
# No network, no Kaggle download, no imbalanced-learn required.
smoke:
	python scripts/smoke.py

# Same smoke against gradient-boosted trees (needs xgboost installed).
smoke-xgb:
	python scripts/smoke.py --model xgboost

test:
	python -m pytest -q

install:
	pip install -r requirements.txt

# Train on the real Kaggle CSV (needs data/raw/creditcard.csv on disk).
train:
	python -m src.train --config configs/default.yaml

benchmark:
	python -m src.benchmark

serve:
	python app.py

clean:
	rm -rf models reports .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
