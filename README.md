# credit-card-fraud-detection

Fraud detection on the kaggle creditcard.csv dataset. Severe class imbalance.
Plan: try logistic regression and xgboost, use SMOTE for oversampling.

## Data

Kaggle: `mlg-ulb/creditcardfraud`. Download with `bash scripts/download_data.sh`
(needs the kaggle CLI configured). The CSV has 284,807 rows. Features V1..V28
are PCA components. Time and Amount are raw. Class is the target (1=fraud).

Only ~492 rows are fraud (~0.17%). This is what makes plain accuracy useless.
