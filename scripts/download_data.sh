#!/usr/bin/env bash
# Download the kaggle creditcard fraud dataset.
# Requires the kaggle CLI: pip install kaggle, plus ~/.kaggle/kaggle.json
set -e
mkdir -p data/raw
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
echo "downloaded to data/raw/creditcard.csv"
