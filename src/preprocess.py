"""Preprocess the creditcard data.

The V1..V28 features are already PCA-transformed and roughly standardized.
The Time and Amount columns are NOT, so we scale them.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def split_xy(df, target="Class"):
    y = df[target].values
    X = df.drop(columns=[target])
    return X, y


def scale_time_amount(X_train, X_test, cols=("Time", "Amount")):
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[list(cols)] = scaler.fit_transform(X_train[list(cols)])
    X_test[list(cols)] = scaler.transform(X_test[list(cols)])
    return X_train, X_test, scaler


def make_split(df, test_size=0.2, random_state=42):
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
