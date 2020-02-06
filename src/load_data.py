"""Load the kaggle creditcard.csv dataset."""
import os
import pandas as pd


DATA_PATH = os.path.join("data", "raw", "creditcard.csv")


def load(path=None):
    if path is None:
        path = DATA_PATH
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    df = load()
    print(df.shape)
    print(df["Class"].value_counts())
