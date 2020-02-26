"""Load the kaggle creditcard.csv dataset."""
import os
import pandas as pd


DATA_PATH = os.path.join("data", "raw", "creditcard.csv")


def load(path=None):
    if path is None:
        path = DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            "could not find %s. run scripts/download_data.sh "
            "(needs the kaggle CLI configured)." % path)
    df = pd.read_csv(path)
    expected = {"Time", "Amount", "Class"} | {"V%d" % i for i in range(1, 29)}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError("input csv missing columns: %s" % sorted(missing))
    return df


if __name__ == "__main__":
    df = load()
    print(df.shape)
    print(df["Class"].value_counts())
