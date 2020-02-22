"""Resampling for the imbalanced fraud problem.

Only fit on training data, never on test. We oversample the minority class
(fraud) to a configurable ratio. SMOTE is the default; we also support
random oversampling and random undersampling for comparison.
"""
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def smote_resample(X_train, y_train, sampling_strategy=0.1, random_state=42, k_neighbors=None):
    # When the minority class is tiny, default k_neighbors=5 fails. Pick a safe
    # k based on the minority count.
    if k_neighbors is None:
        n_min = int(sum(y_train == 1))
        k_neighbors = max(1, min(5, n_min - 1))
    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


def random_over_resample(X_train, y_train, sampling_strategy=0.1, random_state=42):
    s = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    return s.fit_resample(X_train, y_train)


def random_under_resample(X_train, y_train, sampling_strategy=0.5, random_state=42):
    s = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    return s.fit_resample(X_train, y_train)


def resample(strategy, X_train, y_train, sampling_strategy=0.1, random_state=42):
    if strategy == "smote":
        return smote_resample(X_train, y_train, sampling_strategy, random_state)
    if strategy == "random_over":
        return random_over_resample(X_train, y_train, sampling_strategy, random_state)
    if strategy == "random_under":
        return random_under_resample(X_train, y_train, sampling_strategy, random_state)
    if strategy == "none":
        return X_train, y_train
    raise ValueError("unknown resample strategy: %s" % strategy)
