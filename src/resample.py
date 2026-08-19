"""Resampling for the imbalanced fraud problem.

Only fit on training data, never on test. We oversample the minority class
(fraud) to a configurable ratio. SMOTE is the default; we also support
random oversampling and random undersampling for comparison.

imbalanced-learn is an optional dependency. When it is not installed we fall
back to a pure-numpy random oversampler (for "smote" and "random_over") and a
pure-numpy random undersampler (for "random_under"), so the whole pipeline
still runs offline with only numpy/sklearn present. The fallback keeps the
requested minority/majority ratio exactly; it just duplicates or drops rows
instead of synthesizing SMOTE interpolations.
"""
import numpy as np

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except ImportError:  # pragma: no cover - exercised only when imblearn is absent
    SMOTE = RandomOverSampler = RandomUnderSampler = None
    HAS_IMBLEARN = False


def _as_2d(X):
    """Return (array, restore) where restore rebuilds the original container."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        pd = None
    if pd is not None and isinstance(X, pd.DataFrame):
        cols, index_name = list(X.columns), X.index.name
        values = X.to_numpy()

        def restore(arr):
            return pd.DataFrame(arr, columns=cols)

        return values, restore
    arr = np.asarray(X)

    def restore(a):
        return a

    return arr, restore


def _fallback_oversample(X, y, sampling_strategy, random_state):
    """Duplicate random minority rows until minority/majority == sampling_strategy."""
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    values, restore = _as_2d(X)
    maj_mask = y == 0
    min_mask = y == 1
    n_maj = int(maj_mask.sum())
    n_min = int(min_mask.sum())
    target_min = int(round(sampling_strategy * n_maj))
    if target_min <= n_min:
        return restore(values), y
    n_extra = target_min - n_min
    min_idx = np.where(min_mask)[0]
    extra = rng.choice(min_idx, size=n_extra, replace=True)
    new_values = np.vstack([values, values[extra]])
    new_y = np.concatenate([y, y[extra]])
    return restore(new_values), new_y


def _fallback_undersample(X, y, sampling_strategy, random_state):
    """Drop random majority rows until minority/majority == sampling_strategy."""
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    values, restore = _as_2d(X)
    maj_idx = np.where(y == 0)[0]
    min_idx = np.where(y == 1)[0]
    n_min = len(min_idx)
    target_maj = int(round(n_min / sampling_strategy))
    target_maj = min(target_maj, len(maj_idx))
    keep_maj = rng.choice(maj_idx, size=target_maj, replace=False)
    keep = np.sort(np.concatenate([keep_maj, min_idx]))
    return restore(values[keep]), y[keep]


def smote_resample(X_train, y_train, sampling_strategy=0.1, random_state=42, k_neighbors=None):
    # When the minority class is tiny, default k_neighbors=5 fails. Pick a safe
    # k based on the minority count.
    if not HAS_IMBLEARN:
        return _fallback_oversample(X_train, y_train, sampling_strategy, random_state)
    if k_neighbors is None:
        n_min = int(sum(np.asarray(y_train) == 1))
        k_neighbors = max(1, min(5, n_min - 1))
    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


def random_over_resample(X_train, y_train, sampling_strategy=0.1, random_state=42):
    if not HAS_IMBLEARN:
        return _fallback_oversample(X_train, y_train, sampling_strategy, random_state)
    s = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    return s.fit_resample(X_train, y_train)


def random_under_resample(X_train, y_train, sampling_strategy=0.5, random_state=42):
    if not HAS_IMBLEARN:
        return _fallback_undersample(X_train, y_train, sampling_strategy, random_state)
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
