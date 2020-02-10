"""SMOTE resampling for the imbalanced fraud problem.

Only fit SMOTE on the training data, never on test. We oversample the minority
class (fraud) to a configurable ratio.
"""
from imblearn.over_sampling import SMOTE


def smote_resample(X_train, y_train, sampling_strategy=0.1, random_state=42):
    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res
