from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import BaseEstimator

_SUSPICION_THRESHOLD = 1e-6


def evaluate_regression(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    suspicion_threshold: float = _SUSPICION_THRESHOLD,
) -> dict[str, float]:
    y_pred = model.predict(X_test)
    mse = float(metrics.mean_squared_error(y_test, y_pred))
    result = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(metrics.mean_absolute_error(y_test, y_pred)),
        "mape": float(metrics.mean_absolute_percentage_error(y_test, y_pred)),
    }
    if all(value < suspicion_threshold for value in result.values()):
        warnings.warn(
            "All holdout metrics are ~0.0 — possible perfect overfitting. "
            "Check for identifier/leaky categorical columns not dropped by the orchestrator.",
            RuntimeWarning,
            stacklevel=2,
        )
    return result
