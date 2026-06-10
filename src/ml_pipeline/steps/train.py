from __future__ import annotations

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from ml_pipeline.steps.tune import tune_decision_tree


def train_tree_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict | None = None,
) -> tuple[DecisionTreeRegressor, dict, dict]:
    """Train best tree on X_train only (70% split). No refit on 100%."""
    return tune_decision_tree(X_train, y_train, config)
