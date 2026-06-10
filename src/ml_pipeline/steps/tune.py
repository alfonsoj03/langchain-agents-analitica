from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)


def _build_param_grid(raw_grid: dict[str, list]) -> dict[str, list]:
    grid: dict[str, list] = {}
    for key, values in raw_grid.items():
        grid[key] = [None if v is None or v == "null" else v for v in values]
    return grid


def tune_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any] | None = None,
) -> tuple[DecisionTreeRegressor, dict[str, Any], dict[str, Any]]:
    cfg = config or {}
    tuning = cfg.get("tuning", {})
    n_iter = tuning.get("n_iter", 20)
    cv = tuning.get("cv", 3)
    random_state = tuning.get("random_state", 42)
    param_grid = _build_param_grid(
        tuning.get(
            "param_grid",
            {
                "max_depth": [None, 3, 5, 8, 12, 20],
                "min_samples_leaf": [1, 2, 4, 8, 16],
                "min_samples_split": [2, 5, 10, 20],
                "criterion": ["squared_error", "absolute_error"],
            },
        )
    )
    if len(X_train) < 500 and None in param_grid.get("max_depth", []):
        param_grid["max_depth"] = [v for v in param_grid["max_depth"] if v is not None]

    logger.info(
        "Tuning DecisionTreeRegressor: %d candidates, %d-fold CV, %d training rows",
        n_iter,
        cv,
        len(X_train),
    )

    base = DecisionTreeRegressor(random_state=random_state)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        refit=True,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_params = dict(search.best_params_)
    cv_results = search.cv_results_
    top_idx = int(np.argmin(-cv_results["mean_test_score"]))
    tuning_summary = {
        "best_cv_rmse": float(-cv_results["mean_test_score"][top_idx]),
        "n_candidates": int(len(cv_results["params"])),
        "n_iter": int(n_iter),
        "cv_folds": int(cv),
    }
    return search.best_estimator_, best_params, tuning_summary
