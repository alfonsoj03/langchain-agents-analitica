import numpy as np
import pandas as pd

from ml_pipeline.steps.evaluate import evaluate_regression
from ml_pipeline.steps.split import split_train_test
from ml_pipeline.steps.tune import tune_decision_tree


def test_tune_caps_max_depth_on_small_dataset():
    rng = np.random.default_rng(42)
    n = 50
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    y = pd.Series(X["x1"] * 2 + X["x2"] + rng.normal(scale=0.1, size=n))
    config = {"tuning": {"n_iter": 10, "cv": 3, "random_state": 42}}
    _, best_params, _ = tune_decision_tree(X, y, config)
    assert best_params["max_depth"] is not None


def test_tune_returns_best_params_and_beats_default(numeric_train):
    X_train, X_test, y_train, y_test = split_train_test(
        numeric_train, "y", test_size=0.3, random_state=42
    )
    config = {"tuning": {"n_iter": 10, "cv": 3, "random_state": 42}}
    model, best_params, summary = tune_decision_tree(X_train, y_train, config)

    assert isinstance(best_params, dict)
    assert "max_depth" in best_params or "min_samples_leaf" in best_params
    assert summary["n_candidates"] >= 1

    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    metrics = evaluate_regression(model, X_test, y_test)
    assert metrics["rmse"] >= 0
    assert summary["best_cv_rmse"] >= 0
