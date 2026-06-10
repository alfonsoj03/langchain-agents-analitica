import pytest
from sklearn.tree import DecisionTreeRegressor

from ml_pipeline.steps.evaluate import evaluate_regression
from ml_pipeline.steps.split import split_train_test
from ml_pipeline.steps.train import train_tree_model


def test_evaluate_warns_on_perfect_fit(numeric_train):
    X_train, _, y_train, _ = split_train_test(
        numeric_train, "y", test_size=0.3, random_state=42
    )
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    with pytest.warns(RuntimeWarning, match="perfect overfitting"):
        evaluate_regression(model, X_train, y_train)


def test_evaluate_returns_all_metrics(numeric_train):
    X_train, X_test, y_train, y_test = split_train_test(
        numeric_train, "y", test_size=0.3, random_state=42
    )
    config = {"tuning": {"n_iter": 5, "cv": 3, "random_state": 42}}
    model, _, _ = train_tree_model(X_train, y_train, config)
    metrics = evaluate_regression(model, X_test, y_test)
    assert set(metrics.keys()) == {"mse", "rmse", "mae", "mape"}
    assert metrics["rmse"] >= 0
