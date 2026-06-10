from sklearn.tree import DecisionTreeRegressor

from ml_pipeline.tree_report import generate_tree_html_report


def test_tree_report_writes_html(tmp_path, numeric_train):
    X = numeric_train.drop(columns=["y"])
    y = numeric_train["y"]
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X, y)

    path = generate_tree_html_report(
        model,
        feature_names=list(X.columns),
        target="y",
        output_path=tmp_path / "tree.html",
        metrics={"rmse": 1.5, "mae": 1.0, "mse": 2.25, "mape": 0.1},
        best_params={"max_depth": 3, "min_samples_leaf": 1},
    )
    content = path.read_text()

    assert "Decision Tree Model" in content
    assert "Tree Visualization" in content
    assert "Tree Rules (text)" in content
    assert "data:image/png;base64," in content
    assert "Best Hyperparameters" in content
    assert "Model Metrics (30% holdout)" in content
    assert "max_depth" in content
    for col in X.columns:
        assert col in content
