from ml_pipeline.preprocessor import Preprocessor
from ml_pipeline.steps.encode import get_column_kinds
from ml_pipeline.steps.persist import ModelBundle, load_bundle, save_bundle
from ml_pipeline.steps.split import split_train_test
from ml_pipeline.steps.train import train_tree_model


def test_persist_roundtrip(tmp_path, numeric_train, simple_plan):
    preprocessor = Preprocessor(
        target="y",
        plan=simple_plan,
        column_kinds=get_column_kinds(numeric_train, "y"),
    )
    prepared = preprocessor.fit_transform(numeric_train)
    X_train, X_test, y_train, y_test = split_train_test(
        prepared, "y", test_size=0.3, random_state=42
    )
    config = {"tuning": {"n_iter": 5, "cv": 3, "random_state": 42}}
    model, best_params, tuning_summary = train_tree_model(X_train, y_train, config)

    bundle = ModelBundle(
        preprocessor=preprocessor,
        model=model,
        variables=prepared.drop(columns=["y"]).columns.values,
        target="y",
        metrics={"mse": 1.0, "rmse": 1.0, "mae": 1.0, "mape": 0.1},
        best_params=best_params,
        tuning_summary=tuning_summary,
    )
    path = tmp_path / "model.pkl"
    save_bundle(bundle, path)
    loaded = load_bundle(path)
    assert loaded.target == "y"
    assert loaded.best_params == best_params
    assert len(loaded.variables) == len(bundle.variables)
