from ml_pipeline.steps.split import split_train_test
from ml_pipeline.steps.train import train_tree_model


def test_train_fits_on_train_only(numeric_train):
    X_train, X_test, y_train, y_test = split_train_test(
        numeric_train, "y", test_size=0.3, random_state=42
    )
    config = {"tuning": {"n_iter": 5, "cv": 3, "random_state": 42}}
    model, best_params, summary = train_tree_model(X_train, y_train, config)
    assert hasattr(model, "predict")
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
