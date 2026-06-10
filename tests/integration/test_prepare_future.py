import pandas as pd

from ml_pipeline.deployment.prepare_future import prepare_and_predict
from ml_pipeline.pipeline import TrainingPipeline
from ml_pipeline.orchestrator.schemas import ColumnDrop, PipelinePlan


def test_prepare_future_adds_prediction(tmp_path, fixtures_dir, simple_plan):
    pipeline = TrainingPipeline()
    train = pd.read_csv(fixtures_dir / "mixed_train.csv")
    future = pd.read_csv(fixtures_dir / "mixed_future.csv")
    plan = simple_plan
    preprocessor_plan = plan

    from ml_pipeline.preprocessor import Preprocessor
    from ml_pipeline.steps.encode import get_column_kinds
    from ml_pipeline.steps.persist import ModelBundle, save_bundle
    from ml_pipeline.steps.split import split_train_test
    from ml_pipeline.steps.train import train_tree_model
    from ml_pipeline.steps.evaluate import evaluate_regression

    prep = Preprocessor(
        target="y",
        plan=preprocessor_plan,
        column_kinds=get_column_kinds(train, "y"),
    )
    prepared = prep.fit_transform(train)
    X_train, X_test, y_train, y_test = split_train_test(prepared, "y", test_size=0.3, random_state=42)
    model, best_params, tuning_summary = train_tree_model(
        X_train, y_train, {"tuning": {"n_iter": 5, "cv": 3, "random_state": 42}}
    )
    metrics = evaluate_regression(model, X_test, y_test)
    bundle = ModelBundle(
        preprocessor=prep,
        model=model,
        variables=prepared.drop(columns=["y"]).columns.values,
        target="y",
        metrics=metrics,
        best_params=best_params,
        tuning_summary=tuning_summary,
    )
    result = prepare_and_predict(bundle, future)
    assert "prediccion" in result.columns
    assert len(result) == len(future)
    assert result["prediccion"].notna().all()
