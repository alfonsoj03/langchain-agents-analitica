import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ml_pipeline.deployment.api import app, get_bundle
from ml_pipeline.preprocessor import Preprocessor
from ml_pipeline.steps.encode import get_column_kinds
from ml_pipeline.steps.evaluate import evaluate_regression
from ml_pipeline.steps.persist import ModelBundle, save_bundle
from ml_pipeline.steps.split import split_train_test
from ml_pipeline.steps.train import train_tree_model


@pytest.fixture
def client_with_model(tmp_path, mixed_train, simple_plan):
    prep = Preprocessor(
        target="y",
        plan=simple_plan,
        column_kinds=get_column_kinds(mixed_train, "y"),
    )
    prepared = prep.fit_transform(mixed_train)
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
    model_path = tmp_path / "model.pkl"
    save_bundle(bundle, model_path)
    get_bundle(model_path)
    return TestClient(app), mixed_train


def test_health(client_with_model):
    client, _ = client_with_model
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_endpoint(client_with_model, fixtures_dir):
    client, _ = client_with_model
    future = pd.read_csv(fixtures_dir / "mixed_future.csv")
    resp = client.post("/predict", json={"records": future.to_dict(orient="records")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == len(future)
    assert all("prediccion" in r for r in data["records"])
