from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import pytest

from ml_pipeline.config import get_settings
from ml_pipeline.pipeline import TrainingPipeline
from ml_pipeline.steps.persist import load_bundle


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.mark.live_gemini
@pytest.mark.e2e
def test_full_pipeline_and_api(has_gemini_key, project_root, tmp_path):
    if not has_gemini_key:
        pytest.skip("GOOGLE_API_KEY not set")

    data_dir = project_root / "data"
    artifacts = tmp_path / "artifacts"
    settings = get_settings()
    settings.artifacts_dir = artifacts

    pipeline = TrainingPipeline(settings)
    result = pipeline.run(
        train_path=data_dir / "videojuegos.csv",
        future_path=data_dir / "videojuegos-datosFuturos.csv",
        artifacts_dir=artifacts,
    )

    assert result.prepared_dataset_path.exists()
    assert result.model_path.exists()
    assert result.report_path.exists()
    assert result.tree_report_path.exists()

    prepared = pd.read_csv(result.prepared_dataset_path)
    assert len(prepared) > 0

    report = result.report_path.read_text()
    assert "Statistical Description Report" in report

    tree_report = result.tree_report_path.read_text()
    assert "Decision Tree Model" in tree_report

    bundle = load_bundle(result.model_path)
    assert bundle.best_params
    assert set(bundle.metrics.keys()) == {"mse", "rmse", "mae", "mape"}
    assert result.metrics["rmse"] > 1.0, "RMSE ~0 indicates perfect overfitting"
    assert result.metrics["mape"] > 1e-4, "MAPE ~0 indicates perfect overfitting"

    port = _find_free_port()
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    env["MODEL_PATH"] = str(result.model_path)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "ml_pipeline.deployment.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(30):
            try:
                r = httpx.get(f"{base}/health", timeout=1.0)
                if r.status_code == 200:
                    break
            except httpx.RequestError:
                time.sleep(0.2)
        else:
            pytest.fail("API did not start")

        future_df = pd.read_csv(data_dir / "videojuegos-datosFuturos.csv")
        resp = httpx.post(
            f"{base}/predict",
            json={"records": future_df.to_dict(orient="records")},
            timeout=30.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == len(future_df)
        for record in data["records"]:
            assert "prediccion" in record
            assert isinstance(record["prediccion"], (int, float))
    finally:
        proc.terminate()
        proc.wait(timeout=5)
