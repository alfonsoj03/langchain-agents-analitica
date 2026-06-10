"""Pipeline runs on numeric-only, categorical-only, and mixed fixtures without Gemini API."""

import pandas as pd
import pytest

from ml_pipeline.orchestrator.schemas import PipelinePlan
from ml_pipeline.pipeline import TrainingPipeline


@pytest.mark.parametrize(
    "train_file,future_file",
    [
        ("numeric_only_train.csv", "numeric_only_future.csv"),
        ("cat_only_train.csv", "cat_only_future.csv"),
        ("mixed_train.csv", "mixed_future.csv"),
    ],
)
def test_pipeline_generalizes(fixtures_dir, tmp_path, train_file, future_file):
    pipeline = TrainingPipeline()
    result = pipeline.run(
        train_path=fixtures_dir / train_file,
        future_path=fixtures_dir / future_file,
        plan=PipelinePlan(),
        artifacts_dir=tmp_path / train_file,
    )
    assert result.prepared_dataset_path.exists()
    assert result.model_path.exists()
    assert result.report_path.exists()
    assert result.tree_report_path.exists()
    prepared = pd.read_csv(result.prepared_dataset_path)
    assert len(prepared) > 0
    assert result.target in prepared.columns
    assert set(result.metrics.keys()) == {"mse", "rmse", "mae", "mape"}
