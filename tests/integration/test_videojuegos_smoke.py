import pytest

from ml_pipeline.config import get_settings
from ml_pipeline.pipeline import TrainingPipeline


@pytest.mark.live_gemini
def test_videojuegos_pipeline_metrics_are_non_zero(has_gemini_key, project_root, tmp_path):
    if not has_gemini_key:
        pytest.skip("GOOGLE_API_KEY not set")

    settings = get_settings()
    settings.artifacts_dir = tmp_path

    result = TrainingPipeline(settings).run(
        train_path=project_root / "data" / "videojuegos.csv",
        future_path=project_root / "data" / "videojuegos-datosFuturos.csv",
        artifacts_dir=tmp_path,
    )

    assert result.metrics["rmse"] > 1.0, "RMSE ~0 indicates perfect overfitting"
    assert result.metrics["mape"] > 1e-4, "MAPE ~0 indicates perfect overfitting"
