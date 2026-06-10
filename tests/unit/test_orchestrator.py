import pytest

from ml_pipeline.config import get_settings
from ml_pipeline.orchestrator.agent import OrchestratorAgent
from ml_pipeline.orchestrator.schemas import PipelinePlan
from ml_pipeline.profiling import profile_dataframe
from ml_pipeline.steps.ingest import detect_target, load_csv


@pytest.mark.live_gemini
def test_orchestrator_returns_valid_plan(has_gemini_key, fixtures_dir):
    if not has_gemini_key:
        pytest.skip("GOOGLE_API_KEY not set")

    train = load_csv(fixtures_dir / "mixed_train.csv")
    future = load_csv(fixtures_dir / "mixed_future.csv")
    target = detect_target(train, future)
    profile = profile_dataframe(train, target=target)

    agent = OrchestratorAgent(get_settings())
    plan = agent.plan(profile, target)

    assert isinstance(plan, PipelinePlan)
    assert plan.model_family == "tree"
    assert all(d.name != target for d in plan.columns_to_drop)


@pytest.mark.live_gemini
def test_orchestrator_drops_high_eta_column_on_videojuegos(has_gemini_key, project_root):
    if not has_gemini_key:
        pytest.skip("GOOGLE_API_KEY not set")

    train = load_csv(project_root / "data" / "videojuegos.csv")
    future = load_csv(project_root / "data" / "videojuegos-datosFuturos.csv")
    target = detect_target(train, future)
    profile = profile_dataframe(train, target=target)

    agent = OrchestratorAgent(get_settings())
    plan = agent.plan(profile, target)

    dropped = {d.name for d in plan.columns_to_drop}
    assert "videojuego" in dropped
