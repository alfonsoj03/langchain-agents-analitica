from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

from ml_pipeline.config import Settings
from ml_pipeline.orchestrator.schemas import OutlierConfig, PipelinePlan
from ml_pipeline.steps.encode import get_column_kinds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"

load_dotenv(PROJECT_ROOT / ".env", override=False)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def has_gemini_key() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY"))


@pytest.fixture(scope="session")
def has_cursor_key() -> bool:
    return bool(os.getenv("CURSOR_API_KEY"))


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(
        project_root=PROJECT_ROOT,
        artifacts_dir=tmp_path / "artifacts",
        config_path=PROJECT_ROOT / "config" / "pipeline_config.yaml",
    )
    return s


@pytest.fixture
def simple_plan() -> PipelinePlan:
    return PipelinePlan(
        columns_to_drop=[],
        outliers=OutlierConfig(apply=False, method="none"),
        model_family="tree",
    )


def column_kinds_for(df: pd.DataFrame, target: str) -> dict[str, str]:
    return get_column_kinds(df, target)


@pytest.fixture
def numeric_train(fixtures_dir) -> pd.DataFrame:
    return pd.read_csv(fixtures_dir / "numeric_only_train.csv")


@pytest.fixture
def mixed_train(fixtures_dir) -> pd.DataFrame:
    return pd.read_csv(fixtures_dir / "mixed_train.csv")


@pytest.fixture
def cat_train(fixtures_dir) -> pd.DataFrame:
    return pd.read_csv(fixtures_dir / "cat_only_train.csv")


@pytest.fixture
def mixed_future(fixtures_dir) -> pd.DataFrame:
    return pd.read_csv(fixtures_dir / "mixed_future.csv")
