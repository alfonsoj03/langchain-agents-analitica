from __future__ import annotations

import pandas as pd

from ml_pipeline.orchestrator.schemas import PipelinePlan


def apply_feature_selection(
    df: pd.DataFrame, plan: PipelinePlan, target: str
) -> tuple[pd.DataFrame, list[str]]:
    drop_names = [d.name for d in plan.columns_to_drop if d.name != target]
    dropped = [c for c in drop_names if c in df.columns]
    return df.drop(columns=dropped, errors="ignore"), dropped
