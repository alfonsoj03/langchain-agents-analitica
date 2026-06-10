import pandas as pd

from ml_pipeline.orchestrator.schemas import ColumnDrop, PipelinePlan
from ml_pipeline.steps.feature_select import apply_feature_selection


def test_drops_planned_column():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "y": [5, 6]})
    plan = PipelinePlan(columns_to_drop=[ColumnDrop(name="a", reason="test")])
    result, dropped = apply_feature_selection(df, plan, "y")
    assert "a" not in result.columns
    assert dropped == ["a"]
    assert list(result.columns) == ["b", "y"]


def test_never_drops_target():
    df = pd.DataFrame({"a": [1, 2], "y": [5, 6]})
    plan = PipelinePlan(columns_to_drop=[ColumnDrop(name="y", reason="test")])
    result, dropped = apply_feature_selection(df, plan, "y")
    assert "y" in result.columns
    assert dropped == []
