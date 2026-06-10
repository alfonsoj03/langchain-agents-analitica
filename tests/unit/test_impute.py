import pandas as pd

from ml_pipeline.steps.encode import get_column_kinds
from ml_pipeline.steps.impute import impute_dataframe


def test_impute_numeric_median():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "z"]})
    kinds = get_column_kinds(df, target="b")
    result, fills = impute_dataframe(df, kinds)
    assert result["a"].isna().sum() == 0
    assert fills["a"] == 2.0


def test_impute_categorical_mode():
    df = pd.DataFrame({"cat": ["a", None, "a", "b"], "y": [1, 2, 3, 4]})
    kinds = get_column_kinds(df, target="y")
    result, fills = impute_dataframe(df, kinds)
    assert result["cat"].isna().sum() == 0
    assert fills["cat"] == "a"
