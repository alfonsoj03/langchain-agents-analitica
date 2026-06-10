import pandas as pd

from ml_pipeline.profiling import profile_dataframe
from ml_pipeline.steps.encode import get_column_kinds


def test_profile_categorical_has_cardinality_ratio_and_eta_squared(project_root):
    df = pd.read_csv(project_root / "data" / "videojuegos.csv")
    profile = profile_dataframe(df, target="Presupuesto para invertir")
    videojuego = next(c for c in profile["columns"] if c["name"] == "videojuego")
    assert "cardinality_ratio" in videojuego
    assert "target_eta_squared" in videojuego
    assert videojuego["target_eta_squared"] > 0.1


def test_profile_has_required_keys(numeric_train):
    profile = profile_dataframe(numeric_train, target="y")
    assert profile["n_rows"] == len(numeric_train)
    assert profile["n_columns"] == len(numeric_train.columns)
    assert profile["target"] == "y"
    assert len(profile["columns"]) == 3
    assert "correlation" in profile


def test_profile_column_kinds(numeric_train):
    profile = profile_dataframe(numeric_train)
    kinds = {c["name"]: c["kind"] for c in profile["columns"]}
    assert kinds["x1"] == "numeric"
    assert kinds["y"] == "numeric"


def test_profile_uses_provided_column_kinds(project_root):
    df = pd.read_csv(project_root / "data" / "videojuegos.csv")
    target = "Presupuesto para invertir"
    column_kinds = get_column_kinds(df, target)
    profile = profile_dataframe(df, target=target, column_kinds=column_kinds)
    kinds = {c["name"]: c["kind"] for c in profile["columns"] if c["name"] != target}
    assert kinds == column_kinds
