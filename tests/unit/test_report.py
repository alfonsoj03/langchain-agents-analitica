from pathlib import Path

import pandas as pd

from ml_pipeline.report import _correlation_dataframe, generate_html_report


def test_report_writes_html(tmp_path, mixed_train):
    path = generate_html_report(
        mixed_train, "y", tmp_path / "report.html", metrics={"rmse": 1.5}
    )
    content = path.read_text()
    assert "Statistical Description Report" in content
    assert "Numeric Statistical Summary" in content
    assert "Categorical Statistical Summary" in content
    assert "Variable Charts" in content
    assert "Feature-Feature Correlation Matrix" in content
    assert "Feature vs. Target Correlation" in content
    assert "Model Metrics" in content


def test_correlation_dataframe_encodes_categoricals(tmp_path):
    root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / "data" / "videojuegos.csv")
    target = "Presupuesto para invertir"
    _, feature_cols = _correlation_dataframe(df, target)
    assert any(c.startswith("videojuego_") for c in feature_cols)


def test_raw_report_before_dropping(tmp_path):
    root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / "data" / "videojuegos.csv")
    path = generate_html_report(
        df,
        "Presupuesto para invertir",
        tmp_path / "report_raw.html",
        subtitle="Generated before feature selection and preprocessing",
    )
    content = path.read_text()
    assert "Generated before feature selection and preprocessing" in content
    assert "Feature-Feature Correlation Matrix" in content
    assert "videojuego" in content
