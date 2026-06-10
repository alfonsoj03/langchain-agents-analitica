from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _numeric_summary_html(df: pd.DataFrame) -> str:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return ""
    return (
        "<h2>Numeric Statistical Summary</h2>"
        + numeric_df.describe().to_html(classes="table")
    )


def _categorical_summary_html(df: pd.DataFrame) -> str:
    cat_df = df.select_dtypes(exclude="number")
    if cat_df.empty:
        return ""
    return (
        "<h2>Categorical Statistical Summary</h2>"
        + cat_df.describe().to_html(classes="table")
    )


def _variable_charts_html(
    df: pd.DataFrame, numeric_cols: list[str], cat_cols: list[str]
) -> str:
    sections: list[str] = []

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        df[col].hist(ax=ax, bins=20, edgecolor="black")
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        sections.append(
            f'<h3>Variable: {col}</h3>'
            f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />'
        )

    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        df[col].value_counts().plot(kind="barh", ax=ax)
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel("Count")
        sections.append(
            f'<h3>Variable: {col}</h3>'
            f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />'
        )

    if not sections:
        return ""
    return "<h2>Variable Charts</h2>" + "".join(sections)


def _correlation_dataframe(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, list[str]]:
    """Build a numeric frame for correlation (categoricals one-hot encoded)."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols and c != target]

    parts: list[pd.DataFrame] = []
    feature_numeric = [c for c in numeric_cols if c != target]
    if feature_numeric:
        parts.append(df[feature_numeric])
    if cat_cols:
        parts.append(pd.get_dummies(df[cat_cols], drop_first=False, dtype=float))

    if parts:
        corr_df = pd.concat(parts, axis=1)
    else:
        corr_df = pd.DataFrame(index=df.index)

    if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        corr_df[target] = df[target]

    feature_cols = [c for c in corr_df.columns if c != target]
    return corr_df, feature_cols


def _feature_feature_corr_html(df: pd.DataFrame, feature_cols: list[str]) -> str:
    if len(feature_cols) < 2:
        return ""

    corr = df[feature_cols].corr()
    n = len(corr.columns)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n * 0.8)))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title("Feature-Feature Correlation Matrix")

    for i in range(n):
        for j in range(n):
            value = corr.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    return (
        "<h2>Feature-Feature Correlation Matrix</h2>"
        f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />'
    )


def _feature_target_corr_html(
    df: pd.DataFrame, feature_cols: list[str], target: str
) -> str:
    if target not in df.columns or not feature_cols:
        return ""

    corr_with_target = df[feature_cols].corrwith(df[target]).dropna()
    if corr_with_target.empty:
        return ""

    corr_with_target = corr_with_target.reindex(
        corr_with_target.abs().sort_values(ascending=False).index
    )

    fig, ax = plt.subplots(figsize=(8, max(4, len(corr_with_target) * 0.4)))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in corr_with_target.values]
    corr_with_target.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"Feature vs. Target Correlation ({target})")
    ax.set_xlabel("Correlation")
    ax.set_xlim(-1, 1)
    ax.axvline(0, color="black", linewidth=0.8)

    return (
        "<h2>Feature vs. Target Correlation</h2>"
        f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />'
    )


def generate_html_report(
    df: pd.DataFrame,
    target: str,
    output_path: str | Path,
    metrics: dict[str, float] | None = None,
    subtitle: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    logger.info(
        "Building HTML report: %d rows, %d numeric + %d categorical columns, target=%s",
        len(df),
        len(numeric_cols),
        len(cat_cols),
        target,
    )

    corr_df, feature_cols = _correlation_dataframe(df, target)

    logger.info("Generating statistical summaries")
    summary_html = _numeric_summary_html(df) + _categorical_summary_html(df)
    logger.info("Generating distribution charts for %d variable(s)", len(numeric_cols) + len(cat_cols))
    charts_html = _variable_charts_html(df, numeric_cols, cat_cols)
    logger.info("Generating correlation matrices (%d features)", len(feature_cols))
    corr_feature_html = _feature_feature_corr_html(corr_df, feature_cols)
    corr_target_html = _feature_target_corr_html(corr_df, feature_cols, target)

    metrics_html = ""
    if metrics:
        rows = "".join(
            f"<tr><td>{k}</td><td>{v:.6f}</td></tr>" for k, v in metrics.items()
        )
        metrics_html = (
            "<h2>Model Metrics (30% holdout)</h2>"
            f"<table class='table'><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ML Pipeline Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; }}
    .table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
    .table th, .table td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; }}
    img {{ max-width: 100%; margin-bottom: 2rem; }}
    h3 {{ margin-top: 1.5rem; }}
  </style>
</head>
<body>
  <h1>Statistical Description Report</h1>
  {f"<p><em>{subtitle}</em></p>" if subtitle else ""}
  <h2>Dataset Overview</h2>
  <p>Rows: {len(df)} | Columns: {len(df.columns)} | Target: {target}</p>
  {summary_html}
  {charts_html}
  {corr_feature_html}
  {corr_target_html}
  {metrics_html}
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path
