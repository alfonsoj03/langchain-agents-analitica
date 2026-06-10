from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def _infer_column_kind(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _eta_squared(series: pd.Series, target: pd.Series) -> float:
    groups = target.groupby(series)
    ss_between = sum(len(g) * (g.mean() - target.mean()) ** 2 for _, g in groups)
    ss_total = ((target - target.mean()) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def profile_dataframe(
    df: pd.DataFrame,
    target: str | None = None,
    column_kinds: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic JSON-serializable profile for the orchestrator."""
    columns: list[dict[str, Any]] = []
    numeric_cols: list[str] = []
    n_rows = len(df)
    target_series = (
        df[target]
        if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target])
        else None
    )

    for col in df.columns:
        series = df[col]
        if col == target:
            kind = _infer_column_kind(series)
        elif column_kinds and col in column_kinds:
            kind = column_kinds[col]
        else:
            kind = _infer_column_kind(series)
        info: dict[str, Any] = {
            "name": col,
            "dtype": str(series.dtype),
            "kind": kind,
            "null_count": int(series.isna().sum()),
            "null_pct": float(series.isna().mean()),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if kind == "numeric":
            numeric_cols.append(col)
            info.update(
                {
                    "min": float(series.min()) if series.notna().any() else None,
                    "max": float(series.max()) if series.notna().any() else None,
                    "mean": float(series.mean()) if series.notna().any() else None,
                    "std": float(series.std()) if series.notna().any() else None,
                }
            )
        else:
            top = series.value_counts(dropna=True).head(5)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}
            info["cardinality_ratio"] = float(info["unique_count"] / n_rows) if n_rows else 0.0
            if target_series is not None and col != target:
                info["target_eta_squared"] = _eta_squared(series, target_series)
        columns.append(info)

    corr: dict[str, dict[str, float]] = {}
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr(numeric_only=True)
        for row_col in corr_df.columns:
            corr[row_col] = {
                col: float(corr_df.loc[row_col, col])
                for col in corr_df.columns
                if not np.isnan(corr_df.loc[row_col, col])
            }

    profile = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": columns,
        "target": target,
        "correlation": corr,
    }
    return profile


def profile_to_json(profile: dict[str, Any]) -> str:
    return json.dumps(profile, indent=2, default=str)
