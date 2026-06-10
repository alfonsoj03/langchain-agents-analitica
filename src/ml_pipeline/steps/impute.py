from __future__ import annotations

import pandas as pd


def impute_column(series: pd.Series, strategy: str) -> pd.Series:
    if series.isna().sum() == 0:
        return series
    if strategy == "median":
        return series.fillna(series.median())
    if strategy == "mode":
        mode_val = series.mode()
        fill = mode_val.iloc[0] if len(mode_val) else series.dropna().iloc[0] if series.notna().any() else 0
        return series.fillna(fill)
    return series


def impute_dataframe(
    df: pd.DataFrame,
    column_kinds: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    result = df.copy()
    fill_values: dict[str, object] = {}
    for col in result.columns:
        if col in column_kinds:
            kind = column_kinds[col]
        elif pd.api.types.is_numeric_dtype(result[col]):
            kind = "numeric"
        else:
            kind = "categorical"
        strategy = "median" if kind == "numeric" else "mode"
        if result[col].isna().sum() == 0:
            continue
        if strategy == "median":
            fill_values[col] = result[col].median()
        else:
            mode_val = result[col].mode()
            fill_values[col] = (
                mode_val.iloc[0]
                if len(mode_val)
                else result[col].dropna().iloc[0]
                if result[col].notna().any()
                else 0
            )
        result[col] = impute_column(result[col], strategy)
    return result, fill_values
