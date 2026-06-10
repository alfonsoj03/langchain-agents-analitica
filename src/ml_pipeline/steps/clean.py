from __future__ import annotations

import pandas as pd


def clip_outliers_iqr(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    result = df.copy()
    for col in numeric_cols:
        if col not in result.columns:
            continue
        series = result[col]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        result[col] = series.clip(lower=lower, upper=upper)
    return result
