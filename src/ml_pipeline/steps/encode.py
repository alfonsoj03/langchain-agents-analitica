from __future__ import annotations

import pandas as pd


def get_column_kinds(df: pd.DataFrame, target: str) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            kinds[col] = "numeric"
        else:
            kinds[col] = "categorical"
    return kinds


def encode_features(df: pd.DataFrame, target: str, column_kinds: dict[str, str]) -> pd.DataFrame:
    feature_df = df.drop(columns=[target], errors="ignore")
    numeric_cols = [c for c in feature_df.columns if column_kinds.get(c) == "numeric"]
    cat_cols = [c for c in feature_df.columns if column_kinds.get(c) == "categorical"]

    parts: list[pd.DataFrame] = []
    if numeric_cols:
        parts.append(feature_df[numeric_cols])
    if cat_cols:
        dummies = pd.get_dummies(feature_df[cat_cols], drop_first=False, dtype=int)
        parts.append(dummies)

    if not parts:
        encoded = pd.DataFrame(index=feature_df.index)
    else:
        encoded = pd.concat(parts, axis=1)

    result = encoded.copy()
    result[target] = df[target].values
    return result
