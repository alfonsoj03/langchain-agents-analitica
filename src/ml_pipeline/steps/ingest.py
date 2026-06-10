from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";")
    return df


def detect_target(train: pd.DataFrame, future: pd.DataFrame | None = None) -> str:
    if future is not None:
        train_cols = set(train.columns)
        future_cols = set(future.columns)
        diff = train_cols - future_cols
        if len(diff) == 1:
            return next(iter(diff))
        if len(diff) > 1:
            numeric_diff = [
                c for c in diff if pd.api.types.is_numeric_dtype(train[c])
            ]
            if len(numeric_diff) == 1:
                return numeric_diff[0]
    numeric_cols = [
        c for c in train.columns if pd.api.types.is_numeric_dtype(train[c])
    ]
    if numeric_cols:
        return numeric_cols[-1]
    return train.columns[-1]
