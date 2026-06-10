from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from ml_pipeline.orchestrator.schemas import PipelinePlan
from ml_pipeline.steps.clean import clip_outliers_iqr
from ml_pipeline.steps.encode import encode_features
from ml_pipeline.steps.feature_select import apply_feature_selection
from ml_pipeline.steps.impute import impute_dataframe


@dataclass
class Preprocessor:
    target: str
    plan: PipelinePlan
    column_kinds: dict[str, str]
    dropped_columns: list[str] = field(default_factory=list)
    fill_values: dict[str, object] = field(default_factory=dict)
    variables: np.ndarray = field(default_factory=lambda: np.array([]))
    drop_first_map: dict[str, bool] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        working, dropped = apply_feature_selection(df, self.plan, self.target)
        self.dropped_columns = dropped
        if dropped:
            reason_map = {d.name: d.reason for d in self.plan.columns_to_drop}
            drop_details = ", ".join(
                f"{name} ({reason_map.get(name, 'planned')})" for name in dropped
            )
            logger.info("Dropping %d column(s): %s", len(dropped), drop_details)
        else:
            logger.info("No columns dropped by feature selection")

        self.column_kinds = {
            c: k
            for c, k in self.column_kinds.items()
            if c in working.columns and c != self.target
        }
        numeric_cols = [c for c, k in self.column_kinds.items() if k == "numeric"]
        cat_cols = [c for c, k in self.column_kinds.items() if k == "categorical"]
        logger.info(
            "Column types: %d numeric, %d categorical (target=%s excluded)",
            len(numeric_cols),
            len(cat_cols),
            self.target,
        )

        if self.plan.outliers.apply and self.plan.outliers.method == "iqr_clip":
            logger.info("Clipping outliers (IQR) on %d numeric column(s): %s", len(numeric_cols), numeric_cols)
            working = clip_outliers_iqr(working, numeric_cols)
        else:
            logger.info("Outlier handling: none")

        working, self.fill_values = impute_dataframe(working, self.column_kinds)
        if self.fill_values:
            impute_details = ", ".join(
                f"{col}={'median' if self.column_kinds.get(col) == 'numeric' else 'mode'}→{val!r}"
                for col, val in self.fill_values.items()
            )
            logger.info(
                "Imputed %d column(s) (median for numeric, mode for categorical): %s",
                len(self.fill_values),
                impute_details,
            )
        else:
            logger.info("No imputation required")

        logger.info("Normalization: not applied")

        encoded = encode_features(working, self.target, self.column_kinds)
        feature_cols = [c for c in encoded.columns if c != self.target]
        self.variables = np.array(feature_cols)

        if cat_cols:
            logger.info(
                "One-hot encoding applied to categorical column(s): %s",
                cat_cols,
            )
            logger.info(
                "Columns after one-hot encoding (%d): %s",
                len(feature_cols),
                list(feature_cols),
            )
        else:
            logger.info("One-hot encoding: not applied (no categorical features)")
            logger.info(
                "Feature columns (%d): %s",
                len(feature_cols),
                list(feature_cols),
            )
        return self

    def transform(self, df: pd.DataFrame, include_target: bool = False) -> pd.DataFrame:
        working = df.copy()
        working = working.drop(columns=self.dropped_columns, errors="ignore")

        for col, val in self.fill_values.items():
            if col in working.columns:
                working[col] = working[col].fillna(val)

        if self.plan.outliers.apply and self.plan.outliers.method == "iqr_clip":
            numeric_cols = [c for c, k in self.column_kinds.items() if k == "numeric"]
            working = clip_outliers_iqr(working, numeric_cols)

        if include_target and self.target in working.columns:
            encoded = encode_features(working, self.target, self.column_kinds)
            features = encoded.drop(columns=[self.target])
            features = features.reindex(columns=self.variables, fill_value=0)
            result = features.copy()
            result[self.target] = encoded[self.target].values
            return result

        feature_df = working.drop(columns=[self.target], errors="ignore")
        cat_cols = [c for c in feature_df.columns if self.column_kinds.get(c) == "categorical"]
        num_cols = [c for c in feature_df.columns if self.column_kinds.get(c) == "numeric"]

        parts: list[pd.DataFrame] = []
        if num_cols:
            parts.append(feature_df[num_cols])
        if cat_cols:
            dummies = pd.get_dummies(feature_df[cat_cols], drop_first=False, dtype=int)
            parts.append(dummies)

        if parts:
            features = pd.concat(parts, axis=1)
        else:
            features = pd.DataFrame(index=feature_df.index)

        return features.reindex(columns=self.variables, fill_value=0)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df, include_target=True)
