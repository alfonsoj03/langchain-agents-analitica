from __future__ import annotations

import pandas as pd

from ml_pipeline.steps.persist import ModelBundle


def prepare_and_predict(bundle: ModelBundle, future_df: pd.DataFrame) -> pd.DataFrame:
    X = bundle.preprocessor.transform(future_df, include_target=False)
    predictions = bundle.model.predict(X)
    result = future_df.copy()
    pred_col = bundle.metadata.get("prediction_column", "prediccion")
    result[pred_col] = predictions
    return result
