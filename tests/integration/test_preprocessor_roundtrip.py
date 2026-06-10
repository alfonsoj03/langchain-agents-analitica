import pandas as pd

from ml_pipeline.preprocessor import Preprocessor
from ml_pipeline.steps.encode import get_column_kinds


def test_preprocessor_reindex_matches_variables(mixed_train, mixed_future, simple_plan):
    target = "y"
    prep = Preprocessor(
        target=target,
        plan=simple_plan,
        column_kinds=get_column_kinds(mixed_train, target),
    )
    prepared = prep.fit_transform(mixed_train)
    future_features = prep.transform(mixed_future, include_target=False)
    train_features = prepared.drop(columns=[target])
    assert list(future_features.columns) == list(train_features.columns)
    assert future_features.shape[1] == train_features.shape[1]
