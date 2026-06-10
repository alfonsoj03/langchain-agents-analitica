from ml_pipeline.steps.split import split_train_test


def test_split_70_30(mixed_train):
    X_train, X_test, y_train, y_test = split_train_test(mixed_train, "y", test_size=0.3, random_state=42)
    total = len(mixed_train)
    assert len(X_train) + len(X_test) == total
    assert abs(len(X_test) / total - 0.3) < 0.01
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
