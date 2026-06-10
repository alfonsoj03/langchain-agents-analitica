from ml_pipeline.steps.encode import encode_features, get_column_kinds


def test_encode_mixed_produces_dummies(mixed_train):
    kinds = get_column_kinds(mixed_train, target="y")
    encoded = encode_features(mixed_train, "y", kinds)
    assert "y" in encoded.columns
    assert any("category" in c for c in encoded.columns)
    assert "age" in encoded.columns
    assert "score" in encoded.columns


def test_encode_numeric_only(numeric_train):
    kinds = get_column_kinds(numeric_train, target="y")
    encoded = encode_features(numeric_train, "y", kinds)
    assert set(encoded.columns) == {"x1", "x2", "y"}


def test_encode_categorical_only(cat_train):
    kinds = get_column_kinds(cat_train, target="y")
    encoded = encode_features(cat_train, "y", kinds)
    assert "y" in encoded.columns
    assert len([c for c in encoded.columns if c != "y"]) > 0
