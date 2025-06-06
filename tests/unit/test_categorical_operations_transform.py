import pandas as pd
import pytest

from src.feature_engineering.transformations.categorical_operations import CategoricalOperationsTransform

def test_one_hot_encoding():
    df = pd.DataFrame({"color": ["red", "blue", "green"]})
    transform = CategoricalOperationsTransform(
        new_column_name="encoded",
        source_columns=["color"],
        param={"operation": "encodage_oneHot"}
    )
    result = transform.transform(df)
    assert "color_red" in result.columns
    assert "color_blue" in result.columns
    assert "color_green" in result.columns

def test_label_encoding():
    df = pd.DataFrame({"fruit": ["apple", "banana", "apple", "orange"]})
    transform = CategoricalOperationsTransform(
        new_column_name="encoded_fruit",
        source_columns=["fruit"],
        param={"operation": "label_encoding"}
    )
    result = transform.transform(df)
    assert "encoded_fruit" in result.columns
    assert result["encoded_fruit"].nunique() == 3

def test_target_encoding():
    df = pd.DataFrame({
        "category": ["A", "B", "A", "B", "C"],
        "target": [1, 0, 1, 1, 0]
    })
    transform = CategoricalOperationsTransform(
        new_column_name="encoded_target",
        source_columns=["category"],
        param={"operation": "target_encoding"}
    )
    result = transform.transform(df)
    assert "encoded_target" in result.columns
    assert result["encoded_target"].iloc[0] == result["encoded_target"].iloc[2]

def test_invalid_param_structure():
    with pytest.raises(ValueError):
        CategoricalOperationsTransform(
            new_column_name="x",
            source_columns=["col"],
            param=None
        )

def test_unsupported_operation():
    with pytest.raises(ValueError):
        CategoricalOperationsTransform(
            new_column_name="x",
            source_columns=["col"],
            param={"operation": "unknown"}
        )
