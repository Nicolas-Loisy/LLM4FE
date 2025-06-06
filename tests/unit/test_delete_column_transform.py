import pandas as pd
from src.feature_engineering.transformations.delete_column import DeleteColumnTransform

def test_delete_column_success():
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    transform = DeleteColumnTransform(new_column_name="", source_columns=["col1"])
    result = transform.transform(df)
    assert "col1" not in result.columns
    assert "col2" in result.columns

def test_delete_column_not_found():
    df = pd.DataFrame({"col2": [3, 4]})
    transform = DeleteColumnTransform(new_column_name="", source_columns=["col1"])
    result = transform.transform(df)
    assert result.equals(df)

def test_delete_column_wrong_source_columns():
    transform = DeleteColumnTransform(new_column_name="", source_columns=["col1", "col2"])
    assert not transform.valid
