import pandas as pd
from src.feature_engineering.transformations.text_processing import TextProcessingTransform

def test_length_operation():
    df = pd.DataFrame({"text": ["hello", "world!"]})
    transform = TextProcessingTransform(new_column_name="length", source_columns=["text"], param={"operation": "length"})
    assert transform.valid
    result = transform.transform(df)
    assert "length" in result.columns
    assert result["length"].tolist() == [5, 6]

def test_word_count_operation():
    df = pd.DataFrame({"text": ["hello world", "a b c"]})
    transform = TextProcessingTransform(new_column_name="word_count", source_columns=["text"], param={"operation": "word_count"})
    assert transform.valid
    result = transform.transform(df)
    assert result["word_count"].tolist() == [2, 3]

def test_keyword_operation():
    df = pd.DataFrame({"text": ["hello world", "foo bar"]})
    transform = TextProcessingTransform(new_column_name="has_foo", source_columns=["text"], param={"operation": "keyword", "keyword": "foo"})
    assert transform.valid
    result = transform.transform(df)
    assert result["has_foo"].tolist() == [0, 1]

def test_tfidf_operation():
    df = pd.DataFrame({"text": ["foo bar baz", "foo foo qux"]})
    transform = TextProcessingTransform(new_column_name="tfidf", source_columns=["text"], param={"operation": "tfidf", "max_features": 2})
    assert transform.valid
    result = transform.transform(df)
    tfidf_cols = [col for col in result.columns if col.startswith("tfidf_")]
    assert len(tfidf_cols) == 2
    assert result[tfidf_cols].shape[0] == len(df)

def test_invalid_operation_invalid_flag():
    transform = TextProcessingTransform(new_column_name="invalid", source_columns=["text"], param={"operation": "invalid"})
    assert not transform.valid
