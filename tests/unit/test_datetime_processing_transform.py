import pandas as pd
from src.feature_engineering.transformations.datetime_processing import DateTimeProcessingTransform

def test_extract_year():
    df = pd.DataFrame({"date": ["2021-01-01", "2022-03-15"]})
    transform = DateTimeProcessingTransform("year", ["date"], {"operation": "year"})
    assert transform.valid
    result = transform.transform(df)
    assert result["year"].tolist() == [2021, 2022]

def test_extract_month():
    df = pd.DataFrame({"date": ["2021-01-01", "2022-03-15"]})
    transform = DateTimeProcessingTransform("month", ["date"], {"operation": "month"})
    assert transform.valid
    result = transform.transform(df)
    assert result["month"].tolist() == [1, 3]

def test_extract_day():
    df = pd.DataFrame({"date": ["2021-01-01", "2022-03-15"]})
    transform = DateTimeProcessingTransform("day", ["date"], {"operation": "day"})
    assert transform.valid
    result = transform.transform(df)
    assert result["day"].tolist() == [1, 15]

def test_extract_weekday():
    df = pd.DataFrame({"date": ["2021-01-01", "2022-03-15"]})
    transform = DateTimeProcessingTransform("weekday", ["date"], {"operation": "weekday"})
    assert transform.valid
    result = transform.transform(df)
    assert result["weekday"].tolist() == [4, 1]

def test_days_diff_with_two_columns():
    df = pd.DataFrame({
        "start": ["2021-01-01", "2021-01-05"],
        "end": ["2021-01-10", "2021-01-08"]
    })
    transform = DateTimeProcessingTransform("diff", ["start", "end"], {"operation": "days_diff"})
    assert transform.valid
    result = transform.transform(df)
    assert result["diff"].tolist() == [9, 3]

def test_days_diff_with_one_column():
    df = pd.DataFrame({"date": ["2021-01-01", "2021-01-02"]})
    transform = DateTimeProcessingTransform("diff", ["date"], {"operation": "days_diff"})
    # ici valide malgré un seul source_column (OK selon code)
    assert transform.valid
    result = transform.transform(df)
    assert "diff" not in result.columns

def test_days_diff_missing_second_column():
    df = pd.DataFrame({"start": ["2021-01-01", "2021-01-05"]})
    transform = DateTimeProcessingTransform("diff", ["start", "end"], {"operation": "days_diff"})
    assert transform.valid
    result = transform.transform(df)
    assert "diff" not in result.columns

def test_period_operation():
    df = pd.DataFrame({"date": ["2021-01-15", "2021-02-10"]})
    transform = DateTimeProcessingTransform("period", ["date"], {"operation": "period", "freq": "M"})
    assert transform.valid
    result = transform.transform(df)
    assert result["period"].tolist() == ["2021-01", "2021-02"]

def test_missing_column_skips_transformation():
    df = pd.DataFrame({"other_date": ["2021-01-01"]})
    transform = DateTimeProcessingTransform("year", ["date"], {"operation": "year"})
    assert transform.valid
    result = transform.transform(df)
    assert "year" not in result.columns

def test_invalid_operation_in_init():
    transform = DateTimeProcessingTransform("test", ["date"], {"operation": "invalid"})
    assert not transform.valid
