import pandas as pd
import numpy as np
from src.feature_engineering.transformations.math_operations import MathOperationsTransform

def test_log_operation():
    df = pd.DataFrame({"value": [0, 1, np.e]})
    transform = MathOperationsTransform("log_value", ["value"], {"operation": "log"})
    result = transform.transform(df)
    expected = np.log1p(df["value"])
    assert np.allclose(result["log_value"], expected)

def test_sqrt_operation():
    df = pd.DataFrame({"value": [-1, 0, 4]})
    transform = MathOperationsTransform("sqrt_value", ["value"], {"operation": "sqrt"})
    result = transform.transform(df)
    expected = np.sqrt(df["value"].clip(lower=0))
    assert np.allclose(result["sqrt_value"], expected)

def test_square_operation():
    df = pd.DataFrame({"value": [2, -3, 4]})
    transform = MathOperationsTransform("square_value", ["value"], {"operation": "square"})
    result = transform.transform(df)
    expected = df["value"] ** 2
    assert (result["square_value"] == expected).all()

def test_diff_operation():
    df = pd.DataFrame({"a": [5, 8], "b": [3, 2]})
    transform = MathOperationsTransform("diff", ["a", "b"], {"operation": "diff"})
    result = transform.transform(df)
    expected = df["a"] - df["b"]
    assert (result["diff"] == expected).all()

def test_ratio_operation():
    df = pd.DataFrame({"a": [10, 8], "b": [2, 0]})
    transform = MathOperationsTransform("ratio", ["a", "b"], {"operation": "ratio"})
    result = transform.transform(df)
    expected = df["a"] / df["b"].replace(0, np.nan)
    assert np.allclose(result["ratio"], expected, equal_nan=True)

def test_mean_operation():
    df = pd.DataFrame({"x": [2, 4], "y": [6, 10]})
    transform = MathOperationsTransform("mean_val", ["x", "y"], {"operation": "mean"})
    result = transform.transform(df)
    expected = df[["x", "y"]].mean(axis=1)
    assert np.allclose(result["mean_val"], expected)

def test_addition_operation():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    transform = MathOperationsTransform("sum", ["x", "y"], {"operation": "addition"})
    result = transform.transform(df)
    expected = df[["x", "y"]].sum(axis=1)
    assert (result["sum"] == expected).all()

def test_multiply_operation():
    df = pd.DataFrame({"x": [2, 3], "y": [4, 5]})
    transform = MathOperationsTransform("product", ["x", "y"], {"operation": "multiply"})
    result = transform.transform(df)
    expected = df["x"] * df["y"]
    assert (result["product"] == expected).all()
