"""
Unit tests for Descriptive Statistics Layer (statistics.py).
"""
import pytest
import pandas as pd
import numpy as np
from modules.statistics import (
    compute_numerical_statistics,
    compute_categorical_statistics,
    compute_dataset_statistics
)


def test_numerical_statistics_accuracy():
    data = [10, 20, 30, 40, 50]
    s = pd.Series(data)
    stats = compute_numerical_statistics(s, "test_num")
    
    assert stats["count"] == 5
    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["range"] == 40.0
    assert stats["q25"] == 20.0
    assert stats["q75"] == 40.0
    assert stats["iqr"] == 20.0
    assert stats["skewness"] == 0.0
    assert stats["is_constant"] is False


def test_constant_numerical_statistics():
    data = [42.0, 42.0, 42.0, 42.0]
    s = pd.Series(data)
    stats = compute_numerical_statistics(s, "constant_num")
    
    assert stats["count"] == 4
    assert stats["mean"] == 42.0
    assert stats["std"] == 0.0
    assert stats["range"] == 0.0
    assert stats["iqr"] == 0.0
    assert stats["is_constant"] is True


def test_all_nan_numerical_statistics():
    s = pd.Series([np.nan, np.nan])
    stats = compute_numerical_statistics(s, "all_nan")
    assert stats["count"] == 0
    assert stats["mean"] is None
    assert stats["is_constant"] is True


def test_categorical_statistics_frequencies():
    data = ["Red", "Blue", "Red", "Green", "Red", "Blue"]
    s = pd.Series(data)
    stats = compute_categorical_statistics(s, "colors")
    
    assert stats["count"] == 6
    assert stats["unique_categories"] == 3
    assert stats["mode"] == "Red"
    assert stats["mode_frequency"] == 3
    assert stats["mode_percentage"] == 50.0
    assert stats["frequencies"]["Red"] == 3
    assert stats["frequencies"]["Blue"] == 2
    assert stats["frequencies"]["Green"] == 1
    assert stats["is_constant"] is False
