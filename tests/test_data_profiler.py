"""
Unit tests for Dataset Profiler and Type Detection Layers.
"""
import pytest
import pandas as pd
import numpy as np
from modules.data_profiler import profile_dataset
from utils.type_detection import classify_dataset_columns, is_boolean_series, is_identifier_column


def test_profile_basic_dimensions():
    df = pd.DataFrame({
        "num": [1, 2, 3, 4, 5],
        "cat": ["A", "B", "A", "B", "C"],
        "constant": [99, 99, 99, 99, 99]
    })
    profile = profile_dataset(df)
    assert profile["total_rows"] == 5
    assert profile["total_columns"] == 3
    assert profile["duplicate_count"] == 0
    assert "constant" in profile["constant_columns"]


def test_duplicate_row_detection():
    df = pd.DataFrame({
        "x": [1, 2, 1, 1],
        "y": ["a", "b", "a", "a"]
    })
    profile = profile_dataset(df)
    assert profile["duplicate_count"] == 2
    assert profile["duplicate_percentage"] == 50.0


def test_type_detection_classification():
    df = pd.DataFrame({
        "user_id": [f"ID_{i}" for i in range(100)],
        "revenue": np.random.normal(5000, 500, 100),
        "is_active": [True, False] * 50,
        "department": ["Sales", "HR", "IT", "Sales"] * 25,
        "review_text": ["This is a very detailed customer feedback sentence number " + str(i) for i in range(100)]
    })
    classifications = classify_dataset_columns(df)
    
    assert classifications["user_id"]["semantic_type"] == "identifier"
    assert classifications["revenue"]["semantic_type"] == "numerical"
    assert classifications["is_active"]["semantic_type"] == "boolean"
    assert classifications["department"]["semantic_type"] == "categorical"
    assert classifications["review_text"]["semantic_type"] == "text"


def test_empty_column_profiling():
    df = pd.DataFrame({
        "valid_col": [10, 20, 30],
        "empty_col": [np.nan, np.nan, np.nan]
    })
    profile = profile_dataset(df)
    assert "empty_col" in profile["all_null_columns"]
