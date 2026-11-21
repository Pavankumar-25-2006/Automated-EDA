"""
Unit tests for Data Quality Scoring (data_quality.py).
"""
import pytest
import pandas as pd
import numpy as np
from modules.data_profiler import profile_dataset
from modules.missing_values import analyze_missing_values
from modules.data_quality import calculate_data_quality_score


def test_perfect_clean_data_quality_score():
    df = pd.DataFrame({
        "num1": range(100),
        "num2": range(100, 200),
        "cat1": ["A", "B"] * 50
    })
    prof = profile_dataset(df)
    missing = analyze_missing_values(df, prof["type_groups"])
    quality = calculate_data_quality_score(df, prof, missing)

    assert 95.0 <= quality["overall_score"] <= 100.0
    assert quality["grade"] == "A"
    assert quality["rating"] == "Excellent"
    assert quality["component_scores"]["completeness"] == 100.0
    assert quality["component_scores"]["uniqueness"] == 100.0
    assert quality["component_scores"]["consistency"] == 100.0


def test_score_penalties_on_dirty_data():
    # 50% missingness, 50% duplicates, empty columns
    df = pd.DataFrame({
        "a": [1, 1, 1, 1, np.nan, np.nan, np.nan, np.nan],
        "empty": [np.nan] * 8,
        "constant": [10] * 8
    })
    prof = profile_dataset(df)
    missing = analyze_missing_values(df, prof["type_groups"])
    quality = calculate_data_quality_score(df, prof, missing)

    # Score should be significantly penalized
    assert quality["overall_score"] < 70.0
    assert quality["grade"] in ["C", "D", "F"]
    assert len(quality["issues"]) > 0
    assert quality["component_scores"]["completeness"] < 100.0
