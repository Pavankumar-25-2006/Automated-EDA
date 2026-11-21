"""
Unit tests for Outlier Detection (outlier_detection.py).
"""
import pytest
import pandas as pd
import numpy as np
from modules.outlier_detection import detect_column_iqr_outliers, analyze_outliers


def test_iqr_outlier_detection():
    # Normal distribution with 2 extreme outliers
    data = [10, 11, 10, 12, 11, 10, 12, 11, 10, 11, 100, -50]
    s = pd.Series(data)
    res = detect_column_iqr_outliers(s, "test_outliers")

    assert res["outlier_count"] == 2
    assert res["upper_outliers_count"] == 1
    assert res["lower_outliers_count"] == 1
    assert res["outlier_percentage"] > 0
    assert res["status"] == "Outliers detected"


def test_no_outliers_in_uniform_data():
    s = pd.Series(range(20))
    res = detect_column_iqr_outliers(s, "uniform")
    assert res["outlier_count"] == 0
    assert res["outlier_percentage"] == 0.0


def test_zero_iqr_handling():
    s = pd.Series([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    res = detect_column_iqr_outliers(s, "zero_iqr")
    assert res["iqr"] == 0.0
    assert res["outlier_count"] == 0
    assert "Zero IQR" in res["status"]
