"""
Unit tests for Correlation and Relationship Analysis (correlation.py).
"""
import pytest
import pandas as pd
import numpy as np
from modules.correlation import compute_correlations


def test_strong_linear_correlation():
    x = np.linspace(10, 100, 50)
    y = 2.5 * x + np.random.normal(0, 1, 50)
    df = pd.DataFrame({"x": x, "y": y})

    res = compute_correlations(df, ["x", "y"])
    assert res["applied"] is True
    assert len(res["top_pairs"]) == 1
    pair = res["top_pairs"][0]
    assert pair["pearson_corr"] > 0.95
    assert pair["strength"] == "Strong"
    assert pair["direction"] == "Positive"


def test_inverse_correlation():
    x = np.linspace(10, 100, 50)
    y = -3.0 * x + np.random.normal(0, 1, 50)
    df = pd.DataFrame({"x": x, "y": y})

    res = compute_correlations(df, ["x", "y"])
    assert res["applied"] is True
    pair = res["top_pairs"][0]
    assert pair["pearson_corr"] < -0.95
    assert pair["direction"] == "Negative"


def test_insufficient_features_correlation():
    df = pd.DataFrame({"single_col": [1, 2, 3]})
    res = compute_correlations(df, ["single_col"])
    assert res["applied"] is False
    assert "At least 2" in res["reason"]
