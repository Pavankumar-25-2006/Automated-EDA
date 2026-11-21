"""
Unit tests for Intelligent Insight Generation (insight_generator.py).
"""
import pytest
import pandas as pd
import numpy as np
from modules.data_profiler import profile_dataset
from modules.statistics import compute_dataset_statistics
from modules.missing_values import analyze_missing_values
from modules.outlier_detection import analyze_outliers
from modules.correlation import compute_correlations
from modules.data_quality import calculate_data_quality_score
from modules.insight_generator import generate_insights


def test_insight_generation_rules():
    # Construct a dataset with skewness, outliers, and class dominance
    n = 200
    skewed_num = np.random.exponential(scale=10.0, size=n)
    # Add an outlier
    skewed_num[0] = 500.0
    dominated_cat = ["Active"] * 180 + ["Inactive"] * 20

    df = pd.DataFrame({
        "skewed_col": skewed_num,
        "status": dominated_cat
    })

    prof = profile_dataset(df)
    stats_res = compute_dataset_statistics(df, prof["type_groups"])
    missing = analyze_missing_values(df, prof["type_groups"])
    outliers = analyze_outliers(df, prof["type_groups"]["numerical"])
    corr = compute_correlations(df, prof["type_groups"]["numerical"])
    quality = calculate_data_quality_score(df, prof, missing)

    insights = generate_insights(prof, stats_res, missing, outliers, corr, quality)

    assert "overview" in insights
    assert len(insights["overview"]) > 0
    assert "data_quality" in insights
    assert len(insights["data_quality"]) > 0

    # Categorical pattern should detect class dominance
    cat_insights = insights["categorical_patterns"]
    assert any("dominance" in item["title"].lower() for item in cat_insights)

    # Numerical pattern should detect skewness
    num_insights = insights["numerical_patterns"]
    assert any("skew" in item["title"].lower() for item in num_insights)
