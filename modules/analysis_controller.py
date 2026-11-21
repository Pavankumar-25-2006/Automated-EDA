"""
Analysis Controller Layer.
Coordinates and orchestrates the end-to-end automated exploratory data analysis pipeline.
Ensures graceful degradation if an individual analysis module encounters an edge case.
"""
from typing import Dict, Any, Optional
import datetime
import pandas as pd

from modules.data_profiler import profile_dataset
from modules.statistics import compute_dataset_statistics
from modules.missing_values import analyze_missing_values
from modules.outlier_detection import analyze_outliers
from modules.correlation import compute_correlations
from modules.categorical_analysis import analyze_categorical_features
from modules.data_quality import calculate_data_quality_score
from modules.insight_generator import generate_insights
from modules.recommendations import generate_recommendations
from modules.visualizations import generate_all_visualizations


class AnalysisPipelineResult:
    """Consolidated container for complete AutoEDA execution results."""
    def __init__(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
        profiling: Dict[str, Any],
        statistics: Dict[str, Any],
        missing: Dict[str, Any],
        outliers: Dict[str, Any],
        correlations: Dict[str, Any],
        categorical: Dict[str, Any],
        data_quality: Dict[str, Any],
        insights: Dict[str, Any],
        recommendations: list,
        visualizations: Dict[str, Any],
        timestamp: str
    ):
        self.df = df
        self.metadata = metadata
        self.profiling = profiling
        self.statistics = statistics
        self.missing = missing
        self.outliers = outliers
        self.correlations = correlations
        self.categorical = categorical
        self.data_quality = data_quality
        self.insights = insights
        self.recommendations = recommendations
        self.visualizations = visualizations
        self.timestamp = timestamp


def run_full_analysis(df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> AnalysisPipelineResult:
    """
    Run complete automated exploratory data analysis pipeline on the given DataFrame.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = metadata or {
        "file_name": "uploaded_dataset.csv",
        "row_count": len(df),
        "column_count": len(df.columns)
    }

    # Step 1: Profiling
    profiling = profile_dataset(df)
    type_groups = profiling["type_groups"]
    num_cols = type_groups.get("numerical", [])
    cat_cols = type_groups.get("categorical", []) + type_groups.get("boolean", [])

    # Step 2: Missing Values
    try:
        missing = analyze_missing_values(df, type_groups)
    except Exception as e:
        missing = {
            "total_missing_cells": int(df.isna().sum().sum()),
            "overall_missing_percentage": 0.0,
            "rows_with_missing": 0,
            "rows_with_missing_percentage": 0.0,
            "columns_with_missing_count": 0,
            "summary_table": [],
            "recommendations": [{"type": "error", "message": f"Missing value analysis error: {str(e)}"}],
            "high_missing_columns": [],
            "completely_empty_columns": []
        }

    # Step 3: Statistics
    try:
        stats_res = compute_dataset_statistics(df, type_groups)
    except Exception as e:
        stats_res = {
            "numerical_statistics": {},
            "categorical_statistics": {},
            "numerical_summary_table": [],
            "categorical_summary_table": []
        }

    # Step 4: Outliers
    try:
        outliers = analyze_outliers(df, num_cols)
    except Exception as e:
        outliers = {
            "iqr_results": {},
            "summary_table": [],
            "columns_with_outliers": [],
            "columns_with_outliers_count": 0,
            "total_outlier_instances": 0,
            "ml_anomalies": {"applied": False, "reason": str(e)}
        }

    # Step 5: Correlations
    try:
        correlations = compute_correlations(df, num_cols)
    except Exception as e:
        correlations = {
            "applied": False,
            "reason": str(e),
            "top_pairs": [],
            "strong_positive_pairs": [],
            "strong_negative_pairs": [],
            "moderate_pairs": []
        }

    # Step 6: Categorical Analysis
    try:
        categorical = analyze_categorical_features(df, cat_cols)
    except Exception as e:
        categorical = {
            "column_details": {},
            "summary_table": [],
            "imbalanced_columns": [],
            "high_cardinality_columns": []
        }

    # Step 7: Data Quality Score
    try:
        data_quality = calculate_data_quality_score(df, profiling, missing)
    except Exception as e:
        data_quality = {
            "overall_score": 50.0,
            "grade": "C",
            "rating": "Average",
            "weights": {},
            "component_scores": {"completeness": 50.0, "uniqueness": 50.0, "consistency": 50.0},
            "issues": [f"Scoring error: {str(e)}"],
            "suggested_improvements": []
        }

    # Step 8: Insights
    try:
        insights = generate_insights(profiling, stats_res, missing, outliers, correlations, data_quality)
    except Exception as e:
        insights = {
            "overview": [{"category": "Overview", "priority": "info", "title": "Overview", "message": f"Rows: {len(df)}", "basis": ""}],
            "data_quality": [],
            "numerical_patterns": [],
            "categorical_patterns": [],
            "relationships": [],
            "anomalies": []
        }

    # Step 9: Recommendations
    try:
        recs = generate_recommendations(profiling, stats_res, missing, outliers, correlations)
    except Exception as e:
        recs = [{"category": "General", "priority": "Low", "detected": "Error", "why_it_matters": "", "action": str(e)}]

    # Step 10: Visualizations
    try:
        visualizations = generate_all_visualizations(df, profiling, stats_res, missing, correlations, data_quality)
    except Exception as e:
        visualizations = {}

    return AnalysisPipelineResult(
        df=df,
        metadata=meta,
        profiling=profiling,
        statistics=stats_res,
        missing=missing,
        outliers=outliers,
        correlations=correlations,
        categorical=categorical,
        data_quality=data_quality,
        insights=insights,
        recommendations=recs,
        visualizations=visualizations,
        timestamp=timestamp
    )
