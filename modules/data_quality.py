"""
Data Quality Scoring Layer.
Calculates a transparent, explainable heuristic data quality score (0 to 100)
across Completeness, Uniqueness, and Structural Consistency dimensions.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from config import (
    QUALITY_WEIGHT_COMPLETENESS,
    QUALITY_WEIGHT_UNIQUENESS,
    QUALITY_WEIGHT_CONSISTENCY,
)
from utils.helpers import safe_divide


def calculate_data_quality_score(
    df: pd.DataFrame,
    profiling_results: Dict[str, Any],
    missing_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate an explainable data quality score with transparent weights and component scores.
    """
    total_rows = profiling_results.get("total_rows", len(df))
    total_cols = profiling_results.get("total_columns", len(df.columns))
    total_cells = total_rows * total_cols

    if total_cells == 0:
        return {
            "overall_score": 0.0,
            "grade": "F",
            "rating": "Invalid",
            "component_scores": {
                "completeness": 0.0,
                "uniqueness": 0.0,
                "consistency": 0.0
            },
            "issues": ["Empty dataset"],
            "suggested_improvements": ["Upload a non-empty dataset."],
            "limitations": "Cannot evaluate an empty dataset."
        }

    # 1. Completeness Score (0 - 100)
    total_missing = missing_results.get("total_missing_cells", 0)
    completeness_ratio = max(0.0, 1.0 - safe_divide(total_missing, total_cells))
    completeness_score = round(completeness_ratio * 100.0, 2)

    # 2. Uniqueness Score (0 - 100)
    duplicate_count = profiling_results.get("duplicate_count", 0)
    uniqueness_ratio = max(0.0, 1.0 - safe_divide(duplicate_count, total_rows))
    uniqueness_score = round(uniqueness_ratio * 100.0, 2)

    # 3. Structural Consistency Score (0 - 100)
    # Start at 100 and deduct for structural defects
    consistency_deductions = 0.0
    issues = []
    suggestions = []

    # Penalty for 100% empty columns (15 pts per empty col, max 40 pts)
    empty_cols = profiling_results.get("all_null_columns", [])
    if empty_cols:
        deduction = min(40.0, len(empty_cols) * 15.0)
        consistency_deductions += deduction
        issues.append(f"{len(empty_cols)} column(s) are completely empty: {empty_cols}")
        suggestions.append(f"Remove completely empty columns ({empty_cols}) to save memory and avoid model errors.")

    # Penalty for constant columns (10 pts per constant col, max 30 pts)
    constant_cols = profiling_results.get("constant_columns", [])
    # Only count if not already in empty_cols
    non_null_constant = [c for c in constant_cols if c not in empty_cols]
    if non_null_constant:
        deduction = min(30.0, len(non_null_constant) * 10.0)
        consistency_deductions += deduction
        issues.append(f"{len(non_null_constant)} column(s) contain only one constant value: {non_null_constant}")
        suggestions.append(f"Evaluate removing constant column(s) ({non_null_constant}) as they carry zero informational variance.")

    # Penalty for columns with extreme missingness (> 50%)
    high_missing = missing_results.get("high_missing_columns", [])
    high_missing_non_empty = [c for c in high_missing if c not in empty_cols]
    if high_missing_non_empty:
        deduction = min(20.0, len(high_missing_non_empty) * 5.0)
        consistency_deductions += deduction
        issues.append(f"{len(high_missing_non_empty)} column(s) have >30% missing values.")
        suggestions.append("Investigate high-missing columns for imputation or omission.")

    if duplicate_count > 0:
        dup_pct = profiling_results.get("duplicate_percentage", 0)
        issues.append(f"Found {duplicate_count:,} duplicate rows ({dup_pct}%).")
        suggestions.append("Verify if duplicate rows represent repeated transactions or ingestion glitches before deduplication.")

    if total_missing > 0:
        miss_pct = missing_results.get("overall_missing_percentage", 0)
        issues.append(f"{total_missing:,} total missing cells ({miss_pct}% of dataset).")
        suggestions.append("Apply appropriate numerical (median/mean) or categorical (mode/Unknown) imputation.")

    consistency_score = round(max(0.0, 100.0 - consistency_deductions), 2)

    # Weighted Overall Score
    overall_score = round(
        (QUALITY_WEIGHT_COMPLETENESS * completeness_score) +
        (QUALITY_WEIGHT_UNIQUENESS * uniqueness_score) +
        (QUALITY_WEIGHT_CONSISTENCY * consistency_score),
        1
    )

    # Determine Rating & Grade
    if overall_score >= 90.0:
        grade = "A"
        rating = "Excellent"
    elif overall_score >= 80.0:
        grade = "B"
        rating = "Good"
    elif overall_score >= 70.0:
        grade = "C"
        rating = "Fair"
    elif overall_score >= 50.0:
        grade = "D"
        rating = "Needs Attention"
    else:
        grade = "F"
        rating = "Poor"

    if not issues:
        issues.append("No significant data hygiene defects identified.")
        suggestions.append("Dataset is clean and well-structured for immediate exploratory analysis or modeling.")

    return {
        "overall_score": overall_score,
        "grade": grade,
        "rating": rating,
        "weights": {
            "completeness": QUALITY_WEIGHT_COMPLETENESS,
            "uniqueness": QUALITY_WEIGHT_UNIQUENESS,
            "consistency": QUALITY_WEIGHT_CONSISTENCY
        },
        "component_scores": {
            "completeness": completeness_score,
            "uniqueness": uniqueness_score,
            "consistency": consistency_score
        },
        "issues": issues,
        "suggested_improvements": suggestions,
        "limitations": (
            "This score is a transparent heuristic composite measuring completeness, row uniqueness, and structural "
            "consistency. It reflects general data readiness rather than domain-specific semantic validity or ground truth accuracy."
        )
    }
