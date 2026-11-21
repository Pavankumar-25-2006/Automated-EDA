"""
Correlation and Association Analysis Layer.
Computes Pearson (linear) and Spearman (rank-order) correlation matrices,
extracts significant associations, and isolates top correlated feature pairs.
Never conflates correlation with causation.
"""
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from config import (
    CORRELATION_STRONG_THRESHOLD,
    CORRELATION_MODERATE_THRESHOLD,
    CORRELATION_MAX_VARS_HEATMAP,
    TOP_CORRELATED_PAIRS_COUNT
)
from utils.helpers import to_serializable


def compute_correlations(
    df: pd.DataFrame,
    numerical_columns: List[str]
) -> Dict[str, Any]:
    """
    Compute Pearson and Spearman correlation matrices and highlight top relationships.
    """
    # Filter to eligible numerical columns with at least 2 non-null values and non-zero variance
    valid_cols = []
    for col in numerical_columns:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) >= 3 and s.nunique() > 1:
                valid_cols.append(col)

    if len(valid_cols) < 2:
        return {
            "applied": False,
            "reason": "At least 2 non-constant numerical columns are required for correlation analysis.",
            "pearson_matrix": None,
            "spearman_matrix": None,
            "top_pairs": [],
            "strong_positive_pairs": [],
            "strong_negative_pairs": [],
            "moderate_pairs": [],
            "visualized_columns": []
        }

    sub_df = df[valid_cols].astype(float)

    # Compute matrices
    pearson_corr = sub_df.corr(method="pearson")
    spearman_corr = sub_df.corr(method="spearman")

    # Extract deduplicated pairs (i < j)
    pairs = []
    for i in range(len(valid_cols)):
        for j in range(i + 1, len(valid_cols)):
            col1 = valid_cols[i]
            col2 = valid_cols[j]
            p_val = pearson_corr.loc[col1, col2]
            s_val = spearman_corr.loc[col1, col2]

            if pd.isna(p_val):
                p_val = 0.0
            if pd.isna(s_val):
                s_val = 0.0

            abs_p = abs(p_val)
            if abs_p >= CORRELATION_STRONG_THRESHOLD:
                strength = "Strong"
            elif abs_p >= CORRELATION_MODERATE_THRESHOLD:
                strength = "Moderate"
            else:
                strength = "Weak"

            pairs.append({
                "feature_a": col1,
                "feature_b": col2,
                "pearson_corr": round(float(p_val), 4),
                "spearman_corr": round(float(s_val), 4),
                "abs_pearson": round(float(abs_p), 4),
                "strength": strength,
                "direction": "Positive" if p_val >= 0 else "Negative"
            })

    # Sort pairs by absolute Pearson correlation descending
    pairs.sort(key=lambda x: x["abs_pearson"], reverse=True)

    strong_pos = [p for p in pairs if p["pearson_corr"] >= CORRELATION_STRONG_THRESHOLD]
    strong_neg = [p for p in pairs if p["pearson_corr"] <= -CORRELATION_STRONG_THRESHOLD]
    moderate = [
        p for p in pairs
        if CORRELATION_MODERATE_THRESHOLD <= abs(p["pearson_corr"]) < CORRELATION_STRONG_THRESHOLD
    ]

    # For heatmap display, limit to top variables if dataset is too wide
    visualized_cols = valid_cols
    is_limited = False
    if len(valid_cols) > CORRELATION_MAX_VARS_HEATMAP:
        is_limited = True
        # Select features involved in the strongest correlations or highest variance
        top_features = set()
        for p in pairs[:CORRELATION_MAX_VARS_HEATMAP]:
            top_features.add(p["feature_a"])
            top_features.add(p["feature_b"])
            if len(top_features) >= CORRELATION_MAX_VARS_HEATMAP:
                break
        visualized_cols = list(top_features)[:CORRELATION_MAX_VARS_HEATMAP]

    return {
        "applied": True,
        "valid_columns": valid_cols,
        "visualized_columns": visualized_cols,
        "is_heatmap_limited": is_limited,
        "pearson_matrix": pearson_corr.loc[visualized_cols, visualized_cols].to_dict(),
        "spearman_matrix": spearman_corr.loc[visualized_cols, visualized_cols].to_dict(),
        "full_pearson_dict": pearson_corr.to_dict(),
        "top_pairs": pairs[:TOP_CORRELATED_PAIRS_COUNT],
        "strong_positive_pairs": strong_pos,
        "strong_negative_pairs": strong_neg,
        "moderate_pairs": moderate,
        "causation_disclaimer": (
            "Statistical Note: High correlation indicates an empirical linear or monotonic association. "
            "It does not prove or imply a cause-and-effect relationship."
        )
    }
