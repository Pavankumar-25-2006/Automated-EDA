"""
Categorical Frequency and Cardinality Analysis Layer.
Evaluates category balance, dominant classes, rare categories, and bivariate associations.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from utils.helpers import safe_divide


def analyze_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str]
) -> Dict[str, Any]:
    """
    Perform deep categorical distribution analysis across all categorical columns.
    """
    total_rows = len(df)
    results = {}
    summary_table = []
    imbalanced_columns = []
    high_cardinality_columns = []

    for col in categorical_columns:
        if col not in df.columns:
            continue

        series = df[col].dropna().astype(str)
        valid_count = len(series)

        if valid_count == 0:
            continue

        val_counts = series.value_counts()
        unique_count = len(val_counts)
        uniqueness_ratio = safe_divide(unique_count, valid_count)

        # Mode and dominance
        mode_val = str(val_counts.index[0])
        mode_freq = int(val_counts.iloc[0])
        mode_pct = safe_divide(mode_freq, valid_count) * 100.0

        # Dominant category check (> 70%)
        is_imbalanced = mode_pct >= 70.0
        if is_imbalanced:
            imbalanced_columns.append({
                "column": col,
                "dominant_category": mode_val,
                "percentage": round(mode_pct, 2)
            })

        # Rare categories (< 1% of non-null)
        rare_cats = val_counts[val_counts / valid_count < 0.01].index.tolist()
        rare_cats_count = len(rare_cats)

        # High cardinality (> 30 categories)
        is_high_card = unique_count > 30
        if is_high_card:
            high_cardinality_columns.append(col)

        # Top 5 categories
        top_5 = [
            {"category": str(cat), "count": int(cnt), "percentage": round(safe_divide(cnt, valid_count) * 100.0, 2)}
            for cat, cnt in val_counts.head(5).items()
        ]

        col_result = {
            "column_name": col,
            "valid_count": valid_count,
            "unique_categories": unique_count,
            "uniqueness_ratio": round(uniqueness_ratio, 4),
            "mode_category": mode_val,
            "mode_frequency": mode_freq,
            "mode_percentage": round(mode_pct, 2),
            "is_imbalanced": is_imbalanced,
            "rare_categories_count": rare_cats_count,
            "is_high_cardinality": is_high_card,
            "top_categories": top_5
        }

        results[col] = col_result
        summary_table.append(col_result)

    summary_table.sort(key=lambda x: x["unique_categories"])

    return {
        "column_details": results,
        "summary_table": summary_table,
        "imbalanced_columns": imbalanced_columns,
        "high_cardinality_columns": high_cardinality_columns
    }
