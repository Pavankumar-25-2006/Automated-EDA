"""
Dataset Profiling Layer.
Profiles dataset dimensions, memory consumption, column types, uniqueness, duplicates,
constant columns, and high cardinality columns.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from utils.type_detection import classify_dataset_columns, get_column_groups_by_type
from utils.helpers import format_bytes, safe_divide


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform structural profiling of the input DataFrame.
    Returns a comprehensive profiling dictionary.
    """
    total_rows = len(df)
    total_cols = len(df.columns)

    # Memory usage
    try:
        memory_bytes = int(df.memory_usage(deep=True).sum())
    except Exception:
        memory_bytes = int(df.memory_usage().sum())
    memory_formatted = format_bytes(memory_bytes)

    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())
    duplicate_percentage = safe_divide(duplicate_count, total_rows) * 100.0

    # Semantic column classifications
    type_classifications = classify_dataset_columns(df)
    type_groups = get_column_groups_by_type(type_classifications)

    # Detailed column-by-column profiling
    columns_profile = []
    constant_columns = []
    high_cardinality_columns = []
    all_null_columns = []

    for col in df.columns:
        series = df[col]
        meta = type_classifications[col]
        missing_count = int(series.isna().sum())
        missing_percentage = safe_divide(missing_count, total_rows) * 100.0
        unique_count = int(series.nunique(dropna=True))
        valid_count = total_rows - missing_count
        uniqueness_ratio = safe_divide(unique_count, valid_count)

        is_constant = meta["is_constant"]
        if is_constant:
            constant_columns.append(col)

        is_all_null = meta["is_all_null"]
        if is_all_null:
            all_null_columns.append(col)

        # High cardinality in non-numerical / non-identifier columns
        is_high_cardinality = (
            meta["semantic_type"] == "categorical"
            and (unique_count > 50 or uniqueness_ratio > 0.5)
        )
        if is_high_cardinality:
            high_cardinality_columns.append(col)

        # Sample values (first 3 distinct non-null)
        sample_vals = series.dropna().unique()[:3].tolist()
        sample_str = ", ".join(str(v) for v in sample_vals)

        columns_profile.append({
            "column_name": col,
            "original_dtype": meta["original_dtype"],
            "semantic_type": meta["semantic_type"],
            "missing_count": missing_count,
            "missing_percentage": round(missing_percentage, 2),
            "unique_count": unique_count,
            "uniqueness_ratio": round(uniqueness_ratio, 4),
            "is_constant": is_constant,
            "is_all_null": is_all_null,
            "is_high_cardinality": is_high_cardinality,
            "type_reason": meta.get("reason", ""),
            "sample_values": sample_str
        })

    type_counts = {k: len(v) for k, v in type_groups.items()}

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "memory_bytes": memory_bytes,
        "memory_formatted": memory_formatted,
        "duplicate_count": duplicate_count,
        "duplicate_percentage": round(duplicate_percentage, 2),
        "columns_profile": columns_profile,
        "type_classifications": type_classifications,
        "type_groups": type_groups,
        "type_counts": type_counts,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "all_null_columns": all_null_columns,
    }
