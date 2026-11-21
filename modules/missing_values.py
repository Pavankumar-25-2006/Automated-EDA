"""
Missing-Value Analysis Layer.
Analyzes missing data counts, percentages, empty columns, affected rows,
and generates non-destructive, conditional preprocessing guidance.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from utils.helpers import safe_divide


def analyze_missing_values(df: pd.DataFrame, type_groups: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Perform comprehensive missing-value audit on DataFrame.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    missing_per_col = df.isna().sum()
    total_missing = int(missing_per_col.sum())
    overall_missing_pct = safe_divide(total_missing, total_cells) * 100.0

    # Rows with any missing value
    rows_with_missing = int(df.isna().any(axis=1).sum())
    rows_with_missing_pct = safe_divide(rows_with_missing, total_rows) * 100.0

    summary_table = []
    columns_with_missing = []
    completely_empty_columns = []
    high_missing_columns = []  # > 30%

    for col in df.columns:
        col_missing = int(missing_per_col[col])
        missing_pct = safe_divide(col_missing, total_rows) * 100.0

        if col_missing > 0:
            columns_with_missing.append(col)

        if missing_pct >= 99.99:
            completely_empty_columns.append(col)
        elif missing_pct >= 30.0:
            high_missing_columns.append(col)

        # Inferred semantic type
        col_type = "numerical" if col in type_groups.get("numerical", []) else (
            "categorical" if col in type_groups.get("categorical", []) else "other"
        )

        summary_table.append({
            "column_name": col,
            "missing_count": col_missing,
            "missing_percentage": round(missing_pct, 2),
            "valid_count": total_rows - col_missing,
            "valid_percentage": round(100.0 - missing_pct, 2),
            "semantic_type": col_type
        })

    # Sort summary table by missing percentage descending
    summary_table.sort(key=lambda x: x["missing_percentage"], reverse=True)

    # Generate conditional recommendations
    recommendations = []
    if total_missing == 0:
        recommendations.append({
            "type": "positive",
            "message": "Dataset has complete integrity with 0 missing values across all columns."
        })
    else:
        if completely_empty_columns:
            recommendations.append({
                "type": "warning",
                "message": (
                    f"Columns {completely_empty_columns} are 100% missing. "
                    "These provide zero informational variance and should be removed before modeling."
                )
            })

        for item in summary_table:
            col = item["column_name"]
            pct = item["missing_percentage"]
            col_type = item["semantic_type"]

            if 0 < pct < 5.0:
                rec_text = "median imputation" if col_type == "numerical" else "mode imputation or explicit 'Unknown' category"
                recommendations.append({
                    "type": "info",
                    "message": (
                        f"Column '{col}' has low missingness ({pct}%). "
                        f"Consider simple {rec_text} or dropping missing rows if sample size permits."
                    )
                })
            elif 5.0 <= pct < 30.0:
                rec_text = "iterative/KNN imputation or distribution-preserving sampling" if col_type == "numerical" else "an explicit 'Missing' category"
                recommendations.append({
                    "type": "warning",
                    "message": (
                        f"Column '{col}' has moderate missingness ({pct}%). "
                        f"Evaluate whether data is Missing Completely at Random (MCAR). Consider {rec_text}."
                    )
                })
            elif 30.0 <= pct < 100.0:
                recommendations.append({
                    "type": "warning",
                    "message": (
                        f"Column '{col}' has high missingness ({pct}%). "
                        "Investigate whether this feature carries critical domain signal. "
                        "If missingness itself is informative, create a missing indicator flag; otherwise consider feature omission."
                    )
                })

    return {
        "total_missing_cells": total_missing,
        "total_cells": total_cells,
        "overall_missing_percentage": round(overall_missing_pct, 2),
        "rows_with_missing": rows_with_missing,
        "rows_with_missing_percentage": round(rows_with_missing_pct, 2),
        "columns_with_missing_count": len(columns_with_missing),
        "columns_with_missing": columns_with_missing,
        "completely_empty_columns": completely_empty_columns,
        "high_missing_columns": high_missing_columns,
        "summary_table": summary_table,
        "recommendations": recommendations
    }
