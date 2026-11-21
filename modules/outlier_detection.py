"""
Outlier and Anomaly Detection Layer.
Provides statistical outlier detection via the Interquartile Range (IQR) method
and optional machine-learning-based multivariate anomaly detection via Isolation Forest.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from config import IQR_MULTIPLIER, ANOMALY_CONTAMINATION, ANOMALY_RANDOM_STATE
from utils.helpers import safe_divide


def detect_column_iqr_outliers(series: pd.Series, col_name: str, multiplier: float = IQR_MULTIPLIER) -> Dict[str, Any]:
    """
    Detect univariate outliers using Tukey's Interquartile Range (IQR) method.
    """
    clean_series = series.dropna().astype(float)
    valid_count = len(clean_series)

    if valid_count < 4:
        return {
            "column_name": col_name,
            "valid_count": valid_count,
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_bound": None,
            "upper_bound": None,
            "lower_outliers_count": 0,
            "upper_outliers_count": 0,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "min_val": float(clean_series.min()) if valid_count > 0 else None,
            "max_val": float(clean_series.max()) if valid_count > 0 else None,
            "status": "Insufficient sample size (N < 4) for IQR detection"
        }

    q1 = float(np.percentile(clean_series, 25))
    q3 = float(np.percentile(clean_series, 75))
    iqr = float(q3 - q1)

    if iqr == 0.0:
        # Zero IQR: heavily concentrated distribution or constant
        return {
            "column_name": col_name,
            "valid_count": valid_count,
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": 0.0,
            "lower_bound": round(q1, 4),
            "upper_bound": round(q3, 4),
            "lower_outliers_count": 0,
            "upper_outliers_count": 0,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "min_val": float(clean_series.min()),
            "max_val": float(clean_series.max()),
            "status": "Zero IQR: Distribution is concentrated at a single value"
        }

    lower_bound = float(q1 - multiplier * iqr)
    upper_bound = float(q3 + multiplier * iqr)

    lower_mask = clean_series < lower_bound
    upper_mask = clean_series > upper_bound

    lower_count = int(lower_mask.sum())
    upper_count = int(upper_mask.sum())
    total_outliers = lower_count + upper_count
    outlier_pct = safe_divide(total_outliers, valid_count) * 100.0

    return {
        "column_name": col_name,
        "valid_count": valid_count,
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "lower_outliers_count": lower_count,
        "upper_outliers_count": upper_count,
        "outlier_count": total_outliers,
        "outlier_percentage": round(outlier_pct, 2),
        "min_val": round(float(clean_series.min()), 4),
        "max_val": round(float(clean_series.max()), 4),
        "status": "Outliers detected" if total_outliers > 0 else "No IQR outliers"
    }


def detect_isolation_forest_anomalies(
    df: pd.DataFrame,
    numerical_columns: List[str],
    contamination: float = ANOMALY_CONTAMINATION
) -> Dict[str, Any]:
    """
    Advanced multivariate anomaly detection using Scikit-learn's Isolation Forest.
    Discloses methodology as an unsupervised ML algorithm.
    """
    if len(numerical_columns) < 2 or len(df) < 20:
        return {
            "applied": False,
            "reason": "Isolation Forest requires at least 2 numerical columns and >= 20 observations."
        }

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.impute import SimpleImputer

        num_df = df[numerical_columns].copy()
        # Drop constant columns
        valid_cols = [c for c in numerical_columns if num_df[c].nunique(dropna=True) > 1]
        if len(valid_cols) < 2:
            return {
                "applied": False,
                "reason": "Fewer than 2 non-constant numerical columns available."
            }

        # Impute missing values for the model with median
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(num_df[valid_cols])

        iso = IsolationForest(
            contamination=contamination,
            random_state=ANOMALY_RANDOM_STATE,
            n_estimators=100
        )
        preds = iso.fit_predict(X)  # -1 for anomaly, 1 for inlier
        scores = iso.decision_function(X)

        anomaly_indices = np.where(preds == -1)[0].tolist()
        anomaly_count = len(anomaly_indices)
        anomaly_percentage = safe_divide(anomaly_count, len(df)) * 100.0

        return {
            "applied": True,
            "model": "Isolation Forest (Scikit-Learn)",
            "contamination_parameter": contamination,
            "evaluated_features": valid_cols,
            "anomaly_count": anomaly_count,
            "anomaly_percentage": round(anomaly_percentage, 2),
            "sample_anomaly_indices": anomaly_indices[:10],
            "methodology_note": (
                "Isolation Forest isolates multivariate anomalies using random decision splits. "
                "Points with shorter path lengths are flagged as potential multivariate anomalies."
            )
        }
    except Exception as e:
        return {
            "applied": False,
            "reason": f"Anomaly detection error: {str(e)}"
        }


def analyze_outliers(
    df: pd.DataFrame,
    numerical_columns: List[str]
) -> Dict[str, Any]:
    """
    Consolidated outlier analysis combining IQR and Isolation Forest.
    """
    iqr_results = {}
    summary_table = []
    columns_with_outliers = []
    total_outlier_instances = 0

    for col in numerical_columns:
        if col in df.columns:
            res = detect_column_iqr_outliers(df[col], col)
            iqr_results[col] = res
            summary_table.append(res)
            if res["outlier_count"] > 0:
                columns_with_outliers.append(col)
                total_outlier_instances += res["outlier_count"]

    # Sort by outlier percentage descending
    summary_table.sort(key=lambda x: x["outlier_percentage"], reverse=True)

    # Optional ML Anomaly Detection
    ml_anomalies = detect_isolation_forest_anomalies(df, numerical_columns)

    return {
        "iqr_results": iqr_results,
        "summary_table": summary_table,
        "columns_with_outliers": columns_with_outliers,
        "columns_with_outliers_count": len(columns_with_outliers),
        "total_outlier_instances": total_outlier_instances,
        "ml_anomalies": ml_anomalies
    }
