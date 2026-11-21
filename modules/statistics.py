"""
Descriptive Statistical Analysis Layer.
Computes comprehensive parametric and non-parametric statistics for numerical and categorical variables.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats

from utils.helpers import safe_divide, to_serializable


def compute_numerical_statistics(series: pd.Series, col_name: str) -> Dict[str, Any]:
    """
    Compute descriptive statistics for a single numerical column.
    Safely handles constant columns, small samples, and edge cases.
    """
    clean_series = series.dropna().astype(float)
    valid_count = len(clean_series)

    if valid_count == 0:
        return {
            "column_name": col_name,
            "count": 0,
            "mean": None,
            "median": None,
            "mode": None,
            "std": None,
            "variance": None,
            "min": None,
            "max": None,
            "range": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "iqr": None,
            "skewness": None,
            "kurtosis": None,
            "cv": None,
            "is_constant": True,
            "notes": "No valid numerical values present."
        }

    c_mean = float(clean_series.mean())
    c_median = float(clean_series.median())
    
    # Mode
    mode_series = clean_series.mode()
    c_mode = float(mode_series.iloc[0]) if not mode_series.empty else None

    c_min = float(clean_series.min())
    c_max = float(clean_series.max())
    c_range = float(c_max - c_min)

    # Variance and std
    c_var = float(clean_series.var(ddof=1)) if valid_count > 1 else 0.0
    c_std = float(clean_series.std(ddof=1)) if valid_count > 1 else 0.0

    # Percentiles
    q25 = float(np.percentile(clean_series, 25))
    q50 = float(np.percentile(clean_series, 50))
    q75 = float(np.percentile(clean_series, 75))
    iqr = float(q75 - q25)

    # Skewness and Kurtosis
    if valid_count >= 3 and c_std > 0:
        c_skew = float(stats.skew(clean_series, bias=False))
        c_kurt = float(stats.kurtosis(clean_series, bias=False))  # Fisher kurtosis (normal == 0)
    else:
        c_skew = 0.0 if c_std == 0 else None
        c_kurt = 0.0 if c_std == 0 else None

    # Coefficient of variation (CV = std / abs(mean))
    if c_mean != 0 and c_std is not None:
        cv = safe_divide(c_std, abs(c_mean))
    else:
        cv = None

    is_constant = (c_min == c_max)

    return {
        "column_name": col_name,
        "count": valid_count,
        "mean": round(c_mean, 4),
        "median": round(c_median, 4),
        "mode": round(c_mode, 4) if c_mode is not None else None,
        "std": round(c_std, 4),
        "variance": round(c_var, 4),
        "min": round(c_min, 4),
        "max": round(c_max, 4),
        "range": round(c_range, 4),
        "q25": round(q25, 4),
        "q50": round(q50, 4),
        "q75": round(q75, 4),
        "iqr": round(iqr, 4),
        "skewness": round(c_skew, 4) if c_skew is not None else None,
        "kurtosis": round(c_kurt, 4) if c_kurt is not None else None,
        "cv": round(cv, 4) if cv is not None else None,
        "is_constant": is_constant,
        "notes": "Constant column" if is_constant else ("Small sample (N < 30)" if valid_count < 30 else "Normal")
    }


def compute_categorical_statistics(series: pd.Series, col_name: str, max_display: int = 10) -> Dict[str, Any]:
    """
    Compute category frequencies, mode, and distribution for a categorical column.
    """
    clean_series = series.dropna().astype(str)
    valid_count = len(clean_series)

    if valid_count == 0:
        return {
            "column_name": col_name,
            "count": 0,
            "unique_categories": 0,
            "mode": None,
            "mode_frequency": 0,
            "mode_percentage": 0.0,
            "top_categories": [],
            "frequencies": {},
            "percentages": {},
            "is_constant": True
        }

    val_counts = clean_series.value_counts()
    unique_categories = len(val_counts)
    mode_val = str(val_counts.index[0])
    mode_freq = int(val_counts.iloc[0])
    mode_pct = safe_divide(mode_freq, valid_count) * 100.0

    top_cats = []
    freq_dict = {}
    pct_dict = {}

    for cat, freq in val_counts.head(max_display).items():
        pct = safe_divide(freq, valid_count) * 100.0
        top_cats.append({
            "category": str(cat),
            "frequency": int(freq),
            "percentage": round(pct, 2)
        })
        freq_dict[str(cat)] = int(freq)
        pct_dict[str(cat)] = round(pct, 2)

    return {
        "column_name": col_name,
        "count": valid_count,
        "unique_categories": unique_categories,
        "mode": mode_val,
        "mode_frequency": mode_freq,
        "mode_percentage": round(mode_pct, 2),
        "top_categories": top_cats,
        "frequencies": freq_dict,
        "percentages": pct_dict,
        "is_constant": (unique_categories == 1)
    }


def compute_dataset_statistics(
    df: pd.DataFrame,
    type_groups: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics across all eligible columns in the dataset.
    """
    numerical_cols = type_groups.get("numerical", [])
    categorical_cols = type_groups.get("categorical", []) + type_groups.get("boolean", [])

    num_stats = {}
    for col in numerical_cols:
        if col in df.columns:
            num_stats[col] = compute_numerical_statistics(df[col], col)

    cat_stats = {}
    for col in categorical_cols:
        if col in df.columns:
            cat_stats[col] = compute_categorical_statistics(df[col], col)

    return {
        "numerical_statistics": num_stats,
        "categorical_statistics": cat_stats,
        "numerical_summary_table": list(num_stats.values()),
        "categorical_summary_table": [
            {
                "column_name": s["column_name"],
                "count": s["count"],
                "unique_categories": s["unique_categories"],
                "mode": s["mode"],
                "mode_frequency": s["mode_frequency"],
                "mode_percentage": s["mode_percentage"],
                "is_constant": s["is_constant"]
            }
            for s in cat_stats.values()
        ]
    }
