"""
Helper functions for formatting, numerical safeguards, and serialization.
"""
from typing import Any, Union
import numpy as np
import pandas as pd


def format_number(val: Any, decimals: int = 2) -> str:
    """Format a numerical value safely with thousands separator or handle nulls."""
    if val is None or pd.isna(val):
        return "N/A"
    try:
        fval = float(val)
        if np.isinf(fval):
            return "Inf" if fval > 0 else "-Inf"
        if np.isnan(fval):
            return "NaN"
        if fval.is_integer():
            return f"{int(fval):,}"
        return f"{fval:,.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def format_percentage(val: Any, decimals: int = 1) -> str:
    """Format a ratio/percentage safely."""
    if val is None or pd.isna(val):
        return "N/A"
    try:
        fval = float(val)
        if np.isnan(fval) or np.isinf(fval):
            return "N/A"
        return f"{fval:.{decimals}f}%"
    except (ValueError, TypeError):
        return str(val)


def format_bytes(num_bytes: Union[int, float]) -> str:
    """Format memory consumption into human-readable B, KB, MB, GB."""
    if num_bytes is None or pd.isna(num_bytes):
        return "0 B"
    try:
        b = float(num_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if abs(b) < 1024.0:
                return f"{b:3.2f} {unit}"
            b /= 1024.0
        return f"{b:.2f} PB"
    except Exception:
        return "0 B"


def safe_divide(numerator: Union[int, float], denominator: Union[int, float], default: float = 0.0) -> float:
    """Safely divide two numbers handling division by zero or invalid types."""
    try:
        num = float(numerator)
        den = float(denominator)
        if den == 0 or np.isnan(den) or np.isnan(num):
            return default
        return num / den
    except (ZeroDivisionError, ValueError, TypeError):
        return default


def truncate_string(val: Any, max_len: int = 40) -> str:
    """Truncate string with ellipsis if it exceeds max_len."""
    s = str(val) if val is not None else ""
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def to_serializable(val: Any) -> Any:
    """Recursively convert NumPy, Pandas, or special objects to standard Python types."""
    if val is None:
        return None
    if isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32, np.float16)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return str(val)
    if isinstance(val, (pd.Series, np.ndarray)):
        return [to_serializable(x) for x in val]
    if isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    if isinstance(val, list):
        return [to_serializable(x) for x in val]
    return val
