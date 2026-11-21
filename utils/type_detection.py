"""
Intelligent Semantic Type Detection Module.
Classifies dataset columns conservatively into:
- numerical
- categorical
- boolean
- datetime
- text
- identifier
- unsupported
"""
import re
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from config import (
    MAX_CARDINALITY_FOR_CATEGORICAL,
    MAX_CATEGORICAL_UNIQUE_RATIO,
    MAX_TEXT_AVG_WORD_COUNT,
    MAX_SAMPLE_ROWS_FOR_TYPE_INFERENCE,
)
from utils.helpers import safe_divide


IDENTIFIER_KEYWORDS = {
    "id", "uuid", "guid", "ssn", "vin", "key", "code", "zip", "postal",
    "phone", "mobile", "isbn", "account", "ticket", "serial", "hash", "token"
}

BOOLEAN_TRUE_VALUES = {"true", "1", "1.0", "yes", "y", "t"}
BOOLEAN_FALSE_VALUES = {"false", "0", "0.0", "no", "n", "f"}


def is_identifier_column(series: pd.Series, col_name: str, total_rows: int) -> bool:
    """Determine if a column is an identifier rather than continuous data."""
    clean_name = re.sub(r"[^a-zA-Z0-9]", "", col_name.lower())
    non_null_series = series.dropna()
    valid_count = len(non_null_series)
    if valid_count == 0:
        return False

    unique_count = non_null_series.nunique()
    uniqueness_ratio = safe_divide(unique_count, valid_count)

    # Keyword match in column name
    has_keyword = any(
        kw == clean_name or clean_name.endswith(f"_{kw}") or clean_name.endswith(kw) or clean_name.startswith(f"{kw}_")
        for kw in IDENTIFIER_KEYWORDS
    )

    # 100% unique or high uniqueness with identifier keywords
    if has_keyword and uniqueness_ratio >= 0.75:
        return True

    # Do not treat multi-word free-form text as an identifier
    if is_text_series(series):
        return False

    # High cardinality string with row-level uniqueness in reasonably sized datasets
    if total_rows >= 20 and uniqueness_ratio == 1.0:
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            # Check maximum string length: identifiers are typically short codes (<= 50 chars)
            max_len = non_null_series.astype(str).str.len().max()
            if max_len <= 50:
                return True
        if pd.api.types.is_integer_dtype(series) and has_keyword:
            return True

    # Check for UUID / hash regex
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        sample = non_null_series.head(20).astype(str)
        uuid_pattern = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
        if all(uuid_pattern.match(val.strip()) for val in sample if val.strip()):
            return True

    return False


def is_boolean_series(series: pd.Series) -> bool:
    """Detect if a series represents boolean values."""
    if pd.api.types.is_bool_dtype(series):
        return True

    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    unique_vals = non_null.unique()
    if len(unique_vals) not in (1, 2):
        return False

    # Check string representation
    str_vals = {str(v).strip().lower() for v in unique_vals}
    all_known_bool = (str_vals.issubset(BOOLEAN_TRUE_VALUES | BOOLEAN_FALSE_VALUES))
    has_matching_pair = any(
        str_vals.issubset(pair) for pair in [
            {"true", "false"},
            {"1", "0"},
            {"1.0", "0.0"},
            {"yes", "no"},
            {"y", "n"},
            {"t", "f"}
        ]
    )
    return all_known_bool and has_matching_pair


def is_datetime_series(series: pd.Series) -> bool:
    """Detect if an object/string series represents dates or timestamps."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # Numeric series should not be automatically inferred as datetime
    if pd.api.types.is_numeric_dtype(series):
        return False

    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    # Sample to avoid heavy regex / parsing overhead on huge datasets
    sample = non_null.head(min(len(non_null), MAX_SAMPLE_ROWS_FOR_TYPE_INFERENCE)).astype(str).str.strip()
    
    # Must look like a date/time (contains delimiters like -, /, :)
    date_delim_pattern = re.compile(r"[\-\/\:]")
    sample_matching_pattern = sum(bool(date_delim_pattern.search(val)) for val in sample)
    if sample_matching_pattern / len(sample) < 0.8:
        return False

    try:
        # Attempt conversion on sample
        parsed = pd.to_datetime(sample, errors="coerce")
        valid_ratio = parsed.notna().sum() / len(sample)
        return valid_ratio >= 0.85
    except Exception:
        return False


def is_text_series(series: pd.Series) -> bool:
    """Detect free-form unstructured text (e.g. reviews, comments, descriptions)."""
    if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
        return False

    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    sample = non_null.head(100).astype(str)
    # Average word count per string
    word_counts = [len(s.split()) for s in sample]
    avg_words = np.mean(word_counts) if word_counts else 0

    return avg_words >= MAX_TEXT_AVG_WORD_COUNT


def detect_column_type(series: pd.Series, col_name: str, total_rows: int) -> Dict[str, Any]:
    """
    Conservatively infer the semantic type of a single column.
    Returns metadata dictionary including semantic_type and flags.
    """
    total_count = len(series)
    non_null = series.dropna()
    valid_count = len(non_null)
    unique_count = non_null.nunique()
    uniqueness_ratio = safe_divide(unique_count, valid_count)
    is_constant = (unique_count == 1) if valid_count > 0 else False
    is_all_null = (valid_count == 0)

    # 1. Check for all null or empty
    if is_all_null:
        return {
            "semantic_type": "unsupported",
            "original_dtype": str(series.dtype),
            "is_constant": False,
            "is_all_null": True,
            "unique_count": 0,
            "uniqueness_ratio": 0.0,
            "reason": "All values are missing"
        }

    # 2. Check for boolean
    if is_boolean_series(series):
        return {
            "semantic_type": "boolean",
            "original_dtype": str(series.dtype),
            "is_constant": is_constant,
            "is_all_null": False,
            "unique_count": unique_count,
            "uniqueness_ratio": uniqueness_ratio,
            "reason": "Binary boolean representation"
        }

    # 3. Check for identifier
    if is_identifier_column(series, col_name, total_rows):
        return {
            "semantic_type": "identifier",
            "original_dtype": str(series.dtype),
            "is_constant": is_constant,
            "is_all_null": False,
            "unique_count": unique_count,
            "uniqueness_ratio": uniqueness_ratio,
            "reason": "Identifier or key pattern detected"
        }

    # 4. Check for datetime
    if is_datetime_series(series):
        return {
            "semantic_type": "datetime",
            "original_dtype": str(series.dtype),
            "is_constant": is_constant,
            "is_all_null": False,
            "unique_count": unique_count,
            "uniqueness_ratio": uniqueness_ratio,
            "reason": "Timestamp/Date format detected"
        }

    # 5. Check for free-form text
    if is_text_series(series):
        return {
            "semantic_type": "text",
            "original_dtype": str(series.dtype),
            "is_constant": is_constant,
            "is_all_null": False,
            "unique_count": unique_count,
            "uniqueness_ratio": uniqueness_ratio,
            "reason": "Multi-word free-form text"
        }

    # 6. Check for numerical
    if pd.api.types.is_numeric_dtype(series):
        # Even if numeric, low cardinality integers may be treated as discrete/categorical for EDA
        if unique_count <= 10 and total_rows >= 30 and not pd.api.types.is_float_dtype(series):
            return {
                "semantic_type": "categorical",
                "original_dtype": str(series.dtype),
                "is_constant": is_constant,
                "is_all_null": False,
                "unique_count": unique_count,
                "uniqueness_ratio": uniqueness_ratio,
                "reason": "Low-cardinality discrete numeric codes"
            }
        return {
            "semantic_type": "numerical",
            "original_dtype": str(series.dtype),
            "is_constant": is_constant,
            "is_all_null": False,
            "unique_count": unique_count,
            "uniqueness_ratio": uniqueness_ratio,
            "reason": "Continuous or discrete numerical measurements"
        }

    # 7. Check for categorical
    if (
        isinstance(series.dtype, pd.CategoricalDtype)
        or unique_count <= MAX_CARDINALITY_FOR_CATEGORICAL
        or uniqueness_ratio <= MAX_CATEGORICAL_UNIQUE_RATIO
    ):
        return {
            "semantic_type": "categorical",
            "original_dtype": str(series.dtype),
            "is_constant": is_constant,
            "is_all_null": False,
            "unique_count": unique_count,
            "uniqueness_ratio": uniqueness_ratio,
            "reason": "Discrete categorical groups"
        }

    # Default fallback
    return {
        "semantic_type": "categorical" if unique_count < 100 else "unsupported",
        "original_dtype": str(series.dtype),
        "is_constant": is_constant,
        "is_all_null": False,
        "unique_count": unique_count,
        "uniqueness_ratio": uniqueness_ratio,
        "reason": "High-cardinality string or complex representation"
    }


def classify_dataset_columns(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Classify all columns in a DataFrame into semantic types with metadata."""
    total_rows = len(df)
    results = {}
    for col in df.columns:
        results[col] = detect_column_type(df[col], str(col), total_rows)
    return results


def get_column_groups_by_type(type_classifications: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Group column names by their inferred semantic types."""
    groups = {
        "numerical": [],
        "categorical": [],
        "boolean": [],
        "datetime": [],
        "text": [],
        "identifier": [],
        "unsupported": []
    }
    for col, meta in type_classifications.items():
        stype = meta.get("semantic_type", "unsupported")
        if stype in groups:
            groups[stype].append(col)
        else:
            groups["unsupported"].append(col)
    return groups
