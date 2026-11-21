"""
Data Ingestion and Validation Layer.
Handles CSV file parsing, encoding detection, delimiter sniffing, validation, and error reporting.
"""
import io
import csv
from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path
import pandas as pd

from config import (
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    ALLOWED_EXTENSIONS,
    SUPPORTED_ENCODINGS,
    COMMON_DELIMITERS,
)
from utils.helpers import format_bytes


class DataLoaderResult:
    """Structured response container for data ingestion."""
    def __init__(
        self,
        success: bool,
        df: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        self.success = success
        self.df = df
        self.metadata = metadata or {}
        self.error_message = error_message


def validate_file_metadata(file_name: str, file_size_bytes: int) -> Tuple[bool, Optional[str]]:
    """Validate file extension and size constraints."""
    ext = Path(file_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Only CSV files are supported."

    if file_size_bytes == 0:
        return False, "The uploaded file is empty (0 bytes)."

    if file_size_bytes > MAX_UPLOAD_SIZE_BYTES:
        return False, (
            f"File size ({format_bytes(file_size_bytes)}) exceeds the maximum allowed limit "
            f"of {MAX_UPLOAD_SIZE_MB} MB."
        )

    return True, None


def detect_encoding(raw_bytes: bytes) -> str:
    """Detect text encoding using supported candidates."""
    for enc in SUPPORTED_ENCODINGS:
        try:
            raw_bytes.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    # Default fallback
    return "latin-1"


def detect_delimiter(sample_text: str) -> str:
    """Sniff CSV delimiter using Python's csv.Sniffer or frequency heuristic."""
    if not sample_text.strip():
        return ","
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text[:4096], delimiters="".join(COMMON_DELIMITERS))
        if dialect.delimiter in COMMON_DELIMITERS:
            return dialect.delimiter
    except Exception:
        pass

    # Heuristic fallback: count delimiters on the first line
    first_line = sample_text.splitlines()[0] if sample_text.splitlines() else ""
    counts = {d: first_line.count(d) for d in COMMON_DELIMITERS}
    best_del = max(counts, key=counts.get)
    return best_del if counts[best_del] > 0 else ","


def deduplicate_column_names(columns: list) -> list:
    """Ensure column names are distinct and meaningful strings."""
    seen = {}
    new_cols = []
    for i, col in enumerate(columns):
        col_str = str(col).strip()
        if not col_str or col_str.lower().startswith("unnamed:"):
            col_str = f"column_{i + 1}"
        
        if col_str in seen:
            seen[col_str] += 1
            new_cols.append(f"{col_str}_{seen[col_str]}")
        else:
            seen[col_str] = 0
            new_cols.append(col_str)
    return new_cols


def load_csv_data(
    file_source: Union[io.BytesIO, bytes, str, Path],
    file_name: str = "dataset.csv",
    file_size_bytes: Optional[int] = None
) -> DataLoaderResult:
    """
    Load and validate a CSV dataset from a file path, byte stream, or raw bytes.
    Does not persist the dataset to disk.
    """
    try:
        # Extract raw bytes
        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            if not path.exists():
                return DataLoaderResult(False, error_message=f"File not found: {path}")
            raw_bytes = path.read_bytes()
            file_name = path.name
            file_size_bytes = len(raw_bytes)
        elif isinstance(file_source, io.BytesIO):
            file_source.seek(0)
            raw_bytes = file_source.read()
            if file_size_bytes is None:
                file_size_bytes = len(raw_bytes)
        elif isinstance(file_source, bytes):
            raw_bytes = file_source
            if file_size_bytes is None:
                file_size_bytes = len(raw_bytes)
        elif hasattr(file_source, "read"):
            # Streamlit UploadedFile or file-like object
            file_source.seek(0)
            raw_bytes = file_source.read()
            if file_size_bytes is None:
                file_size_bytes = len(raw_bytes)
        else:
            return DataLoaderResult(False, error_message="Unsupported file source type.")

        # Validate metadata
        valid, err = validate_file_metadata(file_name, file_size_bytes)
        if not valid:
            return DataLoaderResult(False, error_message=err)

        # Detect encoding
        encoding = detect_encoding(raw_bytes)
        try:
            decoded_text = raw_bytes.decode(encoding)
        except Exception as e:
            return DataLoaderResult(
                False,
                error_message=f"Failed to decode file with encoding '{encoding}': {str(e)}"
            )

        # Check for empty content
        if not decoded_text.strip():
            return DataLoaderResult(False, error_message="The CSV file contains no readable text content.")

        # Sniff delimiter
        delimiter = detect_delimiter(decoded_text)

        # Parse CSV with Pandas
        string_buffer = io.StringIO(decoded_text)
        try:
            df = pd.read_csv(
                string_buffer,
                sep=delimiter,
                encoding=encoding,
                on_bad_lines="error",
                low_memory=False
            )
        except Exception:
            # Fallback to python engine if C engine fails on complex delimiters/quotes
            string_buffer.seek(0)
            try:
                df = pd.read_csv(
                    string_buffer,
                    sep=delimiter,
                    encoding=encoding,
                    engine="python",
                    on_bad_lines="skip"
                )
            except Exception as parse_err:
                return DataLoaderResult(
                    False,
                    error_message=f"Failed to parse CSV: {str(parse_err)}"
                )

        # Validate rows and columns
        if df.empty:
            return DataLoaderResult(
                False,
                error_message="The dataset is empty. It contains 0 rows or no data records."
            )

        if len(df.columns) == 0:
            return DataLoaderResult(
                False,
                error_message="The dataset has no identifiable columns."
            )

        # Sanitize and deduplicate columns
        df.columns = deduplicate_column_names(df.columns.tolist())

        # Metadata
        metadata = {
            "file_name": file_name,
            "file_size_bytes": file_size_bytes,
            "file_size_formatted": format_bytes(file_size_bytes),
            "encoding": encoding,
            "delimiter": delimiter,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist()
        }

        return DataLoaderResult(True, df=df, metadata=metadata)

    except Exception as exc:
        return DataLoaderResult(
            False,
            error_message=f"Unexpected error while loading dataset: {str(exc)}"
        )
