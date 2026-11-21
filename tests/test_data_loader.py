"""
Unit tests for Data Ingestion Layer (data_loader.py).
"""
import io
import pytest
import pandas as pd
from modules.data_loader import (
    load_csv_data,
    validate_file_metadata,
    detect_encoding,
    detect_delimiter,
    deduplicate_column_names,
)


def test_valid_csv_loading(tmp_path):
    csv_file = tmp_path / "valid.csv"
    csv_file.write_text("col_a,col_b\n1,2\n3,4", encoding="utf-8")
    
    result = load_csv_data(str(csv_file))
    assert result.success is True
    assert result.df is not None
    assert len(result.df) == 2
    assert list(result.df.columns) == ["col_a", "col_b"]
    assert result.metadata["row_count"] == 2
    assert result.metadata["column_count"] == 2


def test_empty_file_rejection(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("", encoding="utf-8")
    
    result = load_csv_data(str(csv_file))
    assert result.success is False
    assert "empty" in result.error_message.lower()


def test_invalid_extension(tmp_path):
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("a,b\n1,2", encoding="utf-8")
    
    result = load_csv_data(str(txt_file))
    assert result.success is False
    assert "extension" in result.error_message.lower()


def test_semicolon_delimiter_handling():
    data = b"col1;col2;col3\n10;20;30\n40;50;60"
    result = load_csv_data(data, file_name="semi.csv")
    assert result.success is True
    assert result.metadata["delimiter"] == ";"
    assert len(result.df) == 2
    assert len(result.df.columns) == 3


def test_tab_delimiter_handling():
    data = b"a\tb\tc\n1\t2\t3\n4\t5\t6"
    result = load_csv_data(data, file_name="tab.csv")
    assert result.success is True
    assert result.metadata["delimiter"] == "\t"
    assert len(result.df.columns) == 3


def test_duplicate_column_headers():
    cols = ["age", "salary", "age", "age", "unnamed: 4", ""]
    deduped = deduplicate_column_names(cols)
    assert len(deduped) == len(cols)
    assert len(set(deduped)) == len(cols)
    assert deduped[0] == "age"
    assert deduped[2] == "age_1"
    assert deduped[3] == "age_2"


def test_file_size_limit_rejection():
    valid, err = validate_file_metadata("test.csv", 101 * 1024 * 1024)
    assert valid is False
    assert "exceeds" in err.lower()
