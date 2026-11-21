"""
Integration tests for the end-to-end Analysis Controller across Scenarios A, B, C, D, E.
"""
from pathlib import Path
import pytest
import pandas as pd
from modules.analysis_controller import run_full_analysis
from modules.report_generator import generate_html_report
from config import SAMPLE_DATASETS_DIR


@pytest.mark.parametrize("csv_filename", [
    "scenario_a_housing_clean.csv",
    "scenario_b_employee_churn.csv",
    "scenario_c_customer_dirty.csv",
    "scenario_d_financial_risk.csv",
    "scenario_e_iot_sensors.csv"
])
def test_full_pipeline_on_all_scenarios(csv_filename):
    file_path = SAMPLE_DATASETS_DIR / csv_filename
    assert file_path.exists(), f"Sample dataset {csv_filename} is missing"

    df = pd.read_csv(file_path)
    metadata = {
        "file_name": csv_filename,
        "row_count": len(df),
        "column_count": len(df.columns)
    }

    result = run_full_analysis(df, metadata)

    # Core assertions
    assert result.df is not None
    assert result.profiling["total_rows"] == len(df)
    assert result.profiling["total_columns"] == len(df.columns)
    assert 0.0 <= result.data_quality["overall_score"] <= 100.0
    assert result.data_quality["grade"] in ["A", "B", "C", "D", "F"]
    assert len(result.insights["overview"]) > 0

    # Ensure HTML report generates without errors
    html_report = generate_html_report(result)
    assert isinstance(html_report, str)
    assert len(html_report) > 500
    assert "<!DOCTYPE html>" in html_report
    assert csv_filename in html_report
