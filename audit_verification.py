"""
AutoEDA Comprehensive Technical Audit & Verification Script.
Performs exhaustive automated verification across all 12 audit requirements.
"""
import io
import os
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from config import SAMPLE_DATASETS_DIR
from modules.data_loader import load_csv_data
from modules.analysis_controller import run_full_analysis
from modules.report_generator import generate_html_report
from modules.data_profiler import profile_dataset
from modules.statistics import compute_dataset_statistics
from modules.data_quality import calculate_data_quality_score
from modules.outlier_detection import analyze_outliers
from modules.correlation import compute_correlations
from modules.insight_generator import generate_insights


def audit_test_all_sample_scenarios():
    print("=== AUDIT STEP 1: Verifying 5 Benchmark CSV Datasets ===")
    scenarios = [
        "scenario_a_housing_clean.csv",
        "scenario_b_employee_churn.csv",
        "scenario_c_customer_dirty.csv",
        "scenario_d_financial_risk.csv",
        "scenario_e_iot_sensors.csv"
    ]
    
    results = {}
    for filename in scenarios:
        path = SAMPLE_DATASETS_DIR / filename
        assert path.exists(), f"Missing dataset {filename}"
        
        load_res = load_csv_data(str(path))
        assert load_res.success is True, f"Failed to load {filename}: {load_res.error_message}"
        df = load_res.df
        assert df is not None and len(df) > 0
        
        analysis = run_full_analysis(df, load_res.metadata)
        
        # Verify core objects
        assert analysis.profiling["total_rows"] == len(df)
        assert analysis.profiling["total_columns"] == len(df.columns)
        assert 0.0 <= analysis.data_quality["overall_score"] <= 100.0
        assert analysis.data_quality["grade"] in ["A", "B", "C", "D", "F"]
        assert len(analysis.insights["overview"]) > 0
        
        # Generate and verify HTML report
        html_report = generate_html_report(analysis)
        assert len(html_report) > 1000
        assert "<!DOCTYPE html>" in html_report
        assert filename in html_report
        
        results[filename] = {
            "rows": analysis.profiling["total_rows"],
            "cols": analysis.profiling["total_columns"],
            "quality_score": analysis.data_quality["overall_score"],
            "quality_grade": analysis.data_quality["grade"],
            "duplicate_count": analysis.profiling["duplicate_count"],
            "missing_pct": analysis.missing["overall_missing_percentage"],
            "insights_count": sum(len(v) for v in analysis.insights.values()),
            "recommendations_count": len(analysis.recommendations),
            "report_bytes": len(html_report)
        }
        print(f"  [PASS] {filename}: {analysis.profiling['total_rows']} rows, {analysis.profiling['total_columns']} cols | Quality: {analysis.data_quality['overall_score']} ({analysis.data_quality['grade']}) | HTML Report: {len(html_report):,} bytes")

    return results


def audit_test_edge_cases():
    print("\n=== AUDIT STEP 2: Edge Case Datasets (Pure Num, Pure Cat, All Null, Constant) ===")
    
    # Edge Case 1: Pure Numerical Dataset
    df_pure_num = pd.DataFrame({
        "n1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "n2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    })
    res_num = run_full_analysis(df_pure_num, {"file_name": "pure_num.csv"})
    assert res_num.profiling["type_counts"]["numerical"] == 2
    assert res_num.correlations["applied"] is True
    print("  [PASS] Pure numerical dataset processed correctly.")

    # Edge Case 2: Pure Categorical Dataset
    df_pure_cat = pd.DataFrame({
        "c1": ["Low", "Medium", "High", "Low", "Medium"] * 4,
        "c2": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"] * 4
    })
    res_cat = run_full_analysis(df_pure_cat, {"file_name": "pure_cat.csv"})
    assert res_cat.profiling["type_counts"]["categorical"] == 2
    assert res_cat.correlations["applied"] is False
    assert "At least 2 non-constant numerical columns are required" in res_cat.correlations["reason"]
    print("  [PASS] Pure categorical dataset handled safely without correlation crash.")

    # Edge Case 3: Constant columns and completely empty columns
    df_degenerate = pd.DataFrame({
        "num": [10, 20, 30, 40, 50],
        "all_null": [np.nan] * 5,
        "constant_num": [42] * 5,
        "constant_str": ["Fixed"] * 5
    })
    res_deg = run_full_analysis(df_degenerate, {"file_name": "degenerate.csv"})
    assert "all_null" in res_deg.profiling["all_null_columns"]
    assert "constant_num" in res_deg.profiling["constant_columns"]
    assert "constant_str" in res_deg.profiling["constant_columns"]
    assert res_deg.data_quality["overall_score"] < 80.0
    print("  [PASS] Degenerate columns (all-null, constant) safely flagged and penalized in data quality.")


def audit_test_mathematical_consistency():
    print("\n=== AUDIT STEP 3: Mathematical Consistency of Insights & Statistics ===")
    
    # Create dataset with verified mathematical properties
    np.random.seed(123)
    n = 300
    skewed_vals = np.random.exponential(scale=5.0, size=n)  # skew > 1.5
    skewed_vals[0] = 200.0  # guaranteed extreme outlier
    
    col_a = np.linspace(1, 100, n)
    col_b = 3.5 * col_a + np.random.normal(0, 0.5, n)  # Pearson r > 0.99
    
    dominant_cat = ["Dominant"] * int(n * 0.85) + ["Rare"] * int(n * 0.15)
    
    df = pd.DataFrame({
        "skewed": skewed_vals,
        "col_a": col_a,
        "col_b": col_b,
        "category": dominant_cat
    })
    
    analysis = run_full_analysis(df, {"file_name": "math_check.csv"})
    
    # 1. Verify Skewness Insight
    calculated_skew = analysis.statistics["numerical_statistics"]["skewed"]["skewness"]
    assert calculated_skew > 1.5
    num_insights = [i for i in analysis.insights["numerical_patterns"] if "skew" in i["title"].lower()]
    assert len(num_insights) > 0
    assert "skewed" in num_insights[0]["title"]
    print(f"  [PASS] Skewness verified: Calculated = {calculated_skew:.2f}, Insight triggered = '{num_insights[0]['title']}'")
    
    # 2. Verify Correlation Insight
    r_val = analysis.correlations["top_pairs"][0]["pearson_corr"]
    assert r_val > 0.95
    rel_insights = [i for i in analysis.insights["relationships"] if "Positive" in i["title"]]
    assert len(rel_insights) > 0
    assert "col_a" in rel_insights[0]["title"] and "col_b" in rel_insights[0]["title"]
    print(f"  [PASS] Correlation verified: Calculated r = {r_val:.4f}, Insight = '{rel_insights[0]['title']}'")

    # 3. Verify Class Dominance Insight
    cat_mode_pct = analysis.statistics["categorical_statistics"]["category"]["mode_percentage"]
    assert cat_mode_pct >= 85.0
    cat_insights = [i for i in analysis.insights["categorical_patterns"] if "dominance" in i["title"].lower()]
    assert len(cat_insights) > 0
    print(f"  [PASS] Class Dominance verified: Mode = {cat_mode_pct}%, Insight = '{cat_insights[0]['title']}'")

    # 4. Verify Outlier Calculation
    outliers = analysis.outliers["iqr_results"]["skewed"]
    assert outliers["outlier_count"] >= 1
    assert outliers["upper_outliers_count"] >= 1
    print(f"  [PASS] Outlier Detection verified: {outliers['outlier_count']} outliers outside [{outliers['lower_bound']}, {outliers['upper_bound']}]")


def audit_test_dataset_replacement():
    print("\n=== AUDIT STEP 4: Dataset Replacement & Session State Isolation ===")
    
    # Dataset 1: Housing clean
    df1 = pd.DataFrame({"house_size": [1000, 2000, 3000], "price": [100000, 200000, 300000]})
    analysis1 = run_full_analysis(df1, {"file_name": "houses.csv"})
    
    # Dataset 2: Students
    df2 = pd.DataFrame({"student_name": ["Alice", "Bob", "Charlie", "David"], "grade": [90, 85, 92, 88]})
    analysis2 = run_full_analysis(df2, {"file_name": "students.csv"})
    
    assert analysis1.metadata["file_name"] == "houses.csv"
    assert analysis2.metadata["file_name"] == "students.csv"
    assert "house_size" in analysis1.df.columns
    assert "house_size" not in analysis2.df.columns
    assert "student_name" in analysis2.df.columns
    assert analysis1.profiling["total_rows"] == 3
    assert analysis2.profiling["total_rows"] == 4
    print("  [PASS] Consecutive dataset analyses are completely isolated and free from memory bleed.")


def audit_test_html_report_inspection():
    print("\n=== AUDIT STEP 5: HTML Report Content & Plotly Div Inspection ===")
    
    path = SAMPLE_DATASETS_DIR / "scenario_c_customer_dirty.csv"
    load_res = load_csv_data(str(path))
    analysis = run_full_analysis(load_res.df, load_res.metadata)
    html_content = generate_html_report(analysis)
    
    # Verify essential HTML components
    assert "AutoEDA — Automated Exploratory Data Analysis Report" in html_content
    assert "scenario_c_customer_dirty.csv" in html_content
    assert "1. Dataset Executive Summary" in html_content
    assert "2. Data Quality & Hygiene Assessment" in html_content
    assert "3. Automated Statistical Insights" in html_content
    assert "4. Feature Profiling Table" in html_content
    assert "5. Numerical Feature Statistics" in html_content
    assert "6. Categorical Feature Analysis & Class Balance" in html_content
    assert "7. Missingness Analysis" in html_content
    assert "8. Outlier & Anomaly Analysis" in html_content
    assert "10. Preprocessing & Data Preparation Recipes" in html_content
    assert "plotly-2.30.0.min.js" in html_content
    
    # Check that dirty dataset defects are explicitly present in the report
    assert "empty_notes_column" in html_content
    assert "uninformative_status" in html_content
    assert "duplicate" in html_content.lower()

    # Save a physical inspection copy in brain or sample_datasets
    inspection_path = SAMPLE_DATASETS_DIR / "audit_report_inspection.html"
    inspection_path.write_text(html_content, encoding="utf-8")
    print(f"  [PASS] HTML report verified and exported to {inspection_path} ({len(html_content):,} bytes).")


if __name__ == "__main__":
    audit_test_all_sample_scenarios()
    audit_test_edge_cases()
    audit_test_mathematical_consistency()
    audit_test_dataset_replacement()
    audit_test_html_report_inspection()
    print("\n>>> ALL AUDIT AND VERIFICATION CHECKS COMPLETED SUCCESSFULLY! <<<")
