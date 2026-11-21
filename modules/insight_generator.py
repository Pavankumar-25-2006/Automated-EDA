"""
Intelligent Insight Generation Layer.
Rule-based deterministic engine that converts computed mathematical and statistical
findings into precise, human-readable insights with priority classifications.
Operates completely offline without external or paid APIs.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def generate_insights(
    profiling_results: Dict[str, Any],
    stats_results: Dict[str, Any],
    missing_results: Dict[str, Any],
    outlier_results: Dict[str, Any],
    corr_results: Dict[str, Any],
    quality_results: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate structured, prioritized, human-readable statistical insights.
    Categories:
    - overview
    - data_quality
    - numerical_patterns
    - categorical_patterns
    - relationships
    - anomalies
    """
    insights = {
        "overview": [],
        "data_quality": [],
        "numerical_patterns": [],
        "categorical_patterns": [],
        "relationships": [],
        "anomalies": []
    }

    # 1. Dataset Overview Insights
    total_rows = profiling_results.get("total_rows", 0)
    total_cols = profiling_results.get("total_columns", 0)
    mem_str = profiling_results.get("memory_formatted", "N/A")
    type_counts = profiling_results.get("type_counts", {})

    type_summary_parts = []
    for stype, count in type_counts.items():
        if count > 0:
            type_summary_parts.append(f"{count} {stype}")
    type_desc = ", ".join(type_summary_parts) if type_summary_parts else "unspecified types"

    insights["overview"].append({
        "category": "Dataset Overview",
        "priority": "info",
        "title": "Dataset Scale and Ingestion Profile",
        "message": (
            f"The dataset contains {total_rows:,} rows and {total_cols} columns, "
            f"occupying approximately {mem_str} of system memory."
        ),
        "basis": f"Calculated exact shape ({total_rows}, {total_cols}) and deep memory footprint."
    })

    insights["overview"].append({
        "category": "Dataset Overview",
        "priority": "info",
        "title": "Semantic Type Distribution",
        "message": f"Identified column distribution: {type_desc}.",
        "basis": "Conservative semantic type classification heuristic."
    })

    constant_cols = profiling_results.get("constant_columns", [])
    if constant_cols:
        insights["overview"].append({
            "category": "Dataset Overview",
            "priority": "warning",
            "title": "Zero-Variance Columns Detected",
            "message": (
                f"Columns {constant_cols} exhibit zero variance (all non-null values are identical). "
                "These columns contribute zero discriminating power to machine learning models."
            ),
            "basis": f"Unique non-null count is 1 for columns: {constant_cols}."
        })

    # 2. Data Quality Insights
    q_score = quality_results.get("overall_score", 0.0)
    q_grade = quality_results.get("grade", "N/A")
    q_rating = quality_results.get("rating", "N/A")
    dup_count = profiling_results.get("duplicate_count", 0)
    dup_pct = profiling_results.get("duplicate_percentage", 0.0)
    tot_miss = missing_results.get("total_missing_cells", 0)
    miss_pct = missing_results.get("overall_missing_percentage", 0.0)

    insights["data_quality"].append({
        "category": "Data Quality",
        "priority": "important" if q_score < 75.0 else "info",
        "title": f"Overall Health Rating: Grade {q_grade} ({q_score}/100 - {q_rating})",
        "message": (
            f"The dataset scored {q_score}/100 on the automated hygiene index. "
            f"Completeness is {quality_results['component_scores']['completeness']}%, "
            f"Uniqueness is {quality_results['component_scores']['uniqueness']}%, and "
            f"Structural Consistency is {quality_results['component_scores']['consistency']}%."
        ),
        "basis": "Weighted composite formula: 50% Completeness + 25% Uniqueness + 25% Structural Consistency."
    })

    if dup_count > 0:
        insights["data_quality"].append({
            "category": "Data Quality",
            "priority": "warning" if dup_pct > 5.0 else "info",
            "title": "Duplicate Rows Present",
            "message": f"Detected {dup_count:,} duplicate records representing {dup_pct}% of the dataset.",
            "basis": f"Exact DataFrame row-level duplicate match count: {dup_count}."
        })

    if tot_miss > 0:
        high_miss = missing_results.get("high_missing_columns", [])
        if high_miss:
            insights["data_quality"].append({
                "category": "Data Quality",
                "priority": "important",
                "title": "Severe Missingness in Key Features",
                "message": (
                    f"Columns {high_miss} exceed 30% missingness. Overall dataset missingness is {miss_pct}%."
                ),
                "basis": f"Missing value ratios exceeded 30% threshold in {len(high_miss)} columns."
            })
        else:
            insights["data_quality"].append({
                "category": "Data Quality",
                "priority": "info",
                "title": "Minor Missingness Detected",
                "message": f"Dataset has {tot_miss:,} missing values ({miss_pct}% overall cell sparsity).",
                "basis": "Missing cell summation across all features."
            })

    # 3. Numerical Distribution Patterns
    num_stats = stats_results.get("numerical_statistics", {})
    for col, st in num_stats.items():
        skew = st.get("skewness")
        kurt = st.get("kurtosis")
        std = st.get("std")
        cv = st.get("cv")

        if skew is not None and not np.isnan(skew):
            if skew > 1.5:
                insights["numerical_patterns"].append({
                    "category": "Numerical Patterns",
                    "priority": "warning",
                    "title": f"Strong Positive Skewness in '{col}'",
                    "message": (
                        f"Feature '{col}' exhibits significant right-skew (skewness = {skew:.2f}). "
                        "The tail extends towards higher values; consider log, square root, or Box-Cox transformations for linear models."
                    ),
                    "basis": f"Fisher-Pearson skewness coefficient ({skew:.2f}) > 1.5."
                })
            elif skew < -1.5:
                insights["numerical_patterns"].append({
                    "category": "Numerical Patterns",
                    "priority": "warning",
                    "title": f"Strong Negative Skewness in '{col}'",
                    "message": (
                        f"Feature '{col}' exhibits significant left-skew (skewness = {skew:.2f}). "
                        "Values are concentrated on the higher end with an extended left tail."
                    ),
                    "basis": f"Fisher-Pearson skewness coefficient ({skew:.2f}) < -1.5."
                })

        if kurt is not None and not np.isnan(kurt) and kurt > 3.0:
            insights["numerical_patterns"].append({
                "category": "Numerical Patterns",
                "priority": "info",
                "title": f"Leptokurtic (Heavy-Tailed) Distribution in '{col}'",
                "message": (
                    f"Feature '{col}' has excess kurtosis of {kurt:.2f}, indicating heavy tails and a sharp peak compared to normal distributions."
                ),
                "basis": f"Excess kurtosis ({kurt:.2f}) > 3.0."
            })

        if cv is not None and not np.isnan(cv) and cv > 1.5:
            insights["numerical_patterns"].append({
                "category": "Numerical Patterns",
                "priority": "info",
                "title": f"High Relative Dispersion in '{col}'",
                "message": (
                    f"Feature '{col}' exhibits high relative variance (Coefficient of Variation = {cv:.2f}). "
                    "Scale varies widely relative to the mean; standard scaling may be beneficial."
                ),
                "basis": f"Coefficient of variation (std / |mean| = {cv:.2f}) > 1.5."
            })

    # 4. Categorical Patterns
    cat_stats = stats_results.get("categorical_statistics", {})
    for col, cst in cat_stats.items():
        mode_pct = cst.get("mode_percentage", 0.0)
        mode_val = cst.get("mode")
        n_unique = cst.get("unique_categories", 0)

        if mode_pct >= 80.0 and n_unique > 1:
            insights["categorical_patterns"].append({
                "category": "Categorical Patterns",
                "priority": "warning",
                "title": f"Severe Class Dominance in '{col}'",
                "message": (
                    f"In column '{col}', category '{mode_val}' accounts for {mode_pct}% of all observations. "
                    "This high concentration creates extreme class imbalance."
                ),
                "basis": f"Dominant category frequency percentage ({mode_pct}%) >= 80%."
            })
        elif n_unique > 30:
            insights["categorical_patterns"].append({
                "category": "Categorical Patterns",
                "priority": "info",
                "title": f"High Cardinality in '{col}'",
                "message": (
                    f"Column '{col}' contains {n_unique} distinct categories. "
                    "One-hot encoding may induce significant dimensional explosion; consider target encoding or grouping rare categories."
                ),
                "basis": f"Unique category count ({n_unique}) > 30."
            })

    # 5. Relationship / Correlation Insights
    if corr_results.get("applied", False):
        strong_pos = corr_results.get("strong_positive_pairs", [])
        strong_neg = corr_results.get("strong_negative_pairs", [])

        for pair in strong_pos[:5]:
            insights["relationships"].append({
                "category": "Relationships",
                "priority": "important",
                "title": f"Strong Positive Linear Association: {pair['feature_a']} & {pair['feature_b']}",
                "message": (
                    f"Features '{pair['feature_a']}' and '{pair['feature_b']}' exhibit strong positive correlation "
                    f"(Pearson r = {pair['pearson_corr']:.2f}, Spearman ρ = {pair['spearman_corr']:.2f}). "
                    "Increases in one correspond to linear increases in the other. Check for multicollinearity in regression models."
                ),
                "basis": f"Pearson r ({pair['pearson_corr']}) >= {pair['pearson_corr']} threshold."
            })

        for pair in strong_neg[:5]:
            insights["relationships"].append({
                "category": "Relationships",
                "priority": "important",
                "title": f"Strong Inverse Association: {pair['feature_a']} & {pair['feature_b']}",
                "message": (
                    f"Features '{pair['feature_a']}' and '{pair['feature_b']}' exhibit strong inverse correlation "
                    f"(Pearson r = {pair['pearson_corr']:.2f}, Spearman ρ = {pair['spearman_corr']:.2f})."
                ),
                "basis": f"Pearson r ({pair['pearson_corr']}) <= -0.70 threshold."
            })

        if not strong_pos and not strong_neg:
            insights["relationships"].append({
                "category": "Relationships",
                "priority": "info",
                "title": "No Severe Multicollinearity Detected",
                "message": "All evaluated numerical feature pairs show correlation coefficients below |0.70|.",
                "basis": "Exhaustive pairwise Pearson correlation matrix evaluation."
            })

    # 6. Outliers and Anomalies
    out_table = outlier_results.get("summary_table", [])
    for row in out_table:
        if row.get("outlier_count", 0) > 0 and row.get("outlier_percentage", 0) > 1.0:
            col = row["column_name"]
            pct = row["outlier_percentage"]
            cnt = row["outlier_count"]
            insights["anomalies"].append({
                "category": "Anomalies",
                "priority": "warning" if pct > 5.0 else "info",
                "title": f"Tukey IQR Outliers in '{col}'",
                "message": (
                    f"Detected {cnt:,} potential outlier records ({pct}% of column) in '{col}' outside the range "
                    f"[{row['lower_bound']:.2f}, {row['upper_bound']:.2f}]."
                ),
                "basis": f"Values outside Q1 - 1.5*IQR and Q3 + 1.5*IQR (IQR = {row['iqr']:.2f})."
            })

    ml_anom = outlier_results.get("ml_anomalies", {})
    if ml_anom.get("applied", False):
        anom_cnt = ml_anom.get("anomaly_count", 0)
        anom_pct = ml_anom.get("anomaly_percentage", 0.0)
        insights["anomalies"].append({
            "category": "Anomalies",
            "priority": "important",
            "title": "Multivariate Anomalies via Isolation Forest",
            "message": (
                f"Isolation Forest flagged {anom_cnt:,} observations ({anom_pct}% of dataset) as multivariate anomalies "
                f"across {len(ml_anom.get('evaluated_features', []))} numerical dimensions."
            ),
            "basis": f"Unsupervised Isolation Forest ensemble (contamination parameter = {ml_anom.get('contamination_parameter')})."
        })

    return insights
