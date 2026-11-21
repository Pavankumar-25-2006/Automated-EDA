"""
Automated Preprocessing Recommendation Layer.
Generates actionable, non-destructive data preparation recipes based on actual statistical properties.
Explains what was detected, why it matters, and suggested candidate actions.
"""
from typing import Dict, Any, List


def generate_recommendations(
    profiling_results: Dict[str, Any],
    stats_results: Dict[str, Any],
    missing_results: Dict[str, Any],
    outlier_results: Dict[str, Any],
    corr_results: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Generate comprehensive, explainable preprocessing recommendations.
    """
    recs = []

    # 1. Missing Value Strategy
    summary_missing = missing_results.get("summary_table", [])
    num_stats = stats_results.get("numerical_statistics", {})

    for item in summary_missing:
        col = item["column_name"]
        pct = item["missing_percentage"]
        stype = item["semantic_type"]

        if pct == 100.0:
            recs.append({
                "category": "Missing Data",
                "priority": "High",
                "detected": f"Column '{col}' is completely empty (100% missing values).",
                "why_it_matters": "Empty columns contain zero information and cause training failures in ML estimators.",
                "action": f"Drop column '{col}' from the feature matrix before model ingestion."
            })
        elif pct >= 40.0:
            recs.append({
                "category": "Missing Data",
                "priority": "Medium",
                "detected": f"Column '{col}' exhibits severe sparsity ({pct}% missing).",
                "why_it_matters": "Imputing over 40% of observations may introduce synthetic bias and distort covariance.",
                "action": (
                    f"Consider creating a binary missingness indicator column ('{col}_is_missing') "
                    "or omit the feature if it is non-essential."
                )
            })
        elif pct > 0:
            if stype == "numerical":
                # Check skewness
                skew = num_stats.get(col, {}).get("skewness", 0) or 0
                if abs(skew) > 1.0:
                    strategy = "median imputation (robust to distributional skew)"
                else:
                    strategy = "mean or median imputation"
                recs.append({
                    "category": "Missing Data",
                    "priority": "Medium",
                    "detected": f"Numerical column '{col}' has {pct}% missing values.",
                    "why_it_matters": "Most standard machine learning models cannot process NaN values directly.",
                    "action": f"Apply {strategy} using SimpleImputer or domain-specific baseline."
                })
            else:
                recs.append({
                    "category": "Missing Data",
                    "priority": "Medium",
                    "detected": f"Categorical column '{col}' has {pct}% missing values.",
                    "why_it_matters": "Missing categorical entries can cause inference errors or missing class labels.",
                    "action": f"Impute with the most frequent value (mode) or create an explicit 'Unknown' category."
                })

    # 2. Duplicate Records
    dup_count = profiling_results.get("duplicate_count", 0)
    dup_pct = profiling_results.get("duplicate_percentage", 0.0)
    if dup_count > 0:
        recs.append({
            "category": "Deduplication",
            "priority": "High" if dup_pct > 5.0 else "Low",
            "detected": f"Found {dup_count:,} duplicate records ({dup_pct}% of total dataset).",
            "why_it_matters": (
                "Unintentional duplicates overweight identical data points, biasing training and inflating "
                "evaluation metrics through train-test data leakage."
            ),
            "action": "Inspect duplicate rows to verify validity. If uninformative duplicates, apply `df.drop_duplicates()`."
        })

    # 3. Constant (Zero-Variance) Features
    constant_cols = profiling_results.get("constant_columns", [])
    if constant_cols:
        recs.append({
            "category": "Feature Selection",
            "priority": "High",
            "detected": f"Zero-variance constant columns detected: {constant_cols}.",
            "why_it_matters": "Constant features provide zero mathematical information and cause singularity in linear models.",
            "action": f"Remove {constant_cols} using `VarianceThreshold(threshold=0.0)` or drop manually."
        })

    # 4. Identifier Columns
    type_groups = profiling_results.get("type_groups", {})
    id_cols = type_groups.get("identifier", [])
    if id_cols:
        recs.append({
            "category": "Feature Selection",
            "priority": "High",
            "detected": f"High-cardinality identifier columns detected: {id_cols}.",
            "why_it_matters": "Identifiers (IDs, keys, hashes) carry no generalizable pattern and can cause extreme overfitting.",
            "action": f"Exclude {id_cols} from model training features (retain solely for index lookup or tracking)."
        })

    # 5. Numerical Feature Scaling
    num_cols = type_groups.get("numerical", [])
    if len(num_cols) >= 2:
        # Check variance/range disparity
        ranges = [num_stats[c].get("range", 0) or 0 for c in num_cols if c in num_stats]
        if ranges and (max(ranges) / (min(ranges) + 1e-6) > 50):
            has_outliers = outlier_results.get("columns_with_outliers_count", 0) > 0
            scaler_rec = "RobustScaler (IQR-based, resistant to outliers)" if has_outliers else "StandardScaler (Z-score normalization)"
            recs.append({
                "category": "Feature Scaling",
                "priority": "Medium",
                "detected": f"Significant scale disparity detected across {len(num_cols)} numerical features (e.g. ranges differ by >50x).",
                "why_it_matters": (
                    "Distance-based algorithms (KNN, SVM, K-Means) and gradient-based models (Neural Nets, Logistic Regression) "
                    "are sensitive to feature magnitudes."
                ),
                "action": f"Apply {scaler_rec} before fitting distance- or gradient-based models."
            })

    # 6. Distribution Skewness & Power Transformations
    skewed_cols = [c for c, st in num_stats.items() if abs(st.get("skewness") or 0) > 1.5]
    if skewed_cols:
        recs.append({
            "category": "Feature Transformation",
            "priority": "Low",
            "detected": f"Substantial distributional skewness in {len(skewed_cols)} column(s): {skewed_cols}.",
            "why_it_matters": "Parametric linear algorithms assume symmetric, near-normal residual errors.",
            "action": f"Evaluate `np.log1p()`, `PowerTransformer(method='yeo-johnson')`, or quantile transformations."
        })

    # 7. Categorical Encoding
    cat_cols = type_groups.get("categorical", [])
    if cat_cols:
        cat_stats = stats_results.get("categorical_statistics", {})
        low_card_cols = [c for c in cat_cols if cat_stats.get(c, {}).get("unique_categories", 0) <= 10]
        high_card_cols = [c for c in cat_cols if cat_stats.get(c, {}).get("unique_categories", 0) > 10]

        if low_card_cols:
            recs.append({
                "category": "Categorical Encoding",
                "priority": "Medium",
                "detected": f"Low-cardinality categorical features (<= 10 categories): {low_card_cols}.",
                "why_it_matters": "Non-numeric categories must be vectorized into machine-readable numeric matrices.",
                "action": f"Apply `OneHotEncoder(drop='first', sparse_output=False)` to prevent dummy variable trap."
            })
        if high_card_cols:
            recs.append({
                "category": "Categorical Encoding",
                "priority": "Medium",
                "detected": f"High-cardinality categorical features (> 10 categories): {high_card_cols}.",
                "why_it_matters": "One-hot encoding high-cardinality features causes severe curse of dimensionality and memory strain.",
                "action": "Consider Target Encoding, Frequency Encoding, or grouping infrequent categories into 'Other'."
            })

    # 8. Outlier Management
    outlier_cols = outlier_results.get("columns_with_outliers", [])
    if outlier_cols:
        recs.append({
            "category": "Outlier Treatment",
            "priority": "Medium",
            "detected": f"Tukey IQR outliers present in {len(outlier_cols)} feature(s): {outlier_cols}.",
            "why_it_matters": (
                "Extreme outliers can distort mean, standard deviation, and regression coefficients. "
                "However, outliers may represent legitimate high-value transactions or edge conditions."
            ),
            "action": (
                "Do NOT delete rows blindly. Consider Winsorization (clipping at 1st and 99th percentiles) "
                "or utilize tree-based ensembles (Random Forest, LightGBM) that are inherently robust to monotonic outliers."
            )
        })

    # 9. Multicollinearity Handling
    if corr_results.get("applied", False):
        strong_pos = corr_results.get("strong_positive_pairs", [])
        if strong_pos:
            top_p = strong_pos[0]
            recs.append({
                "category": "Multicollinearity",
                "priority": "Medium",
                "detected": f"High linear correlation detected between '{top_p['feature_a']}' and '{top_p['feature_b']}' (r = {top_p['pearson_corr']:.2f}).",
                "why_it_matters": (
                    "Multicollinearity inflates variance of coefficient estimates in linear models, making weights unstable and uninterpretable."
                ),
                "action": "Consider dropping one of the collinear features, performing PCA, or using Ridge/Lasso regularization."
            })

    return recs
