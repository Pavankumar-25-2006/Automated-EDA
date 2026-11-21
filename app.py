"""
AutoEDA — Advanced Automated Exploratory Data Analysis System
Main Streamlit Application.
"""
from pathlib import Path
import hashlib
import streamlit as st
import pandas as pd

from config import (
    APP_NAME,
    APP_SUBTITLE,
    APP_VERSION,
    SAMPLE_DATASETS_DIR
)
from modules.data_loader import load_csv_data
from modules.analysis_controller import run_full_analysis, AnalysisPipelineResult
from modules.report_generator import generate_html_report

# Page Configuration
st.set_page_config(
    page_title=f"{APP_NAME} — Automated EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        padding: 24px 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 28px;
        font-weight: 700;
    }
    .main-header p {
        color: #E2E8F0;
        margin: 5px 0 0 0;
        font-size: 14.5px;
    }
    .metric-container {
        background-color: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 12.5px;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 4px;
    }
    .insight-card {
        padding: 14px 18px;
        border-radius: 6px;
        border-left: 4px solid #2563EB;
        background: #F8FAFC;
        margin-bottom: 12px;
    }
    .insight-card.warning {
        border-left-color: #F59E0B;
        background: #FFFBEB;
    }
    .insight-card.important {
        border-left-color: #EF4444;
        background: #FEF2F2;
    }
    .insight-title {
        font-weight: 600;
        font-size: 14.5px;
        color: #1E293B;
    }
    .insight-desc {
        font-size: 13.5px;
        color: #334155;
        margin-top: 4px;
    }
    .insight-basis {
        font-size: 12px;
        color: #64748B;
        font-style: italic;
        margin-top: 4px;
    }
    .rec-box {
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 16px;
        background: white;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


def load_cached_analysis(file_bytes: bytes, file_name: str) -> AnalysisPipelineResult:
    """Load dataset and execute full analysis pipeline."""
    load_res = load_csv_data(file_bytes, file_name=file_name)
    if not load_res.success:
        raise ValueError(load_res.error_message)
    return run_full_analysis(load_res.df, load_res.metadata)


def main():
    # Header Banner
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 {APP_NAME}</h1>
        <p>{APP_SUBTITLE} &bull; AIML Open-Ended Micro Project (College Demonstration)</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Instructions, Sample Datasets & Info
    with st.sidebar:
        st.header("⚙️ Controls & Presets")
        st.markdown("""
        **Zero-Configuration Automated EDA:**
        Upload any CSV file to automatically inspect structure, distributions, missingness, outliers, correlations, and data quality.
        """)
        
        # Sample datasets selector for instant live testing
        st.subheader("📁 Load Sample Dataset")
        sample_files = list(SAMPLE_DATASETS_DIR.glob("*.csv")) if SAMPLE_DATASETS_DIR.exists() else []
        sample_options = ["None (Upload Custom CSV)"] + [f.name for f in sample_files]
        selected_sample = st.selectbox("Select demonstration preset:", sample_options)

        st.divider()
        st.markdown(f"""
        **Application Details:**
        - **Engine:** Python 3.11 / Pandas / SciPy / Plotly
        - **Anomaly Detection:** Tukey IQR + Isolation Forest
        - **Quality Model:** Explainable 3-Axis Heuristic
        - **Mode:** 100% Offline (No paid APIs)
        - **Version:** {APP_VERSION}
        """)

    # Data Ingestion Source
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV format):",
        type=["csv"],
        help="Upload any standard CSV file up to 100 MB."
    )

    df_bytes = None
    file_name = None

    if uploaded_file is not None:
        df_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
    elif selected_sample != "None (Upload Custom CSV)":
        sample_path = SAMPLE_DATASETS_DIR / selected_sample
        if sample_path.exists():
            df_bytes = sample_path.read_bytes()
            file_name = sample_path.name

    if df_bytes is None:
        # Initial Landing State
        st.info("👆 Please upload a CSV dataset or select a sample preset from the sidebar to begin automated analysis.")
        
        # Feature Highlights Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            ### 🔍 Intelligent Profiling
            - Semantic type classification (Categorical, Numerical, ID, Datetime)
            - Structural sparsity & zero-variance column detection
            - Memory footprint & row duplication analysis
            """)
        with col2:
            st.markdown("""
            ### 📈 Advanced Analytics
            - Parametric & non-parametric statistics
            - Tukey IQR & Isolation Forest multivariate outliers
            - Pearson & Spearman correlation heatmaps
            """)
        with col3:
            st.markdown("""
            ### 💡 Actionable Insights
            - Explainable 0-100 Data Quality Health Score
            - Deterministic statistical finding generator
            - ML-ready preprocessing recommendations & HTML report
            """)
        return

    # Execute Analysis with Smart Caching
    data_hash = hashlib.sha256(df_bytes).hexdigest()
    cache_key = (data_hash, file_name)

    if (
        "current_analysis" not in st.session_state
        or st.session_state.get("current_cache_key") != cache_key
    ):
        with st.spinner("Analyzing dataset structure, statistics, correlations, and hygiene..."):
            try:
                analysis_obj = load_cached_analysis(df_bytes, file_name)
                st.session_state["current_analysis"] = analysis_obj
                st.session_state["current_cache_key"] = cache_key
            except Exception as e:
                st.error(f"❌ Analysis Failed: {str(e)}")
                return

    analysis = st.session_state["current_analysis"]

    # Top Metric Summary Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{analysis.profiling['total_rows']:,}</div>
            <div class="metric-label">Observations</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{analysis.profiling['total_columns']:,}</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{analysis.missing['overall_missing_percentage']}%</div>
            <div class="metric-label">Missing Cells</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{analysis.profiling['duplicate_count']:,}</div>
            <div class="metric-label">Duplicates ({analysis.profiling['duplicate_percentage']}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{analysis.data_quality['overall_score']} <span style="font-size:16px;">({analysis.data_quality['grade']})</span></div>
            <div class="metric-label">Quality Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Navigation Tabs
    tabs = st.tabs([
        "📊 Overview & Health",
        "📋 Data Profiling",
        "📐 Descriptive Statistics",
        "❓ Missing Values",
        "⚠️ Outliers & Anomalies",
        "🔗 Correlations",
        "📈 Visualizations",
        "💡 Automated Insights",
        "🛠️ Preprocessing Recipes",
        "📄 Export Report"
    ])

    # TAB 1: OVERVIEW & HEALTH
    with tabs[0]:
        st.subheader("Dataset Ingestion & Health Profile")
        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            if "quality_gauge" in analysis.visualizations:
                st.plotly_chart(analysis.visualizations["quality_gauge"], use_container_width=True)
        with col_g2:
            if "quality_components" in analysis.visualizations:
                st.plotly_chart(analysis.visualizations["quality_components"], use_container_width=True)

        st.markdown("#### Quality Assessment Details")
        st.markdown(f"**Heuristic Rating:** `{analysis.data_quality['rating']}` (Grade {analysis.data_quality['grade']})")
        st.markdown(f"*{analysis.data_quality['limitations']}*")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Identified Quality Issues:**")
            for iss in analysis.data_quality["issues"]:
                st.warning(iss, icon="⚠️")
        with c2:
            st.markdown("**Recommended Cleanliness Actions:**")
            for sug in analysis.data_quality["suggested_improvements"]:
                st.info(sug, icon="💡")

        st.markdown("#### Dataset Preview (First 10 Rows)")
        st.dataframe(analysis.df.head(10), use_container_width=True)

    # TAB 2: DATA PROFILING
    with tabs[1]:
        st.subheader("Feature Structure and Semantic Classification")
        st.write(
            f"Dataset Memory Usage: **{analysis.profiling['memory_formatted']}** | "
            f"Semantic Types: **{analysis.profiling['type_counts']}**"
        )
        
        prof_df = pd.DataFrame(analysis.profiling["columns_profile"])
        st.dataframe(
            prof_df[[
                "column_name", "semantic_type", "original_dtype", 
                "missing_count", "missing_percentage", "unique_count", 
                "is_constant", "is_high_cardinality", "type_reason", "sample_values"
            ]],
            use_container_width=True
        )

        if analysis.profiling["constant_columns"]:
            st.error(f"Zero-variance constant columns detected: {analysis.profiling['constant_columns']}")
        if analysis.profiling["all_null_columns"]:
            st.error(f"Completely empty columns detected: {analysis.profiling['all_null_columns']}")
        if analysis.profiling["high_cardinality_columns"]:
            st.warning(f"High-cardinality categorical columns detected: {analysis.profiling['high_cardinality_columns']}")

    # TAB 3: DESCRIPTIVE STATISTICS
    with tabs[2]:
        st.subheader("Descriptive Statistical Measurements")
        
        num_summary = analysis.statistics.get("numerical_summary_table", [])
        if num_summary:
            st.markdown("#### Numerical Variables (Parametric & Non-Parametric)")
            st.dataframe(pd.DataFrame(num_summary), use_container_width=True)
            with st.expander("ℹ️ Glossary of Statistical Metrics"):
                st.markdown("""
                - **IQR (Interquartile Range):** Q75 - Q25; measures middle 50% spread, robust to outliers.
                - **Skewness:** Fisher-Pearson coefficient. 0 = symmetric, > +1 = right-skewed tail, < -1 = left-skewed tail.
                - **Kurtosis:** Fisher excess kurtosis. 0 = normal (mesokurtic), > 0 = heavy tails (leptokurtic).
                - **CV (Coefficient of Variation):** std / |mean|; relative dispersion independent of scale.
                """)
        else:
            st.info("No numerical features available for continuous statistics.")

        cat_summary = analysis.statistics.get("categorical_summary_table", [])
        if cat_summary:
            st.markdown("#### Categorical Variables (Frequency & Modes)")
            st.dataframe(pd.DataFrame(cat_summary), use_container_width=True)

        cat_card_summary = analysis.categorical.get("summary_table", [])
        if cat_card_summary:
            st.markdown("#### Categorical Class Distribution & Balance")
            st.dataframe(pd.DataFrame([
                {
                    "column_name": r["column_name"],
                    "valid_count": r["valid_count"],
                    "unique_categories": r["unique_categories"],
                    "mode_category": r["mode_category"],
                    "mode_frequency": r["mode_frequency"],
                    "mode_percentage": f"{r['mode_percentage']}%",
                    "rare_categories_count": r["rare_categories_count"],
                    "is_imbalanced": "⚠️ Yes (≥70%)" if r["is_imbalanced"] else "Balanced",
                    "is_high_cardinality": "⚠️ Yes (>30)" if r["is_high_cardinality"] else "No"
                }
                for r in cat_card_summary
            ]), use_container_width=True)

            if analysis.categorical.get("imbalanced_columns"):
                for imb in analysis.categorical["imbalanced_columns"]:
                    st.warning(
                        f"Severe class imbalance in '{imb['column']}': Dominant mode '{imb['dominant_category']}' "
                        f"accounts for {imb['percentage']}% of records.",
                        icon="⚠️"
                    )

    # TAB 4: MISSING VALUES
    with tabs[3]:
        st.subheader("Missing Value Audit & Sparsity")
        st.write(
            f"Total missing cells: **{analysis.missing['total_missing_cells']:,}** "
            f"({analysis.missing['overall_missing_percentage']}%) across "
            f"**{analysis.missing['columns_with_missing_count']}** columns. "
            f"Rows affected: **{analysis.missing['rows_with_missing']:,}** ({analysis.missing['rows_with_missing_percentage']}%)."
        )

        if "missing_values" in analysis.visualizations and analysis.visualizations["missing_values"] is not None:
            st.plotly_chart(analysis.visualizations["missing_values"], use_container_width=True)

        st.markdown("#### Feature Missingness Summary Table")
        st.dataframe(pd.DataFrame(analysis.missing["summary_table"]), use_container_width=True)

        st.markdown("#### Imputation Guidance")
        for rec in analysis.missing["recommendations"]:
            if rec["type"] == "positive":
                st.success(rec["message"])
            elif rec["type"] == "warning":
                st.warning(rec["message"])
            else:
                st.info(rec["message"])

    # TAB 5: OUTLIERS & ANOMALIES
    with tabs[4]:
        st.subheader("Outlier & Anomaly Detection")
        st.markdown("""
        **Methodology:**
        - **Tukey's IQR Rule:** Flags observations outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`.
        - **Isolation Forest:** Unsupervised tree ensemble isolating sparse multivariate anomalies.
        """)

        out_summary = analysis.outliers.get("summary_table", [])
        if out_summary:
            st.markdown("#### IQR Univariate Outlier Summary")
            st.dataframe(pd.DataFrame(out_summary), use_container_width=True)
        else:
            st.info("No numerical features eligible for outlier evaluation.")

        ml_anom = analysis.outliers.get("ml_anomalies", {})
        if ml_anom.get("applied", False):
            st.markdown("#### Machine Learning Multivariate Anomaly Detection")
            st.success(
                f"Isolation Forest identified **{ml_anom['anomaly_count']:,}** potential anomalies "
                f"({ml_anom['anomaly_percentage']}% of dataset) across features {ml_anom['evaluated_features']}."
            )
            st.caption(ml_anom["methodology_note"])

    # TAB 6: CORRELATIONS
    with tabs[5]:
        st.subheader("Correlation & Multicollinearity Analysis")
        st.caption(analysis.correlations.get("causation_disclaimer", ""))

        if analysis.correlations.get("applied", False):
            if "pearson_heatmap" in analysis.visualizations:
                st.plotly_chart(analysis.visualizations["pearson_heatmap"], use_container_width=True)
            
            top_pairs = analysis.correlations.get("top_pairs", [])
            if top_pairs:
                st.markdown("#### Top Correlated Feature Pairs")
                st.dataframe(pd.DataFrame(top_pairs), use_container_width=True)

            scatter_plots = analysis.visualizations.get("scatter_plots", [])
            if scatter_plots:
                st.markdown("#### Bivariate Scatter Plots (Top Linear Relationships)")
                sc_cols = st.columns(len(scatter_plots))
                for idx, sc_fig in enumerate(scatter_plots):
                    with sc_cols[idx]:
                        st.plotly_chart(sc_fig, use_container_width=True)
        else:
            st.info(analysis.correlations.get("reason", "Correlation analysis not applicable."))

    # TAB 7: VISUALIZATIONS
    with tabs[6]:
        st.subheader("Automated Feature Distributions")
        
        # Numerical distribution sub-section
        num_dist = analysis.visualizations.get("numerical_distributions", {})
        if num_dist:
            st.markdown("#### Numerical Feature Distributions (Histograms + Box Plots)")
            num_keys = list(num_dist.keys())
            for i in range(0, len(num_keys), 2):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(num_dist[num_keys[i]], use_container_width=True)
                if i + 1 < len(num_keys):
                    with col_b:
                        st.plotly_chart(num_dist[num_keys[i + 1]], use_container_width=True)
        
        # Categorical distribution sub-section
        cat_dist = analysis.visualizations.get("categorical_distributions", {})
        if cat_dist:
            st.markdown("#### Categorical Frequency Distributions")
            cat_keys = list(cat_dist.keys())
            for i in range(0, len(cat_keys), 2):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(cat_dist[cat_keys[i]], use_container_width=True)
                if i + 1 < len(cat_keys):
                    with col_b:
                        st.plotly_chart(cat_dist[cat_keys[i + 1]], use_container_width=True)

    # TAB 8: INTELLIGENT INSIGHTS
    with tabs[7]:
        st.subheader("Rule-Based Statistical Insights")
        st.caption("Insights are deterministically derived from computed statistics without external AI APIs.")
        
        for section, items in analysis.insights.items():
            if items:
                st.markdown(f"### {section.replace('_', ' ').title()}")
                for item in items:
                    p = item["priority"]
                    st.markdown(f"""
                    <div class="insight-card {p}">
                        <div class="insight-title">[{p.upper()}] {item['title']}</div>
                        <div class="insight-desc">{item['message']}</div>
                        <div class="insight-basis">Basis: {item['basis']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # TAB 9: PREPROCESSING RECIPES
    with tabs[8]:
        st.subheader("Actionable Preprocessing Recipes")
        st.markdown("Recommended preprocessing operations prior to machine learning modeling:")

        if analysis.recommendations:
            for rec in analysis.recommendations:
                p_color = "#EF4444" if rec["priority"] == "High" else ("#F59E0B" if rec["priority"] == "Medium" else "#3B82F6")
                st.markdown(f"""
                <div class="rec-box">
                    <span style="background:{p_color}; color:white; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:bold;">
                        {rec['priority']} Priority
                    </span>
                    <strong style="margin-left: 8px; font-size:15px; color:#1E293B;">{rec['category']}: {rec['detected']}</strong>
                    <div style="margin: 6px 0; color:#475569; font-size:13.5px;">
                        <strong>Why it matters:</strong> {rec['why_it_matters']}
                    </div>
                    <div style="color:#2563EB; font-weight:600; font-size:13.5px;">
                        &rarr; Action: {rec['action']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No critical preprocessing operations required.")

    # TAB 10: EXPORT REPORT
    with tabs[9]:
        st.subheader("Download Comprehensive EDA Report")
        st.markdown("""
        Generate a self-contained, interactive HTML report with embedded Plotly charts,
        profiling summaries, descriptive statistics, and recommendations.
        """)

        html_report = generate_html_report(analysis)
        report_file_name = f"AutoEDA_Report_{Path(file_name).stem}.html"

        st.download_button(
            label="📥 Download Complete HTML Report",
            data=html_report,
            file_name=report_file_name,
            mime="text/html",
            help="Download standalone HTML report readable in any web browser."
        )

        st.success(f"Report prepared successfully for '{file_name}' ({len(html_report):,} bytes).")


if __name__ == "__main__":
    main()
