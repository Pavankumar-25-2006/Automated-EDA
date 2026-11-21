"""
Automatic Visualization Engine Layer.
Intelligently generates Plotly interactive charts based on inferred semantic column types,
statistical properties, and data quality metrics.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config import (
    MAX_DISTRIBUTION_PLOTS,
    MAX_CATEGORICAL_PLOTS,
    PLOT_THEME,
    PLOT_COLOR_PRIMARY,
    PLOT_COLOR_SECONDARY,
    PLOT_COLOR_ACCENT
)


def create_quality_gauge_chart(score: float, grade: str) -> go.Figure:
    """Create a modern Plotly Gauge chart for Data Quality Score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Data Quality Health Index (Grade {grade})", "font": {"size": 20, "color": "#1E293B"}},
        number={"suffix": "/100", "font": {"size": 36, "color": "#0F172A"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
            "bar": {"color": "#2563EB", "thickness": 0.28},
            "bgcolor": "#F1F5F9",
            "borderwidth": 1,
            "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0, 50], "color": "#FEE2E2"},
                {"range": [50, 75], "color": "#FEF3C7"},
                {"range": [75, 90], "color": "#E0E7FF"},
                {"range": [90, 100], "color": "#DCFCE7"}
            ],
            "threshold": {
                "line": {"color": "#10B981", "width": 4},
                "thickness": 0.75,
                "value": score
            }
        }
    ))
    fig.update_layout(
        template=PLOT_THEME,
        height=280,
        margin={"l": 25, "r": 25, "t": 60, "b": 25}
    )
    return fig


def create_quality_components_chart(component_scores: Dict[str, float]) -> go.Figure:
    """Create a horizontal bar chart displaying component quality scores."""
    labels = ["Completeness (50%)", "Uniqueness (25%)", "Consistency (25%)"]
    vals = [
        component_scores.get("completeness", 0.0),
        component_scores.get("uniqueness", 0.0),
        component_scores.get("consistency", 0.0)
    ]
    colors = ["#3B82F6", "#10B981", "#8B5CF6"]

    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        orientation="h",
        marker={"color": colors, "line": {"width": 1, "color": "#CBD5E1"}},
        text=[f"{v:.1f}%" for v in vals],
        textposition="auto"
    ))
    fig.update_layout(
        title="Component Breakdown Scores",
        template=PLOT_THEME,
        xaxis={"range": [0, 105], "title": "Score (%)"},
        yaxis={"autorange": "reversed"},
        height=220,
        margin={"l": 20, "r": 20, "t": 40, "b": 20}
    )
    return fig


def create_missing_values_chart(missing_summary: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """Create a bar chart showing missing value percentages per column."""
    filtered = [m for m in missing_summary if m["missing_count"] > 0]
    if not filtered:
        return None

    # Sort descending
    filtered = sorted(filtered, key=lambda x: x["missing_percentage"], reverse=True)[:25]
    cols = [m["column_name"] for m in filtered]
    pcts = [m["missing_percentage"] for m in filtered]
    counts = [m["missing_count"] for m in filtered]

    fig = go.Figure(go.Bar(
        x=cols,
        y=pcts,
        marker={"color": "#EF4444", "line": {"width": 1, "color": "#B91C1C"}},
        text=[f"{p:.1f}% ({c:,})" for p, c in zip(pcts, counts)],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Missing: %{y:.2f}%<br>Count: %{text}<extra></extra>"
    ))
    fig.update_layout(
        title="Missing Values by Feature (% of Total Observations)",
        template=PLOT_THEME,
        xaxis_title="Column",
        yaxis_title="Missingness Percentage (%)",
        yaxis={"range": [0, max(pcts) * 1.18 + 5]},
        height=380,
        margin={"l": 40, "r": 20, "t": 50, "b": 60}
    )
    return fig


def create_distribution_chart(series: pd.Series, col_name: str) -> go.Figure:
    """Create an interactive histogram with marginal box plot for a numerical feature."""
    clean = series.dropna().astype(float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        clean = pd.Series([0.0])

    fig = px.histogram(
        x=clean,
        marginal="box",
        nbins=min(40, max(15, int(np.sqrt(len(clean))))),
        color_discrete_sequence=["#3B82F6"],
        labels={"x": col_name, "count": "Frequency"}
    )
    fig.update_layout(
        title=f"Distribution & Quartiles: {col_name}",
        template=PLOT_THEME,
        xaxis_title=col_name,
        yaxis_title="Frequency",
        height=340,
        margin={"l": 30, "r": 30, "t": 50, "b": 40}
    )
    return fig


def create_categorical_bar_chart(cat_stat: Dict[str, Any]) -> go.Figure:
    """Create a frequency bar chart for a categorical feature."""
    col_name = cat_stat["column_name"]
    top_cats = cat_stat.get("top_categories", [])
    if not top_cats:
        cats = ["(No Data)"]
        counts = [0]
        pcts = [0.0]
    else:
        cats = [c["category"] for c in top_cats]
        counts = [c["frequency"] for c in top_cats]
        pcts = [c["percentage"] for c in top_cats]

    fig = go.Figure(go.Bar(
        x=cats,
        y=counts,
        marker={"color": "#6366F1", "line": {"width": 1, "color": "#4338CA"}},
        text=[f"{p:.1f}%" for p in pcts],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Count: %{y:,}<br>Percentage: %{text}<extra></extra>"
    ))
    fig.update_layout(
        title=f"Category Distribution: {col_name} (Top {len(cats)})",
        template=PLOT_THEME,
        xaxis_title="Category",
        yaxis_title="Observation Count",
        yaxis={"range": [0, max(counts) * 1.18 + 5] if counts and max(counts) > 0 else [0, 10]},
        height=340,
        margin={"l": 30, "r": 30, "t": 50, "b": 60}
    )
    return fig


def create_correlation_heatmap(corr_matrix_dict: Dict[str, Dict[str, float]], method: str = "Pearson") -> Optional[go.Figure]:
    """Create an interactive correlation heatmap with diverging color palette."""
    if not corr_matrix_dict:
        return None

    df_corr = pd.DataFrame(corr_matrix_dict)
    cols = df_corr.columns.tolist()

    z_vals = df_corr.values
    # Round for display annotations
    text_vals = np.around(z_vals, decimals=2).astype(str)

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=cols,
        y=cols,
        colorscale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        text=text_vals,
        texttemplate="%{text}",
        hoverongaps=False,
        colorbar={"title": f"{method} r", "thickness": 15}
    ))
    fig.update_layout(
        title=f"{method} Correlation Heatmap",
        template=PLOT_THEME,
        height=max(450, min(800, len(cols) * 32)),
        width=max(500, min(900, len(cols) * 35)),
        margin={"l": 60, "r": 40, "t": 60, "b": 60}
    )
    return fig


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, r_val: float) -> go.Figure:
    """Create a scatter plot for high-correlation pairs with optional trendline."""
    clean_df = df[[x_col, y_col]].dropna().copy()
    # Filter out non-finite
    clean_df = clean_df[np.isfinite(clean_df[x_col]) & np.isfinite(clean_df[y_col])]

    if len(clean_df) > 2000:
        plot_df = clean_df.sample(2000, random_state=42)
        note = " (Sampled 2,000 pts for UI speed)"
    else:
        plot_df = clean_df
        note = ""

    # Trendline only if sufficient points and variance
    has_trendline = (
        len(plot_df) >= 4
        and plot_df[x_col].nunique() > 1
        and plot_df[y_col].nunique() > 1
    )

    try:
        if has_trendline:
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                trendline="ols",
                color_discrete_sequence=["#2563EB"],
                opacity=0.6,
                labels={x_col: x_col, y_col: y_col}
            )
        else:
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                color_discrete_sequence=["#2563EB"],
                opacity=0.6,
                labels={x_col: x_col, y_col: y_col}
            )
    except Exception:
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color_discrete_sequence=["#2563EB"],
            opacity=0.6,
            labels={x_col: x_col, y_col: y_col}
        )

    fig.update_layout(
        title=f"{x_col} vs {y_col} (r = {r_val:.2f}){note}",
        template=PLOT_THEME,
        height=360,
        margin={"l": 30, "r": 30, "t": 50, "b": 40}
    )
    return fig


def generate_all_visualizations(
    df: pd.DataFrame,
    profiling_results: Dict[str, Any],
    stats_results: Dict[str, Any],
    missing_results: Dict[str, Any],
    corr_results: Dict[str, Any],
    quality_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Intelligently generate all dashboard and report charts using sensible automated limits.
    """
    visualizations = {}

    # 1. Quality charts
    visualizations["quality_gauge"] = create_quality_gauge_chart(
        quality_results["overall_score"],
        quality_results["grade"]
    )
    visualizations["quality_components"] = create_quality_components_chart(
        quality_results["component_scores"]
    )

    # 2. Missing values chart
    visualizations["missing_values"] = create_missing_values_chart(
        missing_results.get("summary_table", [])
    )

    # 3. Numerical distributions (capped at MAX_DISTRIBUTION_PLOTS)
    num_cols = profiling_results.get("type_groups", {}).get("numerical", [])
    dist_charts = {}
    for col in num_cols[:MAX_DISTRIBUTION_PLOTS]:
        if col in df.columns and df[col].dropna().nunique() > 1:
            try:
                dist_charts[col] = create_distribution_chart(df[col], col)
            except Exception:
                pass
    visualizations["numerical_distributions"] = dist_charts

    # 4. Categorical frequency charts (capped at MAX_CATEGORICAL_PLOTS)
    cat_cols = profiling_results.get("type_groups", {}).get("categorical", [])
    cat_stats = stats_results.get("categorical_statistics", {})
    cat_charts = {}
    for col in cat_cols[:MAX_CATEGORICAL_PLOTS]:
        if col in cat_stats:
            try:
                cat_charts[col] = create_categorical_bar_chart(cat_stats[col])
            except Exception:
                pass
    visualizations["categorical_distributions"] = cat_charts

    # 5. Correlation Heatmap
    if corr_results.get("applied", False):
        p_matrix = corr_results.get("pearson_matrix")
        if p_matrix:
            visualizations["pearson_heatmap"] = create_correlation_heatmap(p_matrix, "Pearson")
        s_matrix = corr_results.get("spearman_matrix")
        if s_matrix:
            visualizations["spearman_heatmap"] = create_correlation_heatmap(s_matrix, "Spearman")

        # Top scatter plots (up to 3 pairs)
        scatter_plots = []
        top_pairs = corr_results.get("top_pairs", [])
        for pair in top_pairs[:3]:
            col_a = pair["feature_a"]
            col_b = pair["feature_b"]
            r_val = pair["pearson_corr"]
            if col_a in df.columns and col_b in df.columns:
                try:
                    sc_fig = create_scatter_plot(df, col_a, col_b, r_val)
                    scatter_plots.append(sc_fig)
                except Exception:
                    pass
        visualizations["scatter_plots"] = scatter_plots

    return visualizations
