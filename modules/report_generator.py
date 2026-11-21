"""
Automated Report Generation Layer.
Produces a beautiful, self-contained, responsive HTML EDA report with embedded Plotly visualizations,
tables, insights, and preprocessing recommendations.
Includes proper HTML escaping to prevent script injection.
"""
import html
from typing import Dict, Any, List
from jinja2 import Template

from modules.analysis_controller import AnalysisPipelineResult
from utils.helpers import format_number, format_percentage


HTML_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AutoEDA Report - {{ metadata.file_name | e }}</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    :root {
      --primary: #2563EB;
      --primary-dark: #1D4ED8;
      --bg-page: #F8FAFC;
      --bg-card: #FFFFFF;
      --text-main: #0F172A;
      --text-muted: #64748B;
      --border: #E2E8F0;
      --success: #10B981;
      --warning: #F59E0B;
      --danger: #EF4444;
      --radius: 8px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background-color: var(--bg-page);
      color: var(--text-main);
      line-height: 1.6;
      padding: 30px 20px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
      color: white;
      padding: 32px 36px;
      border-radius: var(--radius);
      margin-bottom: 24px;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    header h1 { font-size: 28px; font-weight: 700; margin-bottom: 6px; }
    header p { opacity: 0.9; font-size: 15px; }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
    }
    .badge-info { background: #E0E7FF; color: #3730A3; }
    .badge-warning { background: #FEF3C7; color: #92400E; }
    .badge-important { background: #FEE2E2; color: #991B1B; }
    .badge-success { background: #DCFCE7; color: #166534; }
    
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);
    }
    .card h2 {
      font-size: 20px;
      font-weight: 600;
      color: #1E293B;
      margin-bottom: 16px;
      border-bottom: 2px solid #F1F5F9;
      padding-bottom: 10px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }
    .metric-card {
      background: #F8FAFC;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
      text-align: center;
    }
    .metric-value { font-size: 26px; font-weight: 700; color: var(--primary); }
    .metric-label { font-size: 13px; color: var(--text-muted); font-weight: 500; text-transform: uppercase; }
    
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13.5px;
    }
    th, td {
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    th {
      background-color: #F8FAFC;
      color: #475569;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 11.5px;
      letter-spacing: 0.05em;
    }
    tr:hover { background-color: #F8FAFC; }
    
    .insight-item {
      padding: 14px 18px;
      border-radius: 6px;
      border-left: 4px solid var(--primary);
      background: #F8FAFC;
      margin-bottom: 12px;
    }
    .insight-item.warning { border-left-color: var(--warning); background: #FFFBEB; }
    .insight-item.important { border-left-color: var(--danger); background: #FEF2F2; }
    .insight-title { font-weight: 600; font-size: 14.5px; margin-bottom: 4px; display: flex; align-items: center; justify-content: space-between; }
    .insight-body { font-size: 13.5px; color: #334155; }
    .insight-basis { font-size: 12px; color: var(--text-muted); font-style: italic; margin-top: 4px; }
    
    .rec-item {
      padding: 14px 18px;
      border: 1px solid var(--border);
      border-radius: 6px;
      margin-bottom: 12px;
    }
    .rec-header { font-weight: 600; font-size: 14.5px; color: #1E293B; margin-bottom: 4px; }
    .rec-action { font-weight: 600; color: var(--primary); margin-top: 6px; font-size: 13.5px; }

    .chart-container { margin: 20px 0; }
    footer {
      text-align: center;
      margin-top: 40px;
      padding: 20px;
      font-size: 13px;
      color: var(--text-muted);
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>AutoEDA — Automated Exploratory Data Analysis Report</h1>
      <p>Dataset: <strong>{{ metadata.file_name | e }}</strong> | Generated on {{ timestamp | e }} | Engine Version 2.0.0</p>
    </header>

    <!-- Overview Metrics -->
    <div class="card">
      <h2>1. Dataset Executive Summary</h2>
      <div class="metric-grid">
        <div class="metric-card">
          <div class="metric-value">{{ profiling.total_rows | e }}</div>
          <div class="metric-label">Total Observations</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{{ profiling.total_columns | e }}</div>
          <div class="metric-label">Total Features</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{{ profiling.memory_formatted | e }}</div>
          <div class="metric-label">Memory Footprint</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{{ missing.overall_missing_percentage | e }}%</div>
          <div class="metric-label">Total Missingness</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{{ profiling.duplicate_count | e }} ({{ profiling.duplicate_percentage | e }}%)</div>
          <div class="metric-label">Duplicate Rows</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{{ data_quality.overall_score | e }} / 100</div>
          <div class="metric-label">Quality Score (Grade {{ data_quality.grade | e }})</div>
        </div>
      </div>
      <p style="font-size: 13.5px; color: var(--text-muted);">
        Semantic Composition: 
        {% for stype, cnt in profiling.type_counts.items() if cnt > 0 %}
          <strong>{{ cnt }}</strong> {{ stype }}{% if not loop.last %}, {% endif %}
        {% endfor %}
      </p>
    </div>

    <!-- Data Quality -->
    <div class="card">
      <h2>2. Data Quality & Hygiene Assessment</h2>
      <p style="margin-bottom: 12px; font-size: 14px;">Overall Score: <strong>{{ data_quality.overall_score | e }} / 100 ({{ data_quality.rating | e }})</strong></p>
      
      {% if quality_gauge_html %}
      <div class="chart-container">
        {{ quality_gauge_html | safe }}
      </div>
      {% endif %}

      <div style="margin-top: 16px;">
        <h4 style="font-size: 14px; margin-bottom: 6px;">Identified Hygiene Issues:</h4>
        <ul style="margin-left: 20px; font-size: 13.5px;">
          {% for issue in data_quality.issues %}
            <li>{{ issue | e }}</li>
          {% endfor %}
        </ul>
      </div>

      <div style="margin-top: 14px;">
        <h4 style="font-size: 14px; margin-bottom: 6px;">Suggested Data Cleanliness Actions:</h4>
        <ul style="margin-left: 20px; font-size: 13.5px; color: #15803D;">
          {% for sug in data_quality.suggested_improvements %}
            <li>{{ sug | e }}</li>
          {% endfor %}
        </ul>
      </div>
    </div>

    <!-- Intelligent Insights -->
    <div class="card">
      <h2>3. Automated Statistical Insights</h2>
      {% for section, items in insights.items() %}
        {% if items %}
          <h3 style="font-size: 16px; margin: 16px 0 8px 0; color: #334155; text-transform: capitalize;">{{ section.replace('_', ' ') | e }}</h3>
          {% for item in items %}
            <div class="insight-item {{ item.priority | e }}">
              <div class="insight-title">
                <span>{{ item.title | e }}</span>
                <span class="badge badge-{{ item.priority | e }}">{{ item.priority | e }}</span>
              </div>
              <div class="insight-body">{{ item.message | e }}</div>
              <div class="insight-basis">Basis: {{ item.basis | e }}</div>
            </div>
          {% endfor %}
        {% endif %}
      {% endfor %}
    </div>

    <!-- Column Profiling -->
    <div class="card">
      <h2>4. Feature Profiling Table</h2>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th>Feature Name</th>
              <th>Raw Dtype</th>
              <th>Semantic Type</th>
              <th>Missing Count</th>
              <th>Missing %</th>
              <th>Unique Vals</th>
              <th>Properties</th>
            </tr>
          </thead>
          <tbody>
            {% for col in profiling.columns_profile %}
            <tr>
              <td><strong>{{ col.column_name | e }}</strong></td>
              <td><code>{{ col.original_dtype | e }}</code></td>
              <td><span class="badge badge-info">{{ col.semantic_type | e }}</span></td>
              <td>{{ col.missing_count | e }}</td>
              <td>{{ col.missing_percentage | e }}%</td>
              <td>{{ col.unique_count | e }}</td>
              <td>
                {% if col.is_constant %}<span class="badge badge-warning">Constant</span>{% endif %}
                {% if col.is_all_null %}<span class="badge badge-important">All Null</span>{% endif %}
                {% if col.is_high_cardinality %}<span class="badge badge-warning">High Card.</span>{% endif %}
                {% if not col.is_constant and not col.is_all_null and not col.is_high_cardinality %}-{% endif %}
              </td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Numerical Statistics -->
    {% if statistics.numerical_summary_table %}
    <div class="card">
      <h2>5. Numerical Feature Statistics</h2>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th>Feature</th>
              <th>Count</th>
              <th>Mean</th>
              <th>Std Dev</th>
              <th>Min</th>
              <th>Q25</th>
              <th>Median</th>
              <th>Q75</th>
              <th>Max</th>
              <th>IQR</th>
              <th>Skewness</th>
            </tr>
          </thead>
          <tbody>
            {% for row in statistics.numerical_summary_table %}
            <tr>
              <td><strong>{{ row.column_name | e }}</strong></td>
              <td>{{ row.count | e }}</td>
              <td>{{ row.mean | e }}</td>
              <td>{{ row.std | e }}</td>
              <td>{{ row.min | e }}</td>
              <td>{{ row.q25 | e }}</td>
              <td>{{ row.median | e }}</td>
              <td>{{ row.q75 | e }}</td>
              <td>{{ row.max | e }}</td>
              <td>{{ row.iqr | e }}</td>
              <td>{{ row.skewness | e }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
    {% endif %}

    <!-- Missing Values -->
    {% if missing.columns_with_missing_count > 0 %}
    <div class="card">
      <h2>6. Missingness Analysis</h2>
      <p style="font-size: 14px; margin-bottom: 12px;">
        Total Missing Cells: <strong>{{ missing.total_missing_cells | e }}</strong> 
        ({{ missing.overall_missing_percentage | e }}% of dataset). 
        Rows affected: <strong>{{ missing.rows_with_missing | e }}</strong> ({{ missing.rows_with_missing_percentage | e }}%).
      </p>
      {% if missing_chart_html %}
      <div class="chart-container">
        {{ missing_chart_html | safe }}
      </div>
      {% endif %}
    </div>
    {% endif %}

    <!-- Outlier Detection -->
    {% if outliers.summary_table %}
    <div class="card">
      <h2>7. Outlier & Anomaly Analysis</h2>
      <p style="font-size: 14px; margin-bottom: 12px;">
        Statistical detection via Tukey's IQR Method (1.5x Multiplier). 
        Features with outliers: <strong>{{ outliers.columns_with_outliers_count | e }}</strong>.
      </p>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th>Feature</th>
              <th>Valid Obs</th>
              <th>IQR</th>
              <th>Lower Bound</th>
              <th>Upper Bound</th>
              <th>Outlier Count</th>
              <th>Outlier %</th>
            </tr>
          </thead>
          <tbody>
            {% for row in outliers.summary_table %}
            <tr>
              <td><strong>{{ row.column_name | e }}</strong></td>
              <td>{{ row.valid_count | e }}</td>
              <td>{{ row.iqr | e }}</td>
              <td>{{ row.lower_bound | e }}</td>
              <td>{{ row.upper_bound | e }}</td>
              <td>{{ row.outlier_count | e }}</td>
              <td>{{ row.outlier_percentage | e }}%</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% if outliers.ml_anomalies.applied %}
      <div style="margin-top: 16px; background: #F8FAFC; padding: 14px; border-radius: 6px;">
        <h4 style="font-size: 14px;">Multivariate Anomaly Detection (Isolation Forest)</h4>
        <p style="font-size: 13.5px; color: var(--text-muted);">
          Flagged <strong>{{ outliers.ml_anomalies.anomaly_count | e }}</strong> records 
          ({{ outliers.ml_anomalies.anomaly_percentage | e }}% of dataset) as potential multivariate outliers.
        </p>
      </div>
      {% endif %}
    </div>
    {% endif %}

    <!-- Correlation Analysis -->
    {% if correlations.applied %}
    <div class="card">
      <h2>8. Correlation & Relationship Analysis</h2>
      <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 14px;">{{ correlations.causation_disclaimer | e }}</p>
      {% if pearson_heatmap_html %}
      <div class="chart-container">
        {{ pearson_heatmap_html | safe }}
      </div>
      {% endif %}
      
      {% if correlations.top_pairs %}
      <h4 style="font-size: 14px; margin: 16px 0 8px 0;">Top Correlated Feature Pairs:</h4>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th>Feature A</th>
              <th>Feature B</th>
              <th>Pearson r</th>
              <th>Spearman &rho;</th>
              <th>Strength</th>
              <th>Direction</th>
            </tr>
          </thead>
          <tbody>
            {% for pair in correlations.top_pairs %}
            <tr>
              <td><strong>{{ pair.feature_a | e }}</strong></td>
              <td><strong>{{ pair.feature_b | e }}</strong></td>
              <td>{{ pair.pearson_corr | e }}</td>
              <td>{{ pair.spearman_corr | e }}</td>
              <td>{{ pair.strength | e }}</td>
              <td>{{ pair.direction | e }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% endif %}
    </div>
    {% endif %}

    <!-- Preprocessing Recommendations -->
    {% if recommendations %}
    <div class="card">
      <h2>9. Preprocessing & Data Preparation Recipes</h2>
      <p style="font-size: 14px; color: var(--text-muted); margin-bottom: 14px;">
        Actionable guidance tailored to the empirical distribution and structural hygiene of this dataset:
      </p>
      {% for rec in recommendations %}
      <div class="rec-item">
        <div class="rec-header">
          <span class="badge badge-{% if rec.priority == 'High' %}important{% elif rec.priority == 'Medium' %}warning{% else %}info{% endif %}">
            {{ rec.priority | e }}
          </span>
          <span style="margin-left: 8px;"><strong>{{ rec.category | e }}:</strong> {{ rec.detected | e }}</span>
        </div>
        <div style="font-size: 13.5px; color: #475569; margin: 4px 0;"><strong>Why it matters:</strong> {{ rec.why_it_matters | e }}</div>
        <div class="rec-action">&rarr; Recommended Action: {{ rec.action | e }}</div>
      </div>
      {% endfor %}
    </div>
    {% endif %}

    <footer>
      <p>AutoEDA — Advanced Automated Exploratory Data Analysis System</p>
      <p>Developed with Python, Streamlit, Pandas, SciPy, and Plotly. Offline Analysis Engine.</p>
    </footer>
  </div>
</body>
</html>
"""


def generate_html_report(analysis_result: AnalysisPipelineResult) -> str:
    """
    Render self-contained HTML EDA report string with embedded Plotly charts and tables.
    """
    # Convert Plotly figures to HTML divs
    viz = analysis_result.visualizations
    quality_gauge_html = ""
    if "quality_gauge" in viz and viz["quality_gauge"] is not None:
        quality_gauge_html = viz["quality_gauge"].to_html(
            include_plotlyjs=False, full_html=False
        )

    missing_chart_html = ""
    if "missing_values" in viz and viz["missing_values"] is not None:
        missing_chart_html = viz["missing_values"].to_html(
            include_plotlyjs=False, full_html=False
        )

    pearson_heatmap_html = ""
    if "pearson_heatmap" in viz and viz["pearson_heatmap"] is not None:
        pearson_heatmap_html = viz["pearson_heatmap"].to_html(
            include_plotlyjs=False, full_html=False
        )

    template = Template(HTML_REPORT_TEMPLATE)
    rendered_html = template.render(
        metadata=analysis_result.metadata,
        timestamp=analysis_result.timestamp,
        profiling=analysis_result.profiling,
        statistics=analysis_result.statistics,
        missing=analysis_result.missing,
        outliers=analysis_result.outliers,
        correlations=analysis_result.correlations,
        categorical=analysis_result.categorical,
        data_quality=analysis_result.data_quality,
        insights=analysis_result.insights,
        recommendations=analysis_result.recommendations,
        quality_gauge_html=quality_gauge_html,
        missing_chart_html=missing_chart_html,
        pearson_heatmap_html=pearson_heatmap_html
    )
    return rendered_html
