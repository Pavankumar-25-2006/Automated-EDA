# AutoEDA — Advanced Automated Exploratory Data Analysis System

**An AIML Open-Ended Micro Project for Engineering College Demonstration**  
Developed with Python 3.11, Streamlit, Pandas, NumPy, SciPy, Scikit-learn, and Plotly.

---

## 1. Project Abstract

In real-world data science and machine learning workflows, manual Exploratory Data Analysis (EDA) accounts for a substantial fraction of project development time. Analysts frequently perform repetitive scripting tasks: validating file encodings, checking dimensional constraints, inferring variable types, calculating descriptive statistics, auditing missingness, spotting anomalies, generating distribution plots, and formulating preprocessing steps.

**AutoEDA** is an intelligent, zero-configuration automated exploratory data analysis system. The user performs only **one mandatory action: upload any standard CSV dataset**. Upon upload, AutoEDA deterministically inspects the data structure, classifies semantic data types, calculates parametric and non-parametric statistics, audits missing values, performs Tukey's IQR and Isolation Forest anomaly detection, models linear and rank correlation, computes a 3-axis explainable Data Quality Score (0–100), synthesizes prioritized human-readable statistical insights, recommends ML preprocessing recipes, and renders an interactive dashboard alongside a standalone, downloadable HTML report.

---

## 2. Problem Statement

Manual data exploration suffers from several key bottlenecks:
1. **Human Inconsistency:** Analysts may overlook subtle data quality defects such as low-cardinality discrete codes masquerading as continuous features, severe class imbalance, or non-linear multicollinearity.
2. **Repetitive Boilerplate:** Writing boilerplate code for quartile calculations, missing-value bar plots, and heatmap generation slows down project lifecycles.
3. **Black-Box Profilers:** Many commercial or third-party automated EDA libraries operate as monolithic, unmodifiable black boxes or require external cloud/API tokens.
4. **Data Leakage & Preprocessing Errors:** Beginners frequently apply inappropriate imputations (e.g., mean imputation on extreme log-normal distributions) or fail to isolate identifiers prior to model fitting.

---

## 3. Project Objectives

- **Zero-Configuration Automation:** Enable complete exploratory analysis from a single CSV upload without requiring manual column selection, coding, or prior domain configuration.
- **Conservative Semantic Type Inference:** Accurately classify columns into Numerical, Categorical, Boolean, Datetime, Text, Identifier, or Unsupported categories without modifying underlying data.
- **Explainable Data Quality Metric:** Implement a transparent, mathematically grounded 0–100 Data Quality Health Score evaluated across Completeness (50%), Uniqueness (25%), and Structural Consistency (25%).
- **Dual-Layer Anomaly Detection:** Combine robust statistical outlier detection (Tukey's IQR method) with unsupervised machine learning (Scikit-Learn's Isolation Forest).
- **Offline Intelligent Insights:** Generate structured, prioritized statistical findings derived strictly from computed measurements without relying on external paid LLM APIs.
- **Actionable Preprocessing Guidance:** Automatically formulate conditional recommendations for imputation, scaling, categorical encoding, power transformations, and feature selection.
- **Portability & Export:** Generate self-contained, interactive HTML reports with embedded Plotly charts for offline sharing.

---

## 4. Key Features

| Category | Capability |
| :--- | :--- |
| **Data Ingestion** | Dynamic encoding detection (UTF-8, Latin-1, CP1252), delimiter sniffing (`,`, `;`, `\t`, `\|`), file validation, header deduplication. |
| **Dataset Profiling** | Dimensions, memory consumption, semantic type classification, duplicate row tracking, constant column detection. |
| **Descriptive Statistics** | Parametric (Mean, Std Dev, Variance) and non-parametric (Median, IQR, Percentiles, Skewness, Kurtosis, Coefficient of Variation). |
| **Missing Value Audit** | Cell sparsity, row-level impact, high-missingness alerts (>30%), empty column detection, conditional imputation strategies. |
| **Outlier Detection** | Univariate Tukey IQR ($1.5 \times \text{IQR}$) and multivariate unsupervised Isolation Forest. |
| **Correlation Analysis** | Pearson linear correlation, Spearman rank correlation, interactive heatmaps with auto-filtering for wide matrices, scatter plots. |
| **Intelligent Insights** | Deterministic finding generator organizing insights into Overview, Quality, Numerical Patterns, Categorical Dominance, Relationships, and Anomalies. |
| **Data Quality Score** | Explainable 0–100 heuristic index with letter grades (A–F), component breakdown, and hygiene suggestions. |
| **Exportable Report** | Complete, standalone HTML report with responsive styling and embedded interactive Plotly figures. |

---

## 5. Technology Stack

- **Language:** Python 3.11 (Compatible with Python 3.11 - 3.15)
- **Application Framework:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Mathematical & Statistical Engine:** [SciPy](https://scipy.org/)
- **Machine Learning (Multivariate Anomaly Detection):** [Scikit-learn](https://scikit-learn.org/) (Isolation Forest)
- **Interactive Visualizations:** [Plotly Express & Plotly Graph Objects](https://plotly.com/python/)
- **Templating Engine:** [Jinja2](https://palletsprojects.com/p/jinja/)
- **Testing Framework:** [Pytest](https://docs.pytest.org/)

---

## 6. System Architecture

AutoEDA adheres to a strict 6-layer modular architecture ensuring complete separation between business logic and the presentation layer:

```text
AutoEDA/
│
├── app.py                      # Layer 1: Streamlit Presentation Layer & Dashboard
├── config.py                   # Centralized application settings & thresholds
├── requirements.txt            # Package dependencies
├── README.md                   # Complete technical documentation
├── .gitignore                  # Git ignore definitions
├── pytest.ini                  # Pytest configuration
├── generate_samples.py         # Synthetic sample data generator
│
├── modules/                    # Core Business & Analytical Engine
│   ├── __init__.py
│   ├── data_loader.py          # Layer 2: File ingestion, encoding, validation
│   ├── data_profiler.py        # Layer 3: Structural profiling & duplicate audits
│   ├── statistics.py           # Layer 4: Parametric & non-parametric statistics
│   ├── missing_values.py       # Layer 4: Missingness audit & sparsity analysis
│   ├── outlier_detection.py    # Layer 4 & 5: Tukey IQR & Isolation Forest
│   ├── correlation.py          # Layer 4: Pearson & Spearman correlation matrices
│   ├── categorical_analysis.py # Layer 4: Category frequencies, modes, balance
│   ├── data_quality.py         # Layer 5: Explainable 0-100 hygiene scoring
│   ├── insight_generator.py    # Layer 5: Rule-based statistical insight synthesis
│   ├── recommendations.py      # Layer 5: Actionable ML preprocessing recipes
│   ├── visualizations.py       # Layer 5: Intelligent Plotly chart generator
│   ├── analysis_controller.py  # Orchestrator: Pipeline controller & facade
│   └── report_generator.py     # Layer 6: Self-contained HTML report export
│
├── utils/                      # Common Utilities & Inference
│   ├── __init__.py
│   ├── helpers.py              # Formatting, numeric guards, serialization
│   └── type_detection.py       # Semantic column type classifier
│
├── sample_datasets/            # Demonstration & Benchmark Datasets
│   ├── README.md
│   ├── scenario_a_housing_clean.csv
│   ├── scenario_b_employee_churn.csv
│   ├── scenario_c_customer_dirty.csv
│   ├── scenario_d_financial_risk.csv
│   └── scenario_e_iot_sensors.csv
│
├── tests/                      # Automated Test Suite (Pytest)
│   ├── test_data_loader.py
│   ├── test_data_profiler.py
│   ├── test_statistics.py
│   ├── test_data_quality.py
│   ├── test_outlier_detection.py
│   ├── test_correlation.py
│   ├── test_insights.py
│   └── test_integration_controller.py
│
└── .streamlit/
    └── config.toml             # Streamlit visual theme configuration
```

---

## 7. Statistical & Analytical Methodology

### 7.1 Semantic Type Detection
Columns are evaluated conservatively:
- **Numerical:** Float or integer features with $>10$ unique values (or continuous floats).
- **Categorical:** Categorical dtypes, strings with cardinality $\le 50$, or low unique ratios ($\le 5\%$).
- **Boolean:** Binary flags matching $\{0, 1\}$, $\{\text{True}, \text{False}\}$, $\{\text{Yes}, \text{No}\}$, or $\{\text{Y}, \text{N}\}$.
- **Datetime:** Valid ISO date/time format matched across $>85\%$ of non-null entries.
- **Text:** Free-form string entries with average word count $\ge 3.0$ and length $> 50$.
- **Identifier:** Alphanumeric strings or sequential IDs matching pattern keywords (`id`, `uuid`, `code`, `key`) with $>75\%$ uniqueness.

### 7.2 Outlier Detection
1. **Tukey's IQR Method (Default Statistical Method):**
   - First Quartile ($Q_1$) = 25th percentile
   - Third Quartile ($Q_3$) = 75th percentile
   - $\text{IQR} = Q_3 - Q_1$
   - $\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$
   - $\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$
   - Observations $< \text{Lower Bound}$ or $> \text{Upper Bound}$ are cataloged as univariate outliers.
2. **Isolation Forest (Machine Learning Anomaly Detection):**
   - An unsupervised tree ensemble from Scikit-Learn.
   - Operates on multidimensional numerical matrices (imputing missing values with median for isolation trees).
   - Isolates observations by randomly selecting a feature and split value. Anomalies have significantly shorter tree paths.
   - Configured with a default contamination rate $\alpha = 0.05$.

### 7.3 Correlation & Multicollinearity
- **Pearson Linear Correlation:** Measures linear bivariate associations ($r \in [-1, 1]$).
- **Spearman Rank Correlation:** Evaluates monotonic relationships ($\rho \in [-1, 1]$), resistant to non-linear scales and outliers.
- **Heatmap Guard:** Datasets with $>25$ numerical features are intelligently pruned to display the top 25 features with highest variance or association strength to avoid unreadable heatmaps.
- **Association Disclaimer:** The system explicitly notes that correlation demonstrates empirical co-movement, never causation.

### 7.4 Explainable Data Quality Score
The Data Quality Score is a transparent index between 0 and 100:
$$\text{Score} = (0.50 \times \text{Completeness}) + (0.25 \times \text{Uniqueness}) + (0.25 \times \text{Consistency})$$
- **Completeness ($50\%$):** Proportion of non-null cells across the entire matrix:
  $$\text{Completeness} = \left(1 - \frac{\text{Total Missing Cells}}{\text{Total Rows} \times \text{Total Columns}}\right) \times 100$$
- **Uniqueness ($25\%$):** Proportion of non-duplicate records:
  $$\text{Uniqueness} = \left(1 - \frac{\text{Duplicate Rows}}{\text{Total Rows}}\right) \times 100$$
- **Structural Consistency ($25\%$):** Evaluates structural hygiene with penalties for:
  - 100% empty columns ($-15$ pts each, max $-40$)
  - Constant zero-variance columns ($-10$ pts each, max $-30$)
  - Severe sparsity columns with $>30\%$ missingness ($-5$ pts each, max $-20$)

---

## 8. Rule-Based Intelligent Insights

AutoEDA generates human-readable statistical conclusions strictly through deterministic logic based on computed metrics (no paid AI APIs, 100% offline). Every insight cites its mathematical basis:
- **Skewness Insights:** Triggered when Fisher-Pearson skewness exceeds $|1.5|$, recommending appropriate log or power transformations.
- **Kurtosis Insights:** Identifies heavy-tailed (leptokurtic) distributions when excess kurtosis $> 3.0$.
- **Class Dominance:** Flags severe categorical imbalance when a single mode accounts for $\ge 70\%$ of observations.
- **Multicollinearity:** Detects strong linear redundancy when $|r| \ge 0.70$.
- **Hygiene Alerts:** Prioritizes warnings based on sparsity, zero variance, and row duplication.

---

## 9. Installation & Windows Setup Guide

### 9.1 Prerequisites
- Windows 10 or Windows 11
- Python 3.11 (or higher) installed on your system.

### 9.2 Virtual Environment Setup
Open PowerShell or Command Prompt in the project root directory:

```powershell
# 1. Clone or navigate to the project directory
cd "d:\Projects\Mini Projects\AIML Micro"

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
.\venv\Scripts\Activate.ps1
# (In CMD: .\venv\Scripts\activate.bat)

# 4. Upgrade pip and install dependencies
pip install -r requirements.txt
```

---

## 10. Running the Application

Launch the Streamlit web dashboard:

```powershell
.\venv\Scripts\streamlit.exe run app.py
```

The application will launch automatically in your default browser at:
`http://localhost:8501`

### 10.1 Demonstrating AutoEDA
1. **Upload a Dataset:** Click **"Browse files"** and select any CSV file (or select one of the built-in scenario presets from the sidebar).
2. **Automatic Execution:** AutoEDA automatically ingests, profiles, and analyzes the dataset.
3. **Explore Dashboard Tabs:**
   - **Overview & Health:** High-level summary, gauge chart, data quality score.
   - **Data Profiling:** Column-by-column semantic type table.
   - **Descriptive Statistics:** Parametric and non-parametric summary tables.
   - **Missing Values:** Sparsity chart and conditional imputation advice.
   - **Outliers & Anomalies:** Tukey IQR bounds and Isolation Forest ML detections.
   - **Correlations:** Heatmaps and scatter plots.
   - **Visualizations:** Distribution histograms and category frequency plots.
   - **Automated Insights:** Prioritized, explainable findings.
   - **Preprocessing Recipes:** Concrete ML preparation actions.
   - **Export Report:** Download the full self-contained HTML report.

---

## 11. Automated Testing

The project includes an extensive test suite built with **Pytest**:

```powershell
# Run the entire test suite
.\venv\Scripts\python.exe -m pytest -v
```

### Test Coverage Highlights
- `test_data_loader.py`: CSV ingestion, encoding fallback, delimiter sniffing, empty upload rejection, duplicate header deduplication.
- `test_data_profiler.py`: Dimension counting, duplicate detection, semantic type detection, constant column isolation.
- `test_statistics.py`: Parametric and non-parametric calculations, zero-variance handling, all-NaN series safety.
- `test_data_quality.py`: Quality score calculations across clean and corrupted datasets.
- `test_outlier_detection.py`: Tukey IQR calculation, zero-IQR edge cases, outlier counts.
- `test_correlation.py`: Pearson & Spearman matrices, positive/negative association isolation.
- `test_insights.py`: Rule-based deterministic insight synthesis.
- `test_integration_controller.py`: End-to-end integration across Scenarios A, B, C, D, and E, plus HTML report verification.

---

## 12. Project Limitations & Ethical Disclaimers

1. **Heuristic Nature of Data Quality:** The Data Quality Score is a heuristic measure of structural hygiene, completeness, and consistency. It does not certify domain-specific ground-truth accuracy.
2. **Correlation vs. Causation:** High Pearson or Spearman coefficients indicate mathematical co-movement, never direct causality.
3. **Outlier Interpretation:** Outliers are statistically unusual observations; they may represent genuine high-value events rather than data collection errors. They should not be deleted without domain review.
4. **Offline Processing:** All data processing occurs entirely in local memory; no user data is persisted to disk or sent to external servers.

---

## 13. Future Enhancements

- Support for Parquet, Feather, and Excel (.xlsx) file formats.
- Interactive user-driven imputation preview (downloading imputed datasets).
- Multi-table relational schema join analysis.
- Automated feature importance ranking using Random Forest / XGBoost baselines.

---

**Developed for the AIML College Open-Ended Micro Project Demonstration.**
