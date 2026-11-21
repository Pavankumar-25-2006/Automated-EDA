"""
Configuration settings for AutoEDA application.
Centralized parameters for thresholds, file limits, scoring weights, and visualization constraints.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATASETS_DIR = BASE_DIR / "sample_datasets"

# Upload and Ingestion Limits
MAX_UPLOAD_SIZE_MB = 100
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = [".csv"]
SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
COMMON_DELIMITERS = [",", ";", "\t", "|"]

# Type Detection Thresholds
MAX_CARDINALITY_FOR_CATEGORICAL = 50
MAX_CATEGORICAL_UNIQUE_RATIO = 0.05
MAX_TEXT_AVG_WORD_COUNT = 3.0
MAX_SAMPLE_ROWS_FOR_TYPE_INFERENCE = 1000

# Statistical Analysis Settings
IQR_MULTIPLIER = 1.5
CORRELATION_STRONG_THRESHOLD = 0.70
CORRELATION_MODERATE_THRESHOLD = 0.40
CORRELATION_MAX_VARS_HEATMAP = 25
TOP_CORRELATED_PAIRS_COUNT = 10
MAX_CATEGORIES_DISPLAY = 15

# Data Quality Scoring Weights (Transparent formula sum = 1.0)
QUALITY_WEIGHT_COMPLETENESS = 0.50
QUALITY_WEIGHT_UNIQUENESS = 0.25
QUALITY_WEIGHT_CONSISTENCY = 0.25

# Visualization Settings
MAX_DISTRIBUTION_PLOTS = 12
MAX_CATEGORICAL_PLOTS = 10
PLOT_COLOR_PRIMARY = "#1f77b4"
PLOT_COLOR_SECONDARY = "#ff7f0e"
PLOT_COLOR_ACCENT = "#2ca02c"
PLOT_THEME = "plotly_white"

# Anomaly Detection Settings (Isolation Forest)
ANOMALY_CONTAMINATION = 0.05
ANOMALY_RANDOM_STATE = 42

# Report Settings
APP_NAME = "AutoEDA"
APP_SUBTITLE = "Advanced Automated Exploratory Data Analysis System"
APP_VERSION = "2.0.0"
