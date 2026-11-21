# Sample Datasets for AutoEDA Demonstration

This directory contains 5 synthetic, realistic datasets engineered to demonstrate and test AutoEDA across a wide variety of data distributions, structural flaws, and real-world edge cases.

---

### 1. `scenario_a_housing_clean.csv` (Scenario A: Clean Numerical Data)
- **Domain:** Real estate housing valuation.
- **Observations:** 500 rows, 6 numerical features (`square_feet`, `bedrooms`, `bathrooms`, `year_built`, `lot_size`, `price`).
- **Characteristics:** Complete data (0% missing), no duplicate rows, continuous variables with natural correlations.

### 2. `scenario_b_employee_churn.csv` (Scenario B: Mixed Numerical & Categorical)
- **Domain:** Human Resources workforce analytics.
- **Observations:** 600 rows, 10 mixed features.
- **Characteristics:** Categorical features (`department`, `education`, `gender`, `remote_work`), numerical features (`age`, `years_experience`, `monthly_income`), binary target (`churn`), and an identifier column (`employee_id`).

### 3. `scenario_c_customer_dirty.csv` (Scenario C: Missing Data & Duplicates)
- **Domain:** Customer retail transactions.
- **Observations:** 425 rows, 7 features.
- **Characteristics:**
  - 25 duplicate rows injected.
  - Column `empty_notes_column` is 100% missing values.
  - Column `uninformative_status` is constant ("Active").
  - Column `total_spending` has ~15% missing entries.
  - Tests data hygiene scoring, empty column warnings, and imputation recommendations.

### 4. `scenario_d_financial_risk.csv` (Scenario D: Strong Correlations & Outliers)
- **Domain:** Corporate financial metrics.
- **Observations:** 450 rows, 6 features.
- **Characteristics:**
  - Extreme collinearity (Pearson r > 0.90 between revenue, operating cost, and gross profit).
  - Substantial heavy-tailed skewness and extreme Tukey IQR outliers injected in revenue and cost.
  - Multi-dimensional outliers for testing Isolation Forest.

### 5. `scenario_e_iot_sensors.csv` (Scenario E: Edge Cases & Unusual Column Types)
- **Domain:** Industrial IoT telemetry logs.
- **Observations:** 300 rows, 6 features.
- **Characteristics:**
  - Formatted ISO timestamps (`timestamp`) for datetime inference.
  - String UUIDs (`sensor_uuid`) for identifier detection.
  - Multi-word log strings (`log_message`) for text inference.
  - Boolean flag (`is_alert`).
  - Discrete HTTP status codes (`200, 201, 400, 500`) for low-cardinality discrete code detection.
