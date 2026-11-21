"""
Generate comprehensive synthetic sample datasets for AutoEDA demonstration and testing.
Covers Scenarios A, B, C, D, and E.
"""
from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_datasets"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# 1. Scenario A: Clean Numerical Dataset (Housing Prices)
n_a = 500
square_feet = np.random.normal(1800, 450, n_a).clip(600, 4500)
bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_a, p=[0.05, 0.25, 0.45, 0.20, 0.05])
bathrooms = (bedrooms * 0.75 + np.random.uniform(0.5, 1.5, n_a)).round(1)
year_built = np.random.randint(1960, 2024, size=n_a)
lot_size = square_feet * np.random.uniform(2.0, 5.0, n_a)
price = (
    square_feet * 180 +
    bedrooms * 25000 +
    bathrooms * 18000 +
    (year_built - 1960) * 1200 +
    np.random.normal(0, 15000, n_a)
).round(-2)

df_a = pd.DataFrame({
    "square_feet": square_feet.round(1),
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "year_built": year_built,
    "lot_size": lot_size.round(1),
    "price": price
})
df_a.to_csv(SAMPLE_DIR / "scenario_a_housing_clean.csv", index=False)

# 2. Scenario B: Mixed Numerical and Categorical Dataset (Employee Churn)
n_b = 600
departments = ["Sales", "Engineering", "Marketing", "HR", "Finance", "Support"]
edu_levels = ["Bachelor", "Master", "PhD", "High School"]
genders = ["Male", "Female", "Non-Binary"]
dept = np.random.choice(departments, size=n_b, p=[0.30, 0.35, 0.12, 0.05, 0.08, 0.10])
edu = np.random.choice(edu_levels, size=n_b, p=[0.55, 0.30, 0.08, 0.07])
gender = np.random.choice(genders, size=n_b, p=[0.48, 0.48, 0.04])
age = np.random.randint(22, 62, size=n_b)
experience = (age - 21).clip(0, 40)
monthly_income = (experience * 2800 + np.random.normal(45000, 12000, n_b)).clip(30000, 220000).round(2)
remote_work = np.random.choice(["Yes", "No"], size=n_b, p=[0.65, 0.35])
satisfaction_score = np.random.choice([1, 2, 3, 4, 5], size=n_b, p=[0.1, 0.15, 0.3, 0.3, 0.15])
churn = np.random.choice([0, 1], size=n_b, p=[0.82, 0.18])

df_b = pd.DataFrame({
    "employee_id": [f"EMP_{i+1:04d}" for i in range(n_b)],
    "age": age,
    "gender": gender,
    "department": dept,
    "education": edu,
    "years_experience": experience,
    "monthly_income": monthly_income,
    "remote_work": remote_work,
    "satisfaction_score": satisfaction_score,
    "churn": churn
})
df_b.to_csv(SAMPLE_DIR / "scenario_b_employee_churn.csv", index=False)

# 3. Scenario C: Dataset with Missing Values and Duplicates (Customer Feedback Dirty)
n_c = 400
cust_ids = [f"CUST_{i+1:04d}" for i in range(n_c)]
ratings = np.random.choice([1, 2, 3, 4, 5, np.nan], size=n_c, p=[0.1, 0.15, 0.25, 0.25, 0.15, 0.10])
spending = np.random.exponential(scale=150.0, size=n_c).round(2)
region = np.random.choice(["North", "South", "East", "West", None], size=n_c, p=[0.25, 0.25, 0.2, 0.2, 0.1])
signup_date = pd.date_range("2023-01-01", periods=n_c, freq="D").strftime("%Y-%m-%d").tolist()

df_c = pd.DataFrame({
    "customer_id": cust_ids,
    "signup_date": signup_date,
    "region": region,
    "rating": ratings,
    "total_spending": spending,
    "empty_notes_column": [np.nan] * n_c,  # 100% missing column
    "uninformative_status": ["Active"] * n_c  # Constant column
})
# Inject some missing values into spending
mask_missing_spend = np.random.choice([True, False], size=n_c, p=[0.15, 0.85])
df_c.loc[mask_missing_spend, "total_spending"] = np.nan

# Inject duplicate rows (repeat 25 rows)
df_c = pd.concat([df_c, df_c.iloc[:25]], ignore_index=True)
df_c.to_csv(SAMPLE_DIR / "scenario_c_customer_dirty.csv", index=False)

# 4. Scenario D: Strong Correlations and Extreme Outliers (Financial Risk)
n_d = 450
revenue = np.random.lognormal(mean=10.5, sigma=1.0, size=n_d)
# Highly correlated features (revenue & cost & profit)
operating_cost = revenue * 0.72 + np.random.normal(0, 5000, n_d)
gross_profit = revenue - operating_cost
tax_payable = gross_profit * 0.25 + np.random.normal(0, 800, n_d)
debt_ratio = np.random.beta(a=2, b=5, size=n_d).round(3)

# Inject extreme outliers in revenue and cost
outlier_indices = np.random.choice(n_d, size=15, replace=False)
revenue[outlier_indices] *= 8.0
operating_cost[outlier_indices] *= 7.5

df_d = pd.DataFrame({
    "company_ticker": [f"CORP_{i+1:03d}" for i in range(n_d)],
    "annual_revenue": revenue.round(2),
    "operating_cost": operating_cost.round(2),
    "gross_profit": gross_profit.round(2),
    "tax_payable": tax_payable.round(2),
    "debt_ratio": debt_ratio
})
df_d.to_csv(SAMPLE_DIR / "scenario_d_financial_risk.csv", index=False)

# 5. Scenario E: Edge Cases & Unusual Column Types (IoT Sensor Logs)
n_e = 300
timestamps = pd.date_range("2024-03-01 00:00:00", periods=n_e, freq="15min").strftime("%Y-%m-%d %H:%M:%S")
sensor_uuids = [f"uuid-00{i}-sensor" for i in range(n_e)]
temperature = np.random.normal(24.5, 3.2, size=n_e).round(2)
is_alert_triggered = np.random.choice([True, False], size=n_e, p=[0.08, 0.92])
error_log_messages = np.random.choice([
    "Normal operation cycle completed without warnings",
    "Warning: Temperature threshold exceeded minor boundary",
    "Packet loss detected on wireless sensor gateway port 8080",
    "Sensor recalibration routine initiated by telemetry service"
], size=n_e)
status_code = np.random.choice([200, 201, 400, 500], size=n_e, p=[0.75, 0.15, 0.07, 0.03])

df_e = pd.DataFrame({
    "sensor_uuid": sensor_uuids,
    "timestamp": timestamps,
    "ambient_temperature": temperature,
    "is_alert": is_alert_triggered,
    "log_message": error_log_messages,
    "http_status_code": status_code
})
df_e.to_csv(SAMPLE_DIR / "scenario_e_iot_sensors.csv", index=False)

print("Sample datasets generated successfully!")
