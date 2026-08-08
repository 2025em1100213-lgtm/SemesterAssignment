"""
Data loading utilities.

Use case: BINARY CLASSIFICATION - subscription churn prediction.
Target label: `churn` (1 = customer churned, 0 = retained)

Real data source: IBM/Kaggle "Telco Customer Churn" dataset (7,043 customers,
one row per customer, from a fictional telecom provider). Public dataset,
no PII, widely used as a churn-prediction benchmark:
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Mirror used to fetch it programmatically (identical contents, official IBM
sample-data repo, no Kaggle auth required):
    https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

`load_telco_raw()` below performs the minimal cleaning + renaming needed to
match this project's internal schema (used identically by ingest, train, and
serving). `generate_synthetic_customers()` is kept as a small, fast,
network-free data source used only by the unit-test fixtures in
tests/conftest.py so CI doesn't depend on an external download.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CONTRACT_MAP = {
    "Month-to-month": "month_to_month",
    "One year": "one_year",
    "Two year": "two_year",
}
PAYMENT_MAP = {
    "Electronic check": "electronic_check",
    "Mailed check": "mailed_check",
    "Bank transfer (automatic)": "bank_transfer",
    "Credit card (automatic)": "credit_card",
}
INTERNET_MAP = {
    "DSL": "dsl",
    "Fiber optic": "fiber_optic",
    "No": "none",
}
YES_NO_COLS = [
    ("OnlineSecurity", "online_security"),
    ("OnlineBackup", "online_backup"),
    ("DeviceProtection", "device_protection"),
    ("TechSupport", "tech_support"),
    ("StreamingTV", "streaming_tv"),
    ("StreamingMovies", "streaming_movies"),
    ("MultipleLines", "multiple_lines"),
    ("Partner", "has_partner"),
    ("Dependents", "has_dependents"),
    ("PaperlessBilling", "paperless_billing"),
]

REQUIRED_INTERNAL_COLUMNS = [
    "customer_id", "snapshot_date", "tenure_months", "contract_type",
    "payment_method", "internet_service", "monthly_charges", "total_charges",
    "online_security", "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies", "multiple_lines", "senior_citizen",
    "has_partner", "has_dependents", "paperless_billing", "churn",
]


def load_telco_raw(path: str) -> pd.DataFrame:
    """Load the raw Telco-Customer-Churn.csv and clean/rename it into this
    project's internal schema (REQUIRED_INTERNAL_COLUMNS).

    Cleaning steps (documented per assignment requirement):
      - `TotalCharges` ships as a string with 11 blank values (all belong to
        customers with tenure==0, i.e. brand-new customers who haven't been
        billed yet) -> coerced to numeric, blanks filled with 0.0.
      - Yes/No columns (and the three-way "No internet service"/"No phone
        service" variants) are collapsed to a clean 0/1 flag.
      - Categorical fields (Contract, PaymentMethod, InternetService) are
        mapped to fixed, lowercase, snake_case vocabularies so the encoding
        logic in src/features.py never has to special-case raw label text.
    """
    df = pd.read_csv(path)

    out = pd.DataFrame()
    out["customer_id"] = df["customerID"]
    out["tenure_months"] = df["tenure"].astype(int)
    out["contract_type"] = df["Contract"].map(CONTRACT_MAP)
    out["payment_method"] = df["PaymentMethod"].map(PAYMENT_MAP)
    out["internet_service"] = df["InternetService"].map(INTERNET_MAP)
    out["monthly_charges"] = df["MonthlyCharges"].astype(float)
    out["total_charges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    out["senior_citizen"] = df["SeniorCitizen"].astype(int)

    for raw_col, clean_col in YES_NO_COLS:
        out[clean_col] = df[raw_col].apply(lambda v: 1 if v == "Yes" else 0).astype(int)

    out["churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # snapshot_date is not present in the static Kaggle export (it's a
    # one-time extract, not a dated warehouse table). scripts/ingest.py's
    # `stage_daily_batches()` assigns synthetic dates when simulating the
    # "daily file lands" pattern; give every row a placeholder here so the
    # schema is always complete even if this frame is used directly.
    out["snapshot_date"] = "unassigned"

    return out[REQUIRED_INTERNAL_COLUMNS]


def generate_synthetic_customers(n_rows: int, start_date: str, seed: int = 42) -> pd.DataFrame:
    """Small, fast, network-free synthetic generator used ONLY by unit-test
    fixtures (tests/conftest.py) so CI can train+serve without downloading
    the real dataset. Mirrors REQUIRED_INTERNAL_COLUMNS exactly so the
    feature pipeline sees the same shape it sees from load_telco_raw()."""
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(0, 72, size=n_rows)
    contract_type = rng.choice(["month_to_month", "one_year", "two_year"], size=n_rows, p=[0.55, 0.25, 0.20])
    payment_method = rng.choice(
        ["electronic_check", "mailed_check", "bank_transfer", "credit_card"], size=n_rows
    )
    internet_service = rng.choice(["dsl", "fiber_optic", "none"], size=n_rows, p=[0.35, 0.45, 0.20])

    base_charge = {"dsl": 55, "fiber_optic": 85, "none": 25}
    monthly_charges = np.array([base_charge[p] for p in internet_service]) + rng.normal(0, 10, size=n_rows)
    monthly_charges = np.clip(monthly_charges, 15, None)
    total_charges = monthly_charges * tenure_months + rng.normal(0, 20, size=n_rows)
    total_charges = np.clip(total_charges, 0, None)

    def yn(p):
        return rng.binomial(1, p, size=n_rows)

    online_security = yn(0.3)
    online_backup = yn(0.35)
    device_protection = yn(0.35)
    tech_support = yn(0.3)
    streaming_tv = yn(0.4)
    streaming_movies = yn(0.4)
    multiple_lines = yn(0.4)
    senior_citizen = yn(0.16)
    has_partner = yn(0.48)
    has_dependents = yn(0.3)
    paperless_billing = yn(0.6)

    logit = (
        -1.0
        + 1.4 * (contract_type == "month_to_month")
        - 0.9 * (contract_type == "two_year")
        + 0.01 * (monthly_charges - 60)
        - 0.015 * tenure_months
        - 0.35 * (online_security + tech_support)
        + 0.3 * senior_citizen
        - 0.3 * has_partner
        + rng.normal(0, 0.6, size=n_rows)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n_rows)],
            "snapshot_date": start_date,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "monthly_charges": monthly_charges.round(2),
            "total_charges": total_charges.round(2),
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "multiple_lines": multiple_lines,
            "senior_citizen": senior_citizen,
            "has_partner": has_partner,
            "has_dependents": has_dependents,
            "paperless_billing": paperless_billing,
            "churn": churn,
        }
    )
    return df[REQUIRED_INTERNAL_COLUMNS]


def load_processed(path: str) -> pd.DataFrame:
    """Load the append-only processed training table from disk."""
    return pd.read_csv(path)
