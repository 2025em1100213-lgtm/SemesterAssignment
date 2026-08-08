"""
Shared feature engineering module.

CRITICAL DESIGN DECISION (training/serving skew):
This is the SINGLE source of truth for turning raw customer fields into
model-ready features. Both `scripts/train.py` (offline/batch) and
`serving/main.py` (online) import `FeatureEngineer` from here and call
the exact same `.transform()` method. There is no second implementation
of this logic anywhere else in the repo.

    training pipeline  ---\\
                            >--- FeatureEngineer.transform() ---> same columns, same math
    online /predict    ---/

This mirrors the "shared preprocessing module" pattern described in the
assignment brief (as opposed to a duplicated online feature service).

Offline vs online split (documented per assignment requirement):
  - OFFLINE (would live in a feature store in prod, refreshed daily):
    none currently required -- tenure_bucket and num_addon_services are
    derived from fields (tenure, service flags) that only change on a
    billing-cycle cadence, so if this were high-QPS we would precompute
    them nightly and look them up instead of recomputing per request.
  - ONLINE (cheap, computed per-request from the raw payload, safe to
    compute synchronously in the API): every feature below, because all
    are simple arithmetic/encoding operations on fields already present
    on a single customer record -- no historical join or rolling
    aggregate is required for any of them today.

Raw fields are the cleaned Telco Customer Churn columns produced by
src/data.py (load_telco_raw / generate_synthetic_customers) -- see that
module's docstring for the public dataset source and cleaning steps.
"""
from __future__ import annotations
from typing import List
import pandas as pd

CONTRACT_ORDER = ["month_to_month", "one_year", "two_year"]
PAYMENT_METHODS = ["electronic_check", "mailed_check", "bank_transfer", "credit_card"]
INTERNET_SERVICES = ["none", "dsl", "fiber_optic"]

ADDON_SERVICE_COLUMNS = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]

# Final, ordered list of columns the model is trained/served on.
FEATURE_COLUMNS: List[str] = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "senior_citizen",
    "has_partner",
    "has_dependents",
    "paperless_billing",
    "multiple_lines",
    "charges_per_tenure_month",   # 1. ratio feature
    "num_addon_services",         # 2. aggregation (count) feature
    "is_high_value",              # 3. threshold encoding
    "tenure_bucket",              # 4. binned feature
    "is_new_high_risk",           # 5. interaction/composite feature
    "contract_type_encoded",      # 6. ordinal encoding
    "payment_method_encoded",     # 7. categorical encoding
    "internet_service_encoded",   # 8. categorical encoding
]


class FeatureEngineer:
    """Stateless, deterministic feature transform.

    Deliberately has NO fit step that depends on the training set
    distribution (no target leakage, no scaler fit on train-only stats)
    so that calling .transform() on a single request at serving time
    produces bit-identical output to calling it on the training frame.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Guard against divide-by-zero for brand-new customers (tenure=0).
        safe_tenure = out["tenure_months"].clip(lower=1)

        # --- non-trivial feature 1: ratio ---
        out["charges_per_tenure_month"] = (out["total_charges"] / safe_tenure).round(3)

        # --- non-trivial feature 2: aggregation (count of add-on services) ---
        out["num_addon_services"] = out[ADDON_SERVICE_COLUMNS].sum(axis=1).astype(int)

        # --- non-trivial feature 3: threshold / encoding ---
        out["is_high_value"] = (out["monthly_charges"] > 70).astype(int)

        # --- non-trivial feature 4: binning ---
        out["tenure_bucket"] = pd.cut(
            out["tenure_months"],
            bins=[-1, 6, 12, 24, 48, 1000],
            labels=[0, 1, 2, 3, 4],
        ).astype(int)

        # --- non-trivial feature 5: interaction / composite risk flag ---
        # New customers on a month-to-month contract are the single
        # strongest churn segment in telecom churn data -- flag it
        # explicitly rather than relying on the model to rediscover the
        # interaction between two separate raw columns.
        out["is_new_high_risk"] = (
            (out["contract_type"] == "month_to_month") & (out["tenure_months"] <= 12)
        ).astype(int)

        # --- categorical encodings (fixed vocabularies -> no skew risk) ---
        out["contract_type_encoded"] = out["contract_type"].apply(
            lambda x: CONTRACT_ORDER.index(x) if x in CONTRACT_ORDER else -1
        )
        out["payment_method_encoded"] = out["payment_method"].apply(
            lambda x: PAYMENT_METHODS.index(x) if x in PAYMENT_METHODS else -1
        )
        out["internet_service_encoded"] = out["internet_service"].apply(
            lambda x: INTERNET_SERVICES.index(x) if x in INTERNET_SERVICES else -1
        )

        return out[FEATURE_COLUMNS]

    def transform_single(self, payload: dict) -> pd.DataFrame:
        """Convenience wrapper used by the FastAPI /predict endpoint so a
        single JSON request goes through the identical code path as a
        batch DataFrame."""
        return self.transform(pd.DataFrame([payload]))
