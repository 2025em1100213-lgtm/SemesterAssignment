"""Feature engineering tests.

Focus areas: output schema stability, training-serving consistency,
numerical safety checks, and deterministic encodings.
"""

import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.features import FeatureEngineer, FEATURE_COLUMNS


def make_raw_row(**overrides):
    row = {
        "tenure_months": 8,
        "contract_type": "month_to_month",
        "payment_method": "electronic_check",
        "internet_service": "fiber_optic",
        "monthly_charges": 95.5,
        "total_charges": 764.0,
        "online_security": 0,
        "online_backup": 0,
        "device_protection": 0,
        "tech_support": 0,
        "streaming_tv": 1,
        "streaming_movies": 1,
        "multiple_lines": 1,
        "senior_citizen": 0,
        "has_partner": 0,
        "has_dependents": 0,
        "paperless_billing": 1,
    }
    row.update(overrides)
    return row


def test_transform_returns_all_expected_columns():
    fe = FeatureEngineer()
    df = pd.DataFrame([make_raw_row()])
    out = fe.transform(df)
    assert list(out.columns) == FEATURE_COLUMNS
    assert len(out) == 1


def test_transform_single_matches_batch_transform():
    """Guards against training/serving skew: transforming one row via the
    single-record helper must equal transforming it as part of a batch."""
    fe = FeatureEngineer()
    row = make_raw_row()

    batch_out = fe.transform(pd.DataFrame([row])).iloc[0]
    single_out = fe.transform_single(row).iloc[0]

    for col in FEATURE_COLUMNS:
        assert batch_out[col] == single_out[col], f"mismatch on {col}"


def test_no_divide_by_zero_for_new_customer():
    fe = FeatureEngineer()
    row = make_raw_row(tenure_months=0, total_charges=0.0)
    out = fe.transform(pd.DataFrame([row]))
    assert out["charges_per_tenure_month"].iloc[0] == 0.0
    assert not out.isnull().any().any()


def test_is_high_value_threshold():
    fe = FeatureEngineer()
    cheap = fe.transform(pd.DataFrame([make_raw_row(monthly_charges=30)]))
    pricey = fe.transform(pd.DataFrame([make_raw_row(monthly_charges=150)]))
    assert cheap["is_high_value"].iloc[0] == 0
    assert pricey["is_high_value"].iloc[0] == 1


def test_categorical_encodings_are_deterministic():
    fe = FeatureEngineer()
    out1 = fe.transform(pd.DataFrame([make_raw_row(contract_type="two_year")]))
    out2 = fe.transform(pd.DataFrame([make_raw_row(contract_type="two_year")]))
    assert out1["contract_type_encoded"].iloc[0] == out2["contract_type_encoded"].iloc[0]


def test_num_addon_services_counts_correctly():
    fe = FeatureEngineer()
    row = make_raw_row(online_security=1, online_backup=1, device_protection=0,
                        tech_support=0, streaming_tv=1, streaming_movies=0)
    out = fe.transform(pd.DataFrame([row]))
    assert out["num_addon_services"].iloc[0] == 3


def test_is_new_high_risk_flag():
    fe = FeatureEngineer()
    new_mtm = fe.transform(pd.DataFrame([make_raw_row(contract_type="month_to_month", tenure_months=3)]))
    loyal = fe.transform(pd.DataFrame([make_raw_row(contract_type="two_year", tenure_months=3)]))
    old_mtm = fe.transform(pd.DataFrame([make_raw_row(contract_type="month_to_month", tenure_months=40)]))
    assert new_mtm["is_new_high_risk"].iloc[0] == 1
    assert loyal["is_new_high_risk"].iloc[0] == 0
    assert old_mtm["is_new_high_risk"].iloc[0] == 0
