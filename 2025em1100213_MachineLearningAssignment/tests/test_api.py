"""API contract tests for health and prediction endpoints.

These tests validate response shape, input validation behavior, and a
basic business-direction sanity check for churn scores.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from serving.main import app


@pytest.fixture(scope="module")
def client():
    # Using the context manager form triggers the app's lifespan
    # (startup/shutdown) events, which is what loads the model artifacts.
    with TestClient(app) as c:
        yield c

VALID_PAYLOAD = {
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


def test_health_endpoint_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_response_shape(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert isinstance(body["model_version"], str)
    assert isinstance(body["promoted_model_type"], str)


def test_predict_rejects_invalid_categorical_value(client):
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["contract_type"] = "lifetime"  # not in allowed enum
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_rejects_negative_numeric_value(client):
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["monthly_charges"] = -10
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_high_risk_profile_scores_higher_than_low_risk_profile(client):
    """Sanity check that the model direction makes business sense: a new,
    month-to-month, no-add-on-services customer should score higher churn
    risk than a loyal, fully-subscribed, two-year contract customer."""
    high_risk = dict(VALID_PAYLOAD, contract_type="month_to_month", tenure_months=2,
                      online_security=0, online_backup=0, device_protection=0,
                      tech_support=0, has_partner=0, has_dependents=0)
    low_risk = dict(VALID_PAYLOAD, contract_type="two_year", tenure_months=60,
                     online_security=1, online_backup=1, device_protection=1,
                     tech_support=1, has_partner=1, has_dependents=1)

    p_high = client.post("/predict", json=high_risk).json()["churn_probability"]
    p_low = client.post("/predict", json=low_risk).json()["churn_probability"]
    assert p_high > p_low
