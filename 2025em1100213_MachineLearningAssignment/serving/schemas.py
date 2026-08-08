"""Pydantic request/response schemas for the serving API.

These models define the public contract for `/predict` and `/health`.
Validation here protects the feature pipeline from malformed payloads and
keeps API behavior explicit for clients and tests. Field set mirrors the
cleaned Telco Customer Churn schema produced by src/data.py.
"""
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal


class CustomerPayload(BaseModel):
    tenure_months: int = Field(..., ge=0, le=200)
    contract_type: Literal["month_to_month", "one_year", "two_year"]
    payment_method: Literal["electronic_check", "mailed_check", "bank_transfer", "credit_card"]
    internet_service: Literal["none", "dsl", "fiber_optic"]
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    online_security: int = Field(..., ge=0, le=1)
    online_backup: int = Field(..., ge=0, le=1)
    device_protection: int = Field(..., ge=0, le=1)
    tech_support: int = Field(..., ge=0, le=1)
    streaming_tv: int = Field(..., ge=0, le=1)
    streaming_movies: int = Field(..., ge=0, le=1)
    multiple_lines: int = Field(..., ge=0, le=1)
    senior_citizen: int = Field(..., ge=0, le=1)
    has_partner: int = Field(..., ge=0, le=1)
    has_dependents: int = Field(..., ge=0, le=1)
    paperless_billing: int = Field(..., ge=0, le=1)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenure_months": 2,
                "contract_type": "month_to_month",
                "payment_method": "electronic_check",
                "internet_service": "fiber_optic",
                "monthly_charges": 95.5,
                "total_charges": 191.0,
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
        }
    )


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    churn_probability: float
    churn_prediction: int
    model_version: str
    promoted_model_type: str
    latency_ms: float


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: Optional[str] = None
