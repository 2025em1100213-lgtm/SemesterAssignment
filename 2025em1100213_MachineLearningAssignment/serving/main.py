"""
Inference service.

Pattern chosen: ONLINE request-response API (not batch, not hybrid).
Rationale (documented fully in design_doc.md, M2 framing):
  - A human (support agent / retention workflow) is waiting on the result
    when a customer opens a support ticket or hits a cancellation flow.
  - Acceptable latency is low (sub-200ms) since it gates a UI action.
  - The use case is naturally per-event (one customer at a time), not a
    nightly aggregate, so batch scoring would add unacceptable staleness.

Run locally:
    uvicorn serving.main:app --reload --port 8000

Endpoints:
    GET  /health   -> service + model status
    POST /predict  -> churn_probability, churn_prediction, model_version
"""
from __future__ import annotations
import json
import os
import sys
import time
import logging

import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.features import FeatureEngineer  # noqa: E402
from serving.schemas import CustomerPayload, PredictionResponse, HealthResponse  # noqa: E402

MODEL_PATH = "models/current_model.joblib"
SCALER_PATH = "models/scaler.joblib"
REGISTRY_PATH = "models/registry.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("serving")

_state = {"model": None, "scaler": None, "version": None, "model_type": None}
fe = FeatureEngineer()


def _load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        log.warning("Model artifacts not found yet. /predict will 503 until training runs.")
        return
    _state["model"] = joblib.load(MODEL_PATH)
    _state["scaler"] = joblib.load(SCALER_PATH)

    version, model_type = "unknown", "unknown"
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            history = json.load(f)
        if history:
            latest = history[-1]
            version = latest.get("version", "unknown")
            model_type = latest.get("promoted_model", "unknown")
    _state["version"] = version
    _state["model_type"] = model_type
    log.info(f"Loaded model version={version} type={model_type}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(title="Churn Prediction Service", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_state["model"] is not None,
        model_version=_state["version"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerPayload):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run scripts/train.py first.")

    start = time.perf_counter()

    # Same FeatureEngineer used in training -> no training/serving skew.
    X = fe.transform_single(payload.model_dump())
    Xs = _state["scaler"].transform(X)
    proba = float(_state["model"].predict_proba(Xs)[0, 1])
    pred = int(proba >= 0.5)

    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info(f"predict | proba={proba:.4f} | pred={pred} | latency_ms={elapsed_ms:.2f}")

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=pred,
        model_version=_state["version"],
        promoted_model_type=_state["model_type"],
        latency_ms=round(elapsed_ms, 3),
    )


@app.get("/")
def root():
    return {"service": "churn-prediction", "docs": "/docs", "health": "/health"}
