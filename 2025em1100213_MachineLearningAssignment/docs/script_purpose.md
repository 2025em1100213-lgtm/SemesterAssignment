# Script Purpose Index

This quick reference documents the purpose of each executable script and monitoring utility.

## Training and Data Flow

- `scripts/train.py`
  - Purpose: repeatable training pipeline (`load -> split -> train -> evaluate -> promote -> save artifacts`).
  - Outputs: `models/` artifacts, `models/registry.json`, `artifacts/eval/eval_report_*.json|.md`, `artifacts/eval/feature_stats.json`.

- `scripts/ingest.py`
  - Purpose: batch/micro-batch ingestion from `data/incoming/*.csv` into `data/processed/training_data.csv`.
  - Behavior: validates minimal schema, appends/merges, deduplicates, logs file and row counts, moves ingested files to `data/incoming/_ingested/`.

- `src/data.py`
  - Purpose: generate synthetic churn data and load processed training data.
  - Used by: quickstart/demo generation and test/training setup.

- `src/features.py`
  - Purpose: shared feature engineering for both training and serving to avoid training-serving skew.
  - Used by: `scripts/train.py`, `serving/main.py`, and monitoring checks.

## Serving and Performance

- `serving/main.py`
  - Purpose: FastAPI inference service with `/predict` and `/health` endpoints.
  - Behavior: loads promoted model/scaler, computes features, returns prediction and model version metadata.

- `serving/schemas.py`
  - Purpose: request/response contracts and validation rules for serving endpoints.

- `scripts/load_test.py`
  - Purpose: basic latency and throughput measurements against `/predict`.
  - Outputs: console report and `artifacts/eval/load_test_report.json`.

## Monitoring and Retraining

- `monitoring/drift_check.py`
  - Purpose: lightweight data quality and feature drift checks on recent data.
  - Outputs: logs + `artifacts/eval/monitoring_report.json`.

- `monitoring/retrain_trigger.py`
  - Purpose: retraining decision logic from staleness/performance/drift signals.
  - Includes: executable decision function and pseudocode for scheduler integration.

## Tests

- `tests/conftest.py`
  - Purpose: session fixture that generates synthetic data and trains once for stable test setup.

- `tests/test_features.py`
  - Purpose: validates feature columns, deterministic transforms, and skew protections.

- `tests/test_api.py`
  - Purpose: validates API contract, input validation, and risk-direction sanity behavior.

## How to run each script

Run all commands from the project root (`mini_ml_system/`).

Activate virtual environment (PowerShell):

  .\.venv\Scripts\Activate.ps1

Train models and write artifacts:

  .\.venv\Scripts\python.exe scripts\train.py

Ingest new CSV data from `data/incoming/`:

  .\.venv\Scripts\python.exe scripts\ingest.py

Generate a demo daily CSV and ingest it:

  .\.venv\Scripts\python.exe scripts\ingest.py --generate-demo-day 2024-02-01 --n-rows 300

Start serving API (FastAPI):

  .\.venv\Scripts\python.exe -m uvicorn serving.main:app --reload --port 8000

Run load test against the API:

  .\.venv\Scripts\python.exe scripts\load_test.py --n 200

Run drift and data-quality check:

  .\.venv\Scripts\python.exe monitoring\drift_check.py

Run retraining trigger logic demo:

  .\.venv\Scripts\python.exe monitoring\retrain_trigger.py

Run test suite:

  .\.venv\Scripts\python.exe -m pytest tests\ -v

Send a single prediction request from terminal:

  $body = @{
    tenure_months = 24
    monthly_charges = 85.5
    total_charges = 2052.0
    contract_type = "month_to_month"
    payment_method = "electronic_check"
    internet_service = "fiber_optic"
    online_security = 0
    online_backup = 0
    device_protection = 0
    tech_support = 0
    streaming_tv = 1
    streaming_movies = 1
    multiple_lines = 1
    senior_citizen = 0
    has_partner = 0
    has_dependents = 0
    paperless_billing = 1
  } | ConvertTo-Json

  Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 10
