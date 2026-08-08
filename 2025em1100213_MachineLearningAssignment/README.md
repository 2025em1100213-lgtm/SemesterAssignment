# Mini Production ML System — Subscription Churn Prediction

A small but complete production-style ML system: batch ingestion → shared feature
engineering → training pipeline with baseline/candidate evaluation and a promotion
guardrail → a FastAPI serving layer → monitoring (drift/data quality) → a retraining
trigger. Built for the "Design and Build a Mini Production ML System" assignment.

See [`docs/design_doc.md`](docs/design_doc.md) for the full write-up (problem framing,
feature design, model choice, serving pattern, monitoring plan, incident scenario,
trade-offs), [`docs/script_purpose.md`](docs/script_purpose.md) for a script-by-script
purpose index, and [`docs/architecture.png`](docs/architecture.png) for the architecture
diagram.

## Project structure

```
mini_ml_system/
├── .github/workflows/ci.yml   # CI: run tests, build docker image
├── configs/
│   └── train_config.yaml      # data paths, feature list, model params, guardrail thresholds
├── data/
│   ├── incoming/               # drop new daily CSVs here for ingest.py to pick up
│   └── processed/
│       └── training_data.csv   # the append-only "training table"
├── models/                     # trained artifacts + registry.json (created by train.py)
├── artifacts/eval/             # eval reports, feature stats, monitoring/load-test reports
├── data/raw/
│   └── Telco-Customer-Churn.csv # real IBM/Kaggle Telco Customer Churn dataset (7,043 rows)
├── src/
│   ├── data.py                 # real-dataset loader/cleaner + small synthetic generator (tests only)
│   └── features.py             # SHARED feature engineering (training + serving import this)
├── scripts/
│   ├── ingest.py                # batch/micro-batch ingestion
│   ├── train.py                 # training pipeline: load -> split -> train -> eval -> save
│   └── load_test.py             # latency/throughput measurement against a running API
├── serving/
│   ├── main.py                  # FastAPI app: /predict, /health
│   └── schemas.py                # request/response pydantic models
├── monitoring/
│   ├── drift_check.py            # data quality + feature drift checks
│   └── retrain_trigger.py        # retraining decision logic (+ pseudocode)
├── tests/
│   ├── test_features.py          # feature engineering unit tests (incl. skew regression test)
│   └── test_api.py               # API contract tests
├── Dockerfile
├── requirements.txt
└── docs/
    ├── design_doc.md
    └── architecture.png
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Stage the real Telco Customer Churn dataset into simulated daily files
#    and ingest it into data/processed/training_data.csv
python scripts/ingest.py --stage-from-source data/raw/Telco-Customer-Churn.csv --n-batches 10

# 2. Train (produces models/, artifacts/eval/, models/registry.json)
python scripts/train.py

# 3. Serve
uvicorn serving.main:app --reload --port 8000
# Try it: open http://127.0.0.1:8000/docs

# 4. Simulate a new day of data arriving and ingest it
python scripts/ingest.py

# 5. Run the monitoring / drift check against the latest processed data
python monitoring/drift_check.py

# 6. Measure API latency/throughput (with the server from step 3 running)
python scripts/load_test.py --n 200

# 7. Run tests
pytest tests/ -v
```

## Key design decisions (details in the design doc)

- **Training/serving skew** is addressed structurally: `src/features.py` defines one
  `FeatureEngineer` class that both `scripts/train.py` and `serving/main.py` import and
  call identically — there is no second copy of the feature logic anywhere.
- **Promotion guardrail**: a candidate model is only promoted over the current baseline if
  it clears a minimum AUC *and* isn't meaningfully worse than the incumbent — see
  `configs/train_config.yaml` → `promotion_guardrail`.
- **Serving pattern**: online request-response (FastAPI), chosen because a human is
  waiting on the result and per-request feature computation is cheap (~11ms measured
  average latency).
- **Retraining** is signal-driven (staleness, performance drop, drift) via
  `monitoring/retrain_trigger.py`, with the decision logic left as a pure, testable
  function rather than wired into a scheduler.
