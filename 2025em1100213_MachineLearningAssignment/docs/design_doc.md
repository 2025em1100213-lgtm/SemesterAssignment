# Design Document: Mini Production ML System — Subscription Churn Prediction

**Code repository:** https://github.com/2025em1100213-lgtm/Assignments/tree/main/2025em1100213_MachineLearningAssignment

## 1. Problem Definition and Metrics

**Use case:** Binary classification. The system predicts whether a telecom
subscription customer will churn (cancel), given their current billing,
contract, and subscribed-services profile. Target label: `churn` in {0, 1},
where 1 means the customer churned.

**Why this use case:** Churn prediction is a canonical production ML
problem because the prediction feeds a real business action (a retention
offer, a support outreach) rather than being consumed passively, forcing
real decisions about latency, staleness, and monitoring -- exactly the
concepts this assignment exercises.

**Primary metrics:** ROC AUC and accuracy.

- **ROC AUC** is the primary model-selection metric because churn classes
  are imbalanced (26.5% positive in the real data) and AUC is
  threshold-independent -- it measures the model's ability to *rank*
  at-risk customers above safe ones, which is what a retention team
  actually needs (they work down a ranked list, not a hard 0.5 cutoff).
- **Accuracy** is reported alongside AUC as a sanity check but is *not*
  the promotion metric, since always predicting "no churn" already scores
  ~73.5% accuracy without being useful.
- In a real deployment we would add **precision@k** (precision among the
  top-k highest-risk customers) since retention budget is limited and the
  business acts on a ranked shortlist -- noted as future work.

## 2. Data and Feature Design

**Data source:** The public **IBM/Kaggle Telco Customer Churn** dataset
(7,043 real customer records, one row per customer, from a fictional
telecom provider) --
https://www.kaggle.com/datasets/blastchar/telco-customer-churn. Fetched
programmatically from IBM's official sample-data mirror on GitHub
(identical contents, no Kaggle auth required):
`raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/.../Telco-Customer-Churn.csv`.
This replaces the synthetic generator used in the first draft of this
project -- every number in this document now reflects a real model trained
on real data.

**Target label:** `Churn` (Yes/No), mapped to `churn` in {0, 1}. Churn rate
in the full dataset is **26.5%** -- a realistic, moderately imbalanced
split typical of subscription businesses.

**Cleaning / assumptions** (`src/data.py: load_telco_raw`):
- `TotalCharges` ships as a string with 11 blank values -- all belong to
  brand-new customers with `tenure == 0` who haven't been billed yet --
  coerced to numeric and filled with 0.0.
- Yes/No and three-way ("No internet service" / "No phone service")
  columns are collapsed to clean 0/1 flags.
- `Contract`, `PaymentMethod`, `InternetService` are mapped to fixed,
  lowercase snake_case vocabularies so encoding logic never special-cases
  raw label text.

**Engineered features** (`src/features.py`, 8 derived columns on top of 8
raw passthrough columns -- 5+ required):

1. `charges_per_tenure_month` -- total spend normalized by tenure (ratio)
2. `num_addon_services` -- count of subscribed add-ons (OnlineSecurity,
   OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
   StreamingMovies) -- aggregation feature
3. `is_high_value` -- threshold encoding of monthly spend (> $70)
4. `tenure_bucket` -- binned tenure (new / established / loyal segments)
5. `is_new_high_risk` -- interaction feature: month-to-month contract AND
   tenure <= 12 months, the single strongest churn segment in this data
6-8. Fixed-vocabulary encodings for `contract_type`, `payment_method`,
   `internet_service`

**Offline vs. online:** All 16 model features are cheap, stateless
functions of fields already on a single customer record -- all computed
**online**, per-request, in the serving path. None require a historical
join or rolling aggregate today; `src/features.py`'s module docstring
flags that a true rolling-window feature (e.g. "90-day support ticket
trend") would move to an **offline feature table + online lookup**
pattern instead.

**Training/serving skew:** The most safety-critical design decision here.
`src/features.py` defines a single `FeatureEngineer` class with a
stateless `.transform()` method; both `scripts/train.py` (batch) and
`serving/main.py` (online `/predict`) import this exact class and call
the exact same method -- there is no second implementation anywhere.
`tests/test_features.py` includes a regression test
(`test_transform_single_matches_batch_transform`) asserting that
transforming one row via the single-record helper produces bit-identical
output to transforming it as part of a batch. The transform is stateless
(scaling is a separate `StandardScaler` fit only on the train split and
persisted alongside the model), so serving-time behavior can never
silently diverge from training-time behavior.

**Data pipeline:** Since the Kaggle export is a single static file (not a
dated warehouse table), `scripts/ingest.py: stage_daily_batches()`
deterministically shuffles and splits the cleaned 7,043 rows into 10
simulated daily files, mirroring what a real "one CSV lands per day"
feed looks like. `ingest()` then performs the actual pipeline step
required by the brief: it scans `data/incoming/` for new files, validates
schema (required columns, >50% null-rate rejection), appends/deduplicates
into `data/processed/training_data.csv`, and logs row counts and dates to
`logs/ingestion.log`. Files move to `data/incoming/_ingested/` after
processing so re-runs don't double-count. This run ingested **7,043 rows
across 10 files** with zero validation failures.

## 3. Model Choice and Evaluation

`scripts/train.py` implements a repeatable pipeline: load -> stratified
train/val/test split (70/15/15) -> fit `StandardScaler` on train only ->
train **baseline** (logistic regression -- simple, fast, interpretable
coefficients, a reasonable first production cut) and **candidate**
(gradient boosting -- captures non-linear interactions, e.g. between
contract type and add-on count) -> evaluate both on validation and
held-out test sets -> save artifacts.

**Promotion guardrail:** candidate is promoted only if
`candidate_val_auc >= 0.80 AND (baseline_val_auc - candidate_val_auc) <= 0.01`.

**Actual results on the real Telco data** (run `v20260807_075729`):

| Model | Val AUC | Val Acc | Test AUC | Test Acc |
|---|---|---|---|---|
| baseline (logistic regression) | 0.8532 | 0.8030 | 0.8148 | 0.7796 |
| candidate (gradient boosting) | 0.8506 | 0.7992 | 0.8248 | 0.7843 |

The candidate cleared the guardrail (0.8506 >= 0.80, and the 0.0026-point
gap to baseline is well under the 0.01 tolerance) and **was promoted**.
Interestingly the candidate's validation AUC is marginally lower than the
baseline's but its held-out test AUC is higher (0.8248 vs 0.8148) --
exactly the kind of small, within-noise difference the guardrail is
designed to tolerate rather than over-fit a promotion decision to. Both
models comfortably exceed the 0.80 floor on real (non-synthetic) data,
unlike the earlier synthetic-data draft where neither model cleared it --
real telecom churn signal turns out to be stronger than the deliberately
noised synthetic generator produced.

Artifacts saved per run: `models/model_baseline.joblib`,
`models/model_candidate.joblib`, `models/scaler.joblib`,
`models/current_model.joblib` (the promoted one, loaded by serving), an
append-only `models/registry.json` (version, metrics, decision,
timestamp), and JSON + Markdown evaluation reports under
`artifacts/eval/`.

## 4. Serving and Inference Pattern

**Pattern chosen: online request-response (FastAPI).** Using the M2
framing:

- **Is a human waiting?** Yes -- a support agent or retention workflow
  triggered by a cancellation click needs the score synchronously.
- **What latency is acceptable?** Sub-second, ideally sub-100ms, since it
  gates a UI interaction. Measured latency below clears this by a wide
  margin.
- **Is the use case naturally batch or streaming?** Naturally per-event --
  one customer scored at an unpredictable moment (whenever they contact
  support or billing), not on a fixed nightly schedule.

A pure batch score-everyone-nightly pattern was rejected because it would
leave scores up to 24 hours stale exactly when they matter most. A hybrid
(precompute + online lookup) was considered but rejected as unnecessary
complexity given how cheap the feature transform and inference are -- the
hybrid pattern earns its complexity when feature computation is
expensive, which is not the case here.

**API:** `POST /predict` accepts a Pydantic-validated JSON payload
(matching the cleaned Telco schema) and returns `churn_probability`,
`churn_prediction`, and `model_version`. `GET /health` reports whether a
model is loaded, for readiness probes.

**Latency/throughput (real measurement):** `scripts/load_test.py` sent
200 sequential requests to a running local instance:
**avg 8.43 ms, p50 8.23 ms, p95 9.47 ms, p99 12.05 ms, 0 errors,
~118.7 req/s single-threaded** -- comfortably within the sub-100ms budget.
Production would re-measure this under concurrent load with a dedicated
tool (k6/locust) behind a multi-worker ASGI server.

**Containerization:** a `Dockerfile` is included; it installs
dependencies, copies `src/`, `serving/`, `configs/`, and `models/`, and
runs `uvicorn`. It does **not** train a model at build/start time --
training is a separate offline job whose output is baked into or mounted
onto the image.

## 5. Data Pipeline and Retraining Strategy

New data flows in through `scripts/ingest.py` on a batch cadence
(simulating a daily job). Retraining is **not** wired to a scheduler in
this assignment, but the decision logic is fully implemented in
`monitoring/retrain_trigger.py` as a pure function
(`should_retrain(signals) -> RetrainDecision`), unit-testable and ready to
drop into a cron job or Airflow DAG. Three signals, combined with OR
logic -- any one alone triggers a retrain:

1. **Staleness** -- >= 14 days since the current promoted model was trained.
2. **Performance regression** -- AUC on recently labeled feedback has
   dropped >= 0.03 from the AUC recorded at promotion time.
3. **Drift** -- the monitoring drift check (below) flagged a feature shift.

Pseudocode for this logic lives directly in `monitoring/retrain_trigger.py`.

## 6. Monitoring Plan and Basic Alerts

Three tiers of metrics, each mapped to who would consume them:

- **Infra** (on-call/SRE dashboard): request latency (avg, p95), error
  rate, request volume. Alert: p95 > 300ms for 5 min, or error rate > 2%
  for 5 min.
- **Data/feature** (ML engineer dashboard): rows ingested per day, null
  rates per required column, mean/std of each engineered feature vs. the
  training reference. Alert: any required column's null rate > 5%, or any
  feature's mean shifts > 20% relative to its training-time mean.
- **Model/business** (product/retention dashboard): AUC on recent labeled
  feedback, predicted positive rate over time, retention-offer conversion
  among flagged customers. Alert: AUC on feedback drops below the
  guardrail floor, or predicted positive rate moves > 2x outside its
  historical range.

**Implemented lightweight check:** `monitoring/drift_check.py` runs a
data-quality pass (missing required columns, null rates, out-of-range
values) and a drift pass comparing each engineered feature's mean on a
"recent batch" against the mean recorded at training time
(`artifacts/eval/feature_stats.json`), flagging any feature whose mean
moved > 20%. Run against the full ingested table: **0 data-quality
issues, 0 drift warnings** (expected, since the "recent batch" here is
the same real data the model was just trained on). Both checks log
warnings and write a JSON summary to
`artifacts/eval/monitoring_report.json`.

## 7. Incident Scenario

**Scenario: upstream billing schema change.** Suppose the billing
system's export job is updated and `MonthlyCharges` starts arriving in
cents instead of dollars, without notice to the ML team.

**Detection:** `monitoring/drift_check.py`, run against the next ingested
batch, would flag two things: the data-quality check would likely catch
`monthly_charges` values wildly out of the expected range (tens of
thousands instead of tens/hundreds), and even if that range check missed
it, the drift check would show `monthly_charges` and
`charges_per_tenure_month` means shifted ~100x relative to the training
reference -- far past the 20% threshold. Features derived from `tenure` or
service flags staying stable while only charge-derived features spike is
itself a useful diagnostic signal that this is upstream data corruption,
not genuine model decay.

**Response:** (1) Immediately quarantine the corrupted batch -- stop
feeding it into the "recent labeled feedback" AUC calculation and the
training data table so it can't silently poison the next training run.
(2) Do **not** retrain on the corrupted data even though the retrain
trigger's drift signal would fire -- a human confirms this is a data bug,
not real-world drift, before any retraining happens. (3) Patch the
ingestion validation to reject or auto-convert malformed units going
forward. (4) Once the upstream fix is confirmed, re-ingest the corrected
batch and resume normal monitoring. This is exactly why the drift check
is a *human-reviewed alert*, not an automatic retrain trigger by itself.

## 8. Key Trade-offs, Limitations, and Future Work

**Trade-offs made for a 2-week assignment scope:**
- The Kaggle export is a one-time static snapshot with no timestamp
  column, so "daily ingestion" is simulated by deterministically
  splitting it (`stage_daily_batches`) rather than pulling from a truly
  dated source -- a real deployment would read from an actual dated
  warehouse table or event stream.
- A file-based model registry (`registry.json`) instead of a real model
  registry service (MLflow, SageMaker Model Registry) -- sufficient to
  demonstrate versioning/promotion logic, but wouldn't scale to multiple
  concurrent training jobs or multi-model serving.
- Synchronous, single-process feature computation -- fine at this
  latency/throughput, but would need caching or a real online feature
  store if any feature required an expensive join.
- The drift check uses simple mean/std shift rather than a statistical
  test (KS-test, population stability index) -- easier to reason about and
  log, at the cost of being less sensitive to shape changes that don't
  move the mean much.

**Limitations:** no authentication/rate-limiting on the API; no A/B
testing or shadow deployment support for comparing candidate vs. baseline
on live traffic before full promotion; the retraining trigger is a pure
function, not wired to an actual scheduler; monitoring alerts are logged,
not sent to a real paging system; the dataset itself has no true
timestamp, so genuine temporal drift can't yet be measured against this
data (only structurally injected drift, as in the incident scenario).

**Future work:** precision@k as a promotion metric; a real feature store
for any future rolling-window features (e.g. actual support-ticket
history, which this dataset doesn't include); shadow-mode candidate
evaluation on live traffic before promotion; wiring
`retrain_trigger.py` into a scheduled job; structured request logging
with a request ID for full traceability from a `/predict` call back to
the model version and feature values that produced it.

## 9. Rubric Coverage Checklist (A-D)

This section maps each assignment requirement directly to implemented
code or docs.

### A. Data and Feature Engineering

- **Data source, target label, assumptions/cleaning:** Sections 1-2;
  real public dataset, cleaning steps documented in `src/data.py`.
- **At least 5 non-trivial features:** `src/features.py` implements 8
  derived features (ratio, aggregation, threshold, binning, interaction,
  and three fixed-vocabulary encodings).
- **Offline vs online feature awareness:** documented in
  `src/features.py` module header and Section 2.
- **Training-serving skew mitigation:** single shared `FeatureEngineer`
  used by both `scripts/train.py` and `serving/main.py`; validated by
  feature tests.
- **Data pipeline ingestion step:** `scripts/ingest.py` stages the real
  dataset into simulated daily CSVs, validates schema, appends/merges
  into `data/processed/training_data.csv`, and logs file/row/date
  metadata (7,043 rows across 10 files, this run).

### B. Model Training and Offline Evaluation

- **Repeatable training pipeline:** `scripts/train.py` performs
  load -> split -> train -> evaluate -> promotion decision -> artifact save.
- **Metrics and justification:** Section 1 explains why ROC AUC is
  primary and accuracy secondary for churn ranking.
- **Baseline vs candidate harness:** trains logistic regression and
  gradient boosting, compares metrics, applies the promotion guardrail.
  Real results in Section 3 (candidate promoted, test AUC 0.8248).
- **Guardrail rule:** configured in `configs/train_config.yaml` under
  `promotion_guardrail`.
- **Saved artifacts:** models in `models/`; evaluation outputs in
  `artifacts/eval/`.

### C. Serving and Inference Pattern

- **Minimal API endpoint:** `serving/main.py` exposes `/predict` and
  `/health` via FastAPI.
- **Versioned response:** `/predict` returns prediction plus
  model version/type.
- **Inference pattern choice:** Section 4 explains why online
  request-response fits this use case.
- **Latency/throughput measurement:** `scripts/load_test.py` reports
  real avg/p50/p95/p99 latency and throughput (Section 4).
- **Containerization:** Docker support via `Dockerfile`.

### D. Monitoring, Data Quality, and Retraining

- **Monitoring plan (infra + data + model/business):** Section 6 --
  dashboards, thresholds, intended owners.
- **Lightweight drift/data quality code:** `monitoring/drift_check.py`
  performs null/range and relative mean-shift drift checks, writing
  `artifacts/eval/monitoring_report.json` (0 issues on this run).
- **Retraining trigger logic:** `monitoring/retrain_trigger.py` defines
  signal-based logic (staleness, performance drop, drift) with pseudocode.
- **Incident scenario:** Section 7 details a schema-change failure mode,
  detection path, and response actions.
