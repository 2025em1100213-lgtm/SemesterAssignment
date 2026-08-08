"""
Retraining trigger logic.

Not wired to a scheduler in this assignment - this defines the decision
function that WOULD be called by a daily cron / Airflow DAG in production.

Three signals combine with OR logic: any one of them alone is enough to
warrant a retrain.

  1. STALENESS   -> we have accumulated >= N days of new labeled data since
                     the current model was trained.
  2. PERFORMANCE -> AUC computed on recent labeled feedback (customers whose
                     actual churn/retain outcome is now known) has dropped by
                     more than X points vs. the value recorded at promotion time.
  3. DRIFT       -> monitoring/drift_check.py flagged a feature drift warning
                     on the most recent batch.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class RetrainSignals:
    days_since_last_training: int
    current_model_promotion_auc: float
    recent_labeled_auc: float | None      # None if not enough labeled feedback yet
    drift_detected: bool


@dataclass
class RetrainDecision:
    should_retrain: bool
    reasons: list[str]


# --- Thresholds (would live in configs/monitoring_config.yaml in production) ---
MAX_DAYS_WITHOUT_RETRAIN = 14
MAX_ALLOWED_AUC_DROP = 0.03


def should_retrain(signals: RetrainSignals) -> RetrainDecision:
    reasons = []

    # Signal 1: staleness
    if signals.days_since_last_training >= MAX_DAYS_WITHOUT_RETRAIN:
        reasons.append(
            f"staleness: {signals.days_since_last_training} days since last training "
            f"(threshold={MAX_DAYS_WITHOUT_RETRAIN})"
        )

    # Signal 2: performance regression on recent labeled feedback
    if signals.recent_labeled_auc is not None:
        auc_drop = signals.current_model_promotion_auc - signals.recent_labeled_auc
        if auc_drop >= MAX_ALLOWED_AUC_DROP:
            reasons.append(
                f"performance drop: AUC fell from {signals.current_model_promotion_auc:.3f} "
                f"to {signals.recent_labeled_auc:.3f} (drop={auc_drop:.3f}, "
                f"threshold={MAX_ALLOWED_AUC_DROP})"
            )

    # Signal 3: feature drift
    if signals.drift_detected:
        reasons.append("feature drift detected by monitoring/drift_check.py")

    return RetrainDecision(should_retrain=len(reasons) > 0, reasons=reasons)


# --- Pseudocode form, as requested by the assignment brief ---
"""
FUNCTION check_retrain_needed():
    signals = gather_signals()  # query registry, labeled feedback table, latest drift report

    IF days_since_last_training >= 14:
        RETRAIN(reason="stale model")

    IF recent_labeled_auc IS NOT NULL AND (promotion_auc - recent_labeled_auc) >= 0.03:
        RETRAIN(reason="performance regression")

    IF latest_drift_report.any_issue_detected == TRUE:
        RETRAIN(reason="feature drift")

    ELSE:
        NO_ACTION()
"""


if __name__ == "__main__":
    # Demo run with made-up numbers.
    demo_signals = RetrainSignals(
        days_since_last_training=16,
        current_model_promotion_auc=0.842,
        recent_labeled_auc=0.803,
        drift_detected=False,
    )
    decision = should_retrain(demo_signals)
    print(f"should_retrain={decision.should_retrain}")
    for r in decision.reasons:
        print(f"  - {r}")
