#!/usr/bin/env python
"""
Lightweight data quality + drift check.

Compares a "recent batch" of raw customer rows against:
  (a) basic data quality rules (nulls, out-of-range values)
  (b) the reference mean/std saved at training time
      (artifacts/eval/feature_stats.json, written by scripts/train.py)

This is intentionally simple (mean/std relative shift), matching the
assignment's ask for "one lightweight check", not a full KS-test /
population stability index implementation.

Usage:
    python monitoring/drift_check.py --recent-batch data/processed/training_data.csv
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.features import FeatureEngineer, FEATURE_COLUMNS  # noqa: E402

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("logs/monitoring.log"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("drift_check")

REQUIRED_RAW_COLUMNS = [
    "tenure_months", "contract_type", "payment_method", "internet_service",
    "monthly_charges", "total_charges", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
    "multiple_lines", "senior_citizen", "has_partner", "has_dependents",
    "paperless_billing",
]

RELATIVE_MEAN_SHIFT_THRESHOLD = 0.20  # 20% relative change in mean flags drift
NULL_RATE_THRESHOLD = 0.05            # >5% nulls in a required column is a data quality issue


def data_quality_check(df: pd.DataFrame) -> list[str]:
    issues = []
    for col in REQUIRED_RAW_COLUMNS:
        if col not in df.columns:
            issues.append(f"MISSING COLUMN: {col}")
            continue
        null_rate = df[col].isnull().mean()
        if null_rate > NULL_RATE_THRESHOLD:
            issues.append(f"HIGH NULL RATE: {col} = {null_rate:.1%}")

    if "monthly_charges" in df.columns:
        out_of_range = ((df["monthly_charges"] < 0) | (df["monthly_charges"] > 1000)).mean()
        if out_of_range > 0:
            issues.append(f"OUT-OF-RANGE monthly_charges: {out_of_range:.1%} of rows")

    if "tenure_months" in df.columns:
        out_of_range = ((df["tenure_months"] < 0) | (df["tenure_months"] > 600)).mean()
        if out_of_range > 0:
            issues.append(f"OUT-OF-RANGE tenure_months: {out_of_range:.1%} of rows")

    return issues


def drift_check(recent_features: pd.DataFrame, reference_stats: dict) -> list[str]:
    warnings = []
    for col in FEATURE_COLUMNS:
        if col not in reference_stats:
            continue
        ref_mean = reference_stats[col]["mean"]
        recent_mean = float(recent_features[col].mean())
        if ref_mean == 0:
            continue
        rel_shift = abs(recent_mean - ref_mean) / abs(ref_mean)
        if rel_shift > RELATIVE_MEAN_SHIFT_THRESHOLD:
            warnings.append(
                f"DRIFT in '{col}': train_mean={ref_mean:.3f} recent_mean={recent_mean:.3f} "
                f"(relative shift={rel_shift:.1%}, threshold={RELATIVE_MEAN_SHIFT_THRESHOLD:.0%})"
            )
    return warnings


def main(recent_batch_path: str, feature_stats_path: str):
    df = pd.read_csv(recent_batch_path)

    log.info(f"Running data quality check on {len(df)} rows from {recent_batch_path}")
    dq_issues = data_quality_check(df)
    if dq_issues:
        for issue in dq_issues:
            log.warning(f"DATA QUALITY ISSUE: {issue}")
    else:
        log.info("Data quality check passed: no issues found.")

    if not os.path.exists(feature_stats_path):
        log.warning(f"No reference feature stats found at {feature_stats_path}. "
                    f"Run scripts/train.py at least once before drift checking.")
        return

    with open(feature_stats_path) as f:
        reference_stats = json.load(f)

    fe = FeatureEngineer()
    recent_features = fe.transform(df)

    drift_warnings = drift_check(recent_features, reference_stats)
    if drift_warnings:
        for w in drift_warnings:
            log.warning(f"DRIFT WARNING: {w}")
    else:
        log.info("Drift check passed: no significant feature drift detected.")

    summary = {
        "n_rows_checked": int(len(df)),
        "data_quality_issues": dq_issues,
        "drift_warnings": drift_warnings,
        "any_issue_detected": bool(dq_issues or drift_warnings),
    }
    with open("artifacts/eval/monitoring_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Monitoring summary written to artifacts/eval/monitoring_report.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent-batch", type=str, default="data/processed/training_data.csv")
    parser.add_argument("--feature-stats", type=str, default="artifacts/eval/feature_stats.json")
    args = parser.parse_args()
    main(args.recent_batch, args.feature_stats)
