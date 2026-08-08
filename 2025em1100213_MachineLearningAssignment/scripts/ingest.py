#!/usr/bin/env python
"""
Simple batch/micro-batch ingestion script.

Simulates the "new daily CSV lands in a directory" pattern:
  1. Scans data/incoming/ for new *.csv files (a real system: S3 prefix / Kafka topic dump)
  2. Validates minimal schema
  3. Appends them to data/processed/training_data.csv
  4. Logs how many rows were ingested and from which file(s) to logs/ingestion.log
  5. Moves processed files to data/incoming/_ingested/ so re-runs don't double count

The real Telco Customer Churn source (see src/data.py) is a single static
export, not a dated warehouse table, so `stage_daily_batches()` below
chunks it into N sequential dated files to simulate what a real daily
ingestion feed would look like landing in data/incoming/ over time.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --stage-from-source data/raw/Telco-Customer-Churn.csv --n-batches 10
"""
from __future__ import annotations
import argparse
import glob
import logging
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data import load_telco_raw, REQUIRED_INTERNAL_COLUMNS  # noqa: E402

INCOMING_DIR = "data/incoming"
INGESTED_DIR = "data/incoming/_ingested"
PROCESSED_PATH = "data/processed/training_data.csv"
LOG_PATH = "logs/ingestion.log"

REQUIRED_COLUMNS = REQUIRED_INTERNAL_COLUMNS

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ingest")


def _validate(df: pd.DataFrame, fname: str) -> bool:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        log.warning(f"SKIP {fname}: missing required columns {missing}")
        return False
    null_frac = df[REQUIRED_COLUMNS].isnull().mean().max()
    if null_frac > 0.5:
        log.warning(f"SKIP {fname}: >50% nulls in at least one required column")
        return False
    return True


def ingest() -> None:
    os.makedirs(INGESTED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

    new_files = sorted(glob.glob(os.path.join(INCOMING_DIR, "*.csv")))
    if not new_files:
        log.info("No new files found in data/incoming/. Nothing to ingest.")
        return

    total_new_rows = 0
    frames = []
    for f in new_files:
        df = pd.read_csv(f)
        if not _validate(df, f):
            continue
        frames.append(df)
        total_new_rows += len(df)
        log.info(f"Read {len(df)} rows from {os.path.basename(f)}")

    if not frames:
        log.info("No valid files to ingest this run.")
        return

    new_data = pd.concat(frames, ignore_index=True)

    if os.path.exists(PROCESSED_PATH):
        existing = pd.read_csv(PROCESSED_PATH)
        before = len(existing)
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["customer_id", "snapshot_date"], keep="last")
        after_dupe_drop = len(combined)
    else:
        before = 0
        combined = new_data
        after_dupe_drop = len(combined)

    combined.to_csv(PROCESSED_PATH, index=False)

    for f in new_files:
        shutil.move(f, os.path.join(INGESTED_DIR, os.path.basename(f)))

    log.info(
        f"INGEST COMPLETE | files={len(new_files)} | new_rows_read={total_new_rows} "
        f"| table_rows_before={before} | table_rows_after={after_dupe_drop} "
        f"| date={datetime.now(timezone.utc).date().isoformat()}"
    )


def stage_daily_batches(source_csv: str, n_batches: int = 10, start_date: str = "2024-01-01",
                         seed: int = 42) -> None:
    """Simulate a daily-arrival feed from the static Telco export: clean it
    once via load_telco_raw(), shuffle deterministically, then split into
    `n_batches` roughly-equal dated CSVs written to data/incoming/. This
    mirrors what a real "one file per day lands from the warehouse export
    job" pipeline would hand to `ingest()`."""
    os.makedirs(INCOMING_DIR, exist_ok=True)

    df = load_telco_raw(source_csv)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    index_chunks = np.array_split(np.arange(len(df)), n_batches)
    base_date = datetime.strptime(start_date, "%Y-%m-%d")

    for i, idx in enumerate(index_chunks):
        chunk = df.iloc[idx]
        date_str = (base_date + timedelta(days=i)).date().isoformat()
        chunk = chunk.copy()
        chunk["snapshot_date"] = date_str
        out_path = os.path.join(INCOMING_DIR, f"customers_{date_str}.csv")
        chunk.to_csv(out_path, index=False)
        log.info(f"Staged {len(chunk)} rows -> {out_path}")

    log.info(f"Staged {len(df)} total rows across {n_batches} simulated daily files "
              f"from source={source_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-from-source", type=str, default=None,
                         help="Path to the raw Telco-Customer-Churn.csv to split into "
                              "simulated daily files before ingesting")
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    args = parser.parse_args()

    if args.stage_from_source:
        stage_daily_batches(args.stage_from_source, n_batches=args.n_batches, start_date=args.start_date)

    ingest()
