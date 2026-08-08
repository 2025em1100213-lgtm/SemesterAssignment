#!/usr/bin/env python
"""
Basic latency/throughput measurement for the /predict endpoint.

Sends N requests to a running instance of the API and reports avg / p50 /
p95 / p99 latency plus overall throughput. This is a small, honest
"even from a small run" measurement per the assignment brief - not a
substitute for a real load-testing tool (locust/k6) in production.

Usage:
    uvicorn serving.main:app --port 8000 &
    python scripts/load_test.py --n 200 --url http://127.0.0.1:8000/predict
"""
from __future__ import annotations
import argparse
import json
import statistics
import time

import requests

SAMPLE_PAYLOAD = {
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


def percentile(data, pct):
    data = sorted(data)
    k = (len(data) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def main(url: str, n: int):
    latencies_ms = []
    errors = 0

    start_all = time.perf_counter()
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=SAMPLE_PAYLOAD, timeout=5)
            r.raise_for_status()
        except Exception:
            errors += 1
            continue
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    total_wall_s = time.perf_counter() - start_all

    report = {
        "n_requests": n,
        "errors": errors,
        "total_wall_seconds": round(total_wall_s, 3),
        "throughput_req_per_sec": round(n / total_wall_s, 2) if total_wall_s > 0 else None,
        "avg_latency_ms": round(statistics.mean(latencies_ms), 2) if latencies_ms else None,
        "p50_latency_ms": round(percentile(latencies_ms, 50), 2) if latencies_ms else None,
        "p95_latency_ms": round(percentile(latencies_ms, 95), 2) if latencies_ms else None,
        "p99_latency_ms": round(percentile(latencies_ms, 99), 2) if latencies_ms else None,
    }
    print(json.dumps(report, indent=2))

    with open("artifacts/eval/load_test_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/predict")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    main(args.url, args.n)
