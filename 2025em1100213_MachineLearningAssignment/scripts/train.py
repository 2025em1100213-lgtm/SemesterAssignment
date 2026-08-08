#!/usr/bin/env python
"""
Training pipeline (repeatable):
  load data -> split train/val/test -> engineer features -> train baseline
  -> train candidate -> evaluate both -> apply promotion guardrail
  -> save artifacts (models, registry, eval report, reference feature stats)

Usage:
    python scripts/train.py --config configs/train_config.yaml
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.features import FeatureEngineer, FEATURE_COLUMNS  # noqa: E402
from src.data import load_processed  # noqa: E402

MODEL_BUILDERS = {
    "logistic_regression": lambda p: LogisticRegression(**p),
    "gradient_boosting": lambda p: GradientBoostingClassifier(**p),
}


def evaluate(model, scaler, X, y) -> dict:
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "n_samples": int(len(y)),
        "positive_rate": round(float(np.mean(y)), 4),
    }


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["artifacts"]["model_dir"], exist_ok=True)
    os.makedirs(cfg["artifacts"]["eval_dir"], exist_ok=True)

    # 1. LOAD
    df = load_processed(cfg["data"]["processed_path"])
    target_col = cfg["data"]["target_col"]

    fe = FeatureEngineer()
    X_all = fe.transform(df)
    y_all = df[target_col].values

    # 2. SPLIT (train / val / test)
    test_size = cfg["data"]["test_size"]
    val_size = cfg["data"]["val_size"]
    rs = cfg["data"]["random_state"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=(test_size + val_size), random_state=rs, stratify=y_all
    )
    relative_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=rs, stratify=y_temp
    )

    # Scaler fit ONLY on train (avoid leakage), reused identically for val/test/serving.
    scaler = StandardScaler().fit(X_train)

    results = {}
    trained_models = {}

    for name in ["baseline", "candidate"]:
        spec = cfg["model"][name]
        builder = MODEL_BUILDERS[spec["type"]]
        model = builder(spec["params"])
        model.fit(scaler.transform(X_train), y_train)

        val_metrics = evaluate(model, scaler, X_val, y_val)
        test_metrics = evaluate(model, scaler, X_test, y_test)

        trained_models[name] = model
        results[name] = {
            "model_type": spec["type"],
            "params": spec["params"],
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        print(f"[{name:9s}] val AUC={val_metrics['roc_auc']:.4f} acc={val_metrics['accuracy']:.4f} "
              f"| test AUC={test_metrics['roc_auc']:.4f} acc={test_metrics['accuracy']:.4f}")

    # 3. PROMOTION GUARDRAIL
    guard = cfg["promotion_guardrail"]
    baseline_auc = results["baseline"]["val_metrics"]["roc_auc"]
    candidate_auc = results["candidate"]["val_metrics"]["roc_auc"]

    meets_min_auc = candidate_auc >= guard["min_auc"]
    not_much_worse = (baseline_auc - candidate_auc) <= guard["max_allowed_regression_vs_baseline"]
    promote_candidate = meets_min_auc and not_much_worse

    decision = {
        "baseline_val_auc": baseline_auc,
        "candidate_val_auc": candidate_auc,
        "rule": f"promote candidate if candidate_auc >= {guard['min_auc']} "
                f"AND (baseline_auc - candidate_auc) <= {guard['max_allowed_regression_vs_baseline']}",
        "meets_min_auc": meets_min_auc,
        "not_much_worse_than_baseline": not_much_worse,
        "promoted_model": "candidate" if promote_candidate else "baseline",
    }
    print(f"\nPROMOTION DECISION: {decision['promoted_model']} "
          f"(candidate promoted = {promote_candidate})")

    # 4. SAVE ARTIFACTS
    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    model_dir = cfg["artifacts"]["model_dir"]

    for name, model in trained_models.items():
        joblib.dump(model, os.path.join(model_dir, f"model_{name}.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))

    promoted_name = decision["promoted_model"]
    joblib.dump(trained_models[promoted_name], os.path.join(model_dir, "current_model.joblib"))

    registry_path = cfg["artifacts"]["registry_path"]
    registry_entry = {
        "version": version,
        "promoted_model": promoted_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": results,
        "decision": decision,
        "feature_columns": FEATURE_COLUMNS,
    }
    history = []
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            try:
                history = json.load(f)
                if not isinstance(history, list):
                    history = [history]
            except json.JSONDecodeError:
                history = []
    history.append(registry_entry)
    with open(registry_path, "w") as f:
        json.dump(history, f, indent=2)

    eval_report_path = os.path.join(cfg["artifacts"]["eval_dir"], f"eval_report_{version}.json")
    with open(eval_report_path, "w") as f:
        json.dump(registry_entry, f, indent=2)

    # Markdown summary too (assignment asks for JSON/CSV/Markdown)
    md_path = os.path.join(cfg["artifacts"]["eval_dir"], f"eval_report_{version}.md")
    with open(md_path, "w") as f:
        f.write(f"# Evaluation Report {version}\n\n")
        f.write(f"**Promoted model:** `{promoted_name}`\n\n")
        f.write("| Model | Val AUC | Val Acc | Test AUC | Test Acc |\n|---|---|---|---|---|\n")
        for name in ["baseline", "candidate"]:
            v = results[name]["val_metrics"]
            t = results[name]["test_metrics"]
            f.write(f"| {name} | {v['roc_auc']} | {v['accuracy']} | {t['roc_auc']} | {t['accuracy']} |\n")
        f.write(f"\n**Guardrail rule:** {decision['rule']}\n")

    # Reference feature statistics for drift monitoring (mean/std per feature on TRAIN split)
    feature_stats = {
        col: {"mean": float(X_train[col].mean()), "std": float(X_train[col].std())}
        for col in FEATURE_COLUMNS
    }
    feature_stats["_meta"] = {"version": version, "n_train_rows": int(len(X_train))}
    with open(cfg["artifacts"]["feature_stats_path"], "w") as f:
        json.dump(feature_stats, f, indent=2)

    print(f"\nSaved model artifacts to {model_dir}/")
    print(f"Saved eval report to {eval_report_path} and {md_path}")
    print(f"Updated registry at {registry_path} (version={version})")
    print(f"Saved reference feature stats to {cfg['artifacts']['feature_stats_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()
    main(args.config)
