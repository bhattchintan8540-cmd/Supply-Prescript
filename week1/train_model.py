"""
Week 1 - trains the delay model against data/shipments.csv and drops the
artifact at the path in week1/config.MODEL_PATH.

    python week1/train_model.py

Also called from week4/retrain.py once decisions start piling up and
drift crosses the threshold. Retraining incorporates eligible resolved
outcomes when their feature snapshots are available.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from week1.config import METRICS_PATH, MODEL_PATH, ROOT_DIR
from week1.delay_model import DelayModel

DATA_PATH = ROOT_DIR / "data" / "shipments.csv"


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return value


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found - run generate_mock_data.py first")

    df = pd.read_csv(DATA_PATH)
    model = DelayModel()
    metrics = model.fit(df)

    clean = _json_safe(metrics)
    for key, value in list(clean.items()):
        if isinstance(value, (int, float)) and key.startswith("n_"):
            clean[key] = int(value)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    importance = model.feature_importance(top_n=10)
    clean["top_features"] = importance
    METRICS_PATH.write_text(json.dumps(clean, indent=2))
    print(f"saved model -> {MODEL_PATH}")
    print(f"saved metrics -> {METRICS_PATH}")
    print(
        {
            k: clean[k]
            for k in (
                "validation_strategy",
                "mae_days",
                "rmse_days",
                "r2_days",
                "baseline_mae_days",
                "mae_lift_bootstrap",
                "auc",
                "pr_auc",
                "baseline_auc",
                "auc_lift_bootstrap",
                "precision",
                "recall",
                "f1",
                "specificity",
                "brier_score",
                "ece",
                "decision_threshold",
                "probability_calibrated",
                "data_is_synthetic",
            )
            if k in clean
        }
    )
    print("top features (regressor):")
    for row in importance:
        print(f"  {row['importance']:.4f}  {row['feature']}")


if __name__ == "__main__":
    main()
