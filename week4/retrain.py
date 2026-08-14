"""
Week 4 - drift detection + retraining with outcome feedback.

Not a scheduler or a Kafka consumer - just a function you'd point cron
(or a GitHub Action, or Retool's scheduled query) at once a week. It:

  1. pulls resolved decisions out of the database
  2. checks how far predicted cost drifted from actual cost
  3. if the average drift crosses RETRAIN_DRIFT_THRESHOLD, rebuilds a
     training frame from shipments.csv PLUS eligible resolved outcomes
     (those with a shipment feature snapshot and an actual delay label)
  4. refits DelayModel and writes the artifact + metrics.json

This is closed-loop learning when feature snapshots were stored at
decision time. Decisions without shipment_features_json still contribute
to the drift trigger only — documented honestly rather than over-claimed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from week1.config import METRICS_PATH, MODEL_PATH, ROOT_DIR, RETRAIN_DRIFT_THRESHOLD
from week1.database import SessionLocal, init_db
from week1.delay_model import DelayModel
from week1 import models
from week1.features import CATEGORICAL_COLS, NUMERIC_COLS

DATA_PATH = ROOT_DIR / "data" / "shipments.csv"
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


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


def average_cost_drift(session) -> float | None:
    resolved = session.query(models.Decision).filter(models.Decision.actual_cost_usd.isnot(None)).all()
    if not resolved:
        return None
    errors = [abs(d.actual_cost_usd - d.predicted_cost_usd) / d.predicted_cost_usd for d in resolved if d.predicted_cost_usd]
    return sum(errors) / len(errors) if errors else None


def outcomes_as_training_rows(session) -> pd.DataFrame:
    """Convert resolved decisions with feature snapshots into shipment-like rows.

    Only rows that carry shipment_features_json AND actual_delay_days AND
    a usable timestamp can enter the training set — otherwise we would
    invent features or break temporal splits with null dates.

    Prefer `created_at` (decision time) as shipment_date, not resolved_at.
    Resolution time would place the label in the future relative to when
    the shipment was actually decided, which leaks hold-out order.
    """
    resolved = (
        session.query(models.Decision)
        .filter(models.Decision.actual_cost_usd.isnot(None))
        .filter(models.Decision.actual_delay_days.isnot(None))
        .filter(models.Decision.shipment_features_json.isnot(None))
        .all()
    )
    rows = []
    for decision in resolved:
        try:
            features = json.loads(decision.shipment_features_json)
        except json.JSONDecodeError:
            continue
        if not all(col in features for col in FEATURE_COLS):
            continue
        row = {col: features[col] for col in FEATURE_COLS}
        stamp = decision.created_at or decision.resolved_at
        if stamp is None:
            continue
        row["actual_delay_days"] = decision.actual_delay_days
        row["shipment_date"] = stamp.date().isoformat()
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=FEATURE_COLS + ["actual_delay_days", "shipment_date"])


def build_training_frame(session) -> tuple[pd.DataFrame, dict]:
    """Shipments CSV ∪ eligible resolved outcomes."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(str(DATA_PATH))
    base = pd.read_csv(DATA_PATH)
    outcomes = outcomes_as_training_rows(session)
    meta = {
        "base_rows": len(base),
        "outcome_rows_added": len(outcomes),
    }
    if outcomes.empty:
        return base, meta
    # Align columns; outcome rows may lack optional training-only fields.
    for col in base.columns:
        if col not in outcomes.columns:
            outcomes[col] = None
    combined = pd.concat([base, outcomes[base.columns]], ignore_index=True)
    return combined, meta


def maybe_retrain(force: bool = False) -> dict:
    init_db()
    session = SessionLocal()
    try:
        drift = average_cost_drift(session)
        if drift is None:
            return {"retrained": False, "reason": "no resolved decisions yet", "drift": None}

        if not force and drift < RETRAIN_DRIFT_THRESHOLD:
            return {
                "retrained": False,
                "reason": f"drift {drift:.1%} under threshold {RETRAIN_DRIFT_THRESHOLD:.0%}",
                "drift": drift,
            }

        if not DATA_PATH.exists():
            return {"retrained": False, "reason": f"{DATA_PATH} missing", "drift": drift}

        df, frame_meta = build_training_frame(session)
        model = DelayModel()
        metrics = model.fit(df, verbose=False)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_PATH)

        clean = _json_safe(metrics)
        for key, value in list(clean.items()):
            if isinstance(value, (int, float)) and key.startswith("n_"):
                clean[key] = int(value)
        clean["top_features"] = model.feature_importance(top_n=10)
        clean["training_frame"] = frame_meta
        METRICS_PATH.write_text(json.dumps(clean, indent=2))

        return {
            "retrained": True,
            "reason": f"drift {drift:.1%} >= threshold {RETRAIN_DRIFT_THRESHOLD:.0%}",
            "drift": drift,
            "metrics": metrics,
            "training_frame": frame_meta,
            "metrics_path": str(METRICS_PATH),
        }
    finally:
        session.close()


if __name__ == "__main__":
    result = maybe_retrain(force="--force" in sys.argv)
    print(result)
