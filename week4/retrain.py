"""
Week 4 - drift detection + retraining with outcome feedback.

Not a scheduler or a Kafka consumer - just a function you'd point cron
(or a GitHub Action, or Retool's scheduled query) at once a week. It:

  1. pulls resolved decisions out of the database
  2. checks how far predicted cost drifted from actual cost
  3. if the average drift crosses RETRAIN_DRIFT_THRESHOLD, rebuilds a
     training frame from shipments.csv PLUS eligible resolved outcomes
     (those with a shipment feature snapshot and an actual delay label)
  4. refits DelayModel and writes the artifact

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

from week1.config import MODEL_PATH, ROOT_DIR, RETRAIN_DRIFT_THRESHOLD
from week1.database import SessionLocal
from week1.delay_model import DelayModel
from week1 import models
from week1.features import CATEGORICAL_COLS, NUMERIC_COLS

DATA_PATH = ROOT_DIR / "data" / "shipments.csv"
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def average_cost_drift(session) -> float | None:
    resolved = session.query(models.Decision).filter(models.Decision.actual_cost_usd.isnot(None)).all()
    if not resolved:
        return None
    errors = [abs(d.actual_cost_usd - d.predicted_cost_usd) / d.predicted_cost_usd for d in resolved if d.predicted_cost_usd]
    return sum(errors) / len(errors) if errors else None


def outcomes_as_training_rows(session) -> pd.DataFrame:
    """Convert resolved decisions with feature snapshots into shipment-like rows.

    Only rows that carry shipment_features_json AND actual_delay_days can
    enter the training set — otherwise we would be inventing features.
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
        row["actual_delay_days"] = decision.actual_delay_days
        # Outcomes are "most recent" observations for temporal splits.
        if decision.resolved_at is not None:
            row["shipment_date"] = decision.resolved_at.date().isoformat()
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=FEATURE_COLS + ["actual_delay_days"])


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

        return {
            "retrained": True,
            "reason": f"drift {drift:.1%} >= threshold {RETRAIN_DRIFT_THRESHOLD:.0%}",
            "drift": drift,
            "metrics": metrics,
            "training_frame": frame_meta,
        }
    finally:
        session.close()


if __name__ == "__main__":
    result = maybe_retrain(force="--force" in sys.argv)
    print(result)
