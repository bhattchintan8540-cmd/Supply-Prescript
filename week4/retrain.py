"""
Week 4 - "continuous learning" piece.

Not a scheduler or a Kafka consumer - just a function you'd point cron
(or a GitHub Action, or Retool's scheduled query) at once a week. It:

  1. pulls resolved decisions out of the database
  2. checks how far predicted cost drifted from actual cost
  3. if the average drift crosses RETRAIN_DRIFT_THRESHOLD, retrains the
     model against the current training set

Real-world caveat: a Decision row doesn't carry the full original
shipment feature set, only enough to compute cost drift. A production
version would join back to shipments via a foreign key so resolved
decisions could be folded directly into the training data; for this
build the retrain step re-fits on data/shipments.csv, which is the
honest scope for a portfolio demo of "the loop closes and retraining
gets triggered."
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from week1.config import MODEL_PATH, ROOT_DIR, RETRAIN_DRIFT_THRESHOLD
from week1.database import SessionLocal
from week1.delay_model import DelayModel
from week1 import models

DATA_PATH = ROOT_DIR / "data" / "shipments.csv"


def average_cost_drift(session) -> float | None:
    resolved = session.query(models.Decision).filter(models.Decision.actual_cost_usd.isnot(None)).all()
    if not resolved:
        return None
    errors = [abs(d.actual_cost_usd - d.predicted_cost_usd) / d.predicted_cost_usd for d in resolved if d.predicted_cost_usd]
    return sum(errors) / len(errors) if errors else None


def maybe_retrain(force: bool = False) -> dict:
    session = SessionLocal()
    try:
        drift = average_cost_drift(session)
    finally:
        session.close()

    if drift is None:
        return {"retrained": False, "reason": "no resolved decisions yet", "drift": None}

    if not force and drift < RETRAIN_DRIFT_THRESHOLD:
        return {"retrained": False, "reason": f"drift {drift:.1%} under threshold {RETRAIN_DRIFT_THRESHOLD:.0%}", "drift": drift}

    if not DATA_PATH.exists():
        return {"retrained": False, "reason": f"{DATA_PATH} missing", "drift": drift}

    df = pd.read_csv(DATA_PATH)
    model = DelayModel()
    metrics = model.fit(df, verbose=False)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    return {"retrained": True, "reason": f"drift {drift:.1%} >= threshold {RETRAIN_DRIFT_THRESHOLD:.0%}", "drift": drift, "metrics": metrics}


if __name__ == "__main__":
    result = maybe_retrain(force="--force" in sys.argv)
    print(result)
