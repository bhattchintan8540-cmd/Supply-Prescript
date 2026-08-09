"""
Week 4 - drift detection + retraining with outcome feedback.

Not a scheduler or a Kafka consumer - just a function you'd point cron
(or a GitHub Action, or Retool's scheduled query) at once a week. It:

  1. pulls resolved decisions out of the database
  2. checks cost forecast drift AND prediction distribution drift
  3. if either signal crosses threshold, rebuilds a training frame from
     shipments.csv PLUS eligible resolved outcomes
     (those with a shipment feature snapshot and an actual delay label)
  4. refits DelayModel and writes the artifact + metrics.json

Drift signals (ruthless mentor version)
--------------------------------------
- Cost MAPE: |actual − predicted| / predicted  (business money error)
- Delay MAE: |actual_delay − predicted_delay| (label error)
- Probability Brier-like: (actual_late − predicted_prob)² mean
- Hard miss rate: share where hard late label disagrees with P≥0.5

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
from week1.features import CATEGORICAL_COLS, DELAY_FLAG_THRESHOLD_DAYS, NUMERIC_COLS

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


def prediction_drift_diagnostics(session) -> dict:
    """Multi-signal drift: cost MAPE is not enough when P(delay) drives spend."""
    resolved = (
        session.query(models.Decision)
        .filter(models.Decision.actual_cost_usd.isnot(None))
        .all()
    )
    if not resolved:
        return {"n_resolved": 0}

    cost_errors = [
        abs(d.actual_cost_usd - d.predicted_cost_usd) / d.predicted_cost_usd
        for d in resolved
        if d.predicted_cost_usd
    ]
    delay_rows = [
        d
        for d in resolved
        if d.actual_delay_days is not None and d.predicted_delay_days is not None
    ]
    delay_mae = (
        sum(abs(d.actual_delay_days - d.predicted_delay_days) for d in delay_rows) / len(delay_rows)
        if delay_rows
        else None
    )
    prob_rows = [
        d
        for d in resolved
        if d.actual_delay_days is not None and d.predicted_delay_probability is not None
    ]
    brier = None
    hard_miss_rate = None
    if prob_rows:
        sq = []
        misses = 0
        for d in prob_rows:
            late = 1.0 if d.actual_delay_days > DELAY_FLAG_THRESHOLD_DAYS else 0.0
            p = float(d.predicted_delay_probability)
            sq.append((late - p) ** 2)
            pred_late = 1.0 if p >= 0.5 else 0.0
            if pred_late != late:
                misses += 1
        brier = sum(sq) / len(sq)
        hard_miss_rate = misses / len(prob_rows)

    return {
        "n_resolved": len(resolved),
        "cost_mape": sum(cost_errors) / len(cost_errors) if cost_errors else None,
        "delay_mae_days": delay_mae,
        "outcome_brier": brier,
        "hard_miss_rate": hard_miss_rate,
    }


def should_retrain(diagnostics: dict, force: bool = False) -> tuple[bool, str]:
    if force:
        return True, "forced"
    cost = diagnostics.get("cost_mape")
    if cost is None:
        return False, "no resolved decisions yet"
    reasons = []
    if cost >= RETRAIN_DRIFT_THRESHOLD:
        reasons.append(f"cost MAPE {cost:.1%} ≥ {RETRAIN_DRIFT_THRESHOLD:.0%}")
    hard = diagnostics.get("hard_miss_rate")
    if hard is not None and hard >= 0.40:
        reasons.append(f"hard miss rate {hard:.1%} ≥ 40%")
    brier = diagnostics.get("outcome_brier")
    if brier is not None and brier >= 0.30:
        reasons.append(f"outcome Brier {brier:.3f} ≥ 0.30")
    delay_mae = diagnostics.get("delay_mae_days")
    if delay_mae is not None and delay_mae >= 3.0:
        reasons.append(f"delay MAE {delay_mae:.2f}d ≥ 3.0d")
    if reasons:
        return True, "; ".join(reasons)
    return False, (
        f"drift under thresholds (cost MAPE {cost:.1%}, "
        f"hard miss {hard}, Brier {brier}, delay MAE {delay_mae})"
    )


def outcomes_as_training_rows(session) -> pd.DataFrame:
    """Convert resolved decisions with feature snapshots into shipment-like rows.

    Only rows that carry shipment_features_json AND actual_delay_days AND
    resolved_at can enter the training set — otherwise we would invent
    features or break temporal splits with null dates.

    IMPORTANT: prefer `created_at` (decision time) as shipment_date, not
    resolved_at. Resolution time would place labels in the future relative
    to when features were observed and pollute temporal validation.
    """
    resolved = (
        session.query(models.Decision)
        .filter(models.Decision.actual_cost_usd.isnot(None))
        .filter(models.Decision.actual_delay_days.isnot(None))
        .filter(models.Decision.shipment_features_json.isnot(None))
        .filter(models.Decision.resolved_at.isnot(None))
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
        stamp = decision.created_at or decision.resolved_at
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
    for col in base.columns:
        if col not in outcomes.columns:
            outcomes[col] = None
    combined = pd.concat([base, outcomes[base.columns]], ignore_index=True)
    return combined, meta


def maybe_retrain(force: bool = False) -> dict:
    init_db()
    session = SessionLocal()
    try:
        diagnostics = prediction_drift_diagnostics(session)
        drift = diagnostics.get("cost_mape")
        do_it, reason = should_retrain(diagnostics, force=force)
        if not do_it:
            return {
                "retrained": False,
                "reason": reason,
                "drift": drift,
                "diagnostics": diagnostics,
            }

        if not DATA_PATH.exists():
            return {
                "retrained": False,
                "reason": f"{DATA_PATH} missing",
                "drift": drift,
                "diagnostics": diagnostics,
            }

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
        clean["drift_diagnostics"] = _json_safe(diagnostics)
        METRICS_PATH.write_text(json.dumps(clean, indent=2))

        return {
            "retrained": True,
            "reason": reason,
            "drift": drift,
            "diagnostics": diagnostics,
            "metrics": metrics,
            "training_frame": frame_meta,
            "metrics_path": str(METRICS_PATH),
        }
    finally:
        session.close()


if __name__ == "__main__":
    result = maybe_retrain(force="--force" in sys.argv)
    print(result)
