"""
Week 4 - drift detection + retraining with outcome feedback.

Not a scheduler or a Kafka consumer - just a function you'd point cron
(or a GitHub Action, or Retool's scheduled query) at once a week. It:

  1. pulls resolved decisions out of the database
  2. checks cost MAPE, delay MAE, hard-miss rate, and outcome Brier
  3. if any signal crosses its threshold, rebuilds a training
     frame from shipments.csv (or the Shipment table) PLUS eligible
     resolved outcomes (feature snapshot + actual delay label)
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

from urllib.error import URLError
from urllib.request import Request, urlopen

from week1.config import (
    METRICS_PATH,
    MODEL_PATH,
    RETRAIN_DELAY_MAE_DAYS,
    RETRAIN_DRIFT_THRESHOLD,
    RETRAIN_HARD_MISS_RATE,
    RETRAIN_OUTCOME_BRIER,
    ROOT_DIR,
)
from week1.database import SessionLocal, init_db
from week1.delay_model import DelayModel
from week1 import models
from week1.features import CATEGORICAL_COLS, DELAY_FLAG_THRESHOLD_DAYS, NUMERIC_COLS

DATA_PATH = ROOT_DIR / "data" / "shipments.csv"
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
API_RELOAD_URL = "http://127.0.0.1:8000/model/reload"


def _reload_running_api() -> bool:
    """Best-effort: drop the in-memory model if uvicorn is already up."""
    try:
        req = Request(API_RELOAD_URL, method="POST", data=b"")
        with urlopen(req, timeout=1.5) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except (URLError, TimeoutError, OSError):
        return False


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


def outcome_drift_signals(session) -> dict | None:
    """Multi-signal drift on resolved decisions.

    Cost MAPE can look healthy while P(delay) — which scales expected
    holding in the MILP — has gone stale. Retrain if *any* signal fires.
    """
    resolved = (
        session.query(models.Decision)
        .filter(models.Decision.actual_cost_usd.isnot(None))
        .all()
    )
    if not resolved:
        return None

    cost_errors = [
        abs(d.actual_cost_usd - d.predicted_cost_usd) / d.predicted_cost_usd
        for d in resolved
        if d.predicted_cost_usd
    ]
    delay_pairs = [
        (d.predicted_delay_days, d.actual_delay_days)
        for d in resolved
        if d.predicted_delay_days is not None and d.actual_delay_days is not None
    ]
    delay_abs = [abs(pred - actual) for pred, actual in delay_pairs]
    brier_terms = []
    for d in resolved:
        if d.predicted_delay_probability is None or d.actual_delay_days is None:
            continue
        y = 1.0 if d.actual_delay_days > DELAY_FLAG_THRESHOLD_DAYS else 0.0
        p = min(1.0, max(0.0, float(d.predicted_delay_probability)))
        brier_terms.append((p - y) ** 2)

    cost_mape = sum(cost_errors) / len(cost_errors) if cost_errors else None
    delay_mae = sum(delay_abs) / len(delay_abs) if delay_abs else None
    hard_miss_rate = (
        sum(1 for err in delay_abs if err > DELAY_FLAG_THRESHOLD_DAYS) / len(delay_abs)
        if delay_abs
        else None
    )
    outcome_brier = sum(brier_terms) / len(brier_terms) if brier_terms else None

    triggers = []
    if cost_mape is not None and cost_mape >= RETRAIN_DRIFT_THRESHOLD:
        triggers.append("cost_mape")
    if delay_mae is not None and delay_mae >= RETRAIN_DELAY_MAE_DAYS:
        triggers.append("delay_mae")
    if hard_miss_rate is not None and hard_miss_rate >= RETRAIN_HARD_MISS_RATE:
        triggers.append("hard_miss_rate")
    if outcome_brier is not None and outcome_brier >= RETRAIN_OUTCOME_BRIER:
        triggers.append("outcome_brier")

    return {
        "n_resolved": len(resolved),
        "cost_mape": cost_mape,
        "delay_mae": delay_mae,
        "hard_miss_rate": hard_miss_rate,
        "outcome_brier": outcome_brier,
        "triggers": triggers,
        "should_retrain": bool(triggers),
    }


def average_cost_drift(session) -> float | None:
    signals = outcome_drift_signals(session)
    if signals is None:
        return None
    return signals["cost_mape"]


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
        feature_date = features.get("shipment_date")
        stamp = decision.created_at or decision.resolved_at
        if feature_date:
            row["shipment_date"] = str(feature_date)[:10]
        elif stamp is not None:
            row["shipment_date"] = stamp.date().isoformat()
        else:
            continue
        row["actual_delay_days"] = decision.actual_delay_days
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=FEATURE_COLS + ["actual_delay_days", "shipment_date"])


def _shipments_from_orm(session) -> pd.DataFrame:
    rows = session.query(models.Shipment).all()
    if not rows:
        return pd.DataFrame()
    records = []
    for row in rows:
        records.append(
            {
                "shipment_date": row.shipment_date,
                "sku": row.sku,
                "supplier": row.supplier,
                "origin_region": row.origin_region,
                "distance_km": row.distance_km,
                "historical_avg_lead_time_days": row.historical_avg_lead_time_days,
                "order_quantity": row.order_quantity,
                "unit_cost_usd": row.unit_cost_usd,
                "is_peak_season": row.is_peak_season,
                "actual_delay_days": row.actual_delay_days,
            }
        )
    return pd.DataFrame(records)


def build_training_frame(session) -> tuple[pd.DataFrame, dict]:
    """Shipments CSV (or ORM seed) ∪ eligible resolved outcomes."""
    source = "csv"
    if DATA_PATH.exists():
        base = pd.read_csv(DATA_PATH)
    else:
        base = _shipments_from_orm(session)
        source = "orm"
        if base.empty:
            raise FileNotFoundError(
                f"{DATA_PATH} missing and shipments table is empty — generate mock data first"
            )
    outcomes = outcomes_as_training_rows(session)
    meta = {
        "base_rows": len(base),
        "base_source": source,
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
        signals = outcome_drift_signals(session)
        if signals is None:
            return {"retrained": False, "reason": "no resolved decisions yet", "drift": None, "signals": None}

        if not force and not signals["should_retrain"]:
            return {
                "retrained": False,
                "reason": (
                    f"signals under thresholds "
                    f"(cost MAPE {signals['cost_mape']}, delay MAE {signals['delay_mae']}, "
                    f"hard-miss {signals['hard_miss_rate']}, Brier {signals['outcome_brier']})"
                ),
                "drift": signals["cost_mape"],
                "signals": signals,
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
        METRICS_PATH.write_text(json.dumps(clean, indent=2))
        api_reloaded = _reload_running_api()

        return {
            "retrained": True,
            "reason": (
                "forced" if force else f"triggers={signals['triggers']}"
            ),
            "drift": signals["cost_mape"],
            "signals": signals,
            "metrics": metrics,
            "training_frame": frame_meta,
            "metrics_path": str(METRICS_PATH),
            "api_reloaded": api_reloaded,
        }
    finally:
        session.close()


if __name__ == "__main__":
    result = maybe_retrain(force="--force" in sys.argv)
    print(result)
