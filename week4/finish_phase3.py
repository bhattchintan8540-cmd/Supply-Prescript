"""
Phase 3 completion check. Uses a throwaway database so the demo file is unchanged.

Prescribe, write the choice back, log an outcome, then read ROI and drift status.
Does not force a model refit.

    python week4/finish_phase3.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_tmp = tempfile.TemporaryDirectory(prefix="phase3-")
os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(_tmp.name) / "phase3.db")

from week1.config import (  # noqa: E402
    RETRAIN_DELAY_MAE_DAYS,
    RETRAIN_DRIFT_THRESHOLD,
    RETRAIN_HARD_MISS_RATE,
    RETRAIN_OUTCOME_BRIER,
)

SAMPLE = {
    "sku": "MICROCHIP-A2",
    "supplier": "NovaChip Manufacturing",
    "origin_region": "Asia Pacific",
    "distance_km": 8800,
    "historical_avg_lead_time_days": 16,
    "order_quantity": 6000,
    "unit_cost_usd": 14.2,
    "is_peak_season": False,
}


def main() -> int:
    from fastapi.testclient import TestClient

    from week3.main import app

    errors: list[str] = []
    with TestClient(app) as client:
        health = client.get("/health")
        if health.status_code != 200:
            errors.append(f"GET /health returned {health.status_code}")

        prescribed = client.post("/prescribe", json={"shipment": SAMPLE})
        if prescribed.status_code != 200:
            errors.append(f"POST /prescribe returned {prescribed.status_code}")
            _report(errors)
            return 1
        body = prescribed.json()
        chosen = next(opt for opt in body["options"] if opt["label"] == "Air Freight")
        created = client.post(
            "/decisions",
            json={
                "shipment_sku": SAMPLE["sku"],
                "predicted_delay_days": body["prediction"]["predicted_delay_days"],
                "predicted_delay_probability": body["prediction"]["predicted_delay_probability"],
                "options": body["options"],
                "chosen_option_label": chosen["label"],
                "budget_cap_usd": body["budget_cap_usd"],
                "shipment_features": SAMPLE,
                "no_action_cost_usd": body["no_action_cost_usd"],
            },
        )
        if created.status_code != 201:
            errors.append(f"POST /decisions returned {created.status_code}")
            _report(errors)
            return 1
        decision = created.json()
        actual_cost = min(decision["predicted_cost_usd"] * 1.05, body["no_action_cost_usd"] * 0.9)
        outcome = client.patch(
            f"/decisions/{decision['id']}/outcome",
            json={"actual_cost_usd": actual_cost, "actual_delay_days": 1.0},
        )
        if outcome.status_code != 200 or outcome.json().get("is_resolved") is not True:
            errors.append("outcome was not recorded")

        roi = client.get("/decisions/roi")
        if roi.status_code != 200 or roi.json().get("decisions_with_counterfactual", 0) < 1:
            errors.append("ROI versus Delay Launch was not available")

        status = client.get("/phase3/status")
        if status.status_code != 200:
            errors.append(f"GET /phase3/status returned {status.status_code}")
        else:
            snapshot = status.json()
            if snapshot.get("resolved_decisions", 0) < 1:
                errors.append("status did not see the resolved decision")
            if "cost_mape" not in snapshot.get("thresholds", {}):
                errors.append("drift thresholds missing")

        skipped = client.post("/phase3/retrain", json={"force": False})
        if skipped.status_code != 200:
            errors.append(f"POST /phase3/retrain returned {skipped.status_code}")

    _report(errors, roi.json() if roi.status_code == 200 else None, status.json() if status.status_code == 200 else None)
    from week1.database import engine

    engine.dispose()
    if errors:
        print("PHASE 3 CHECKS FAILED")
        for item in errors:
            print(f"  - {item}")
        return 1
    print("WEEK4 PHASE3 COMPLETE OK")
    return 0


def _report(errors, roi=None, status=None) -> None:
    print("PARAMETERS")
    print(f"  drift cost_mape threshold={RETRAIN_DRIFT_THRESHOLD:.0%}")
    print(f"  delay_mae threshold={RETRAIN_DELAY_MAE_DAYS}d")
    print(f"  hard_miss threshold={RETRAIN_HARD_MISS_RATE:.0%}")
    print(f"  outcome_brier threshold={RETRAIN_OUTCOME_BRIER}")
    if status:
        print(
            f"  resolved={status.get('resolved_decisions')}  "
            f"cost_mape={status.get('cost_mape')}  "
            f"should_retrain={status.get('should_retrain')}"
        )
    if roi:
        print(
            f"  roi_pct={roi.get('avg_roi_pct')}  "
            f"avoided_loss={roi.get('avg_avoided_loss_usd')}"
        )
    print("LIMITATIONS")
    print("  ROI needs a stored Delay Launch cost. Cost error is not ROI.")
    print("  Decisions without a feature snapshot cannot become training rows.")
    print("  Drift thresholds are portfolio settings, not a calibrated operations policy.")
    print("  This check does not force a refit.")
    print("DEBUG")
    print("  If prescribe returns 503, run python week1/train_model.py")
    print("  If ROI is empty, the decision was saved without no_action_cost_usd")
    print("  Retrain from the dashboard only when a signal is over its threshold.")


if __name__ == "__main__":
    raise SystemExit(main())
