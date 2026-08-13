"""
Week 5 — end-to-end smoke of the closed loop (no uvicorn required).

Runs: generate data → train → prescribe → write-back → outcome → drift check.
Exits non-zero if any stage fails so demos catch breakage early.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week1 import models
from week1.config import METRICS_PATH, MODEL_PATH, ROOT_DIR
from week1.database import SessionLocal, init_db
from week1.delay_model import DelayModel
from week1.generate_mock_data import build
from week2.solver import pure_options, solve_optimal_allocation
from week4.retrain import average_cost_drift, maybe_retrain

DATA_PATH = ROOT_DIR / "data" / "shipments.csv"

SAMPLE = {
    "sku": "MICROCHIP-A2",
    "supplier": "Delta Cove Electronics",
    "origin_region": "Asia Pacific",
    "distance_km": 9500.0,
    "historical_avg_lead_time_days": 18.0,
    "order_quantity": 6000,
    "unit_cost_usd": 14.2,
    "is_peak_season": True,
}


def main() -> int:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        build().to_csv(DATA_PATH, index=False)
        print(f"[ok] generated {DATA_PATH}")
    else:
        print(f"[ok] using existing {DATA_PATH}")

    from week1.train_model import main as train_main

    train_main()
    assert MODEL_PATH.exists(), "model artifact missing after train"
    assert METRICS_PATH.exists(), "metrics.json missing after train"
    print("[ok] trained model + metrics")

    model = DelayModel.load(MODEL_PATH)
    days, prob = model.predict_one(SAMPLE)
    days, prob = round(days, 1), round(prob, 3)
    print(f"[ok] predict days={days} p={prob}")

    budget = 100_000
    max_delay = 5
    options = pure_options(
        unit_cost_usd=SAMPLE["unit_cost_usd"],
        order_quantity=SAMPLE["order_quantity"],
        predicted_delay_days=days,
        budget_cap_usd=budget,
        predicted_delay_probability=prob,
        max_acceptable_delay_days=max_delay,
    )
    blend = solve_optimal_allocation(
        unit_cost_usd=SAMPLE["unit_cost_usd"],
        order_quantity=SAMPLE["order_quantity"],
        predicted_delay_days=days,
        budget_cap_usd=budget,
        max_acceptable_delay_days=max_delay,
        predicted_delay_probability=prob,
    )
    assert blend["status"] == "Optimal" or blend.get("infeasible")
    print(f"[ok] prescribe status={blend['status']} infeasible={blend.get('infeasible')}")

    init_db()
    session = SessionLocal()
    try:
        chosen = next(o for o in options if o["label"] == "Air Freight")
        no_action = next(o for o in options if o["label"] == "Delay Launch")
        decision = models.Decision(
            shipment_sku=SAMPLE["sku"],
            predicted_delay_days=days,
            predicted_delay_probability=prob,
            options_json=json.dumps(options),
            shipment_features_json=json.dumps(SAMPLE),
            chosen_option_label=chosen["label"],
            predicted_cost_usd=chosen["cost_usd"],
            no_action_cost_usd=no_action["cost_usd"],
            budget_cap_usd=budget,
        )
        session.add(decision)
        session.commit()
        session.refresh(decision)
        decision.actual_cost_usd = round(chosen["cost_usd"] * 1.05, 2)
        decision.actual_delay_days = 1.0
        decision.resolved_at = datetime.now(timezone.utc)
        session.commit()
        print(f"[ok] decision {decision.id} resolved")

        drift = average_cost_drift(session)
        print(f"[ok] drift={drift}")
    finally:
        session.close()

    result = maybe_retrain(force=False)
    print(f"[ok] retrain check: {result.get('reason')}")
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
