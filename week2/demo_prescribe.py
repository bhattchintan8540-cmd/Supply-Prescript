"""
End-to-end demo: model prediction + four prescribed options.

Useful when you cannot share a browser screen but still want to show
the full predict → prescribe story in the terminal.

    python week2/demo_prescribe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week1.config import DEFAULT_BUDGET_USD, DEFAULT_MAX_DELAY_DAYS, MODEL_PATH
from week1.delay_model import DelayModel
from week2.solver import pure_options, solve_optimal_allocation

DEMO_SHIPMENT = {
    "sku": "MICROCHIP-A2",
    "supplier": "Delta Cove Electronics",
    "origin_region": "Asia Pacific",
    "distance_km": 9500,
    "historical_avg_lead_time_days": 18,
    "order_quantity": 6000,
    "unit_cost_usd": 14.2,
    "is_peak_season": True,
}


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Train the model first: python week1/train_model.py\nMissing {MODEL_PATH}")

    model = DelayModel.load(MODEL_PATH)
    days, prob = model.predict_one(DEMO_SHIPMENT)
    days, prob = round(days, 1), round(prob, 3)

    budget = 95_000
    max_delay = 5

    print()
    print("SupplyPrescript — predict + prescribe demo")
    print("=" * 64)
    print("Shipment")
    for key, value in DEMO_SHIPMENT.items():
        print(f"  {key}: {value}")
    print()
    print("Model prediction")
    print(f"  expected delay : {days} days")
    print(f"  P(delay > 3d)  : {prob:.1%}")
    print()

    options = pure_options(
        unit_cost_usd=DEMO_SHIPMENT["unit_cost_usd"],
        order_quantity=DEMO_SHIPMENT["order_quantity"],
        predicted_delay_days=days,
        budget_cap_usd=budget,
    )
    blend = solve_optimal_allocation(
        unit_cost_usd=DEMO_SHIPMENT["unit_cost_usd"],
        order_quantity=DEMO_SHIPMENT["order_quantity"],
        predicted_delay_days=days,
        budget_cap_usd=budget,
        max_acceptable_delay_days=max_delay,
    )
    options.append(
        {
            "label": "Optimizer Recommended Split",
            "cost_usd": blend["total_cost_usd"],
            "resulting_delay_days": blend["weighted_avg_delay_days"],
            "within_budget": blend["within_budget"],
            "allocation_units": blend["allocation_units"],
        }
    )

    print(f"Prescribed options (budget ${budget:,.0f}, max delay {max_delay}d)")
    print("-" * 64)
    for opt in options:
        flag = "OK " if opt["within_budget"] else "OVER"
        print(
            f"  [{flag}] {opt['label']:<28} "
            f"${opt['cost_usd']:>10,.2f}   delay {opt['resulting_delay_days']}d"
        )
        if "allocation_units" in opt:
            parts = [
                f"{qty:.0f} via {name.replace('_', ' ')}"
                for name, qty in opt["allocation_units"].items()
                if qty > 0
            ]
            print(f"         split: {', '.join(parts)}")
    print()
    print("Say out loud: the model flagged peak-season risk; the optimizer")
    print("then finds the cheapest mix that still respects the delay ceiling.")
    print()


if __name__ == "__main__":
    main()
