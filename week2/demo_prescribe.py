"""
End-to-end demo: model prediction + four prescribed options.

Uses a high-delay shipment from the real open extract when available
so the terminal story matches the dashboard Demo C path.

    python week2/demo_prescribe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week1.config import MODEL_PATH
from week1.dataset_service import dataset_summary, demo_scenarios_from_data
from week1.delay_model import DelayModel
from week2.solver import pure_options, solve_optimal_allocation

FALLBACK_SHIPMENT = {
    "sku": "ARV-GENERIC-TENOFOVIR-DISOPROXIL-FUMARAT",
    "supplier": "CIPLA LIMITED",
    "origin_region": "Asia Pacific",
    "distance_km": 8450,
    "historical_avg_lead_time_days": 128,
    "order_quantity": 20000,
    "unit_cost_usd": 0.12,
    "is_peak_season": True,
}


def _demo_shipment() -> tuple[dict, float, int]:
    demos = demo_scenarios_from_data()
    if demos:
        risky = next((d for d in demos if d["id"] == "risky"), demos[-1])
        values = risky["values"]
        shipment = {
            k: values[k]
            for k in (
                "sku",
                "supplier",
                "origin_region",
                "distance_km",
                "historical_avg_lead_time_days",
                "order_quantity",
                "unit_cost_usd",
                "is_peak_season",
            )
        }
        budget = float(values.get("budget_cap_usd") or 95_000)
        max_delay = int(values.get("max_acceptable_delay_days") or 5)
        return shipment, budget, max_delay
    return FALLBACK_SHIPMENT, 95_000, 5


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Train the model first: python week1/train_model.py\nMissing {MODEL_PATH}")

    model = DelayModel.load(MODEL_PATH)
    shipment, budget, max_delay = _demo_shipment()
    days, prob = model.predict_one(shipment)
    days, prob = round(days, 1), round(prob, 3)
    summary = dataset_summary()

    print()
    print("SupplyPrescript — predict + prescribe demo (real open data)")
    print("=" * 64)
    if summary.get("available"):
        sources = ", ".join(s["label"] for s in summary.get("sources", []))
        print(f"Dataset: {sources} · {summary['n_rows']:,} rows")
        print()
    print("Shipment")
    for key, value in shipment.items():
        print(f"  {key}: {value}")
    print()
    print("Model prediction")
    print(f"  expected delay : {days} days")
    print(f"  P(delay > 3d)  : {prob:.1%}")
    print()

    options = pure_options(
        unit_cost_usd=shipment["unit_cost_usd"],
        order_quantity=shipment["order_quantity"],
        predicted_delay_days=days,
        budget_cap_usd=budget,
    )
    blend = solve_optimal_allocation(
        unit_cost_usd=shipment["unit_cost_usd"],
        order_quantity=shipment["order_quantity"],
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
    print("Say out loud: the model scored this real shipment from the open")
    print("extract; the optimizer then finds the cheapest mix under the delay ceiling.")
    print()


if __name__ == "__main__":
    main()
