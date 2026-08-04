"""
Live model demo for presentations.

Compares a few realistic shipments side-by-side so you can show the
audience that the delay model reacts to supplier risk and peak season.

    python week1/demo_model.py

Requires data/delay_model.joblib (run train_model.py first).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week1.config import MODEL_PATH
from week1.delay_model import DelayModel

# Curated stories that make the model "come alive" on stage.
SCENARIOS = [
    {
        "name": "Reliable supplier · off-peak",
        "why": "Baseline — Meridian is the most reliable vendor.",
        "shipment": {
            "sku": "SENSOR-IR",
            "supplier": "Meridian Fasteners",
            "origin_region": "North America",
            "distance_km": 2400,
            "historical_avg_lead_time_days": 9,
            "order_quantity": 4000,
            "unit_cost_usd": 8.5,
            "is_peak_season": False,
        },
    },
    {
        "name": "Same order · peak season",
        "why": "Only the peak flag flips — delay risk should jump.",
        "shipment": {
            "sku": "SENSOR-IR",
            "supplier": "Meridian Fasteners",
            "origin_region": "North America",
            "distance_km": 2400,
            "historical_avg_lead_time_days": 9,
            "order_quantity": 4000,
            "unit_cost_usd": 8.5,
            "is_peak_season": True,
        },
    },
    {
        "name": "Risky supplier · long haul",
        "why": "Delta Cove + Asia Pacific — the model should flag trouble.",
        "shipment": {
            "sku": "MICROCHIP-A2",
            "supplier": "Delta Cove Electronics",
            "origin_region": "Asia Pacific",
            "distance_km": 9500,
            "historical_avg_lead_time_days": 18,
            "order_quantity": 8000,
            "unit_cost_usd": 22.0,
            "is_peak_season": False,
        },
    },
    {
        "name": "Risky supplier · peak + long haul",
        "why": "Worst case — expect the highest delay / probability.",
        "shipment": {
            "sku": "MICROCHIP-A2",
            "supplier": "Delta Cove Electronics",
            "origin_region": "Asia Pacific",
            "distance_km": 9500,
            "historical_avg_lead_time_days": 18,
            "order_quantity": 8000,
            "unit_cost_usd": 22.0,
            "is_peak_season": True,
        },
    },
]


def _bar(prob: float, width: int = 20) -> str:
    filled = int(round(prob * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model not found at {MODEL_PATH}\n"
            "Run these first:\n"
            "  python week1/generate_mock_data.py\n"
            "  python week1/train_model.py"
        )

    model = DelayModel.load(MODEL_PATH)
    print()
    print("SupplyPrescript — delay model demo")
    print("=" * 72)
    print(f"Loaded: {MODEL_PATH.name}")
    print()
    print(f"{'Scenario':<36} {'Days':>6}  {'P(late>3d)':>10}  Risk bar")
    print("-" * 72)

    results = []
    for case in SCENARIOS:
        days, prob = model.predict_one(case["shipment"])
        days = round(days, 1)
        prob = round(prob, 3)
        results.append((case, days, prob))
        print(f"{case['name']:<36} {days:>6.1f}  {prob:>9.1%}  {_bar(prob)}")

    print("-" * 72)
    print()
    print("Talking points while you present:")
    for case, days, prob in results:
        print(f"  • {case['name']}: {days} days, {prob:.0%} late risk — {case['why']}")
    print()
    print("Next: open the full prescribe demo")
    print("  uvicorn week3.main:app --reload")
    print("  → http://127.0.0.1:8000/ui/")
    print()


if __name__ == "__main__":
    main()
