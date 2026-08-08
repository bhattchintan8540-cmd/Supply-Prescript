"""
Live model demo for presentations.

Compares a few realistic shipments side-by-side so you can show the
audience that the delay model reacts to supplier / corridor risk.
Scenarios use vendors that appear in the USAID SCMS open dataset.

    python week1/demo_model.py

Requires data/delay_model.joblib (run train_model.py first).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week1.config import MODEL_PATH
from week1.delay_model import DelayModel

# Curated stories grounded in real SCMS vendors / corridors.
# Trinity Biotech has near-zero historical delay; CIPLA / Aurobindo
# sit on the higher-delay Asia→Africa ARV corridor in the open data.
SCENARIOS = [
    {
        "name": "Trinity Biotech · Europe · off-peak",
        "why": "Low-delay HIV rapid-test vendor on a short corridor.",
        "shipment": {
            "sku": "HRDT-UNI-GOLD-HIV-1-2",
            "supplier": "Trinity Biotech, Plc",
            "origin_region": "Europe",
            "distance_km": 6200,
            "historical_avg_lead_time_days": 78,
            "order_quantity": 400,
            "unit_cost_usd": 1.6,
            "is_peak_season": False,
        },
    },
    {
        "name": "Same order · peak season",
        "why": "Only the peak flag flips — watch probability move.",
        "shipment": {
            "sku": "HRDT-UNI-GOLD-HIV-1-2",
            "supplier": "Trinity Biotech, Plc",
            "origin_region": "Europe",
            "distance_km": 6200,
            "historical_avg_lead_time_days": 78,
            "order_quantity": 400,
            "unit_cost_usd": 1.6,
            "is_peak_season": True,
        },
    },
    {
        "name": "Aurobindo · Asia→Africa · long haul",
        "why": "High-volume Indian ARV supplier — higher historical delay.",
        "shipment": {
            "sku": "ARV-GENERIC-EFAVIRENZ",
            "supplier": "Aurobindo Pharma Limited",
            "origin_region": "Asia Pacific",
            "distance_km": 8300,
            "historical_avg_lead_time_days": 120,
            "order_quantity": 11000,
            "unit_cost_usd": 0.12,
            "is_peak_season": False,
        },
    },
    {
        "name": "CIPLA · Asia→Africa · peak",
        "why": "Highest mean delay among large SCMS vendors in the open data.",
        "shipment": {
            "sku": "ARV-GENERIC-TENOFOVIR-DISOPROXIL-FUMARAT",
            "supplier": "CIPLA LIMITED",
            "origin_region": "Asia Pacific",
            "distance_km": 8450,
            "historical_avg_lead_time_days": 128,
            "order_quantity": 20000,
            "unit_cost_usd": 0.12,
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
            "  python week1/ingest_real_data.py\n"
            "  python week1/train_model.py"
        )

    model = DelayModel.load(MODEL_PATH)
    print()
    print("SupplyPrescript — delay model demo")
    print("=" * 72)
    print(f"Loaded: {MODEL_PATH.name}")
    print()
    print(f"{'Scenario':<42} {'Days':>6}  {'P(late>3d)':>10}  Risk bar")
    print("-" * 72)

    results = []
    for case in SCENARIOS:
        days, prob = model.predict_one(case["shipment"])
        days = round(days, 1)
        prob = round(prob, 3)
        results.append((case, days, prob))
        print(f"{case['name']:<42} {days:>6.1f}  {prob:>9.1%}  {_bar(prob)}")

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
