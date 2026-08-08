"""
Week 1 - offline mock data generator (fallback only).

Prefer the real open-data path:

    python week1/ingest_real_data.py

Use this script only when you need a fully offline / deterministic demo
without network access. It fabricates a plausible three-year shipment
log with supplier reliability, peak season, and a Delta Cove shock.

    python week1/generate_mock_data.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "shipments.csv"
RNG_SEED = 42
sys.path.insert(0, str(ROOT))

SUPPLIERS = {
    # name -> (baseline reliability, region)
    "NovaChip Manufacturing": (0.92, "Asia Pacific"),
    "Redline Components": (0.85, "Asia Pacific"),
    "Baltic Precision Parts": (0.88, "Europe"),
    "Meridian Fasteners": (0.95, "North America"),
    "Delta Cove Electronics": (0.78, "Asia Pacific"),
}
SKUS = ["MICROCHIP-A2", "RESISTOR-PACK", "CASING-ALU", "SENSOR-IR", "PCB-STD-4L"]

REGION_BASE_DISTANCE_KM = {
    "Asia Pacific": 9200,
    "Europe": 6800,
    "North America": 2400,
}


def _month_is_peak(month: int) -> bool:
    # Nov/Dec holiday ramp - matches the seasonality bump used elsewhere
    # in the portfolio (see the MetricMind seed data).
    return month in (11, 12)


def build(n_rows: int = 4000, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    rows = []
    supplier_names = list(SUPPLIERS.keys())

    for i in range(n_rows):
        supplier = py_rng.choice(supplier_names)
        reliability, region = SUPPLIERS[supplier]
        sku = py_rng.choice(SKUS)
        month = py_rng.randint(1, 12)
        peak = _month_is_peak(month)

        distance = REGION_BASE_DISTANCE_KM[region] * rng.uniform(0.85, 1.15)
        avg_lead_time = 7 + distance / 900 + rng.normal(0, 1.2)
        avg_lead_time = max(avg_lead_time, 3)

        qty = int(rng.integers(500, 20_000))
        unit_cost = round(float(rng.uniform(0.8, 40)), 2)

        # Bad-quarter shock: Delta Cove has a rough stretch mid-dataset,
        # this is the pattern the model needs to actually pick up on
        # rather than just memorizing supplier averages.
        shock = 0.0
        if supplier == "Delta Cove Electronics" and i % 7 == 0:
            shock = rng.uniform(4, 11)

        peak_penalty = rng.uniform(1.5, 4.0) if peak else 0.0
        base_delay = max(0.0, rng.normal((1 - reliability) * 18, 2.5))
        actual_delay = round(max(0.0, base_delay + peak_penalty + shock), 1)

        rows.append(
            {
                "sku": sku,
                "supplier": supplier,
                "origin_region": region,
                "distance_km": round(distance, 1),
                "historical_avg_lead_time_days": round(avg_lead_time, 1),
                "order_quantity": qty,
                "unit_cost_usd": unit_cost,
                "is_peak_season": peak,
                "actual_delay_days": actual_delay,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    df = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(df)} mock rows -> {OUT_PATH}")
    print(df["actual_delay_days"].describe())
    print("note: prefer `python week1/ingest_real_data.py` for real open data")

    # Keep the DB in sync so train_model's prefer-DB path still works offline.
    try:
        from week1.ingest_real_data import seed_database

        df = df.copy()
        df["data_source"] = "mock"
        seed_database(df, replace=True)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not seed database ({exc})")


if __name__ == "__main__":
    main()
